"""
src/inventory.py — Classical post-processing layer for the iterative hybrid loop.

Implements qubo_formulation.md §§13–15:
- 4-day minimum ullage buffer check using p_{j,m} (the full blocking duration).
- Identification of infeasible (vessel, machine, slot) triples — never entire vessels.
- P3 penalty injection via qubo_builder._add_h_cuts, respecting Eq. (16).
- Orchestration of the full iterative loop (K ≤ 10 iterations, Eq. 17).

Full tank inventory simulation is out of scope.  The stock check is an aggregate
viability gate: completion_slot = start_slot + p_j, where p_{j,m} already encodes
the total pipeline-blocking window (physical_constraints.md §2).

Nothing in this module touches the BQM internals or the sampler directly.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, FrozenSet, List, Set, Tuple

import pandas as pd

from src.config import CONFLICT_SET_R, DAILY_INFLOW_M3, MAX_ITERATIONS, MIN_ULLAGE_DAYS, N_TANKS, PENALTY_ALPHA, PENALTY_BETA, TANK_CAPACITY_M3, INITIAL_TERMINAL_STOCK_M3, SLOT_HOURS, T
from src.qubo_builder import build_qubo, calibrate_penalties
from src.solver import check_feasibility, decode_schedule, run_solver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IterativeResult:
    """
    Container for the output of :func:`run_iterative_loop`.

    Attributes:
        schedule:      Best schedule DataFrame found, from
                       :func:`src.solver.decode_schedule`.  May reflect an
                       infeasible solution if the loop did not converge.
        converged:     ``True`` if F^(k) = ∅ before hitting MAX_ITERATIONS.
                       ``False`` triggers human-planner intervention flag.
        iterations:    Number of solver calls executed (1-based).
        feasibility:   Dict returned by :func:`src.solver.check_feasibility`
                       for the best schedule.
        all_cuts:      Complete set of (vessel_id, machine, slot) triples
                       penalised across all iterations.
        oversaturated: ``True`` if the nomination set is volumetrically
                       impossible — Σ p_j exceeds the planning horizon.
                       When True, convergence is impossible regardless of
                       penalty calibration.
    """

    schedule: pd.DataFrame
    converged: bool
    iterations: int
    feasibility: Dict[str, Any]
    all_cuts: Set[Tuple[str, int, int]]
    oversaturated: bool


# ---------------------------------------------------------------------------
# Over-saturation diagnostic
# ---------------------------------------------------------------------------


def _is_oversaturated(
    nominations_df: pd.DataFrame,
    horizon_slots: int = T,
) -> bool:
    """
    Return True if the nomination set physically cannot fit in the horizon.

    Uses p_j (the full blocking duration per vessel).  If the sum of all
    processing times exceeds *horizon_slots*, no sequential schedule exists —
    convergence is impossible and the nomination set must be reduced before
    resubmitting.

    Args:
        nominations_df: Nomination DataFrame from
            :func:`src.config.generate_nominations`, containing ``p_j``.
        horizon_slots: Total number of macro-slots in the planning horizon.
            Defaults to ``src.config.T``.  Pass an explicit value when using
            a different slot duration (e.g. ``horizon_days * (24 // slot_hours)``).

    Returns:
        ``True`` if Σ p_j > horizon_slots.
    """
    total_p_j: int = int(nominations_df["p_j"].sum())
    if total_p_j > horizon_slots:
        logger.error(
            "Nomination set is over-saturated: Σ p_j = %d > T = %d.  "
            "The planning horizon cannot accommodate all vessels sequentially.  "
            "Reduce the nomination set.",
            total_p_j,
            horizon_slots,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Buffer violation detector — §13 (inventory cuts)
# ---------------------------------------------------------------------------


def check_worst_case_overlaps(
    schedule_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    nominations_df: pd.DataFrame,
    slot_duration_hours: int = SLOT_HOURS,
    min_ullage_days: int = MIN_ULLAGE_DAYS,
    initial_terminal_stock_m3: float = INITIAL_TERMINAL_STOCK_M3,
    n_tanks: int = N_TANKS,
    tank_capacity_m3: float = TANK_CAPACITY_M3,
    daily_inflow_m3: float = DAILY_INFLOW_M3,
) -> Set[Tuple[str, int, int]]:
    """
    Check if the current schedule violates the 4-day minimum ullage buffer.

    The terminal stock evolves as a saw-tooth curve:
    - Rises continuously at the terminal-level daily inflow rate (daily_inflow_m3).
    - Drops instantly by volume_m3 when a vessel completes loading.

    The check validates that between the start of the first loading and the
    completion of the last vessel, the stock never exceeds the safe threshold.
    After the last vessel completes, stock level is unconstrained.

    The safe threshold is:
        safe_threshold = max_capacity - (total_daily_inflow × min_ullage_days)

    This ensures at least min_ullage_days of free space (buffer) exists while
    vessels are actively loading.

    Cut granularity — temporal slots, not individual vessel assignments:
    When stock(t) > safe_threshold at slot t, ALL feasible-slot variables
    x_{j,m,t} for that specific t are flagged — regardless of which vessel
    is currently active. This prevents the solver from escaping a cut by
    moving the offending vessel to an adjacent slot or substituting a
    different vessel into the same slot t, both of which would produce the
    same ullage violation.

    Algorithm:
    1. Find the completion time of the LAST vessel (worst-case).
    2. For each slot t from 0 to last_completion_slot:
       a. Compute projected stock(t) considering:
          - Initial stock + cumulative inflow up to t
          - Minus volumes of vessels completed by t
       b. If stock(t) > safe_threshold, flag ALL (vessel_id, machine, t)
          triples present in variables_df for slot t.
    3. Only check while vessels are active. After the last one completes, no violations.

    Args:
        schedule_df:    Decoded schedule from :func:`src.solver.decode_schedule`,
                        containing columns ``vessel_id``, ``start_slot``, ``p_j``.
        variables_df:   Feasible-slot DataFrame from
                        :func:`src.preprocessing.compute_feasible_slots`,
                        with columns ``vessel_id``, ``machine``, ``slot``.
        nominations_df: Nomination DataFrame from
                        :func:`src.config.generate_nominations`,
                        with columns ``vessel_id``, ``volume_m3``.
        slot_duration_hours: Duration of one macro-slot in hours.  Defaults to
            ``src.config.SLOT_HOURS``.  Pass per-request value to support
            different time discretisations.
        min_ullage_days: Minimum ullage buffer in calendar days.  Defaults to
            ``src.config.MIN_ULLAGE_DAYS``.  Pass per-request value to override.
        initial_terminal_stock_m3: Initial total crude in tanks at slot 0 (m³).
            Defaults to ``src.config.INITIAL_TERMINAL_STOCK_M3``.
        n_tanks: Number of storage tanks.  Defaults to ``src.config.N_TANKS``.
        tank_capacity_m3: Capacity of a single tank (m³).
            Defaults to ``src.config.TANK_CAPACITY_M3``.
        daily_inflow_m3: Terminal-level daily crude inflow from upstream
            fields (m³/day).  Defaults to ``src.config.DAILY_INFLOW_M3``.
            This is the aggregate inflow to the tank farm, NOT the
            per-vessel shipper inflow used for ESD priority.

    Returns:
        Set of ``(vessel_id, machine, slot)`` triples to pass to
        :func:`src.qubo_builder.build_qubo` as the *cuts* argument.
        Each triple corresponds to a QUBO variable x_{vessel_id,machine,slot}
        for slot t where stock(t) > safe_threshold.
        Empty set means no violations — no overflow risk detected.
    """
    if schedule_df.empty:
        return set()

    slots_per_day: float = 24.0 / slot_duration_hours
    max_terminal_capacity_m3: float = n_tanks * tank_capacity_m3

    # Per-vessel lookup table from nominations
    vessel_volume: Dict[str, float] = {
        str(row["vessel_id"]): float(row["volume_m3"])
        for _, row in nominations_df.iterrows()
    }

    # Terminal-level daily inflow from upstream fields (configured by the user).
    # NOT the per-vessel daily_inflow_m3, which is a shipper-level field used
    # only for ESD priority weight calculation.
    total_daily_inflow: float = daily_inflow_m3

    # Safe threshold: tanks must never exceed this level.
    # Keeps min_ullage_days × total_daily_inflow of free space at all times.
    safe_threshold: float = max_terminal_capacity_m3 - (total_daily_inflow * min_ullage_days)

    # Build schedule lookup: vessel_id -> (start_slot, completion_slot)
    # completion_slot = start_slot + p_j: p_{j,m} already encodes total blocking duration.
    sched_lookup: Dict[str, Tuple[int, int]] = {}
    max_completion_slot: int = 0
    for _, row in schedule_df.iterrows():
        vid: str = str(row["vessel_id"])
        t_start: int = int(row["start_slot"])
        p_j: int = int(row["p_j"])
        t_completion: int = t_start + p_j
        sched_lookup[vid] = (t_start, t_completion)
        max_completion_slot = max(max_completion_slot, t_completion)

    flagged: Set[Tuple[str, int, int]] = set()

    # Only check up to the last vessel's completion. After that, stock level is unconstrained.
    if max_completion_slot == 0:
        logger.info("No vessels scheduled, returning empty cuts.")
        return set()

    logger.info(
        "Buffer check: initial_stock=%.0f m³, daily_inflow=%.0f m³/day, "
        "safe_threshold=%.0f m³, max_completion_slot=%d, "
        "scheduled_vessels=%d",
        initial_terminal_stock_m3,
        total_daily_inflow,
        safe_threshold,
        max_completion_slot,
        len(sched_lookup),
    )

    # Debug: log all scheduled vessels
    for vid, (start_slot, completion_slot) in sched_lookup.items():
        volume = vessel_volume.get(vid, 0.0)
        logger.info(
            "  Vessel %s: start=%d, completion=%d, volume=%.0f m³",
            vid, start_slot, completion_slot, volume
        )

    # Walk through time from slot 0 to max_completion_slot and check buffer at each step.
    for t in range(0, max_completion_slot + 1):
        elapsed_days: float = t / slots_per_day
        stock_at_t: float = (
            initial_terminal_stock_m3
            + total_daily_inflow * elapsed_days
        )

        # Subtract volumes of vessels that have already completed by time t
        completed_volume = 0.0
        for other_vid, (start_slot, completion_slot) in sched_lookup.items():
            if completion_slot <= t:
                vol = vessel_volume.get(other_vid, 0.0)
                stock_at_t -= vol
                completed_volume += vol

        # If stock exceeds threshold, flag ALL QUBO variables for slot t.
        # Temporal-granularity cuts: any vessel on any monobuoy at slot t
        # contributes equally to the overflow risk, so the entire slot is closed.
        if stock_at_t > safe_threshold:
            slot_vars = variables_df[variables_df["slot"] == t]
            n_flagged_slot = len(slot_vars)

            logger.warning(
                "Overflow violation at slot t=%d (%.2f days): "
                "stock=%.0f m³ > threshold=%.0f m³ "
                "(initial=%.0f + inflow=%.0f - completed=%.0f) — "
                "flagging %d variable(s) for slot %d",
                t,
                elapsed_days,
                stock_at_t,
                safe_threshold,
                initial_terminal_stock_m3,
                total_daily_inflow * elapsed_days,
                completed_volume,
                n_flagged_slot,
                t,
            )

            for _, row in slot_vars.iterrows():
                flagged.add((str(row["vessel_id"]), int(row["machine"]), t))

    if flagged:
        violated_slots = len({t for _, _, t in flagged})
        logger.warning(
            "Overflow check found %d variable(s) across %d violated slot(s) "
            "where stock exceeds safe threshold "
            "(max_capacity=%.0f m³, safe_threshold=%.0f m³, "
            "total_daily_inflow=%.0f m³/day, buffer_days=%d).",
            len(flagged),
            violated_slots,
            max_terminal_capacity_m3,
            safe_threshold,
            total_daily_inflow,
            min_ullage_days,
        )
    else:
        logger.info(
            "Buffer check PASSED: no overflow violations detected. "
            "Stock never exceeded safe_threshold=%.0f m³.",
            safe_threshold,
        )

    return flagged


# ---------------------------------------------------------------------------
# P3 calibration — Eq. (16): P1 > P2 > P3 > c_max
# ---------------------------------------------------------------------------


def _calibrate_p3(P2: float, n_vessels: int, beta: float = PENALTY_BETA) -> float:
    """
    Derive P3 from P2 satisfying the penalty hierarchy P1 > P2 > P3 > c_max
    (Eq. 16).

    With P2 = α · n · c_max:

        P3 = P2 / β = (α/β) · n · c_max

    Hierarchy guarantees (for β > 1):

    - P3 < P2  always  (P3 = P2/β < P2  when β > 1)
    - P3 > c_max  when  (α/β) · n > 1, i.e. n ≥ 1 for α = 3.0, β = 2.0 ✓
    - P3 < P1  follows transitively from P3 < P2 < P1

    If P3 ≥ P1, the solver finds it cheaper to drop a vessel entirely
    (cost P1) than to place it in a flagged slot (cost P3) — vessels would
    silently disappear from the schedule.

    Args:
        P2:        Overlap penalty from :func:`src.qubo_builder.calibrate_penalties`
                   (= α · n · c_max).
        n_vessels: Number of distinct vessels in the nomination set.
                   Passed for documentation clarity; already embedded in P2.
        beta:      Secondary penalty scaling factor β (Eq. 13c).  Must be > 1
                   to guarantee P3 < P2.  Defaults to
                   :data:`src.config.PENALTY_BETA`.

    Returns:
        P3 = P2 / β satisfying the full hierarchy.
    """
    return P2 / beta


# ---------------------------------------------------------------------------
# Iterative hybrid loop — Eq. (17)
# ---------------------------------------------------------------------------


def run_iterative_loop(
    nominations_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    horizon_slots: int = T,
    alpha: float = PENALTY_ALPHA,
    beta: float = PENALTY_BETA,
    conflict_set: FrozenSet[Tuple[int, int]] = CONFLICT_SET_R,
) -> IterativeResult:
    """
    Orchestrate the full iterative hybrid QUBO loop (qubo_formulation.md §15).

    Each iteration follows Eq. (17):

        H_QUBO^(k+1) = H_obj + H_assign + H_overlap + Σ_{i=1}^{k} H_cuts^(i)

    The loop terminates when:

    - **Converged**: F^(k) = ∅ — no ullage violations detected after solving.
      Returns the schedule with ``converged=True``.
    - **Max iterations**: K = MAX_ITERATIONS reached without convergence.
      Returns the best feasible schedule found (or the last schedule if none
      was feasible) with ``converged=False`` for human-planner intervention.
      Never silently fails.

    Over-saturation is checked once before the loop.  If Σ p_j exceeds the
    planning horizon, the loop is skipped and ``oversaturated=True`` is
    returned immediately.

    Penalty hierarchy (Eq. 16) is enforced inside :func:`src.qubo_builder.build_qubo`
    via assertions before every solver call.

    Args:
        nominations_df: Nomination DataFrame from
            :func:`src.config.generate_nominations`, with columns
            ``vessel_id``, ``p_j``, ``volume_m3`` (needed for buffer check).
        variables_df:   Self-contained feasible-slot DataFrame from
            :func:`src.preprocessing.compute_feasible_slots`, containing
            all columns required by the QUBO builder.
        horizon_slots: Total number of macro-slots in the planning horizon.
            Defaults to ``src.config.T``.  Pass an explicit value when using
            a different slot duration than the default (e.g. experiments with
            24h slots use ``horizon_days * (24 // slot_duration_hours)``).
        alpha: Overlap penalty scaling factor α in P2 = α · n · c_max
            (Eq. 13b).  Defaults to :data:`src.config.PENALTY_ALPHA`.
            Experiment notebooks may override per-run.
        beta: Secondary penalty scaling factor β for P3 = P2 / β (Eq. 13c).
            Must be > 1 to guarantee P3 < P2.  Defaults to
            :data:`src.config.PENALTY_BETA`.
        conflict_set: Resource-conflict set R passed through to
            :func:`src.qubo_builder.build_qubo` and
            :func:`src.solver.check_feasibility`.  Defaults to
            :data:`src.config.CONFLICT_SET_R` (shared pipeline).

    Returns:
        :class:`IterativeResult` with the best schedule, convergence flag,
        iteration count, feasibility report, accumulated cuts, and
        over-saturation diagnostic.
    """
    # -- Pre-loop: over-saturation guard ----------------------------------
    oversaturated: bool = _is_oversaturated(nominations_df, horizon_slots=horizon_slots)
    if oversaturated:
        empty_schedule: pd.DataFrame = pd.DataFrame()
        return IterativeResult(
            schedule=empty_schedule,
            converged=False,
            iterations=0,
            feasibility={
                "is_feasible": False,
                "missing_vessels": nominations_df["vessel_id"].tolist(),
                "duplicate_vessels": [],
                "pipeline_violations": [],
                "total_weighted_tardiness": 0.0,
            },
            all_cuts=set(),
            oversaturated=True,
        )

    # Derive P3 before the loop using calibrate_penalties (no BQM needed).
    P1, P2, P3 = calibrate_penalties(variables_df, alpha=alpha, beta=beta)

    all_cuts: Set[Tuple[str, int, int]] = set()
    best_schedule: pd.DataFrame = pd.DataFrame()
    best_feasibility: Dict[str, Any] = {}
    best_wt: float = float("inf")

    # Initialise so they are always defined when referenced after the loop.
    schedule_df: pd.DataFrame = pd.DataFrame()
    feasibility: Dict[str, Any] = {}

    for k in range(1, MAX_ITERATIONS + 1):
        print(f"\n[inventory] ── Iteration {k}/{MAX_ITERATIONS} ──")

        # Build QUBO: first iteration has no cuts; subsequent ones pass all
        # accumulated cuts with P3.  The hierarchy assert is inside build_qubo.
        bqm, _, _, P3, _ = build_qubo(
            variables_df,
            cuts=all_cuts if all_cuts else None,
            conflict_set=conflict_set,
            alpha=alpha,
            beta=beta,
        )

        # Solve.
        sampleset, _ = run_solver(bqm)
        best_sample: Dict[str, int] = dict(sampleset.first.sample)

        # Decode and check hard constraints.
        schedule_df: pd.DataFrame = decode_schedule(best_sample, variables_df)
        feasibility: Dict[str, Any] = check_feasibility(
            schedule_df, variables_df, conflict_set=conflict_set
        )

        # Track best feasible solution by total weighted tardiness.
        current_wt: float = feasibility["total_weighted_tardiness"]
        if feasibility["is_feasible"] and current_wt < best_wt:
            best_schedule = schedule_df.copy()
            best_feasibility = feasibility.copy()
            best_wt = current_wt
            logger.info(
                "Iteration %d: new best feasible schedule "
                "(total_wt=%.4f).",
                k,
                best_wt,
            )

        # Buffer check: find new infeasible triples F^(k).
        new_cuts: Set[Tuple[str, int, int]] = check_worst_case_overlaps(
            schedule_df, variables_df, nominations_df
        )

        if not new_cuts:
            print(
                f"[inventory] Converged at iteration {k}: "
                f"no ullage violations detected."
            )
            # If we converged but the current schedule is the first feasible one,
            # use it even if best_schedule wasn't updated above.
            if best_schedule.empty and feasibility["is_feasible"]:
                best_schedule = schedule_df.copy()
                best_feasibility = feasibility.copy()
            return IterativeResult(
                schedule=best_schedule if not best_schedule.empty else schedule_df,
                converged=True,
                iterations=k,
                feasibility=best_feasibility if best_feasibility else feasibility,
                all_cuts=all_cuts,
                oversaturated=False,
            )

        # Accumulate cuts for next iteration.
        added: int = len(new_cuts - all_cuts)
        all_cuts |= new_cuts
        print(
            f"[inventory] Iteration {k}: {len(new_cuts)} violation triple(s) "
            f"found ({added} new).  Total accumulated cuts: {len(all_cuts)}."
        )

    # -- Max iterations reached without convergence -----------------------
    print(
        f"\n[inventory] WARNING: did not converge after {MAX_ITERATIONS} iterations.  "
        f"Returning best solution found (converged=False).  "
        f"Human planner intervention required."
    )
    logger.warning(
        "Iterative loop did not converge after %d iterations.  "
        "Total cuts accumulated: %d.  "
        "Best feasible solution found: %s.",
        MAX_ITERATIONS,
        len(all_cuts),
        "yes" if not best_schedule.empty else "no",
    )

    # Fall back to the last decoded schedule if no feasible solution was ever found.
    final_schedule = best_schedule if not best_schedule.empty else schedule_df
    final_feasibility = best_feasibility if best_feasibility else feasibility

    return IterativeResult(
        schedule=final_schedule,
        converged=False,
        iterations=MAX_ITERATIONS,
        feasibility=final_feasibility,
        all_cuts=all_cuts,
        oversaturated=False,
    )
