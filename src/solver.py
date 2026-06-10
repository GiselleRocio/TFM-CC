"""
src/solver.py — Quantum and classical-fallback solving layer.

Responsibilities (strict separation — see src/CLAUDE.md):
- Auto-detect DWAVE_API_TOKEN and select the appropriate sampler.
- Submit the BQM and return the raw SampleSet.
- Decode the best sample into a human-readable schedule DataFrame.
- Run a feasibility check on the decoded schedule.

No BQM construction, no penalty calibration, no tank logic here.
Sampler selection follows CLAUDE.md §SOLVER STRATEGY:
  Primary  : LeapHybridSampler  (requires DWAVE_API_TOKEN in environment)
  Fallback : SimulatedAnnealingSampler  (fully offline, no token needed)
  Exploratory: DWaveSampler (QPU directo) — solo para instancias pequeñas
               (n_vars ≲ 700). El Q-matrix denso por pipeline compartido
               genera chains largas; los resultados pueden degradarse por
               chain breaks. Ver §1.6 y §1.8 de la tesis.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from dotenv import load_dotenv

# Load .env from the repo root so DWAVE_API_TOKEN is available in all contexts
# (Jupyter notebooks, CLI, Docker) without requiring the caller to pre-load it.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import dimod
import pandas as pd

from src.config import (
    CONFLICT_SET_R,
    SA_NUM_READS,
    slot_to_datetime,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler selection
# ---------------------------------------------------------------------------


def _detect_leap_token() -> bool:
    """
    Return True if a D-Wave Leap API token is present in the environment.

    Checks ``DWAVE_API_TOKEN`` in the process environment.  If the project
    uses python-dotenv, the token will already be loaded into ``os.environ``
    before this function is called.
    """
    return bool(os.environ.get("DWAVE_API_TOKEN", "").strip())


def _select_sampler(requested: str | None = None) -> Tuple[Any, str]:
    """
    Instantiate and return the best available sampler plus a display name.

    Selection logic:

    1. If *requested* is ``"simulated_annealing"``, always use SA regardless
       of whether a D-Wave token is present (useful for offline testing).
    2. If *requested* is ``"leap_hybrid"`` (or ``None`` with a token present),
       attempt to instantiate :class:`dwave.system.LeapHybridSampler`.
       Falls back to SA with a warning if the import fails or the token
       is absent.
    3. If *requested* is ``None`` and no token is present, use SA.

    DWaveSampler is explicitly forbidden — see module docstring.

    Args:
        requested: Sampler name from the API request (``"leap_hybrid"`` or
            ``"simulated_annealing"``).  ``None`` preserves the original
            auto-detect behaviour (token present → Leap, else SA).

    Returns:
        Tuple ``(sampler_instance, solver_name)`` where *solver_name* is a
        short human-readable label printed to stdout.
    """
    _req = requested.replace("-", "_").lower() if requested else None
    want_sa   = _req in ("simulated_annealing", "sa")
    want_qpu  = _req in ("qpu", "dwave", "dwavesampler")
    want_leap = _req in ("leap_hybrid", "leaphybrid") or (_req is None and _detect_leap_token())

    if want_qpu:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore[import]
            sampler = EmbeddingComposite(DWaveSampler())
            name = "QPU_EmbeddingComposite"
            return sampler, name
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DWaveSampler (QPU) requested but unavailable (%s).  "
                "Falling back to SimulatedAnnealingSampler.",
                exc,
            )

    if not want_sa and want_leap:
        try:
            from dwave.system import LeapHybridSampler  # type: ignore[import]

            sampler = LeapHybridSampler()
            name = "LeapHybridSampler"
            return sampler, name
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LeapHybridSampler requested but unavailable (%s).  "
                "Falling back to SimulatedAnnealingSampler.",
                exc,
            )

    from dwave.samplers import SimulatedAnnealingSampler  # type: ignore[import]

    sampler = SimulatedAnnealingSampler()
    name = f"SimulatedAnnealingSampler (num_reads={SA_NUM_READS})"
    return sampler, name


# ---------------------------------------------------------------------------
# BQM submission
# ---------------------------------------------------------------------------


def run_solver(
    bqm: dimod.BinaryQuadraticModel,
    requested_sampler: str | None = None,
) -> Tuple[dimod.SampleSet, str]:
    """
    Submit *bqm* to the selected sampler and return the full SampleSet.

    Sampler is chosen automatically via :func:`_select_sampler`; the active
    solver name is printed to stdout so runs are traceable in logs.

    Args:
        bqm: Fully assembled BQM from
            :func:`src.qubo_builder.build_qubo`.
        requested_sampler: Sampler name from the API request
            (``"leap_hybrid"`` or ``"simulated_annealing"``).
            ``None`` preserves auto-detect behaviour.

    Returns:
        Tuple ``(sampleset, solver_name)`` where *sampleset* is the
        :class:`dimod.SampleSet` with all reads sorted by energy (lowest
        first), and *solver_name* is the human-readable label of the sampler
        that was actually used (reflecting any automatic fallback from
        LeapHybrid to SA).

    Raises:
        RuntimeError: If the sampler raises during submission (network
            error, token expired, etc.).
    """
    sampler, solver_name = _select_sampler(requested_sampler)
    print(f"[solver] Active sampler: {solver_name}")
    print(f"[solver] Submitting BQM with {len(bqm.variables)} variables …")

    try:
        if "Leap" in solver_name:
            # LeapHybridSampler requires an explicit time_limit — it cannot run
            # without one.  D-Wave enforces a problem-size-dependent minimum:
            # empirical formula min_time ≈ n_vars / 152 (seconds), observed from
            # D-Wave error messages at large problem sizes (n_vars ~32k → min ~215s).
            # We add 5s margin on top of the computed minimum.
            n_vars: int = len(bqm.variables)
            effective_time_limit: int = max(3, int(n_vars / 152.0) + 5)
            print(
                f"[solver] BQM has {n_vars} variables — time_limit={effective_time_limit}s "
                f"(D-Wave minimum for this instance size)."
            )
            sampleset: dimod.SampleSet = sampler.sample(
                bqm, time_limit=effective_time_limit
            )
        elif "QPU" in solver_name:
            # QPU directo vía EmbeddingComposite. num_reads=100 por defecto;
            # annealing_time=20µs (default Advantage). El embedding se calcula
            # automáticamente — puede fallar si el grafo es demasiado denso.
            n_vars_qpu: int = len(bqm.variables)
            print(
                f"[solver] QPU submit: {n_vars_qpu} logical vars — "
                "embedding may generate long chains on dense graphs."
            )
            sampleset = sampler.sample(bqm, num_reads=100, label="exp03_qpu")
        else:
            sampleset = sampler.sample(bqm, num_reads=SA_NUM_READS)
    except Exception as exc:
        raise RuntimeError(f"Sampler submission failed: {exc}") from exc

    best_energy: float = sampleset.first.energy
    print(
        f"[solver] Done.  Best energy={best_energy:.4f}  "
        f"({len(sampleset)} samples returned)."
    )
    return sampleset, solver_name


# ---------------------------------------------------------------------------
# Schedule decoding
# ---------------------------------------------------------------------------


def decode_schedule(
    sample: Dict[str, int],
    variables_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decode a binary sample into a human-readable schedule DataFrame.

    Iterates over variables set to 1 in *sample*, parses the BQM key
    convention ``f"x_{vessel_id}_{machine}_{slot_t}"`` (see CLAUDE.md
    §VARIABLE NAMING CONVENTION), and enriches each assignment with timing
    and tardiness information from *variables_df*.

    Note: vessel_id must not contain underscores for the key split to work
    correctly.  IDs generated by :func:`src.config.generate_nominations`
    (format ``V{n:02d}``) satisfy this requirement.

    Args:
        sample: Dict mapping BQM variable name → 0 or 1.  Typically
            ``sampleset.first.sample`` from :func:`run_solver`.
        variables_df: Self-contained feasible-slot DataFrame from
            :func:`src.preprocessing.compute_feasible_slots`, containing
            columns ``vessel_id``, ``machine``, ``slot``, ``p_j``,
            ``d_j``, ``w_j``.

    Returns:
        A :class:`pandas.DataFrame` with one row per assigned vessel and
        columns:

        - ``vessel_id``          (str)   — vessel identifier
        - ``machine``            (int)   — assigned monobuoy (1 or 2)
        - ``start_slot``         (int)   — starting macro-slot
        - ``end_slot``           (int)   — completion macro-slot (start + p_j)
        - ``start_dt``           (datetime) — UTC datetime of start
        - ``end_dt``             (datetime) — UTC datetime of completion
        - ``p_j``                (int)   — processing time in macro-slots
        - ``d_j``                (int)   — due slot
        - ``w_j``                (float) — priority weight (ESD)
        - ``tardiness_slots``    (int)   — max(0, end_slot − d_j)
        - ``weighted_tardiness`` (float) — w_j × tardiness_slots
        - ``is_late``            (bool)  — True if tardiness_slots > 0

        Sorted by ``start_slot`` ascending.  May have fewer rows than the
        number of nominations if the solver dropped vessels (infeasible
        sample).
    """
    # Build a lookup index: (vessel_id, machine, slot) → row series.
    idx: Dict[Tuple[str, int, int], pd.Series] = {
        (str(row["vessel_id"]), int(row["machine"]), int(row["slot"])): row
        for _, row in variables_df.iterrows()
    }

    records: List[Dict[str, Any]] = []

    for var, val in sample.items():
        if val != 1:
            continue
        if not var.startswith("x_"):
            continue

        # Parse key: strip the "x_" prefix, then split the suffix.
        # The last two tokens are always machine and slot; everything in
        # between is the vessel_id (which may itself contain underscores).
        # Example: "x_V01_1_4"  → vid_parts=["V01"], machine_s="1", slot_s="4"
        # Example: "x_AB_CD_1_4" → vid_parts=["AB","CD"], machine_s="1", slot_s="4"
        suffix: str = var[2:]  # strip leading "x_"
        suffix_parts: List[str] = suffix.split("_")
        if len(suffix_parts) < 3:
            logger.warning("Unexpected variable key format: %s — skipping.", var)
            continue

        *vid_parts, machine_s, slot_s = suffix_parts
        vessel_id: str = "_".join(vid_parts)
        machine: int = int(machine_s)
        slot_t: int = int(slot_s)

        row = idx.get((vessel_id, machine, slot_t))
        if row is None:
            logger.warning(
                "Variable %s set to 1 but not found in variables_df — "
                "phantom variable, skipping.",
                var,
            )
            continue

        p_j: int = int(row["p_j"])
        d_j: int = int(row["d_j"])
        w_j: float = float(row["w_j"])
        end_slot: int = slot_t + p_j
        tardiness_slots: int = max(0, end_slot - d_j)

        # slot_to_datetime is only valid for the live terminal horizon (T=62).
        # Experiment instances use extended horizons — skip datetime conversion
        # gracefully rather than raising ValueError.
        try:
            start_dt = slot_to_datetime(slot_t)
            end_dt = slot_to_datetime(end_slot)
        except ValueError:
            start_dt = None
            end_dt = None

        records.append(
            {
                "vessel_id": vessel_id,
                "machine": machine,
                "start_slot": slot_t,
                "end_slot": end_slot,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "p_j": p_j,
                "d_j": d_j,
                "w_j": w_j,
                "tardiness_slots": tardiness_slots,
                "weighted_tardiness": w_j * tardiness_slots,
                "is_late": tardiness_slots > 0,
            }
        )

    if not records:
        logger.warning("No variables set to 1 in sample — schedule is empty.")
        return pd.DataFrame(
            columns=[
                "vessel_id", "machine", "start_slot", "end_slot",
                "start_dt", "end_dt", "p_j", "d_j", "w_j",
                "tardiness_slots", "weighted_tardiness", "is_late",
            ]
        )

    schedule: pd.DataFrame = (
        pd.DataFrame(records)
        .sort_values("start_slot")
        .reset_index(drop=True)
    )

    total_wt: float = float(schedule["weighted_tardiness"].sum()) if not schedule.empty else 0.0
    n_late: int = int(schedule["is_late"].sum()) if not schedule.empty else 0
    n_unique_vessels: int = schedule["vessel_id"].nunique() if not schedule.empty else 0
    
    print(
        f"[solver] Decoded {len(schedule)} active variables for {n_unique_vessels} unique vessels.  "
        f"Total weighted tardiness={total_wt:.2f}  "
        f"Late vessels={n_late}/{n_unique_vessels}."
    )
    return schedule


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------


def check_feasibility(
    schedule_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    conflict_set: FrozenSet[Tuple[int, int]] = CONFLICT_SET_R,
) -> Dict[str, Any]:
    """
    Verify that a decoded schedule satisfies both hard constraints.

    Checks two conditions from CLAUDE.md §PHYSICAL CONSTRAINTS:

    **1. Assignment constraint** — every vessel in *variables_df* appears
    exactly once in *schedule_df*.  Missing or duplicate vessels indicate
    an infeasible sample (H_assign violated).

    **2. Pipeline constraint** — no two vessels with conflicting machine
    assignments have overlapping loading intervals.  Two vessels conflict
    only when their ``(machine_a, machine_b)`` pair is in *conflict_set*.
    With the default ``CONFLICT_SET_R`` = {(1,1),(1,2),(2,1),(2,2)} (shared
    pipeline) all vessel pairs conflict regardless of monobuoy.  With an
    independent-pipeline topology (diagonal-only conflict set), vessels on
    different monobuoys are allowed to overlap.  Intervals [start_i, end_i)
    and [start_k, end_k) overlap when start_i < end_k AND start_k < end_i.

    All pairs (i, j) with i < j in the sorted schedule are checked — O(n²)
    in the number of vessels, which is fine because n ≤ 200.

    Args:
        schedule_df:  Decoded schedule from :func:`decode_schedule`.
        variables_df: Feasible-slot DataFrame from preprocessing, used to
            determine the complete set of expected vessel IDs.
        conflict_set: Resource-conflict set R.  Defaults to ``CONFLICT_SET_R``
            (shared pipeline — all machine pairs conflict).  Pass a diagonal
            or partial set for independent / clustered pipeline topologies.

    Returns:
        Dict with keys:

        - ``is_feasible``            (bool)       — True iff all checks pass.
        - ``missing_vessels``        (List[str])  — vessels not in schedule.
        - ``duplicate_vessels``      (List[str])  — vessels assigned > once.
        - ``pipeline_violations``    (List[str])  — human-readable overlap
          strings, one per colliding consecutive pair.
        - ``total_weighted_tardiness`` (float)    — sum of w_j × tardiness_slots
          over all assigned vessels; 0.0 if schedule is empty.

        All violations are also logged at ERROR level with vessel details.
    """
    expected_vessels: List[str] = sorted(
        variables_df["vessel_id"].unique().tolist()
    )
    n_expected: int = len(expected_vessels)

    missing: List[str] = []
    duplicates: List[str] = []
    pipeline_violations: List[str] = []
    total_wt: float = (
        float(schedule_df["weighted_tardiness"].sum())
        if not schedule_df.empty
        else 0.0
    )

    # -- 1. Assignment constraint -----------------------------------------
    if schedule_df.empty:
        logger.error(
            "Assignment check FAILED: schedule is empty, expected %d vessels.",
            n_expected,
        )
        missing = expected_vessels[:]
    else:
        scheduled_vessels: List[str] = schedule_df["vessel_id"].tolist()

        missing = [v for v in expected_vessels if v not in scheduled_vessels]
        if missing:
            logger.error(
                "Assignment check FAILED: %d vessel(s) not scheduled: %s.",
                len(missing),
                missing,
            )

        counts = schedule_df["vessel_id"].value_counts()
        duplicates = counts[counts > 1].index.tolist()
        if duplicates:
            logger.error(
                "Assignment check FAILED: %d vessel(s) assigned multiple times: %s.",
                len(duplicates),
                duplicates,
            )

        # -- 2. Pipeline constraint (topology-aware) ----------------------
        # Only pairs whose (machine_a, machine_b) is in conflict_set are
        # checked — under independent pipelines, vessels on different
        # monobuoys are allowed to overlap.
        sorted_sched: pd.DataFrame = schedule_df.sort_values(
            "start_slot"
        ).reset_index(drop=True)

        n_sched: int = len(sorted_sched)
        for i in range(n_sched):
            row_a = sorted_sched.iloc[i]
            start_a: int = int(row_a["start_slot"])
            end_a: int = int(row_a["end_slot"])
            machine_a: int = int(row_a["machine"])
            for j in range(i + 1, n_sched):
                row_b = sorted_sched.iloc[j]
                start_b: int = int(row_b["start_slot"])
                # sorted by start_slot, so once start_b >= end_a no further
                # j can overlap with i — break early.
                if start_b >= end_a:
                    break
                machine_b: int = int(row_b["machine"])
                if (machine_a, machine_b) not in conflict_set and (
                    machine_b,
                    machine_a,
                ) not in conflict_set:
                    continue
                end_b: int = int(row_b["end_slot"])
                pipeline_violations.append(
                    f"{row_a['vessel_id']}@m{machine_a}"
                    f"[{start_a},{end_a})"
                    f" ∩ "
                    f"{row_b['vessel_id']}@m{machine_b}"
                    f"[{start_b},{end_b})"
                )

        if pipeline_violations:
            logger.error(
                "Pipeline constraint FAILED: %d overlapping interval pair(s): %s.",
                len(pipeline_violations),
                pipeline_violations,
            )

    is_feasible: bool = not (missing or duplicates or pipeline_violations)

    if is_feasible:
        print(
            f"[solver] Feasibility check PASSED: "
            f"{n_expected} vessels assigned, no pipeline conflicts."
        )
    else:
        print(
            "[solver] Feasibility check FAILED — see logger output for details."
        )

    return {
        "is_feasible": is_feasible,
        "missing_vessels": missing,
        "duplicate_vessels": duplicates,
        "pipeline_violations": pipeline_violations,
        "total_weighted_tardiness": total_wt,
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def post_process_schedule(
    schedule_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    shared_pipeline: bool,
) -> pd.DataFrame:
    """
    Post-process a decoded QUBO schedule to improve vessel-slot assignments.

    Two modes depending on the pipeline topology:

    **Shared pipeline** (shared_pipeline=True):
        All monobuoys share one physical pipeline, so the QUBO already
        serialises every vessel globally (no overlaps).  Post-processing
        attempts to alternate monobuoy assignments (M1 → M2 → M1 …) to
        balance port-side operations.  A monobuoy swap is applied only when
        the vessel has a feasible slot at the *same start time* on the other
        monobuoy; otherwise the original assignment is kept.

    **Independent pipeline** (shared_pipeline=False):
        Each monobuoy operates its own pipeline.  Post-processing greedily
        repacks each monobuoy's queue: vessels are sorted by their earliest
        release slot (r_j ≈ first feasible slot) and placed as early as
        possible, minimising tardiness without touching the other monobuoy.

    Args:
        schedule_df:     Decoded schedule from :func:`decode_schedule`.
        variables_df:    Feasible-slot DataFrame from preprocessing.
        shared_pipeline: True if all monobuoys share one pipeline.

    Returns:
        Improved :class:`pandas.DataFrame` with the same schema as
        *schedule_df*, sorted by ``start_slot`` ascending.
        Returns *schedule_df* unchanged if it is empty.
    """
    if schedule_df.empty:
        return schedule_df

    # Build lookup: (vessel_id, machine) → sorted list of feasible start slots.
    feasible: Dict[Tuple[str, int], List[int]] = {}
    for _, row in variables_df.iterrows():
        key = (str(row["vessel_id"]), int(row["machine"]))
        feasible.setdefault(key, []).append(int(row["slot"]))
    for key in feasible:
        feasible[key].sort()

    if shared_pipeline:
        return _alternate_monobuoy_assignment(schedule_df, feasible)
    return _greedy_preferred_slot(schedule_df, feasible)


def _alternate_monobuoy_assignment(
    schedule_df: pd.DataFrame,
    feasible: Dict[Tuple[str, int], List[int]],
) -> pd.DataFrame:
    """Attempt to alternate monobuoy assignments for consecutive vessels.

    Walks the schedule sorted by start_slot.  Whenever two consecutive vessels
    land on the same monobuoy the second vessel is swapped to the other one,
    but only if that monobuoy has a feasible slot at the *identical* start time.
    Timing (start_slot / end_slot) is never changed — only the machine label.

    Args:
        schedule_df: Decoded schedule sorted by start_slot.
        feasible:    Mapping (vessel_id, machine) → sorted feasible slots.

    Returns:
        DataFrame with the same schema as *schedule_df*, potentially with
        updated ``machine`` values.
    """
    machines: List[int] = sorted(schedule_df["machine"].unique().tolist())
    if len(machines) < 2:
        # Only one monobuoy in use — nothing to alternate.
        return schedule_df.sort_values("start_slot").reset_index(drop=True)

    rows: List[Dict[str, Any]] = (
        schedule_df.sort_values("start_slot").reset_index(drop=True).to_dict(orient="records")
    )
    prev_machine: Optional[int] = None

    for i, row in enumerate(rows):
        current_machine: int = int(row["machine"])
        if prev_machine is not None and current_machine == prev_machine:
            # Try every other monobuoy in sorted order.
            for other_m in machines:
                if other_m == current_machine:
                    continue
                if int(row["start_slot"]) in feasible.get((str(row["vessel_id"]), other_m), []):
                    rows[i] = {**row, "machine": other_m}
                    current_machine = other_m
                    break
            # If no swap is feasible, keep original — no change.
        prev_machine = current_machine

    result = pd.DataFrame(rows).sort_values("start_slot").reset_index(drop=True)
    n_swapped = (result["machine"] != schedule_df.sort_values("start_slot").reset_index(drop=True)["machine"]).sum()
    if n_swapped:
        print(f"[post-process] Alternated monobuoy assignment for {n_swapped} vessel(s).")
    return result


def _greedy_preferred_slot(
    schedule_df: pd.DataFrame,
    feasible: Dict[Tuple[str, int], List[int]],
) -> pd.DataFrame:
    """Repack each monobuoy queue to push vessels toward their earliest slot.

    For each monobuoy independently:
      1. Sort assigned vessels by their earliest feasible slot (≈ r_j).
      2. Place each vessel at the first feasible slot ≥ max(r_j, end_of_prev).
      3. Recalculate tardiness and weighted_tardiness accordingly.

    After the per-machine repack, :func:`_rebalance_late_vessels` attempts to
    move late vessels to under-utilised monobuoys when doing so strictly reduces
    their tardiness.  This resolves cases where the QUBO left one monobuoy idle
    while another carried a late vessel.

    This is a purely classical improvement pass that does not affect the QUBO
    solution quality — it resolves degeneracies left by the annealer.

    Args:
        schedule_df: Decoded schedule DataFrame.
        feasible:    Mapping (vessel_id, machine) → sorted feasible slots.

    Returns:
        Repacked DataFrame with updated timing columns.
    """
    machines: List[int] = sorted(schedule_df["machine"].unique().tolist())
    result_rows: List[Dict[str, Any]] = []
    machine_timeline: Dict[int, int] = {m: 0 for m in machines}

    for m in machines:
        machine_df = schedule_df[schedule_df["machine"] == m].copy()

        # Annotate each vessel with its earliest feasible slot (≈ r_j) on this
        # machine so we can sort by arrival order before greedy placement.
        machine_df["_r_j_approx"] = machine_df.apply(
            lambda row: (
                feasible.get((str(row["vessel_id"]), m), [int(row["start_slot"])])[0]
            ),
            axis=1,
        )
        machine_df = machine_df.sort_values("_r_j_approx").reset_index(drop=True)

        timeline_end: int = 0  # slot at which the last placed vessel finishes

        for _, row in machine_df.iterrows():
            vessel_id: str = str(row["vessel_id"])
            p_j: int = int(row["p_j"])
            d_j: int = int(row["d_j"])
            w_j: float = float(row["w_j"])

            # Earliest feasible slot that doesn't overlap with previous vessel.
            available: List[int] = [
                s for s in feasible.get((vessel_id, m), []) if s >= timeline_end
            ]
            best_slot: int = available[0] if available else int(row["start_slot"])

            end_slot: int = best_slot + p_j
            tardiness_slots: int = max(0, end_slot - d_j)

            try:
                start_dt = slot_to_datetime(best_slot)
                end_dt = slot_to_datetime(end_slot)
            except ValueError:
                start_dt = None
                end_dt = None

            result_rows.append(
                {
                    "vessel_id": vessel_id,
                    "machine": m,
                    "start_slot": best_slot,
                    "end_slot": end_slot,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "p_j": p_j,
                    "d_j": d_j,
                    "w_j": w_j,
                    "tardiness_slots": tardiness_slots,
                    "weighted_tardiness": w_j * tardiness_slots,
                    "is_late": tardiness_slots > 0,
                }
            )
            timeline_end = end_slot

        machine_timeline[m] = timeline_end

    if not result_rows:
        return schedule_df

    # Cross-machine balancing: move late vessels to under-utilised monobuoys.
    if len(machines) > 1:
        result_rows = _rebalance_late_vessels(result_rows, feasible, machines, machine_timeline)

    result = pd.DataFrame(result_rows).sort_values("start_slot").reset_index(drop=True)
    old_wt = float(schedule_df["weighted_tardiness"].sum())
    new_wt = float(result["weighted_tardiness"].sum())
    print(
        f"[post-process] Greedy slot repack: weighted tardiness "
        f"{old_wt:.2f} → {new_wt:.2f} "
        f"({'improved' if new_wt < old_wt else 'unchanged'})."
    )
    return result


def _rebalance_late_vessels(
    rows: List[Dict[str, Any]],
    feasible: Dict[Tuple[str, int], List[int]],
    machines: List[int],
    machine_timeline: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Move late vessels to under-utilised monobuoys when it reduces tardiness.

    Iterates late vessels in descending weighted-tardiness order.  For each,
    tries every other monobuoy sorted by earliest available slot (least loaded
    first) and moves the vessel if the new tardiness is strictly smaller.
    Updates per-machine timelines after each successful move.

    Single-pass local-search improvement — O(n_late × n_machines).

    Args:
        rows:             Current schedule rows (one dict per vessel).
        feasible:         Mapping (vessel_id, machine) → sorted feasible slots.
        machines:         Sorted list of monobuoy indices.
        machine_timeline: Mapping machine → end slot of its last vessel.
                          Mutated in place to reflect moves.

    Returns:
        Updated rows list.
    """
    late_indices: List[int] = [
        i for i, r in enumerate(rows) if bool(r["is_late"])
    ]
    if not late_indices:
        return rows

    late_indices.sort(key=lambda i: float(rows[i]["weighted_tardiness"]), reverse=True)

    n_moved: int = 0

    for idx in late_indices:
        row = rows[idx]
        vid: str = str(row["vessel_id"])
        cur_m: int = int(row["machine"])
        cur_tardiness: int = int(row["tardiness_slots"])
        p_j: int = int(row["p_j"])
        d_j: int = int(row["d_j"])
        w_j: float = float(row["w_j"])

        # Try other monobuoys ordered by earliest available slot (least loaded first).
        candidates: List[int] = sorted(
            [m for m in machines if m != cur_m],
            key=lambda m: machine_timeline[m],
        )

        for other_m in candidates:
            slots: List[int] = feasible.get((vid, other_m), [])
            avail: List[int] = [s for s in slots if s >= machine_timeline[other_m]]
            if not avail:
                continue
            new_start: int = avail[0]
            new_end: int = new_start + p_j
            new_tardiness: int = max(0, new_end - d_j)

            if new_tardiness < cur_tardiness:
                try:
                    new_start_dt = slot_to_datetime(new_start)
                    new_end_dt = slot_to_datetime(new_end)
                except ValueError:
                    new_start_dt = None
                    new_end_dt = None

                rows[idx] = {
                    **row,
                    "machine": other_m,
                    "start_slot": new_start,
                    "end_slot": new_end,
                    "start_dt": new_start_dt,
                    "end_dt": new_end_dt,
                    "tardiness_slots": new_tardiness,
                    "weighted_tardiness": w_j * new_tardiness,
                    "is_late": new_tardiness > 0,
                }
                machine_timeline[other_m] = max(machine_timeline[other_m], new_end)
                # Recompute source machine timeline after the vessel is removed.
                machine_timeline[cur_m] = max(
                    (int(r["end_slot"]) for r in rows if int(r["machine"]) == cur_m),
                    default=0,
                )
                n_moved += 1
                break  # vessel moved — proceed to next late vessel

    if n_moved:
        print(
            f"[post-process] Cross-machine balance: {n_moved} vessel(s) moved "
            "to less-loaded monobuoys."
        )

    return rows
