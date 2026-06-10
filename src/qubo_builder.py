"""
src/qubo_builder.py — Binary Quadratic Model construction for the
terminal scheduling problem.

Constructs H_QUBO = H_obj + H_assign + H_overlap [+ H_cuts] following
qubo_formulation.md Equations (3)–(13).

Responsibilities (strict separation — see src/CLAUDE.md):
- Build the BQM from a pre-validated feasible-slot DataFrame.
- Calibrate penalties P1 and P2 deterministically (Eqs. 13a–13b).
- Assert the penalty hierarchy before returning the BQM (Eq. 16).

No I/O, no solving, no tank/inventory logic in this module.
"""

from __future__ import annotations

import bisect
import itertools
import logging
from typing import Collection, Dict, List, Optional, Tuple

import dimod
import pandas as pd

from src.config import (
    CONFLICT_SET_R,
    EPSILON,
    PENALTY_ALPHA,
    PENALTY_BETA,
)

# Conflict set for terminals where monobuoys operate on independent pipelines.
# Only same-machine pairs conflict (a vessel cannot overlap with itself).
#
# WARNING: this constant is hardcoded for a 2-monobuoy terminal (machines 1 and 2).
# For terminals with N monobuoys, build the conflict set dynamically:
#     conflict_set = frozenset((m, m) for m in machines)
# api.py already does this correctly — do not pass this constant directly when
# n_machines != 2.
CONFLICT_SET_INDEPENDENT: FrozenSet[Tuple[int, int]] = frozenset({(1, 1), (2, 2)})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost coefficient — Eq. (3)
# ---------------------------------------------------------------------------


def _cost_coefficient(w_j: float, p_j: int, t: int, d_j: int) -> float:
    """
    Compute the cost coefficient c_{j,m,t} for one BQM variable.

    c_{j,m,t} = w_j * max(0, t + p_j − d_j) + ε * (t + p_j)          (Eq. 3)

    The ε term breaks ties among zero-tardiness solutions in favour of
    earlier completions.  Both ε bounds from Eq. (4) must be verified
    empirically during the penalty calibration sweep — ε does not affect
    the penalty hierarchy.

    Args:
        w_j: Priority weight (Equivalent Stock in Days).
        p_j: Processing time in macro-slots.
        t:   Starting macro-slot index.
        d_j: Due slot (preferred completion deadline).

    Returns:
        Non-negative float cost coefficient.
    """
    tardiness: float = float(max(0, t + p_j - d_j))
    return w_j * tardiness + EPSILON * (t + p_j)


# ---------------------------------------------------------------------------
# Penalty calibration — Eqs. (13a)–(13b)
# ---------------------------------------------------------------------------


def calibrate_penalties(
    variables_df: pd.DataFrame,
    alpha: float = PENALTY_ALPHA,
    beta: float = PENALTY_BETA,
) -> Tuple[float, float, float]:
    """
    Compute P1, P2, and P3 deterministically from the feasible-slot table.

    The three penalties satisfy the hierarchy P1 > P2 > P3 > c_max (Eq. 16):

        P1 = α² * n * c_max     (assignment penalty)              (Eq. 13a)
        P2 = α  * n * c_max     (overlap penalty)                 (Eq. 13b)
        P3 = P2 / β             (inventory-cut penalty)           (Eq. 13c)

    where *n* is the number of distinct vessels, *c_max* is the maximum
    cost coefficient across all instantiated variables, α = *alpha*, and
    β = *beta*.  The ratio P1/P2 = α > 1 ensures assignment constraints are
    penalised more heavily than pipeline-overlap constraints.

    Args:
        variables_df: Feasible-slot DataFrame produced by
            :func:`src.preprocessing.compute_feasible_slots`, containing
            columns ``vessel_id``, ``slot``, ``p_j``, ``d_j``, ``w_j``.
        alpha: Penalty scaling factor α.  Defaults to ``PENALTY_ALPHA`` from
            ``src.config``.  Pass an explicit value to override per-request
            (e.g. from the API's ``RunConfig.alpha`` field).
        beta: Secondary scaling factor β for inventory cuts.  Defaults to
            ``PENALTY_BETA`` from ``src.config``.  Must be > 1 so that
            P3 < P2 (inventory cuts do not outweigh overlap penalties).

    Returns:
        Tuple ``(P1, P2, P3)`` with ``P1 > P2 > P3 > c_max``.

    Raises:
        ValueError: If *variables_df* is empty.
    """
    if variables_df.empty:
        raise ValueError("variables_df is empty — cannot calibrate penalties.")

    n_vessels: int = variables_df["vessel_id"].nunique()

    c_max: float = 0.0
    for _, row in variables_df.iterrows():
        c = _cost_coefficient(
            float(row["w_j"]), int(row["p_j"]), int(row["slot"]), int(row["d_j"])
        )
        if c > c_max:
            c_max = c

    if c_max == 0.0:
        # All feasible slots have zero tardiness.  Use the ε-only term at the
        # latest slot as a non-zero anchor so penalties remain finite and
        # the hierarchy assertion does not trigger a spurious failure.
        logger.warning(
            "All c_jmt == 0 (no tardiness in any feasible slot).  "
            "Using ε-only anchor for penalty calibration."
        )
        max_slot: int = int(variables_df["slot"].max())
        max_p: int = int(variables_df["p_j"].max())
        c_max = EPSILON * (max_slot + max_p)

    P1: float = alpha ** 2 * n_vessels * c_max  # α² · n · c_max  (Eq. 13a)
    P2: float = alpha      * n_vessels * c_max  # α  · n · c_max  (Eq. 13b)
    P3: float = P2 / beta                       # P2 / β          (Eq. 13c)

    logger.info(
        "Penalties calibrated: n=%d, c_max=%.6f, α=%.1f, β=%.1f → P1=%.4f, P2=%.4f, P3=%.4f.",
        n_vessels,
        c_max,
        alpha,
        beta,
        P1,
        P2,
        P3,
    )
    return P1, P2, P3


# ---------------------------------------------------------------------------
# H_obj — Eq. (5): objective diagonal
# ---------------------------------------------------------------------------


def _add_h_obj(
    bqm: dimod.BinaryQuadraticModel,
    variables_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Add H_obj diagonal terms to *bqm* and return the cost-coefficient map.

    H_obj = Σ_{j,m,t} c_{jmt} * x_{jmt}                               (Eq. 5)

    Contributes only to diagonal entries of Q.

    Args:
        bqm:          BQM to update in place.
        variables_df: Feasible-slot DataFrame (output of preprocessing).

    Returns:
        Dict mapping variable name → c_{jmt}, used later to verify the
        penalty hierarchy and to compute c_max for diagnostics.
    """
    costs: Dict[str, float] = {}
    for _, row in variables_df.iterrows():
        var: str = f"x_{row['vessel_id']}_{row['machine']}_{row['slot']}"
        c: float = _cost_coefficient(
            float(row["w_j"]), int(row["p_j"]), int(row["slot"]), int(row["d_j"])
        )
        bqm.add_variable(var, c)
        costs[var] = c
    return costs


# ---------------------------------------------------------------------------
# H_assign — Eq. (6): assignment constraint
# ---------------------------------------------------------------------------


def _add_h_assign(
    bqm: dimod.BinaryQuadraticModel,
    variables_df: pd.DataFrame,
    P1: float,
) -> float:
    """
    Add H_assign terms to *bqm* and return the constant energy offset.

    H_assign = P1 * Σ_j (1 − Σ_{m,t} x_{jmt})²                       (Eq. 6)

    Expanding the square with the binary property x_i² = x_i:

        P1(1 − Σ x_i)² = P1 − P1·Σ x_i + 2·P1·Σ_{i<j} x_i·x_j

    Three contributions:

    - **Constant:** +P1 per vessel — tracked as offset, not stored in Q.
    - **Diagonal:** −P1 per variable x_{jmt}.  Combined with H_obj, the net
      diagonal entry is c_{jmt} − P1, matching Table 1 of the formulation.
    - **Off-diagonal:** +2·P1 for every distinct same-vessel pair
      (x_{jmt}, x_{jm′t′}).  This already handles cross-machine pairs
      (m ≠ m′) — no separate overlap term is needed for them.

    Args:
        bqm:          BQM to update in place.
        variables_df: Feasible-slot DataFrame.
        P1:           Assignment penalty from :func:`calibrate_penalties`.

    Returns:
        Constant energy offset equal to P1 × number of vessels.
    """
    offset: float = 0.0

    for vessel_id, grp in variables_df.groupby("vessel_id", sort=True):
        vessel_vars: List[str] = [
            f"x_{vessel_id}_{row['machine']}_{row['slot']}"
            for _, row in grp.iterrows()
        ]

        offset += P1  # constant: +P1 per vessel

        for var in vessel_vars:
            bqm.add_variable(var, -P1)  # diagonal: −P1

        for var_i, var_j in itertools.combinations(vessel_vars, 2):
            bqm.add_interaction(var_i, var_j, 2.0 * P1)  # off-diagonal: +2·P1

    return offset


# ---------------------------------------------------------------------------
# H_overlap — Eq. (7): mutual exclusion on the shared pipeline
# ---------------------------------------------------------------------------


def _add_h_overlap(
    bqm: dimod.BinaryQuadraticModel,
    variables_df: pd.DataFrame,
    P2: float,
    conflict_set: FrozenSet[Tuple[int, int]] = CONFLICT_SET_R,
) -> int:
    """
    Add H_overlap off-diagonal terms to *bqm* and return the coupler count.

    H_overlap = P2 * Σ_{j<k, (m1,m2)∈R}
                       Σ_{t ∈ T_{j,m1}}
                         Σ_{t′ ∈ T_{k,m2}, α ≤ t′ ≤ β}
                           x_{j,m1,t} · x_{k,m2,t′}                   (Eq. 7)

    Collision bounds are clamped to the feasible slot set of vessel k
    to prevent iterating over slots outside T_{k,m2}:

        α_{k,m2}(t) = max(min T_{k,m2},  t − p_k + 1)                (Eq. 8)
        β_{k,m2}(t) = min(max T_{k,m2},  t + p_j − 1)                (Eq. 9)

    Two vessels collide at (t, t′) when their processing intervals
    [t, t+p_j−1] and [t′, t′+p_k−1] overlap, i.e. t′ ≥ t−p_k+1 and
    t′ ≤ t+p_j−1.  The clamp shrinks this range to the feasible set.

    Only the upper-triangular pair set (j < k, sorted lexicographically
    by vessel_id) is iterated — duplicate couplers must not be added.

    Overlap coupler coefficient is **+P2**, not +2·P2.  There is no
    binomial expansion in H_overlap (see §9 key distinctions).

    Args:
        bqm:          BQM to update in place.
        variables_df: Feasible-slot DataFrame.
        P2:           Overlap penalty from :func:`calibrate_penalties`.

    Returns:
        Total number of overlap couplers added to the BQM.
    """
    # T_jm[(vessel_id, machine)] → sorted list of feasible slots.
    # p_map[vessel_id] → processing time (machine-independent for this terminal).
    T_jm: Dict[Tuple[str, int], List[int]] = {}
    p_map: Dict[str, int] = {}

    for (vessel_id, machine), grp in variables_df.groupby(
        ["vessel_id", "machine"], sort=True
    ):
        T_jm[(vessel_id, machine)] = sorted(grp["slot"].tolist())
        p_map[vessel_id] = int(grp["p_j"].iloc[0])

    vessel_ids: List[str] = sorted(variables_df["vessel_id"].unique().tolist())
    n_couplers: int = 0

    for j_idx, k_idx in itertools.combinations(range(len(vessel_ids)), 2):
        j_id: str = vessel_ids[j_idx]
        k_id: str = vessel_ids[k_idx]
        p_j: int = p_map[j_id]
        p_k: int = p_map[k_id]

        for m1, m2 in conflict_set:
            slots_j: Optional[List[int]] = T_jm.get((j_id, m1))
            slots_k: Optional[List[int]] = T_jm.get((k_id, m2))
            if not slots_j or not slots_k:
                continue

            min_t_k: int = slots_k[0]
            max_t_k: int = slots_k[-1]

            for t in slots_j:
                # Clamped collision bounds (Eqs. 8–9).
                alpha: int = max(min_t_k, t - p_k + 1)
                beta: int = min(max_t_k, t + p_j - 1)
                if alpha > beta:
                    continue  # intervals cannot collide for this t

                # Restrict t′ to slots_k ∩ [alpha, beta] via binary search.
                lo: int = bisect.bisect_left(slots_k, alpha)
                hi: int = bisect.bisect_right(slots_k, beta)

                var_j: str = f"x_{j_id}_{m1}_{t}"
                for t_prime in slots_k[lo:hi]:
                    var_k: str = f"x_{k_id}_{m2}_{t_prime}"
                    bqm.add_interaction(var_j, var_k, P2)  # +P2, not +2·P2
                    n_couplers += 1

    return n_couplers


# ---------------------------------------------------------------------------
# H_cuts — Eq. (15): heuristic inventory cuts (iterative loop)
# ---------------------------------------------------------------------------


def _add_h_cuts(
    bqm: dimod.BinaryQuadraticModel,
    cuts: Collection[Tuple[str, int, int]],
    P3: float,
) -> int:
    """
    Add inventory-cut diagonal penalties to *bqm* (Eq. 15).

    H_cuts = P3 * Σ_{(j,m,t) ∈ F} x_{jmt}                            (Eq. 15)

    Cuts penalise specific (vessel, monobuoy, slot) triples identified by
    the classical inventory layer as causing a 4-day ullage violation.
    Each cut adds only a diagonal bias — Q-matrix density is unaffected.

    P3 must satisfy the penalty hierarchy P3 < P1, asserted in
    :func:`build_qubo`.  If P3 ≥ P1, the solver would prefer dropping a
    vessel entirely (cost P1) over placing it in a flagged slot (cost P3),
    causing vessels to silently vanish from the schedule.

    Args:
        bqm:  BQM to update in place.
        cuts: Iterable of ``(vessel_id, machine, slot)`` triples from the
            classical inventory checker.  Corresponds to F^(k) in Eq. (15).
        P3:   Inventory-cut penalty.

    Returns:
        Number of cut terms added.
    """
    n_cuts: int = 0
    for vessel_id, machine, slot in cuts:
        var: str = f"x_{vessel_id}_{machine}_{slot}"
        if var not in bqm.variables:
            logger.warning(
                "Cut variable %s does not exist in the BQM — "
                "slot was never instantiated (infeasible or outside T_{j,m}).  "
                "Skipping.",
                var,
            )
            continue
        bqm.add_variable(var, P3)
        n_cuts += 1
    return n_cuts


# ---------------------------------------------------------------------------
# Orchestrator — build complete QUBO
# ---------------------------------------------------------------------------


def build_qubo(
    variables_df: pd.DataFrame,
    cuts: Optional[Collection[Tuple[str, int, int]]] = None,
    conflict_set: FrozenSet[Tuple[int, int]] = CONFLICT_SET_R,
    alpha: float = PENALTY_ALPHA,
    beta: float = PENALTY_BETA,
) -> Tuple[dimod.BinaryQuadraticModel, float, float, float, float]:
    """
    Build the complete QUBO Hamiltonian from a feasible-slot table.

    H_QUBO = H_obj + H_assign + H_overlap [+ H_cuts]                  (Eq. 10)

    Construction order:

    1. Calibrate P1 = α²·n·c_max (Eq. 13a), P2 = α·n·c_max (Eq. 13b),
       P3 = P2/β (Eq. 13c).
    2. Add H_obj diagonal — cost coefficients (Eq. 5).
    3. Add H_assign diagonal and off-diagonal — expanded with x²=x (Eq. 6).
    4. Add H_overlap off-diagonal — clamped collision bounds (Eqs. 7–9).
    5. Optionally inject H_cuts diagonal — inventory cuts (Eq. 15).
    6. Assert penalty hierarchy P1 > P2 > P3 > c_max (Eq. 16).
    7. Print Q-matrix statistics.

    Args:
        variables_df: Self-contained feasible-slot DataFrame produced by
            :func:`src.preprocessing.compute_feasible_slots`.  Must contain
            columns ``vessel_id``, ``machine``, ``slot``, ``p_j``, ``d_j``,
            ``w_j``.
        cuts: Optional collection of ``(vessel_id, machine, slot)`` triples
            to penalise with *P3*, corresponding to F^(k) in Eq. (15).
            Pass ``None`` on the first build (iteration k=0, no cuts yet).
        conflict_set: Resource-conflict set R.  Defaults to ``CONFLICT_SET_R``
            (shared pipeline).
        alpha: Penalty scaling factor α.  Defaults to ``PENALTY_ALPHA`` from
            ``src.config``.  Pass an explicit value to override per-request.
        beta: Secondary scaling factor β for inventory cuts (Eq. 13c).
            Defaults to ``PENALTY_BETA`` from ``src.config``.  Must be > 1.

    Returns:
        Tuple ``(bqm, P1, P2, P3, offset)`` where:

        - ``bqm``    — fully assembled :class:`dimod.BinaryQuadraticModel`.
        - ``P1``     — assignment penalty (α² · n · c_max).
        - ``P2``     — overlap penalty   (α  · n · c_max).
        - ``P3``     — inventory-cut penalty (P2 / β).
        - ``offset`` — constant energy shift from H_assign expansion
                       (= P1 × n_vessels); subtract to recover absolute energy.

    Raises:
        ValueError: If *variables_df* is empty.
        AssertionError: If the penalty hierarchy P1 > P2 > P3 > c_max
            is violated (Eq. 16).
    """
    if variables_df.empty:
        raise ValueError("variables_df is empty — cannot build QUBO.")

    # -- 1. Penalty calibration (Eqs. 13a–13c) ----------------------------
    P1, P2, P3 = calibrate_penalties(variables_df, alpha=alpha, beta=beta)

    # -- 2–5. Assemble BQM ------------------------------------------------
    bqm: dimod.BinaryQuadraticModel = dimod.BinaryQuadraticModel(vartype="BINARY")

    costs: Dict[str, float] = _add_h_obj(bqm, variables_df)
    offset: float = _add_h_assign(bqm, variables_df, P1)
    n_overlap: int = _add_h_overlap(bqm, variables_df, P2, conflict_set)

    n_cuts: int = 0
    if cuts:
        n_cuts = _add_h_cuts(bqm, cuts, P3)  # type: ignore[arg-type]

    # -- 6. Penalty hierarchy assertion (Eq. 16): P1 > P2 > P3 > c_max ----
    # Assertions follow the strict chain order so that the first violated
    # link in the chain is the one reported.
    c_max: float = max(costs.values()) if costs else 0.0

    assert P1 > P2, (
        f"Eq. 16 violated: P1 ({P1:.4f}) ≤ P2 ({P2:.4f}).  "
        "Assignment constraints are not dominant — increase α."
    )
    assert P2 > c_max, (
        f"Eq. 16 violated: P2 ({P2:.4f}) ≤ c_max ({c_max:.6f}).  "
        "Overlap penalty does not dominate the objective — increase α."
    )
    assert P3 < P1, (
        f"Eq. 16 violated: P3 ({P3:.4f}) ≥ P1 ({P1:.4f}).  "
        "Solver prefers dropping a vessel over rescheduling it — reduce β."
    )
    assert P3 < P2, (
        f"Eq. 16 violated: P3 ({P3:.4f}) ≥ P2 ({P2:.4f}).  "
        "Inventory cuts must not outweigh overlap penalties — reduce β."
    )
    assert P3 > c_max, (
        f"Eq. 16 violated: P3 ({P3:.4f}) ≤ c_max ({c_max:.6f}).  "
        "Inventory cuts must dominate cost coefficients — increase α or reduce β."
    )

    # -- 7. Q matrix statistics -------------------------------------------
    n_vars: int = len(bqm.variables)
    n_interactions: int = len(bqm.quadratic)
    max_possible: int = n_vars * (n_vars - 1) // 2
    density: float = n_interactions / max_possible if max_possible > 0 else 0.0

    cut_info: str = f", cut terms={n_cuts}" if cuts else ""
    print(
        f"[qubo_builder] BQM assembled: {n_vars} variables, "
        f"{n_interactions} interactions, "
        f"density={density:.6f} "
        f"(overlap couplers={n_overlap}{cut_info})."
    )

    return bqm, P1, P2, P3, offset
