"""
src/preprocessing.py — Nomination validation and feasible slot computation.

Responsibilities (strict separation — see src/CLAUDE.md):
- Validate the nomination DataFrame for structural feasibility before any
  QUBO is constructed.
- Compute the feasible starting-slot set T_{j,m} for every (vessel, monobuoy)
  pair following qubo_formulation.md Eq. (1).
- Return the full (vessel, machine, slot) triple table that defines exactly
  which binary variables will exist in the QUBO.

Nothing in this module touches the BQM, the sampler, or tank inventory.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from src.config import BLOCKED_SLOTS, MACHINES, MIN_ULLAGE_DAYS, T

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Nomination validation
# ---------------------------------------------------------------------------


def validate_nominations(
    nominations: pd.DataFrame,
    horizon_slots: int = T,
    min_ullage_days: int = MIN_ULLAGE_DAYS,
) -> bool:
    """
    Check the nomination DataFrame for structural feasibility.

    Performs four checks in order:

    1. **Schema** — required columns are present with non-null values.
    2. **Horizon bounds** — ``r_j`` is non-negative and ``d_j ≤ horizon_slots``.
    3. **Tight windows** — warns for any vessel whose scheduling window
       ``[r_j, d_j]`` is narrower than ``p_j`` slots (the vessel cannot
       complete on time regardless of slot assignment).
    4. **Minimum ullage buffer** — warns when ``w_j`` (Equivalent Stock in Days)
       is below *min_ullage_days*, flagging nominations that may trigger
       inventory infeasibility cuts in the iterative QUBO loop.

    This function never raises on soft violations; it logs warnings and
    returns ``False`` so the caller can decide whether to abort or proceed
    to QUBO construction.

    Args:
        nominations: DataFrame produced by
            :func:`src.config.generate_nominations`, containing at minimum
            the columns ``vessel_id``, ``r_j``, ``d_j``, ``p_j``, ``w_j``
            (``p_ws_j`` is checked for schema compatibility with experiment
            notebooks but is not used in scheduling logic).
        horizon_slots: Total number of macro-slots in the planning horizon.
            Defaults to ``src.config.T``.  Must match the value passed to
            :func:`compute_feasible_slots` — using different values will
            cause vessels with ``d_j > horizon_slots`` to pass validation
            but have no feasible slots.
        min_ullage_days: Minimum idle-gap threshold in calendar days
            used for the ESD warning.  Defaults to
            ``src.config.MIN_ULLAGE_DAYS``.

    Returns:
        ``True`` if all checks pass with no violations; ``False`` if any
        warning or hard error was found.  Hard errors (missing columns,
        negative release slots) are logged at ERROR level; soft violations
        (tight windows, buffer risk) at WARNING level.

    Raises:
        KeyError: If a required column is entirely absent from the DataFrame
            (distinct from having null values, which is a soft error).
    """
    required_cols: List[str] = [
        "vessel_id", "r_j", "d_j", "p_j", "w_j",
    ]
    feasible: bool = True

    # -- 1. Schema --------------------------------------------------------
    missing = [c for c in required_cols if c not in nominations.columns]
    if missing:
        logger.error("Nomination DataFrame is missing required columns: %s.", missing)
        raise KeyError(missing)

    null_counts = nominations[required_cols].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        logger.error(
            "Nomination DataFrame contains null values: %s.", null_cols.to_dict()
        )
        feasible = False

    n: int = len(nominations)
    if n == 0:
        logger.error("Nomination DataFrame is empty — nothing to schedule.")
        return False

    # -- 2. Horizon bounds ------------------------------------------------
    bad_r = nominations[nominations["r_j"] < 0]
    if not bad_r.empty:
        logger.error(
            "Vessels with negative release slots (r_j < 0): %s.",
            bad_r["vessel_id"].tolist(),
        )
        feasible = False

    bad_d = nominations[nominations["d_j"] > horizon_slots]
    if not bad_d.empty:
        logger.error(
            "Vessels with due slot beyond horizon (d_j > T=%d): %s.",
            horizon_slots,
            bad_d["vessel_id"].tolist(),
        )
        feasible = False

    bad_rd = nominations[nominations["d_j"] < nominations["r_j"] + nominations["p_j"]]
    if not bad_rd.empty:
        logger.error(
            "Vessels where d_j < r_j + p_j (impossible to complete on time): %s.",
            bad_rd["vessel_id"].tolist(),
        )
        feasible = False

    # -- 3. Tight windows (soft) ------------------------------------------
    # Flag vessels where d_j == r_j + p_j: exactly one slot achieves zero
    # tardiness and any delay pushes completion past the due date entirely.
    # Even a single blocked slot at that position leaves no on-time option.
    tight = nominations[nominations["d_j"] == nominations["r_j"] + nominations["p_j"]]
    if not tight.empty:
        logger.warning(
            "Vessels with zero scheduling margin (d_j == r_j + p_j) — "
            "exactly one zero-tardiness slot exists, no buffer: %s.",
            tight["vessel_id"].tolist(),
        )

    # -- 4. 4-day minimum ullage buffer (soft) -----------------------------
    # w_j = stock_acumulado_m3 / daily_inflow_m3 is the Equivalent Stock
    # in Days (ESD) metric.  w_j < 4.0 means the cargador has fewer than
    # 4 days of its own production accumulated in the tanks — below the
    # mandatory minimum buffer threshold, making it a candidate for
    # inventory infeasibility cuts in the iterative QUBO loop.
    # This is a pre-filter warning; tank simulation is out of scope here.
    buffer_risk = nominations[nominations["w_j"] < float(min_ullage_days)]
    if not buffer_risk.empty:
        logger.warning(
            "Vessels with Equivalent Stock in Days below the %d-day buffer "
            "threshold (w_j < %d ESD): %s.  Inventory cuts may be needed.",
            min_ullage_days,
            min_ullage_days,
            buffer_risk["vessel_id"].tolist(),
        )
        # Buffer risk is informational — does not flip feasible to False.

    if feasible:
        logger.info(
            "Nomination validation passed for %d vessels.  "
            "w_j ∈ [%.2f, %.2f] ESD.",
            n,
            float(nominations["w_j"].min()),
            float(nominations["w_j"].max()),
        )
    return feasible


# ---------------------------------------------------------------------------
# 2. Feasible slot computation — T_{j,m}  (Eq. 1)
# ---------------------------------------------------------------------------


def compute_feasible_slots(
    nominations: pd.DataFrame,
    machines: Optional[List[int]] = None,
    blocked_slots_map: Optional[Dict[int, Set[int]]] = None,
    horizon_slots: int = T,
) -> pd.DataFrame:
    """
    Compute the feasible starting-slot set T_{j,m} for every (vessel, monobuoy)
    pair and return the full table of valid (vessel_id, machine, slot) triples.

    Implements qubo_formulation.md Eq. (1):

        T_{j,m} = { t ∈ {0,…,T-1} :
                    t ≥ r_j,
                    t + p_j ≤ T,
                    [t, t + p_j - 1] ∩ B_m = ∅ }

    Only triples returned by this function are ever instantiated as BQM
    variables — infeasible slots are excluded structurally before QUBO
    construction, keeping the Q matrix as sparse as possible.

    The expected BQM variable name for each row is:
        ``f"x_{vessel_id}_{machine}_{slot}"``

    Args:
        nominations: DataFrame produced by
            :func:`src.config.generate_nominations`, containing at minimum
            the columns ``vessel_id``, ``r_j``, ``p_j``.
        machines: List of monobuoy indices to consider.  Defaults to
            ``src.config.MACHINES`` when ``None``.  Pass an explicit list
            to support terminals with more than two monobuoys without
            modifying ``config.py``.
        blocked_slots_map: Mapping from monobuoy index → set of blocked
            macro-slot indices.  Defaults to ``src.config.BLOCKED_SLOTS``
            when ``None``.  Monobuoys absent from the map are treated as
            having no blocked slots.
        horizon_slots: Total number of macro-slots in the planning horizon.
            Defaults to ``src.config.T``.  Pass an explicit value to support
            different slot durations per request without modifying ``config.py``
            (e.g. ``horizon_days * (24 // slot_duration_hours)``).

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``vessel_id`` (str), ``machine`` (int), ``slot`` (int),
        ``p_j`` (int), ``d_j`` (int), ``w_j`` (float),
        sorted by ``(vessel_id, machine, slot)``.  The DataFrame is
        self-contained — ``qubo_builder.py`` does not need to join back
        to the nominations table.  The DataFrame index is reset (0-based integers).

        Total row count equals N_vars — the number of binary variables in
        the QUBO (see qubo_formulation.md Eq. 14).

    Raises:
        ValueError: If *nominations* is empty.

    Side effects:
        Prints a one-line summary of total variables instantiated.
        Logs a WARNING for any (vessel, monobuoy) pair with an empty
        feasible slot set — such a vessel cannot be scheduled on that
        monobuoy and the QUBO will have no variable to assign it there.
    """
    if nominations.empty:
        raise ValueError("nominations DataFrame is empty — nothing to expand.")

    _machines: List[int] = machines if machines is not None else MACHINES
    _blocked: Dict[int, Set[int]] = blocked_slots_map if blocked_slots_map is not None else BLOCKED_SLOTS

    records: list[dict] = []

    for _, row in nominations.iterrows():
        vessel_id: str = str(row["vessel_id"])
        r_j: int = int(row["r_j"])
        p_j: int = int(row["p_j"])
        d_j: int = int(row["d_j"])
        w_j: float = float(row["w_j"])

        for machine in _machines:
            blocked: Set[int] = _blocked.get(machine, set())
            feasible_slots: List[int] = []

            for t in range(r_j, horizon_slots):
                # Horizon bound: processing must finish within the horizon.
                if t + p_j > horizon_slots:
                    break
                # Blocked-slot check: no slot in [t, t+p_j-1] may be blocked.
                if blocked.isdisjoint(range(t, t + p_j)):
                    feasible_slots.append(t)

            if not feasible_slots:
                logger.warning(
                    "Vessel %s has NO feasible slots on monobuoy %d "
                    "(r_j=%d, p_j=%d, T=%d).  "
                    "This vessel cannot be assigned to this monobuoy.",
                    vessel_id,
                    machine,
                    r_j,
                    p_j,
                    horizon_slots,
                )

            for t in feasible_slots:
                records.append(
                    {
                        "vessel_id": vessel_id,
                        "machine": machine,
                        "slot": t,
                        "p_j": p_j,
                        "d_j": d_j,
                        "w_j": w_j,
                    }
                )

    if not records:
        logger.error(
            "No feasible (vessel, machine, slot) triples found.  "
            "The QUBO would have zero variables — check nominations and BLOCKED_SLOTS."
        )
        return pd.DataFrame(columns=["vessel_id", "machine", "slot"])

    df: pd.DataFrame = (
        pd.DataFrame(records)
        .sort_values(["vessel_id", "machine", "slot"])
        .reset_index(drop=True)
    )

    # -- Post-check: every vessel must appear in at least one (machine, slot) --
    # A vessel with zero rows in `df` cannot be assigned by H_assign and will
    # always incur its full penalty.  Surface this as an explicit error rather
    # than letting it silently degrade the QUBO solution quality.
    all_vessel_ids: set[str] = set(nominations["vessel_id"].astype(str))
    scheduled_vessel_ids: set[str] = set(df["vessel_id"].astype(str))
    unschedulable: set[str] = all_vessel_ids - scheduled_vessel_ids
    if unschedulable:
        logger.error(
            "Vessels with NO feasible slot on ANY monobuoy — "
            "they will be absent from the QUBO and cannot be scheduled: %s.  "
            "Check r_j, p_j, T and BLOCKED_SLOTS for these vessels.",
            sorted(unschedulable),
        )

    n_vars: int = len(df)
    n_vessels: int = nominations["vessel_id"].nunique()
    n_schedulable: int = len(scheduled_vessel_ids)
    print(
        f"[preprocessing] {n_vars} BQM variables instantiated "
        f"({n_schedulable}/{n_vessels} vessels schedulable × {len(_machines)} monobuoys, "
        f"T={horizon_slots} slots, blocked windows applied)."
    )

    return df
