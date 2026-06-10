"""
src/config.py — Single source of truth for the scheduling problem.

All QUBO parameters, physical infrastructure constants, and solver
hyperparameters are defined here.  Every other module must import from
this file; never hardcode T, MACHINES, or penalty values elsewhere.

Tank inventory configuration is intentionally excluded — see inventory.py.

References
----------
- qubo_formulation.md — full mathematical specification
- CLAUDE.md           — architecture rules and constraint definitions
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planning horizon
# ---------------------------------------------------------------------------

SLOT_HOURS: int = 12
"""Duration of one macro-slot in hours."""

HORIZON_DAYS: int = 31
"""Planning horizon in calendar days.  T is derived from this value."""

T: int = HORIZON_DAYS * (24 // SLOT_HOURS)
"""
Total number of macro-slots in the planning horizon (= HORIZON_DAYS × 2).

Slot indices run from 0 to T-1.  A variable x_{j,m,t} is only valid
when t + p_j ≤ T (see Eq. 1).  Always derived from HORIZON_DAYS and
SLOT_HOURS — never hardcoded.
"""

PLANNING_START: datetime = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)
"""
Reference UTC timestamp for macro-slot 0.

Slot t corresponds to PLANNING_START + t * SLOT_HOURS hours.
"""

# ---------------------------------------------------------------------------
# Physical infrastructure
# ---------------------------------------------------------------------------

MACHINES: List[int] = [1, 2]
"""
Monobuoy indices at the  terminal.

Indexed 1-based to match the operational naming convention used by the
terminal operator (Monobuoy 1 = M1, Monobuoy 2 = M2).
"""

CONFLICT_SET_R: FrozenSet[Tuple[int, int]] = frozenset(
    {(1, 1), (1, 2), (2, 1), (2, 2)}
)
"""
Conflict set R ⊆ M × M encoding the shared underwater pipeline.

The terminal has two monobuoys (M1, M2) but a single shared submarine
pipeline.  Therefore NO two vessels may load simultaneously, regardless
of which monobuoy each is assigned to.  All four monobuoy-pair
combinations are conflicting:

    R = {(1,1), (1,2), (2,1), (2,2)}                        (Eq. 11)

This makes the effective scheduling problem equivalent to a single-machine
serialisation constraint:  Rm | rj | Σ wj Tj  with m = 1 effective machine.

H_overlap must iterate over all pairs (m1, m2) ∈ R for j < k vessel pairs.
See qubo_formulation.md §7 and §10.
"""

MIN_ULLAGE_DAYS: int = 4
"""
Minimum ullage buffer (receiving capacity) the terminal must maintain at all times, in calendar days.

Operationally: the terminal must keep at least 4 days' worth of available storage
capacity (ullage) free in the tank farm.  If a vessel is loaded too late, the
projected inventory will exceed the safe maximum limit, causing an ullage violation
that halts upstream production.  The iterative QUBO loop uses this threshold to
detect worst-case temporal overlaps between consecutive vessel loading operations
that would leave insufficient time to free up tank capacity before overflow.

This is a viability constraint, NOT a tank model.  Full per-slot inventory
simulation is out of scope for this TFM — see CLAUDE.md §PHYSICAL CONSTRAINTS §2.

Renamed from MIN_STOCK_BUFFER_DAYS to reflect that this represents available
receiving capacity (ullage), not stored inventory buffer.
"""

WORST_CASE_PROCESSING_RATIO: int = 2
"""
DEPRECATED — no longer used in the core scheduling pipeline.

p_{j,m} (processing_slots) now represents the full pipeline-blocking
duration directly, so a separate worst-case multiplier is not needed.
The buffer check in inventory.py uses completion_slot = start_slot + p_j.

Kept here for reference in case experiments notebooks still reference it.
Never enters any QUBO equation or the API layer.
"""

# ---------------------------------------------------------------------------
# Tank farm infrastructure
# ---------------------------------------------------------------------------

N_TANKS: int = 6
"""Number of identical storage tanks at the terminal."""

TANK_CAPACITY_M3: float = 100_000.0
"""Physical volume capacity of a single tank in cubic meters."""

INITIAL_TERMINAL_STOCK_M3: float = 300_000.0
"""Initial total crude oil volume already in the tanks at slot 0 (m³)."""

DAILY_INFLOW_M3: float = 20_000.0
"""Daily crude oil inflow rate from upstream production (m³/day)."""

BLOCKED_SLOTS: Dict[int, Set[int]] = {
    1: {10, 11, 30, 31},
    2: {10, 11, 50, 51},
}
"""
Pre-computed maintenance and tugboat-unavailability windows per monobuoy.

Keys are monobuoy indices (matching MACHINES).  Values are sets of
macro-slot indices during which that monobuoy is unavailable for loading.

These slots are excluded structurally from the feasible set T_{jm} in
preprocessing — no QUBO penalty is needed.  See Eq. (1):

    T_{jm} = { t : t ≥ r_j,  t + p_j ≤ T,  [t, t+p_j-1] ∩ B_m = ∅ }

Defaults encode two synthetic 24-hour maintenance windows per buoy spread
within the 31-day horizon.  Production runs replace this dict with windows
derived from the terminal's actual maintenance schedule.
"""

# ---------------------------------------------------------------------------
# Solver hyperparameters
# ---------------------------------------------------------------------------

PENALTY_ALPHA: float = 3.0
"""
Safety-margin multiplier α for penalty calibration (Eqs. 13a–13b).

Penalties are computed as:

    P1 = α² × n × c_max     (Eq. 13a — assignment penalty)
    P2 = α  × n × c_max     (Eq. 13b — overlap penalty)

where n is the number of vessels and c_max = max_{j,m,t} c_{jmt}.

The ratio P1/P2 = α > 1 ensures assignment constraints are penalised
more heavily than pipeline-overlap constraints.  α = 3.0 provides
headroom against analogue precision effects in the D-Wave annealer.
It is NOT a hyperparameter to tune freely; see the two-phase calibration
sweep in qubo_formulation.md §11.
"""

PENALTY_BETA: float = 2.0
"""
Secondary penalty scaling factor β for inventory-cut calibration (Eq. 13c).

    P3 = P2 / β

Controls how aggressively the hybrid loop penalises slots that triggered an
ullage violation.  With P2 = α·n·c_max:

    P3 = (α/β)·n·c_max

The hierarchy P1 > P2 > P3 > c_max requires β > 1.  The nominal value β = 2.0
guarantees P3 = P2/2 < P2, and P3 > c_max whenever (α/β)·n > 1, i.e. n ≥ 1
for α = 3.0 ✓.

Must be passed to every call to the iterative loop and to P3 derivation inside
the solver loop.  See qubo_formulation.md §11 and §13.
"""

EPSILON: float = 1e-2
"""
Symmetry-breaking perturbation ε added to the cost coefficient (Eq. 3).

    c_{jmt} = w_j · max(0, t + p_j - d_j) + ε · (t + p_j)

ε must simultaneously satisfy (Eq. 4):

    (i)  ε ≪ min_{j,m,t | c_jmt > 0} c_jmt   — commercial optimality preserved
    (ii) ε × (T + max_j p_j) ≳ δ_QPU ≈ 1e-2  — perceptible after D-Wave rescaling

With heterogeneous weights w_j ~ stock/inflow (range ≈ 5–50 days) and
T ≈ 53–67 slots, ε=1e-2 gives ε_eff ≈ 0.61, which is ~12% of w_min≈5
(visible tiebreaker) and << w_min × 1 slot (preserves commercial ordering).
Both bounds must be verified empirically during the penalty calibration sweep.
"""

MAX_ITERATIONS: int = 10
"""
Maximum number of iterative QUBO-update cycles K (Eq. 17).

If the inventory buffer is still violated after K iterations, the best
solution found is surfaced with a converged=False diagnostic flag.
Never silently fail.
"""

LEAP_TIME_LIMIT_S: int = 60
"""
Wall-clock time limit in seconds for LeapHybridSampler.

The primary solver.  Requires DWAVE_API_TOKEN in the environment.
60 s is the minimum recommended value for instances up to ~10,000 variables;
for larger BQMs the solver auto-scales above this floor (see solver.py).
"""

SA_NUM_READS: int = 1_000
"""
Number of reads for the SimulatedAnnealingSampler offline fallback.

Used when DWAVE_API_TOKEN is absent or the Leap cloud is unreachable.
Never use DWaveSampler directly — Q-matrix density makes native QPU
embedding infeasible for this instance size (see CLAUDE.md §SOLVER STRATEGY).
"""

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def slot_to_datetime(slot: int) -> datetime:
    """
    Convert a macro-slot index to its corresponding UTC datetime.

    Slot 0 maps to PLANNING_START; each subsequent slot advances by
    SLOT_HOURS hours.

    Args:
        slot: Non-negative macro-slot index in the range [0, T].
            Passing T is valid (it represents the end of the horizon).

    Returns:
        UTC-aware :class:`datetime` for the start of the given slot.

    Raises:
        ValueError: If *slot* is negative or exceeds T.

    Example:
        >>> slot_to_datetime(0) == PLANNING_START
        True
        >>> slot_to_datetime(2)
        datetime.datetime(2026, 3, 15, 0, 0, tzinfo=datetime.timezone.utc)
    """
    if slot < 0 or slot > T:
        raise ValueError(f"slot must be in [0, {T}], got {slot}.")
    return PLANNING_START + timedelta(hours=slot * SLOT_HOURS)


# ---------------------------------------------------------------------------
# Internal generation parameters (not part of the public API)
# ---------------------------------------------------------------------------

#: Uniform draw range for accumulated stock per cargador in the terminal tanks (m³).
#: Must exceed _VOLUME_MAX_M3 so that vessel cargo is always a subset of available stock.
_STOCK_MIN_M3: float = 150_000.0
_STOCK_MAX_M3: float = 500_000.0

#: Uniform draw range for cargo volume nominations (m³).
#: Represents the portion of the cargador's accumulated stock loaded onto this vessel.
_VOLUME_MIN_M3: float = 50_000.0
_VOLUME_MAX_M3: float = 150_000.0

#: Uniform draw range for each cargador's daily crude oil injection rate (m³/day).
#: Per-cargador: different source routes inject at different throughput rates.
_INFLOW_MIN_M3_DAY: float = 10_000.0
_INFLOW_MAX_M3_DAY: float = 30_000.0

#: Physical loading durations in calendar days: 2 days (small vessels), 4 days (large).
#: Converted to macro-slots using SLOT_HOURS so this stays correct if the time
#: discretisation changes.  With SLOT_HOURS=12 → [4, 8]; with SLOT_HOURS=24 → [2, 4].
_PROC_DAYS: List[int] = [2, 4]
_PROC_OPTIONS: List[int] = [d * (24 // SLOT_HOURS) for d in _PROC_DAYS]

DEFAULT_SEED: int = 8
"""
Default RNG seed for synthetic data generation.

Used by ``generate_nominations`` and the ``/generate`` API endpoint to
guarantee reproducible results across runs.  Pass ``None`` explicitly only
when a non-deterministic draw is intentional (e.g. Monte Carlo studies).
"""


# ---------------------------------------------------------------------------
# Synthetic nominations generator
# ---------------------------------------------------------------------------


def generate_nominations(
    n: int,
    seed: int | None = DEFAULT_SEED,
    n_machines: int | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic nomination table for *n* vessels.

    Produces all vessel-level parameters required by the QUBO formulation
    (qubo_formulation.md §1) and the preprocessing layer:

    - ``r_j``            : release slot — earliest macro-slot at which vessel j
                           may begin loading (integer, ≥ 0).
    - ``d_j``            : due slot — preferred completion deadline in macro-slots
                           (integer, > r_j + p_j).
    - ``p_j``            : processing time in macro-slots ∈ {4, 8}.  4 slots = 2 days
                           (small vessels), 8 slots = 4 days (large vessels).  Equal
                           for both monobuoys in this terminal.
    - ``p_ws_j``              : DEPRECATED — retained in the generated DataFrame for
                               backwards compatibility with experiment notebooks.
                               The core pipeline uses ``p_j`` for the buffer check
                               (completion_slot = start_slot + p_j).
    - ``stock_acumulado_m3``  : crude oil volume accumulated by the cargador in the
                               terminal tanks at the time of nomination (m³).
                               This is the stock level that drives scheduling urgency,
                               not the vessel's cargo size.
    - ``volume_m3``           : cargo volume loaded onto this vessel (m³).  Always
                               ≤ stock_acumulado_m3 — the vessel takes a portion of
                               the cargador's accumulated stock.
    - ``daily_inflow_m3``     : daily crude oil injection rate of this cargador (m³/day).
                               Per-cargador value; different source routes inject at
                               different throughput rates.
    - ``w_j``                 : priority weight = stock_acumulado_m3 / daily_inflow_m3
                               (Equivalent Stock in Days).  Measures how many days of
                               that cargador's own production is sitting in the tanks.
                               Higher value ⟹ more urgent ⟹ should be scheduled earlier.

    Args:
        n: Number of vessels (nominations) to generate.  Must be ≥ 1 and
            small enough that all vessels can be sequentially scheduled
            within the T-slot horizon.
        seed: Seed for ``numpy.random.default_rng``.  Pass ``None`` for a
            non-reproducible draw suitable for Monte Carlo studies.

    Returns:
        A :class:`pandas.DataFrame` with one row per vessel and columns:
        ``vessel_id``, ``r_j``, ``d_j``, ``p_j``, ``stock_acumulado_m3``,
        ``volume_m3``, ``daily_inflow_m3``, ``w_j``
        (plus ``p_ws_j`` retained for experiment notebook compatibility).
        The DataFrame index is reset (0-based integers).

    Raises:
        ValueError: If *n* < 1, or if *n* vessels at maximum processing
            time cannot all fit within the T-slot planning horizon.

    Example:
        >>> df = generate_nominations(n=8, seed=0)
        >>> list(df.columns)
        ['vessel_id', 'r_j', 'd_j', 'p_j', 'p_ws_j', 'stock_acumulado_m3', 'volume_m3', 'daily_inflow_m3', 'w_j']
        >>> (df["w_j"] > 0).all()
        True
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}.")

    max_p: int = max(_PROC_OPTIONS)
    effective_machines: int = n_machines if n_machines is not None else len(MACHINES)
    if n * max_p > T * effective_machines:
        raise ValueError(
            f"Cannot schedule {n} vessels (max p_j = {max_p} slots each) "
            f"within T = {T} slots × {effective_machines} machines.  Reduce n or extend the horizon."
        )

    rng: np.random.Generator = np.random.default_rng(seed)

    # -- Processing times (macro-slots) -----------------------------------
    # Small vessels require 2 days (4 slots); large vessels require 4 days (8 slots).
    p_j: np.ndarray = rng.choice(_PROC_OPTIONS, size=n)

    # Worst-case processing time: used ONLY by the classical inventory check.
    p_ws_j: np.ndarray = WORST_CASE_PROCESSING_RATIO * p_j

    # -- Release slots (no duplicates) ------------------------------------
    # Arrivals are spread across the first two-thirds of the horizon so that
    # vessels have meaningful scheduling flexibility and the problem is
    # non-trivial (not all vessels arriving at slot 0).
    # Sampling without replacement guarantees distinct arrival slots, which
    # avoids degenerate ties in the release-slot ordering.
    horizon_2_3: int = max(n, int(T * 2 / 3))
    r_j: np.ndarray = np.sort(
        rng.choice(horizon_2_3, size=n, replace=False).astype(int)
    )

    # -- Due slots --------------------------------------------------------
    # d_j must allow completion within the horizon: d_j ≥ r_j + p_j.
    # A minimum slack of 2 extra slots (24 h) prevents zero-margin windows
    # where the due slot equals the earliest possible completion — such
    # windows would force maximum tardiness cost on every feasible slot.
    # An additional random slack of 0–8 slots models commercial tolerance.
    slack: np.ndarray = 2 + rng.integers(0, 9, size=n)
    d_j: np.ndarray = np.minimum(r_j + p_j + slack, T)

    # -- Accumulated stock per cargador (m³) ------------------------------
    # How much crude oil the cargador has stored in the terminal tanks at
    # nomination time.  Range [_STOCK_MIN_M3, _STOCK_MAX_M3] always exceeds
    # the vessel's cargo volume, ensuring the physical constraint
    # volume_m3 ≤ stock_acumulado_m3 holds by construction.
    stock_acumulado_m3: np.ndarray = rng.uniform(_STOCK_MIN_M3, _STOCK_MAX_M3, size=n)

    # -- Cargo volumes (m³) -----------------------------------------------
    # Volume loaded onto this specific vessel — a portion of the cargador's
    # accumulated stock.  Independent draw within [_VOLUME_MIN_M3, _VOLUME_MAX_M3],
    # which lies entirely below _STOCK_MIN_M3, so volume_m3 ≤ stock_acumulado_m3.
    volume_m3: np.ndarray = rng.uniform(_VOLUME_MIN_M3, _VOLUME_MAX_M3, size=n)

    # -- Daily injection rate per cargador (m³/day) -----------------------
    # Each cargador injects crude at its own throughput rate depending on
    # the source route.  Varied per nomination to produce a non-trivial
    # spread of ESD priority weights.
    daily_inflow_m3: np.ndarray = rng.uniform(
        _INFLOW_MIN_M3_DAY, _INFLOW_MAX_M3_DAY, size=n
    )

    # -- Priority weights: Equivalent Stock in Days -----------------------
    # w_j = stock_acumulado_m3 / daily_inflow_m3.
    # Measures how many days of that cargador's own production is held in
    # the tanks.  Higher value ⟹ more days of stock waiting ⟹ higher urgency.
    w_j: np.ndarray = stock_acumulado_m3 / daily_inflow_m3

    # -- Vessel IDs -------------------------------------------------------
    vessel_ids: List[str] = [f"V{i + 1:02d}" for i in range(n)]

    df: pd.DataFrame = pd.DataFrame(
        {
            "vessel_id": vessel_ids,
            "r_j": r_j.astype(int),
            "d_j": d_j.astype(int),
            "p_j": p_j.astype(int),
            "p_ws_j": p_ws_j.astype(int),
            "stock_acumulado_m3": stock_acumulado_m3,
            "volume_m3": volume_m3,
            "daily_inflow_m3": daily_inflow_m3,
            "w_j": w_j,
        }
    )

    logger.info(
        "Generated %d synthetic nominations (seed=%s).  "
        "w_j ∈ [%.2f, %.2f] ESD.  r_j ∈ [%d, %d].  p_j values: %s.",
        n,
        seed,
        float(w_j.min()),
        float(w_j.max()),
        int(r_j.min()),
        int(r_j.max()),
        sorted(int(v) for v in np.unique(p_j)),
    )

    return df
