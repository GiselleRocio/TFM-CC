"""
experiment_config.py — Constantes locales de experiments2.

Importa de src/config.py donde corresponde (DEFAULT_SEED, penalizaciones, etc.).
NO modifica src/config.py — es solo lectura.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import (
    DEFAULT_SEED,
    SLOT_HOURS,
    MIN_ULLAGE_DAYS,
    PENALTY_ALPHA,
    PENALTY_BETA,
    EPSILON,
    MAX_ITERATIONS,
    LEAP_TIME_LIMIT_S,
    SA_NUM_READS,
)

# ============================================================================
# Seeds globales
# ============================================================================

SEEDS: list[int] = list(range(5))  # seeds 0–4

# ============================================================================
# Solver constants
# ============================================================================

GUROBI_THREADS: int    = 1      # aislar complejidad algorítmica
GUROBI_TIMELIMIT_S: float = 300.0  # 5 min estrictos por instancia

N_RUNS_SA: int  = 30
N_RUNS_LH: int  = 15
N_RUNS_QPU: int = 15

# ============================================================================
# Exp 2 — SA Sweep grid
# ============================================================================

EXP2_SA_SWEEP = {
    "alpha_grid":        [2.0, 3.0, 5.0, 7.0, 10.0],
    "beta_grid":         [2.0],   # β no afecta factibilidad sin cuts; fijo en 2.0
    "instances":         ["Size_1", "Cong_3"],
    "n_instances":       2,
    "n_seeds_per_sweep": 5,
    "n_runs_sa":         20,
    "sa_num_reads":      200,    # más reads = más diversidad de muestras
    "sa_num_sweeps":     1000,   # más pasos por read = más chances de escapar mínimos
}

# ============================================================================
# Exp 3
# ============================================================================

EXP3_LH_RUNS: int  = 10           # reduced from 25 to fit 60-min QPU budget
EXP3_QPU_RUNS: int = N_RUNS_QPU
EXP3_SA_RUNS_BASELINE: int = 25   # SA on the 7 LH instances (SA is free)

# ============================================================================
# Exp 5
# ============================================================================

EXP5_K_MAX: int  = 10
EXP5_LH_RUNS: int = 3

# ============================================================================
# Exp 6
# ============================================================================

EXP6_LH_RUNS: int = 5
EXP6_INSTANCES: list[str] = ["Size_1", "Cong_3"]

# ============================================================================
# Exp 8
# ============================================================================

EXP8_QPU_INSTANCES: list[str] = ["Tiny_3", "Size_1"]

# ============================================================================
# Misc
# ============================================================================

SAVE_COMMIT_HASH: bool = True
