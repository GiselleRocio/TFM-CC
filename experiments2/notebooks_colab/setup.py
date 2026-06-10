# ================================================================
# CELDA 0 — COLAB BOOTSTRAP  (ejecutar primero, una sola vez)
# ================================================================

DRIVE_TESIS_PATH = "MyDrive/TESIS"

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, shutil, subprocess, sys as _sys, importlib.util as _ilu

DRIVE_TESIS = f"/content/drive/{DRIVE_TESIS_PATH}"

for _desc, _req in [
    ("raiz del proyecto", DRIVE_TESIS),
    ("src/ (solver QUBO)", f"{DRIVE_TESIS}/src"),
    ("shared/ (utilities)", f"{DRIVE_TESIS}/shared"),
]:
    if not os.path.isdir(_req):
        raise FileNotFoundError(f"No encontrado: {_req}  ({_desc})")
print(f"Drive OK  ->  {DRIVE_TESIS}")

_e2    = f"{DRIVE_TESIS}/experiments2"
_e2_sh = f"{_e2}/shared"
os.makedirs(_e2_sh, exist_ok=True)
for _init in [f"{_e2}/__init__.py", f"{_e2_sh}/__init__.py"]:
    if not os.path.exists(_init):
        open(_init, "w").close()

_n = sum(
    1 for _f in os.listdir(f"{DRIVE_TESIS}/shared")
    if _f.endswith(".py")
    and shutil.copy2(f"{DRIVE_TESIS}/shared/{_f}", f"{_e2_sh}/{_f}") is None
)
print(f"  ok  experiments2/shared/ en Drive ({_n} modulos sincronizados desde shared/)")

_PKGS = [
    ("dimod",          "dimod"),
    ("dwave-samplers", "dwave.samplers"),
    ("dwave-system",   "dwave.system"),
    ("openpyxl",       "openpyxl"),
    ("seaborn",        "seaborn"),
]
for _pip_name, _mod in _PKGS:
    if _ilu.find_spec(_mod.split(".")[0]) is None:
        print(f"  instalando {_pip_name}...", end=" ", flush=True)
        subprocess.check_call(
            [_sys.executable, "-m", "pip", "install", "-q", _pip_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("listo")
    else:
        print(f"  ok  {_pip_name}")

if "DWAVE_API_TOKEN" not in os.environ:
    try:
        from google.colab import userdata
        os.environ["DWAVE_API_TOKEN"] = userdata.get("DWAVE_API_TOKEN")
        print("  ok  DWAVE_API_TOKEN (Colab Secrets)")
    except Exception:
        _env_path = f"{DRIVE_TESIS}/.env"
        if os.path.exists(_env_path):
            for _ln in open(_env_path):
                _ln = _ln.strip()
                if _ln and not _ln.startswith("#") and "=" in _ln:
                    _k, _v = _ln.split("=", 1)
                    os.environ[_k.strip()] = _v.strip()
            print("  ok  variables cargadas desde Drive/.env")
        else:
            print("  AVISO: DWAVE_API_TOKEN no configurado.")
else:
    print("  ok  DWAVE_API_TOKEN ya en el entorno")

print("\nBootstrap completo. Ejecutar las celdas siguientes.")


# ================================================================
# CELDA 1: SETUP
# ================================================================
import sys
import math
import logging
from pathlib import Path

try:
    REPO_ROOT = Path(DRIVE_TESIS)
except NameError:
    raise RuntimeError(
        "DRIVE_TESIS no definido. Ejecutar Celda 0 (COLAB BOOTSTRAP) primero."
    )

EXPERIMENTS2_DIR = REPO_ROOT / "experiments2"
SRC_DIR          = REPO_ROOT / "src"

for p in [str(SRC_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("setup")

logger.info("REPO_ROOT      : %s", REPO_ROOT)
logger.info("EXPERIMENTS2   : %s", EXPERIMENTS2_DIR)
logger.info("SRC_DIR        : %s", SRC_DIR)


# ================================================================
# CELDA 2: VERIFICAR DEPENDENCIAS
# ================================================================
import importlib
import importlib.util

REQUIRED_PACKAGES = {
    "numpy":       "numpy",
    "pandas":      "pandas",
    "scipy":       "scipy",
    "openpyxl":    "openpyxl",
    "matplotlib":  "matplotlib",
    "seaborn":     "seaborn",
    "dimod":       "dimod",
    "dwave.system": "dwave-system",
    "gurobipy":    "gurobipy",
}

missing = []
for module_name, pip_name in REQUIRED_PACKAGES.items():
    top_level = module_name.split(".")[0]
    spec   = importlib.util.find_spec(top_level)
    status = "OK" if spec is not None else "MISSING"
    if spec is None:
        missing.append(pip_name)
    print(f"  {'✓' if spec else '✗'}  {module_name:<20} [{status}]")

if missing:
    print(f"\n⚠  Faltan paquetes: {missing}")
    print("   Instalar con: pip install " + " ".join(missing))
else:
    print("\nTodas las dependencias disponibles.")


# ================================================================
# CELDA 3: IMPORTS PRINCIPALES
# ================================================================
from config import DEFAULT_SEED, SLOT_HOURS, T as T_DEFAULT, MACHINES

from experiments2.shared.experiment_config import (
    SEEDS,
    GUROBI_THREADS,
    GUROBI_TIMELIMIT_S,
    EXP2_SA_SWEEP,
    EXP3_LH_RUNS,
    EXP3_SA_RUNS_BASELINE,
    EXP5_LH_RUNS,
    EXP5_K_MAX,
    EXP6_LH_RUNS,
)
from experiments2.shared.io_utils import (
    ensure_directories,
    get_commit_hash,
    save_instances_to_excel,
    INSTANCES_EXCEL,
    RESULTS_DIR,
)

print("Todos los módulos importados correctamente.")


# ================================================================
# CELDA 4: MOSTRAR CONFIGURACIÓN
# ================================================================
print("=" * 60)
print("CONFIGURACIÓN GLOBAL (src/config.py — solo lectura)")
print("=" * 60)
print(f"  DEFAULT_SEED          = {DEFAULT_SEED}")
print(f"  SLOT_HOURS            = {SLOT_HOURS} h")
print(f"  T (horizonte base)    = {T_DEFAULT} slots  ({T_DEFAULT * SLOT_HOURS / 24:.1f} días)")
print(f"  MACHINES              = {MACHINES}")
print()
print("CONFIGURACIÓN DE EXPERIMENTOS (experiments2/shared/experiment_config.py)")
print("-" * 60)
print(f"  SEEDS                 = {SEEDS}")
print(f"  GUROBI_THREADS        = {GUROBI_THREADS}")
print(f"  GUROBI_TIMELIMIT_S    = {GUROBI_TIMELIMIT_S} s")
print(f"  EXP3_LH_RUNS          = {EXP3_LH_RUNS}")
print(f"  EXP3_SA_RUNS_BASELINE = {EXP3_SA_RUNS_BASELINE}")
print(f"  EXP5_LH_RUNS          = {EXP5_LH_RUNS}")
print(f"  EXP5_K_MAX            = {EXP5_K_MAX}")
print(f"  EXP6_LH_RUNS          = {EXP6_LH_RUNS}")
print()
print("EXP2 SA SWEEP GRID")
print("-" * 60)
print(f"  alpha_grid            = {EXP2_SA_SWEEP['alpha_grid']}")
print(f"  beta_grid             = {EXP2_SA_SWEEP['beta_grid']}")
total_sweep = (
    len(EXP2_SA_SWEEP["alpha_grid"])
    * len(EXP2_SA_SWEEP["beta_grid"])
    * EXP2_SA_SWEEP["n_instances"]
    * EXP2_SA_SWEEP["n_seeds_per_sweep"]
    * EXP2_SA_SWEEP["n_runs_sa"]
)
print(f"  Total runs (sin podar) = {total_sweep:,}")


# ================================================================
# CELDA 5: CREAR DIRECTORIOS
# ================================================================
ensure_directories()
print(f"  ✓  {(EXPERIMENTS2_DIR / 'data').relative_to(REPO_ROOT)}")
print(f"  ✓  {RESULTS_DIR.relative_to(REPO_ROOT)}")


# ================================================================
# CELDA 6: COMMIT HASH
# ================================================================
commit_hash = get_commit_hash()
print(f"Commit hash actual : {commit_hash}")


# ================================================================
# CELDA 7: FUNCIONES DE GENERACIÓN DE INSTANCIAS
# ================================================================
import numpy as np
import pandas as pd


def _rho_effective(nominations_df: pd.DataFrame, T: int) -> float:
    return float(nominations_df["p_j"].sum()) / T


def _n_vars_approx(n: int, T: int, vlcc_pct: float = 0.25,
                   p_std: int = 4, p_vlcc: int = 8, n_machines: int = 2) -> int:
    n_vlcc = round(n * vlcc_pct)
    n_std  = n - n_vlcc
    return (n_std * max(0, T - p_std + 1) + n_vlcc * max(0, T - p_vlcc + 1)) * n_machines


def _conflict_density(nominations_df: pd.DataFrame) -> float:
    """Fracción de pares (j,k) con ventanas solapadas."""
    r = nominations_df["r_j"].values
    d = nominations_df["d_j"].values
    n = len(r)
    if n < 2:
        return 0.0
    total     = n * (n - 1) // 2
    conflicts = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if r[i] < d[j] and r[j] < d[i]
    )
    return round(conflicts / total, 4)


def _t_from_rho(n: int, vlcc_pct: float, rho_target: float) -> int:
    n_vlcc = round(n * vlcc_pct)
    n_std  = n - n_vlcc
    total_p = n_vlcc * 8 + n_std * 4
    return math.ceil(total_p / rho_target)


def _make_nominations(n: int, T: int, vlcc_pct: float,
                      r_j_distribution: str, seed: int,
                      min_conflict_density: float = 0.6,
                      collision_target: float | None = None) -> pd.DataFrame:
    p_max = max(8, round(vlcc_pct * 8 + (1 - vlcc_pct) * 4) + 2)
    max_density_seen = 0.0
    _valid_r_j: np.ndarray | None = None
    _valid_p_j: np.ndarray | None = None
    _valid_d_j: np.ndarray | None = None

    for attempt in range(20):
        rng    = np.random.default_rng(seed + attempt)
        n_vlcc = round(n * vlcc_pct)
        n_std  = n - n_vlcc
        p_j    = np.array([8] * n_vlcc + [4] * n_std, dtype=int)
        rng.shuffle(p_j)

        r_max = max(1, T - max(p_j) - 2)
        if collision_target is not None and r_j_distribution == "clustered":
            if attempt >= 15:
                r_max = max(1, int(r_max * 0.25))
            elif attempt >= 10:
                r_max = max(1, int(r_max * 0.50))
            elif attempt >= 5:
                r_max = max(1, int(r_max * 0.75))

        if r_j_distribution == "clustered":
            n_cluster     = max(1, round(n * 0.7))
            n_spread      = n - n_cluster
            cluster_time  = rng.integers(0, max(1, r_max - T // 3))
            spread_times  = rng.integers(0, max(1, r_max + 1), size=max(1, n_spread // 2))
            r_cluster     = np.full(n_cluster, cluster_time)
            r_spread      = rng.choice(spread_times, size=n_spread) if n_spread > 0 else np.array([])
            r_j           = np.concatenate([r_cluster, r_spread])

        elif r_j_distribution == "operational":
            r_op_max = max(1, 2 * T // 3)
            r_j      = rng.integers(0, r_op_max + 1, size=n)

        elif r_j_distribution == "uniform":
            n_batches   = max(1, n // 3)
            batch_times = rng.integers(0, max(1, r_max + 1), size=n_batches)
            r_j         = rng.choice(batch_times, size=n)

        elif r_j_distribution == "random":
            n_batches   = max(1, n // 3)
            batch_times = rng.integers(0, max(1, r_max // 2), size=n_batches)
            r_j         = rng.choice(batch_times, size=n)

        elif r_j_distribution == "bimodal":
            half  = n // 2
            peak1 = max(0, T // 6)
            peak2 = max(0, T // 2)
            r_j   = np.concatenate([np.full(half, peak1), np.full(n - half, peak2)])

        elif r_j_distribution == "hub_cluster":
            n_hub    = max(1, round(n * 0.6))
            n_spread = n - n_hub
            hub_time = max(0, T // 4)
            r_hub    = np.full(n_hub, hub_time)
            r_spread = rng.integers(0, max(1, r_max + 1), size=max(1, n_spread)) if n_spread > 0 else np.array([], dtype=int)
            r_j      = np.concatenate([r_hub, r_spread[:n_spread]])

        elif r_j_distribution == "symmetric_batches":
            K       = 4
            step    = max(1, T // (K + 1))
            batches = [step * (k + 1) for k in range(K)]
            base    = n // K
            counts  = [base] * K
            counts[-1] += n - base * K
            r_j = np.concatenate([np.full(cnt, t) for cnt, t in zip(counts, batches)])

        else:
            raise ValueError(f"r_j_distribution desconocida: {r_j_distribution!r}")

        r_j = np.sort(r_j.astype(int))

        t_free = 0
        fits_in_T = True
        for arr, proc in zip(r_j, p_j):
            start = max(t_free, arr)
            if start + proc > T:
                fits_in_T = False
                break
            t_free = start + proc

        if not fits_in_T:
            continue

        slack_vals = rng.integers(1, 4, size=n)
        d_j = np.minimum(r_j + p_j + slack_vals, T)
        d_j = np.maximum(d_j, r_j + p_j)
        _valid_r_j = r_j.copy()
        _valid_p_j = p_j.copy()
        _valid_d_j = d_j.copy()

        noms_temp = pd.DataFrame({"r_j": r_j, "d_j": d_j, "p_j": p_j})
        current_density = _conflict_density(noms_temp)
        max_density_seen = max(max_density_seen, current_density)

        target = collision_target if collision_target is not None else min_conflict_density
        if current_density >= target:
            break
    else:
        if collision_target is not None:
            raise ValueError(
                f"No se pudo alcanzar collision_target={collision_target} "
                f"para n={n}, T={T} después de 20 intentos. "
                f"Máxima densidad alcanzada: {max_density_seen:.4f}"
            )

    if _valid_r_j is not None:
        r_j = _valid_r_j
        p_j = _valid_p_j
        d_j = _valid_d_j
    else:
        _s  = np.random.default_rng(seed + 20).integers(1, 4, size=n)
        d_j = np.maximum(np.minimum(r_j + p_j + _s, T), r_j + p_j)

    stock_m3  = np.where(
        p_j == 8,
        np.exp(rng.normal(13.0, 0.3, size=n)),
        np.exp(rng.normal(12.5, 0.4, size=n)),
    )
    stock_m3  = np.clip(stock_m3, 150_000, 500_000)
    inflow_m3 = rng.uniform(10_000, 30_000, size=n)
    volume_m3 = rng.uniform(50_000, 150_000, size=n)
    w_j       = stock_m3 / inflow_m3

    return pd.DataFrame({
        "vessel_id":          [f"V{i+1:02d}" for i in range(n)],
        "r_j":                r_j.astype(int),
        "d_j":                d_j.astype(int),
        "p_j":                p_j.astype(int),
        "w_j":                w_j,
        "stock_acumulado_m3": stock_m3,
        "volume_m3":          volume_m3,
        "daily_inflow_m3":    inflow_m3,
    })


def _make_dens_nominations(
    n: int, T: int, vlcc_pct: float, seed: int,
    collision_target: float,
    tol: float = 0.05,
    max_attempts: int = 60,
) -> pd.DataFrame:
    """Genera nominations con collision_density ≈ collision_target (±tol)."""
    n_vlcc = round(n * vlcc_pct)
    n_std  = n - n_vlcc
    avg_p  = (n_vlcc * 8 + n_std * 4) / n
    _threshold = avg_p + 2.0
    window = 0 if collision_target >= 1.0 else max(0, int(2 * _threshold / max(collision_target, 0.05)))

    best_noms = None
    best_diff = float("inf")

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        p_j = np.array([8] * n_vlcc + [4] * n_std, dtype=int)
        rng.shuffle(p_j)
        r_j = np.zeros(n, dtype=int) if window == 0 else np.sort(rng.integers(0, window + 1, size=n))

        t_free, fits = 0, True
        for arr, proc in zip(r_j, p_j):
            start = max(t_free, int(arr))
            if start + proc > T:
                fits = False
                break
            t_free = start + proc
        if not fits:
            window = max(0, window - max(1, window // 5))
            continue

        slack_vals = rng.integers(1, 4, size=n)
        d_j    = np.maximum(np.minimum(r_j + p_j + slack_vals, T), r_j + p_j)
        density = _conflict_density(pd.DataFrame({"r_j": r_j, "d_j": d_j, "p_j": p_j}))
        diff    = abs(density - collision_target)

        if diff < best_diff:
            best_diff = diff
            stock_m3  = np.where(
                p_j == 8,
                np.exp(rng.normal(13.0, 0.3, size=n)),
                np.exp(rng.normal(12.5, 0.4, size=n)),
            )
            stock_m3  = np.clip(stock_m3, 150_000, 500_000)
            inflow_m3 = rng.uniform(10_000, 30_000, size=n)
            volume_m3 = rng.uniform(50_000, 150_000, size=n)
            best_noms = pd.DataFrame({
                "vessel_id":          [f"V{i+1:02d}" for i in range(n)],
                "r_j":                r_j.astype(int),
                "d_j":                d_j.astype(int),
                "p_j":                p_j.astype(int),
                "w_j":                stock_m3 / inflow_m3,
                "stock_acumulado_m3": stock_m3,
                "volume_m3":          volume_m3,
                "daily_inflow_m3":    inflow_m3,
            })
            if diff < tol:
                break

        if density > collision_target:
            window = int(window * 1.2) + 2
        else:
            window = max(0, int(window * 0.85))

    return best_noms


print("Funciones de generación definidas.")


# ================================================================
# CELDA 8: GENERAR EJE 1 (Size)
# ================================================================
SIZE_AXIS = {
    "Size_1": {"N":  8, "mix_vlcc_pct": 0.25,  "rho_target": 0.750},
    "Size_2": {"N": 12, "mix_vlcc_pct": 0.25,  "rho_target": 0.800},
    "Size_3": {"N": 16, "mix_vlcc_pct": 0.25,  "rho_target": 0.800},
    "Size_4": {"N": 20, "mix_vlcc_pct": 0.25,  "rho_target": 0.800},
    "Size_5": {"N": 30, "mix_vlcc_pct": 0.267, "rho_target": 0.800},
    "Size_6": {"N": 40, "mix_vlcc_pct": 0.25,  "rho_target": 0.800},
    "Size_7": {"N": 60, "mix_vlcc_pct": 0.25,  "rho_target": 0.800},
    "Size_8": {"N": 80, "mix_vlcc_pct": 0.25,  "rho_target": 0.800},
}

instances_size: dict = {}
print("Generando Eje 1 (Size) ...")
for label, spec in SIZE_AXIS.items():
    T_calc = _t_from_rho(spec["N"], spec["mix_vlcc_pct"], spec["rho_target"])
    noms = _make_nominations(
        n=spec["N"], T=T_calc,
        vlcc_pct=spec["mix_vlcc_pct"],
        r_j_distribution="operational",
        seed=DEFAULT_SEED,
    )
    rho_eff = _rho_effective(noms, T_calc)
    instances_size[label] = {
        "instance_label":    label,
        "N":                 spec["N"],
        "T":                 T_calc,
        "rho_target":        spec["rho_target"],
        "rho_effective":     rho_eff,
        "mix_vlcc_pct":      spec["mix_vlcc_pct"] * 100,
        "r_j_distribution":  "operational",
        "n_vars_qubo_approx": _n_vars_approx(spec["N"], T_calc, spec["mix_vlcc_pct"]),
        "collision_density": _conflict_density(noms),
        "nominations":       noms,
    }
    print(f"  {label}: N={spec['N']}  T={T_calc}  "
          f"ρ_eff={rho_eff:.3f}  N_vars≈{instances_size[label]['n_vars_qubo_approx']}")


# ================================================================
# CELDA 9: GENERAR EJE 2 (Congestion)
# ================================================================
CONG_AXIS = {
    "Cong_1": {"N":  6, "T_fixed": 62, "mix_vlcc_pct": 0.25},
    "Cong_2": {"N":  8, "T_fixed": 62, "mix_vlcc_pct": 0.25},
    "Cong_3": {"N": 10, "T_fixed": 62, "mix_vlcc_pct": 0.25},
    "Cong_4": {"N": 12, "T_fixed": 62, "mix_vlcc_pct": 0.25},
}

instances_cong: dict = {}
print("Generando Eje 2 (Congestion) ...")
for label, spec in CONG_AXIS.items():
    T_calc = spec["T_fixed"]
    N      = spec["N"]
    noms   = _make_nominations(
        n=N, T=T_calc,
        vlcc_pct=spec["mix_vlcc_pct"],
        r_j_distribution="operational",
        seed=DEFAULT_SEED,
    )
    rho_eff           = _rho_effective(noms, T_calc)
    collision_density = _conflict_density(noms)
    n_vars_approx     = _n_vars_approx(N, T_calc, spec["mix_vlcc_pct"])
    instances_cong[label] = {
        "instance_label":    label,
        "N":                 N,
        "T":                 T_calc,
        "mix_vlcc_pct":      spec["mix_vlcc_pct"] * 100,
        "r_j_distribution":  "operational",
        "collision_density": collision_density,
        "rho_effective":     rho_eff,
        "n_vars_qubo_approx": n_vars_approx,
        "nominations":       noms,
    }
    print(f"  {label}: N={N}  T={T_calc}  "
          f"collision_density={collision_density:.4f}  "
          f"ρ_eff={rho_eff:.3f}  "
          f"N_vars≈{n_vars_approx}")


# ================================================================
# CELDA 10: GENERAR EJE 3 (Structure)
# ================================================================
STRUCT_AXIS = {
    "Struct_1": {"N": 10, "r_j_distribution": "uniform"},
    "Struct_2": {"N": 10, "r_j_distribution": "random"},
    "Struct_3": {"N": 10, "r_j_distribution": "bimodal"},
    "Struct_4": {"N": 10, "r_j_distribution": "hub_cluster"},
    "Struct_5": {"N": 10, "r_j_distribution": "symmetric_batches"},
}

_struct_vlcc = 0.25
_struct_T    = 62

instances_struct: dict = {}
print("Generando Eje 3 (Structure) ...")
for label, spec in STRUCT_AXIS.items():
    N_s  = spec["N"]
    noms = _make_nominations(
        n=N_s, T=_struct_T,
        vlcc_pct=_struct_vlcc,
        r_j_distribution=spec["r_j_distribution"],
        seed=DEFAULT_SEED,
    )
    rho_eff = _rho_effective(noms, _struct_T)
    instances_struct[label] = {
        "instance_label":    label,
        "N":                 N_s,
        "T":                 _struct_T,
        "mix_vlcc_pct":      _struct_vlcc * 100,
        "r_j_distribution":  spec["r_j_distribution"],
        "rho_effective":     rho_eff,
        "n_vars_qubo_approx": _n_vars_approx(N_s, _struct_T, _struct_vlcc),
        "collision_density": _conflict_density(noms),
        "nominations":       noms,
    }
    print(f"  {label}: N={N_s}  T={_struct_T}  dist={spec['r_j_distribution']}  "
          f"ρ_eff={rho_eff:.3f}  N_vars≈{instances_struct[label]['n_vars_qubo_approx']}")


# ================================================================
# CELDA 11: GUARDAR INSTANCIAS
# ================================================================
import datetime

if INSTANCES_EXCEL.exists():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = INSTANCES_EXCEL.with_name(f"instances_backup_{timestamp}.xlsx")
    INSTANCES_EXCEL.rename(backup_path)
    print(f"Backup creado: {backup_path.name}")

save_instances_to_excel({
    "size":       instances_size,
    "congestion": instances_cong,
    "structure":  instances_struct,
})
print(f"Guardado: {INSTANCES_EXCEL}")


# ================================================================
# CELDA 12: VERIFICAR FACTIBILIDAD
# ================================================================
def verify_feasibility(label: str, inst: dict) -> str:
    noms = inst["nominations"]
    T    = inst["T"]

    if int(noms["p_j"].sum()) > T:
        return "INFACTIBLE_CAPACIDAD"

    if ((noms["d_j"] - noms["r_j"]) < noms["p_j"]).any():
        return "INFACTIBLE_VENTANAS"

    jobs   = noms.sort_values("d_j").reset_index(drop=True)
    t_free = 0
    for _, job in jobs.iterrows():
        start  = max(t_free, int(job["r_j"]))
        finish = start + int(job["p_j"])
        if finish > int(job["d_j"]):
            return "TARDANZA_INEVITABLE"
        t_free = finish

    return "FACTIBLE"


print("\nVerificación de factibilidad:")
all_instances = {
    **{f"size/{k}":        v for k, v in instances_size.items()},
    **{f"congestion/{k}":  v for k, v in instances_cong.items()},
    **{f"structure/{k}":   v for k, v in instances_struct.items()},
}
feasibility_map_local: dict[str, str] = {}
for path, inst in all_instances.items():
    status = verify_feasibility(path, inst)
    feasibility_map_local[path] = status
    icon   = "✓" if status in ("FACTIBLE", "TARDANZA_INEVITABLE") else "✗"
    print(f"  {icon}  {path:<25} {status}")

truly_infeasible = [p for p, s in feasibility_map_local.items()
                    if s not in ("FACTIBLE", "TARDANZA_INEVITABLE")]

n_tardanza = sum(1 for v in feasibility_map_local.values() if v == "TARDANZA_INEVITABLE")
n_factible = sum(1 for v in feasibility_map_local.values() if v == "FACTIBLE")

print(f"\nResumen:")
print(f"  FACTIBLE:            {n_factible}")
print(f"  TARDANZA_INEVITABLE: {n_tardanza}  ← instancias con gradiente de comparación")
print(f"  INFACTIBLE:          {len(truly_infeasible)}")

if truly_infeasible:
    print(f"\n⚠  Instancias realmente infactibles: {truly_infeasible}")
    print("   Ajustar parámetros de generación antes de continuar.")
else:
    print("\nSetup completado. Continuar con exp01_gurobi_baseline.ipynb")
