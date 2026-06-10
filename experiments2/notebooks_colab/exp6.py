"""
exp06_topology_variation.py — Experimento 6: Variación de Topología de Pipeline

Pregunta: ¿Cómo afecta la topología del pipeline (shared vs independent) a la
          factibilidad, calidad de solución y estructura BQM en Size_1 y Dens_3?

Topologías:
  "shared"      → CONFLICT_SET_R      = {(1,1),(1,2),(2,1),(2,2)}  (oleoducto compartido)
  "independent" → CONFLICT_SET_INDEPENDENT = {(1,1),(2,2)}          (tuberías independientes)

Solvers por (instancia, topología):
  1. Gurobi-MILP  — shared: pipeline único;  independent: máquinas paralelas
  2. Gurobi-QUBO  — BQP binario con Gurobi sobre el QUBO de cada topología
  3. SA           — 25 runs
  4. LeapHybrid   — EXP6_LH_RUNS=5 runs

Prerequisito: Exp 2 completado (α*, β* en metadata).
  Si no está disponible, usa PENALTY_ALPHA y PENALTY_BETA como fallback.

Outputs:
  results/exp06_topology_variation.xlsx
    hoja: solver_runs      (una fila por solver × instancia × topología × run_id)
    hoja: topology_stats   (métricas de estructura BQM por instancia × topología)
    hoja: metadata

Ejecución:
  Celda 1: SETUP
  Celda 2: LOAD instancias + Exp 2 metadata
  Celda 3: BQM TOPOLOGY STATS — n_vars, n_interactions, q_density, max_degree
  Celda 4: RUN Gurobi-MILP — ambas topologías × 2 instancias
  Celda 5: RUN Gurobi-QUBO — ambas topologías × 2 instancias
  Celda 6: RUN SA           — ambas topologías × 2 instancias (25 runs c/u)
  Celda 7: RUN LeapHybrid   — ambas topologías × 2 instancias (EXP6_LH_RUNS runs c/u)
  Celda 8: ANALYSIS — RPD vs Gurobi-MILP, tabla comparación topologías
  Celda 9: PLOT comparación topologías
"""

# CELDA 0: INSTALL
# %pip install -q dimod dwave-samplers dwave-system openpyxl seaborn gurobipy

# CELDA 1: SETUP (Colab)
# ---- EDITAR SI TU CARPETA TIENE OTRO NOMBRE ---
DRIVE_TESIS_PATH = "MyDrive/TESIS"
# -----------------------------------------------

import os, sys, time, logging, datetime, subprocess
import importlib.util as _ilu
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_TESIS      = f"/content/drive/{DRIVE_TESIS_PATH}"
REPO_ROOT        = Path(DRIVE_TESIS)
EXPERIMENTS2_DIR = REPO_ROOT / "experiments2"

for p in [str(REPO_ROOT / "src"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Instalar dependencias si es necesario
_PKGS = [
    ("dimod",          "dimod"),
    ("dwave-samplers", "dwave.samplers"),
    ("dwave-system",   "dwave.system"),
    ("openpyxl",       "openpyxl"),
    ("seaborn",        "seaborn"),
]
for _pip, _mod in _PKGS:
    _needs_install = _ilu.find_spec(_mod.split(".")[0]) is None
    if not _needs_install and _pip == "dwave-system":
        try:
            from dwave.system import LeapHybridSampler as _LHS  # noqa: F401
        except Exception:
            _needs_install = True
    if _needs_install:
        print(f"  instalando {_pip}...", end=" ", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("listo")
    else:
        print(f"  ok  {_pip}")

# Instalar gurobipy si no esta disponible
if _ilu.find_spec("gurobipy") is None:
    print("  instalando gurobipy...", end=" ", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gurobipy"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("listo")
else:
    print("  ok  gurobipy")

# Forzar import de LeapHybridSampler despues de instalacion
try:
    import importlib, dwave.system as _dws
    importlib.reload(_dws)
    from dwave.system import LeapHybridSampler as _LHS_CHECK  # noqa: F401
    print("  ok  LeapHybridSampler importable")
except Exception as _e:
    print(f"  AVISO: LeapHybridSampler no importable tras instalacion: {_e}")

# Cargar DWAVE_API_TOKEN: primero Colab Secrets, luego TESIS/.env
if "DWAVE_API_TOKEN" not in os.environ:
    try:
        from google.colab import userdata
        os.environ["DWAVE_API_TOKEN"] = userdata.get("DWAVE_API_TOKEN")
        print("  ok  DWAVE_API_TOKEN (Colab Secrets)")
    except Exception:
        _dotenv_path = REPO_ROOT / ".env"
        if _dotenv_path.exists():
            for _ln in _dotenv_path.read_text().splitlines():
                _ln = _ln.strip()
                if _ln and not _ln.startswith("#") and "=" in _ln:
                    _k, _, _v = _ln.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
            print("  ok  credenciales cargadas desde TESIS/.env")
        else:
            print("  AVISO: DWAVE_API_TOKEN no configurado. SA funciona igual; LH necesita el token.")
else:
    print("  ok  DWAVE_API_TOKEN ya en el entorno")

# Cargar credenciales Gurobi WLS: primero Colab Secrets, luego TESIS/.env
for _gk in ("GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID"):
    if _gk not in os.environ:
        try:
            from google.colab import userdata
            _val = userdata.get(_gk)
            if _val:
                os.environ[_gk] = _val
        except Exception:
            pass  # fallback ya aplicado al cargar .env arriba

import numpy as np
import pandas as pd

from config import DEFAULT_SEED, MACHINES, PENALTY_ALPHA, PENALTY_BETA

from experiments2.shared.run_id import new_run_uuid
from experiments2.shared.experiment_config import (
    SEEDS,
    GUROBI_THREADS,
    GUROBI_TIMELIMIT_S,
    EXP6_LH_RUNS,
    EXP6_INSTANCES,
)
from experiments2.shared.io_utils import (
    ensure_directories,
    load_instances_from_excel,
    load_existing_runs,
    append_rows,
    save_metadata,
    load_metadata,
    extract_solver_timing,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("exp06")

ensure_directories()

RUN_UUID  = new_run_uuid()
FILEPATH  = RESULTS_DIR / "exp06_topology_variation.xlsx"
EXP2_PATH = RESULTS_DIR / "exp02_lagrange_calibration.xlsx"

SHEET_RUNS  = "solver_runs"
SHEET_STATS = "topology_stats"

N_RUNS_SA = 25

logger.info("Exp 6 SETUP  run_uuid=%s", RUN_UUID)
logger.info("REPO_ROOT: %s", REPO_ROOT)
logger.info("Output: %s", FILEPATH)


# CELDA 2: LOAD instancias + Exp 2 metadata (α*, β*)

from preprocessing import compute_feasible_slots
from qubo_builder import build_qubo, CONFLICT_SET_INDEPENDENT
from config import CONFLICT_SET_R

# α*, β* desde Exp 2 con fallback a config defaults
_meta2 = load_metadata(EXP2_PATH)
if _meta2 and "alpha_star" in _meta2 and "beta_star" in _meta2:
    alpha_star = float(_meta2["alpha_star"])
    beta_star  = float(_meta2["beta_star"])
    logger.info("α*=%.1f  β*=%.1f  (cargados de Exp 2)", alpha_star, beta_star)
else:
    alpha_star = float(PENALTY_ALPHA)
    beta_star  = float(PENALTY_BETA)
    logger.warning(
        "Exp 2 metadata no disponible — usando fallback: α=%.1f β=%.1f",
        alpha_star, beta_star,
    )

# Cargar instancias EXP6_INSTANCES = ["Size_1", "Cong_3"]
# Size_1 está en eje "size", Cong_3 en eje "congestion"
_size_dict = load_instances_from_excel("size")
_cong_dict = load_instances_from_excel("congestion")

INSTANCE_MAP: dict[str, dict] = {}
for _lbl in EXP6_INSTANCES:
    if _lbl in _size_dict:
        INSTANCE_MAP[_lbl] = _size_dict[_lbl]
    elif _lbl in _cong_dict:
        INSTANCE_MAP[_lbl] = _cong_dict[_lbl]
    else:
        logger.warning("Instancia %s no encontrada — se omitirá", _lbl)

TOPOLOGIES: dict[str, object] = {
    "shared":      CONFLICT_SET_R,
    "independent": CONFLICT_SET_INDEPENDENT,
}

logger.info("Instancias cargadas: %s", list(INSTANCE_MAP.keys()))
logger.info("Topologías: %s", list(TOPOLOGIES.keys()))
logger.info("α*=%.1f  β*=%.1f  N_RUNS_SA=%d  EXP6_LH_RUNS=%d",
            alpha_star, beta_star, N_RUNS_SA, EXP6_LH_RUNS)


# CELDA 3: BQM TOPOLOGY STATS
# Para cada (instancia, topología): n_vars, n_interactions, q_density, max_degree, conflict_set_size
# Guarda en hoja topology_stats (append-safe por (instance_label, topology)).

_existing_stats = load_existing_runs(FILEPATH, SHEET_STATS)
_done_stats: set[tuple[str, str]] = set()
if not _existing_stats.empty:
    _done_stats = set(zip(
        _existing_stats["instance_label"].astype(str),
        _existing_stats["topology"].astype(str),
    ))

_stats_rows: list[dict] = []

for _inst_label, _inst in INSTANCE_MAP.items():
    _noms_s = _inst["nominations"].copy()
    _T_s    = int(_inst["T"])

    _vdf_s = compute_feasible_slots(_noms_s, horizon_slots=_T_s)

    for _topo_name, _conflict_set in TOPOLOGIES.items():
        if (_inst_label, _topo_name) in _done_stats:
            logger.info("  stats skip %s/%s (ya completado)", _inst_label, _topo_name)
            continue

        try:
            _bqm_s, _, _, _, _ = build_qubo(
                _vdf_s, alpha=alpha_star, beta=beta_star,
                conflict_set=_conflict_set,
            )
            _n_vars_s    = len(_bqm_s.variables)
            _n_inter_s   = len(_bqm_s.quadratic)
            _max_e_s     = _n_vars_s * (_n_vars_s - 1) / 2 if _n_vars_s > 1 else 1
            _q_dens_s    = round(_n_inter_s / _max_e_s, 6)
            _max_deg_s   = (
                max(len(_bqm_s.adj[_v]) for _v in _bqm_s.variables)
                if _bqm_s.variables else 0
            )
            _cset_size_s = len(_conflict_set)

            _stats_rows.append({
                "instance_label":    _inst_label,
                "topology":          _topo_name,
                "n_vars":            _n_vars_s,
                "n_interactions":    _n_inter_s,
                "q_density":         _q_dens_s,
                "max_degree":        _max_deg_s,
                "conflict_set_size": _cset_size_s,
            })
            logger.info(
                "  stats %s/%s: n_vars=%d  n_inter=%d  q_dens=%.4f  max_deg=%d",
                _inst_label, _topo_name, _n_vars_s, _n_inter_s, _q_dens_s, _max_deg_s,
            )
        except Exception as _exc:
            logger.error("  stats %s/%s falló: %s", _inst_label, _topo_name, _exc)

if _stats_rows:
    append_rows(FILEPATH, SHEET_STATS, _stats_rows)
    logger.info("topology_stats: %d filas guardadas.", len(_stats_rows))
else:
    logger.info("topology_stats: nada nuevo para guardar.")


# CELDA 4: RUN Gurobi-MILP — ambas topologías × 2 instancias
#
# shared topology     → formulación pipeline único (1 máquina efectiva, idéntica a Exp 1)
#   x[j,t] ∈ {0,1}: buque j empieza en slot t
#   (1) Σ_t x[j,t] = 1                           — asignación
#   (2) Σ_j Σ_{t'≤t<t'+p_j} x[j,t'] ≤ 1         — no solapamiento en pipeline
#   (3) tard[j] ≥ Σ_t (t+p_j-d_j)*x[j,t]        — tardanza linealizada
#
# independent topology → formulación parallel-machine
#   x[j,m,t] ∈ {0,1}: buque j empieza en máquina m en slot t
#   (1) Σ_{m,t} x[j,m,t] = 1                              — asignación única
#   (2) Para cada m, t: Σ_j Σ_{t'≤t<t'+p_j} x[j,m,t'] ≤ 1 — no solapamiento por máquina
#   (3) tard[j] ≥ Σ_{m,t} (t+p_j-d_j)*x[j,m,t]           — tardanza linealizada

import gurobipy as gp
from gurobipy import GRB


def _solve_milp_shared(inst: dict, seed: int) -> dict:
    """Gurobi-MILP topología shared (pipeline único): formulación assignment 1-máquina."""
    _noms    = inst["nominations"].copy()
    _T_inst  = int(inst["T"])
    _vessels = list(_noms["vessel_id"])
    _r = {row["vessel_id"]: int(row["r_j"]) for _, row in _noms.iterrows()}
    _d = {row["vessel_id"]: int(row["d_j"]) for _, row in _noms.iterrows()}
    _p = {row["vessel_id"]: int(row["p_j"]) for _, row in _noms.iterrows()}
    _w = {row["vessel_id"]: float(row["w_j"]) for _, row in _noms.iterrows()}

    _T_viable: dict[str, list[int]] = {
        j: [t for t in range(_r[j], _T_inst - _p[j] + 1)]
        for j in _vessels
    }

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    _wls_id  = os.environ.get("GRB_WLSACCESSID")
    _wls_sec = os.environ.get("GRB_WLSSECRET")
    _wls_lic = os.environ.get("GRB_LICENSEID")
    if _wls_id:  env.setParam("WLSAccessID", _wls_id)
    if _wls_sec: env.setParam("WLSSecret", _wls_sec)
    if _wls_lic: env.setParam("LicenseID", int(_wls_lic))
    env.start()

    model = gp.Model(env=env)
    model.Params.Threads   = GUROBI_THREADS
    model.Params.TimeLimit = GUROBI_TIMELIMIT_S
    model.Params.MIPGap    = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.Seed      = seed

    _x_sh = {
        (j, t): model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}")
        for j in _vessels
        for t in _T_viable[j]
    }
    _tard_sh = {j: model.addVar(lb=0.0, name=f"tard_{j}") for j in _vessels}
    model.update()

    for j in _vessels:
        if _T_viable[j]:
            model.addConstr(
                gp.quicksum(_x_sh[j, t] for t in _T_viable[j]) == 1,
                name=f"assign_{j}",
            )

    for t in range(_T_inst):
        _occupants = [
            _x_sh[j, t2]
            for j in _vessels
            for t2 in _T_viable[j]
            if t2 <= t < t2 + _p[j]
        ]
        if len(_occupants) > 1:
            model.addConstr(gp.quicksum(_occupants) <= 1, name=f"pipe_{t}")

    for j in _vessels:
        if _T_viable[j]:
            model.addConstr(
                _tard_sh[j] >= gp.quicksum(
                    (t2 + _p[j] - _d[j]) * _x_sh[j, t2]
                    for t2 in _T_viable[j]
                ),
                name=f"tard_{j}",
            )

    model.setObjective(
        gp.quicksum(_w[j] * _tard_sh[j] for j in _vessels),
        GRB.MINIMIZE,
    )

    _t0 = time.perf_counter()
    model.optimize()
    _wall = time.perf_counter() - _t0

    _status_map = {
        GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "TimeLimit",
        GRB.INFEASIBLE: "Infeasible", GRB.MEM_LIMIT: "OOM",
    }
    _status  = _status_map.get(model.Status, f"Unknown_{model.Status}")
    _obj_val = model.ObjVal if model.SolCount > 0 else float("inf")
    _mip_gap = model.MIPGap * 100.0 if model.SolCount > 0 else float("nan")
    model.dispose()
    env.dispose()

    return {
        "gurobi_status": _status,
        "obj_value":     _obj_val,
        "mip_gap_pct":   _mip_gap,
        "wall_time_s":   round(_wall, 3),
    }


def _solve_milp_independent(inst: dict, seed: int) -> dict:
    """Gurobi-MILP topología independent: formulación parallel-machine."""
    _noms    = inst["nominations"].copy()
    _T_inst  = int(inst["T"])
    _vessels = list(_noms["vessel_id"])
    _r = {row["vessel_id"]: int(row["r_j"]) for _, row in _noms.iterrows()}
    _d = {row["vessel_id"]: int(row["d_j"]) for _, row in _noms.iterrows()}
    _p = {row["vessel_id"]: int(row["p_j"]) for _, row in _noms.iterrows()}
    _w = {row["vessel_id"]: float(row["w_j"]) for _, row in _noms.iterrows()}
    _mach = MACHINES  # [1, 2]

    # T_viable_pm[j][m]: slots de inicio viables para buque j en máquina m
    _T_viable_pm: dict[str, dict[int, list[int]]] = {
        j: {
            m: [t for t in range(_r[j], _T_inst - _p[j] + 1)]
            for m in _mach
        }
        for j in _vessels
    }

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    _wls_id  = os.environ.get("GRB_WLSACCESSID")
    _wls_sec = os.environ.get("GRB_WLSSECRET")
    _wls_lic = os.environ.get("GRB_LICENSEID")
    if _wls_id:  env.setParam("WLSAccessID", _wls_id)
    if _wls_sec: env.setParam("WLSSecret", _wls_sec)
    if _wls_lic: env.setParam("LicenseID", int(_wls_lic))
    env.start()

    model = gp.Model(env=env)
    model.Params.Threads   = GUROBI_THREADS
    model.Params.TimeLimit = GUROBI_TIMELIMIT_S
    model.Params.MIPGap    = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.Seed      = seed

    _x_pm = {
        (j, m, t): model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}_{t}")
        for j in _vessels
        for m in _mach
        for t in _T_viable_pm[j][m]
    }
    _tard_pm = {j: model.addVar(lb=0.0, name=f"tard_{j}") for j in _vessels}
    model.update()

    # (1) Asignación única: cada buque va a exactamente una (máquina, slot)
    for j in _vessels:
        _all_jmt = [
            _x_pm[j, m, t]
            for m in _mach
            for t in _T_viable_pm[j][m]
        ]
        if _all_jmt:
            model.addConstr(gp.quicksum(_all_jmt) == 1, name=f"assign_{j}")

    # (2) No solapamiento por máquina
    for m in _mach:
        for t in range(_T_inst):
            _occ_m = [
                _x_pm[j, m, t2]
                for j in _vessels
                for t2 in _T_viable_pm[j][m]
                if t2 <= t < t2 + _p[j]
            ]
            if len(_occ_m) > 1:
                model.addConstr(gp.quicksum(_occ_m) <= 1, name=f"pipe_m{m}_t{t}")

    # (3) Tardanza linealizada
    for j in _vessels:
        _tard_expr = gp.quicksum(
            (t + _p[j] - _d[j]) * _x_pm[j, m, t]
            for m in _mach
            for t in _T_viable_pm[j][m]
        )
        if _tard_expr.size() > 0:
            model.addConstr(_tard_pm[j] >= _tard_expr, name=f"tard_{j}")

    model.setObjective(
        gp.quicksum(_w[j] * _tard_pm[j] for j in _vessels),
        GRB.MINIMIZE,
    )

    _t0_pm = time.perf_counter()
    model.optimize()
    _wall_pm = time.perf_counter() - _t0_pm

    _status_map_pm = {
        GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "TimeLimit",
        GRB.INFEASIBLE: "Infeasible", GRB.MEM_LIMIT: "OOM",
    }
    _status_pm  = _status_map_pm.get(model.Status, f"Unknown_{model.Status}")
    _obj_val_pm = model.ObjVal if model.SolCount > 0 else float("inf")
    _mip_gap_pm = model.MIPGap * 100.0 if model.SolCount > 0 else float("nan")
    model.dispose()
    env.dispose()

    return {
        "gurobi_status": _status_pm,
        "obj_value":     _obj_val_pm,
        "mip_gap_pct":   _mip_gap_pm,
        "wall_time_s":   round(_wall_pm, 3),
    }


def _build_run_row(
    solver: str,
    inst_label: str,
    topology: str,
    seed: int,
    run_id: int,
    inst: dict,
    n_vars: int,
    n_inter: int,
    q_dens: float,
    feasible: bool,
    obj_value: float,
    wall_time_s: float,
    best_energy: float = float("nan"),
    energy_gap: float = float("nan"),
    lh_run_time_s: float = float("nan"),
    lh_run_time_us: float = float("nan"),
    gurobi_bqp_status: str = "",
    is_feasible_bqp: bool = False,
) -> dict:
    """Construye la fila estándar para la hoja solver_runs."""
    return {
        "exp_id":            "exp06",
        "run_uuid":          RUN_UUID,
        "instance_label":    inst_label,
        "topology":          topology,
        "solver":            solver,
        "N":                 int(inst["N"]),
        "T":                 int(inst["T"]),
        "rho_effective":     float(inst["rho_effective"]),
        "seed":              seed,
        "run_id":            run_id,
        "feasible":          bool(feasible),
        "obj_value":         float(obj_value),
        "wall_time_s":       round(float(wall_time_s), 3),
        "lh_run_time_s":     float(lh_run_time_s),
        "lh_run_time_us":    float(lh_run_time_us),
        "n_vars":            int(n_vars),
        "q_density":         float(q_dens),
        "n_interactions":    int(n_inter),
        "best_energy":       float(best_energy),
        "energy_gap":        float(energy_gap),
        "alpha":             float(alpha_star),
        "beta":              float(beta_star),
        "gurobi_bqp_status": str(gurobi_bqp_status),
        "is_feasible_bqp":   bool(is_feasible_bqp),
        "run_timestamp":     datetime.datetime.now().isoformat(),
    }


# Cargar done_runs una vez para MILP
_existing_runs = load_existing_runs(FILEPATH, SHEET_RUNS)
if not _existing_runs.empty:
    _milp_rows  = _existing_runs[_existing_runs["solver"] == "Gurobi-MILP"]
    _done_milp: set[tuple[str, str, int]] = set(zip(
        _milp_rows["instance_label"].astype(str),
        _milp_rows["topology"].astype(str),
        _milp_rows["run_id"].astype(int),
    ))
else:
    _done_milp = set()

# Estructura BQM precalculada para reutilizar en Celdas 4-7
_bqm_cache: dict[tuple[str, str], dict] = {}

for _inst_label, _inst in INSTANCE_MAP.items():
    _noms_c = _inst["nominations"].copy()
    _T_c    = int(_inst["T"])
    _vdf_c  = compute_feasible_slots(_noms_c, horizon_slots=_T_c)

    for _topo_name, _conflict_set in TOPOLOGIES.items():
        _bqm_c, _, _, _, _ = build_qubo(
            _vdf_c, alpha=alpha_star, beta=beta_star,
            conflict_set=_conflict_set,
        )
        _nv_c  = len(_bqm_c.variables)
        _ni_c  = len(_bqm_c.quadratic)
        _me_c  = _nv_c * (_nv_c - 1) / 2 if _nv_c > 1 else 1
        _qd_c  = round(_ni_c / _me_c, 6)
        _bqm_cache[(_inst_label, _topo_name)] = {
            "bqm":        _bqm_c,
            "vdf":        _vdf_c,
            "n_vars":     _nv_c,
            "n_inter":    _ni_c,
            "q_density":  _qd_c,
        }

logger.info("BQM cache precalculado: %d entradas.", len(_bqm_cache))

# RUN Gurobi-MILP
for _inst_label, _inst in INSTANCE_MAP.items():
    for _topo_name, _conflict_set in TOPOLOGIES.items():
        # MILP run_id=0 (seed=0 por defecto, sin variación de seed para MILP)
        if (_inst_label, _topo_name, 0) in _done_milp:
            logger.info("  MILP skip %s/%s run_id=0 (ya completado)", _inst_label, _topo_name)
            continue

        logger.info("  MILP %s/%s ...", _inst_label, _topo_name)
        try:
            if _topo_name == "shared":
                _milp_res = _solve_milp_shared(_inst, seed=0)
            else:
                _milp_res = _solve_milp_independent(_inst, seed=0)

            _bqm_info = _bqm_cache[(_inst_label, _topo_name)]
            _is_feas_milp = (_milp_res["gurobi_status"] in ("Optimal", "TimeLimit")
                             and _milp_res["obj_value"] < float("inf"))

            _row_milp = _build_run_row(
                solver="Gurobi-MILP",
                inst_label=_inst_label,
                topology=_topo_name,
                seed=0,
                run_id=0,
                inst=_inst,
                n_vars=_bqm_info["n_vars"],
                n_inter=_bqm_info["n_inter"],
                q_dens=_bqm_info["q_density"],
                feasible=_is_feas_milp,
                obj_value=_milp_res["obj_value"] if _is_feas_milp else float("nan"),
                wall_time_s=_milp_res["wall_time_s"],
                gurobi_bqp_status=_milp_res["gurobi_status"],
                is_feasible_bqp=False,
            )
            append_rows(FILEPATH, SHEET_RUNS, [_row_milp])
            logger.info(
                "    MILP %s/%s  status=%s  obj=%.2f  t=%.2fs",
                _inst_label, _topo_name,
                _milp_res["gurobi_status"],
                _milp_res["obj_value"] if _is_feas_milp else float("nan"),
                _milp_res["wall_time_s"],
            )
        except Exception as _exc_milp:
            logger.error("  MILP %s/%s falló: %s", _inst_label, _topo_name, _exc_milp)

logger.info("Gurobi-MILP completo.")


# CELDA 5: RUN Gurobi-QUBO (BQP) — ambas topologías × 2 instancias
# Usa el BQM de build_qubo para la topología correspondiente.
# Variables binarias Gurobi, objetivo QuadExpr (linear + quadratic terms del BQM).
# Decode con decode_schedule + check_feasibility.

from solver import decode_schedule, check_feasibility

_existing_runs_5 = load_existing_runs(FILEPATH, SHEET_RUNS)
if not _existing_runs_5.empty:
    _qubo_rows  = _existing_runs_5[_existing_runs_5["solver"] == "Gurobi-QUBO"]
    _done_qubo: set[tuple[str, str, int]] = set(zip(
        _qubo_rows["instance_label"].astype(str),
        _qubo_rows["topology"].astype(str),
        _qubo_rows["run_id"].astype(int),
    ))
else:
    _done_qubo = set()


def _solve_gurobi_bqp(inst: dict, inst_label: str, topology: str) -> dict:
    """Resolve el QUBO como BQP binario con Gurobi para la topología indicada."""
    _bqm_info = _bqm_cache[(inst_label, topology)]
    _bqm_bqp  = _bqm_info["bqm"]
    _vdf_bqp  = _bqm_info["vdf"]
    _var_list  = list(_bqm_bqp.variables)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    _wls_id  = os.environ.get("GRB_WLSACCESSID")
    _wls_sec = os.environ.get("GRB_WLSSECRET")
    _wls_lic = os.environ.get("GRB_LICENSEID")
    if _wls_id:  env.setParam("WLSAccessID", _wls_id)
    if _wls_sec: env.setParam("WLSSecret", _wls_sec)
    if _wls_lic: env.setParam("LicenseID", int(_wls_lic))
    env.start()

    model = gp.Model(env=env)
    model.Params.Threads   = GUROBI_THREADS
    model.Params.TimeLimit = GUROBI_TIMELIMIT_S
    model.Params.MIPGap    = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.Seed      = 0

    _x_bqp = {v: model.addVar(vtype=GRB.BINARY, name=f"v{i}") for i, v in enumerate(_var_list)}
    model.update()

    _lin_expr = gp.LinExpr()
    for _v_lin, _bias_lin in _bqm_bqp.linear.items():
        _lin_expr.add(_x_bqp[_v_lin], float(_bias_lin))

    _quad_expr = gp.QuadExpr()
    for (_v1, _v2), _bias_q in _bqm_bqp.quadratic.items():
        _quad_expr.add(_x_bqp[_v1] * _x_bqp[_v2], float(_bias_q))

    model.setObjective(_lin_expr + _quad_expr, GRB.MINIMIZE)

    _t0_bqp = time.perf_counter()
    model.optimize()
    _wall_bqp = time.perf_counter() - _t0_bqp

    _status_map_bqp = {
        GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "TimeLimit",
        GRB.INFEASIBLE: "Infeasible",
    }
    _bqp_status = _status_map_bqp.get(model.Status, f"Unknown_{model.Status}")
    _best_energy_bqp = float("nan")
    _is_feas_bqp     = False
    _obj_bqp         = float("nan")

    if model.SolCount > 0:
        _sample_bqp   = {v: round(_x_bqp[v].X) for v in _var_list}
        _best_energy_bqp = float(model.ObjVal)
        _sched_bqp    = decode_schedule(_sample_bqp, _vdf_bqp)
        _fres_bqp     = check_feasibility(_sched_bqp, _vdf_bqp)
        _is_feas_bqp  = bool(_fres_bqp["is_feasible"])
        if _is_feas_bqp:
            _obj_bqp = float(_fres_bqp["total_weighted_tardiness"])

    model.dispose()
    env.dispose()

    return {
        "gurobi_bqp_status": _bqp_status,
        "is_feasible_bqp":   _is_feas_bqp,
        "obj_value":         _obj_bqp,
        "best_energy":       _best_energy_bqp,
        "wall_time_s":       round(_wall_bqp, 3),
    }


for _inst_label, _inst in INSTANCE_MAP.items():
    for _topo_name in TOPOLOGIES:
        if (_inst_label, _topo_name, 0) in _done_qubo:
            logger.info("  QUBO skip %s/%s run_id=0 (ya completado)", _inst_label, _topo_name)
            continue

        logger.info("  QUBO %s/%s ...", _inst_label, _topo_name)
        try:
            _qubo_res    = _solve_gurobi_bqp(_inst, _inst_label, _topo_name)
            _bqm_info_q5 = _bqm_cache[(_inst_label, _topo_name)]
            _row_qubo    = _build_run_row(
                solver="Gurobi-QUBO",
                inst_label=_inst_label,
                topology=_topo_name,
                seed=0,
                run_id=0,
                inst=_inst,
                n_vars=_bqm_info_q5["n_vars"],
                n_inter=_bqm_info_q5["n_inter"],
                q_dens=_bqm_info_q5["q_density"],
                feasible=_qubo_res["is_feasible_bqp"],
                obj_value=_qubo_res["obj_value"],
                wall_time_s=_qubo_res["wall_time_s"],
                best_energy=_qubo_res["best_energy"],
                gurobi_bqp_status=_qubo_res["gurobi_bqp_status"],
                is_feasible_bqp=_qubo_res["is_feasible_bqp"],
            )
            append_rows(FILEPATH, SHEET_RUNS, [_row_qubo])
            logger.info(
                "    QUBO %s/%s  status=%s  feasible=%s  obj=%.2f  t=%.2fs",
                _inst_label, _topo_name,
                _qubo_res["gurobi_bqp_status"],
                _qubo_res["is_feasible_bqp"],
                _qubo_res["obj_value"] if not np.isnan(_qubo_res["obj_value"]) else float("nan"),
                _qubo_res["wall_time_s"],
            )
        except Exception as _exc_qubo:
            logger.error("  QUBO %s/%s falló: %s", _inst_label, _topo_name, _exc_qubo)

logger.info("Gurobi-QUBO completo.")


# CELDA 6: RUN SA — ambas topologías × 2 instancias (N_RUNS_SA=25 runs c/u)
# Append-safe por (instance_label, topology, run_id).

from dwave.samplers import SimulatedAnnealingSampler as _SASampler6

_existing_runs_6 = load_existing_runs(FILEPATH, SHEET_RUNS)
if not _existing_runs_6.empty:
    _sa_rows_6  = _existing_runs_6[_existing_runs_6["solver"] == "SA"]
    _done_sa6: set[tuple[str, str, int]] = set(zip(
        _sa_rows_6["instance_label"].astype(str),
        _sa_rows_6["topology"].astype(str),
        _sa_rows_6["run_id"].astype(int),
    ))
else:
    _done_sa6 = set()

_sa_sampler6 = _SASampler6()

for _inst_label, _inst in INSTANCE_MAP.items():
    for _topo_name in TOPOLOGIES:
        _key_sa6 = (_inst_label, _topo_name)
        _bqm_info_6 = _bqm_cache[_key_sa6]
        _bqm6       = _bqm_info_6["bqm"]
        _vdf6       = _bqm_info_6["vdf"]
        _nv6        = _bqm_info_6["n_vars"]
        _ni6        = _bqm_info_6["n_inter"]
        _qd6        = _bqm_info_6["q_density"]

        # beta_range: escalar con penalización para evitar patrón de feasibility invertido
        _bqm6_build, _P1_6, *_ = build_qubo(
            compute_feasible_slots(_inst["nominations"].copy(), horizon_slots=int(_inst["T"])),
            alpha=alpha_star, beta=beta_star,
            conflict_set=TOPOLOGIES[_topo_name],
        )
        _beta_range6 = (1.0 / (_P1_6 * 2.0), 10.0)

        n_done_sa6 = sum(1 for r in range(N_RUNS_SA) if (_inst_label, _topo_name, r) in _done_sa6)
        logger.info("  SA %s/%s: %d/%d runs ya completados", _inst_label, _topo_name, n_done_sa6, N_RUNS_SA)

        for _run_id6 in range(N_RUNS_SA):
            if (_inst_label, _topo_name, _run_id6) in _done_sa6:
                continue

            try:
                _t0_sa6 = time.perf_counter()
                _ss6 = _sa_sampler6.sample(
                    _bqm6,
                    num_reads=200,
                    num_sweeps=1000,
                    beta_range=_beta_range6,
                    seed=_run_id6 * 31 + 7,
                )
                _wall_sa6 = time.perf_counter() - _t0_sa6

                _best_sample6  = _ss6.first.sample
                _best_energy6  = float(_ss6.first.energy)
                _sched6        = decode_schedule(_best_sample6, _vdf6)
                _fres6         = check_feasibility(_sched6, _vdf6)
                _is_feas6      = bool(_fres6["is_feasible"])
                _obj6          = float(_fres6["total_weighted_tardiness"]) if _is_feas6 else float("nan")

                _row_sa6 = _build_run_row(
                    solver="SA",
                    inst_label=_inst_label,
                    topology=_topo_name,
                    seed=0,
                    run_id=_run_id6,
                    inst=_inst,
                    n_vars=_nv6,
                    n_inter=_ni6,
                    q_dens=_qd6,
                    feasible=_is_feas6,
                    obj_value=_obj6,
                    wall_time_s=_wall_sa6,
                    best_energy=_best_energy6,
                )
                append_rows(FILEPATH, SHEET_RUNS, [_row_sa6])
                _done_sa6.add((_inst_label, _topo_name, _run_id6))

            except Exception as _exc_sa6:
                logger.error("  SA %s/%s run_id=%d falló: %s",
                             _inst_label, _topo_name, _run_id6, _exc_sa6)

    logger.info("  SA %s: completo.", _inst_label)

logger.info("SA completo.")


# CELDA 7: RUN LeapHybrid — ambas topologías × 2 instancias (EXP6_LH_RUNS=5 runs c/u)
# Append-safe por (instance_label, topology, run_id).

from solver import run_solver

_existing_runs_7 = load_existing_runs(FILEPATH, SHEET_RUNS)
if not _existing_runs_7.empty:
    _lh_rows_7  = _existing_runs_7[_existing_runs_7["solver"] == "LeapHybrid"]
    _done_lh7: set[tuple[str, str, int]] = set(zip(
        _lh_rows_7["instance_label"].astype(str),
        _lh_rows_7["topology"].astype(str),
        _lh_rows_7["run_id"].astype(int),
    ))
else:
    _done_lh7 = set()

for _inst_label, _inst in INSTANCE_MAP.items():
    for _topo_name in TOPOLOGIES:
        _key_lh7    = (_inst_label, _topo_name)
        _bqm_info_7 = _bqm_cache[_key_lh7]
        _bqm7       = _bqm_info_7["bqm"]
        _vdf7       = _bqm_info_7["vdf"]
        _nv7        = _bqm_info_7["n_vars"]
        _ni7        = _bqm_info_7["n_inter"]
        _qd7        = _bqm_info_7["q_density"]

        n_done_lh7 = sum(1 for r in range(EXP6_LH_RUNS) if (_inst_label, _topo_name, r) in _done_lh7)
        logger.info("  LH %s/%s: %d/%d runs ya completados", _inst_label, _topo_name, n_done_lh7, EXP6_LH_RUNS)

        for _run_id7 in range(EXP6_LH_RUNS):
            if (_inst_label, _topo_name, _run_id7) in _done_lh7:
                logger.info("  skip %s/%s run_id=%d", _inst_label, _topo_name, _run_id7)
                continue

            logger.info("  LH %s/%s run_id=%d (n_vars=%d) ...",
                        _inst_label, _topo_name, _run_id7, _nv7)
            try:
                _t0_lh7 = time.perf_counter()
                _ss7, _solver_name7 = run_solver(_bqm7, requested_sampler="leaphybrid")
                _wall_lh7 = time.perf_counter() - _t0_lh7

                _dw7 = extract_solver_timing(_ss7)

                _best_sample7  = _ss7.first.sample
                _best_energy7  = float(_ss7.first.energy)

                # energy_gap: separación energética entre mejores factible e infactible
                _e_feas7, _e_inf7 = [], []
                for _samp7, _en7 in _ss7.data(["sample", "energy"]):
                    _sched_tmp7 = decode_schedule(_samp7, _vdf7)
                    _fres_tmp7  = check_feasibility(_sched_tmp7, _vdf7)
                    (_e_feas7 if _fres_tmp7["is_feasible"] else _e_inf7).append(float(_en7))
                _e_gap7 = (
                    (min(_e_inf7) - min(_e_feas7))
                    if _e_feas7 and _e_inf7
                    else float("nan")
                )

                _sched7   = decode_schedule(_best_sample7, _vdf7)
                _fres7    = check_feasibility(_sched7, _vdf7)
                _is_feas7 = bool(_fres7["is_feasible"])
                _obj7     = float(_fres7["total_weighted_tardiness"]) if _is_feas7 else float("nan")

                _row_lh7 = _build_run_row(
                    solver="LeapHybrid",
                    inst_label=_inst_label,
                    topology=_topo_name,
                    seed=0,
                    run_id=_run_id7,
                    inst=_inst,
                    n_vars=_nv7,
                    n_inter=_ni7,
                    q_dens=_qd7,
                    feasible=_is_feas7,
                    obj_value=_obj7,
                    wall_time_s=_wall_lh7,
                    best_energy=_best_energy7,
                    energy_gap=_e_gap7,
                    lh_run_time_s=_dw7["lh_run_time_s"],
                    lh_run_time_us=_dw7["lh_run_time_us"],
                )
                append_rows(FILEPATH, SHEET_RUNS, [_row_lh7])
                _done_lh7.add((_inst_label, _topo_name, _run_id7))
                logger.info(
                    "    %s  feasible=%s  obj=%.1f  wall=%.1fs  lh_compute=%.1fs",
                    _solver_name7, _is_feas7,
                    _obj7 if not np.isnan(_obj7) else -1,
                    _wall_lh7,
                    _dw7["lh_run_time_s"] if _dw7["lh_run_time_s"] == _dw7["lh_run_time_s"] else float("nan"),
                )

            except Exception as _exc_lh7:
                logger.error("  LH %s/%s run_id=%d falló: %s",
                             _inst_label, _topo_name, _run_id7, _exc_lh7)

logger.info("LeapHybrid completo.")


# CELDA 8: ANALYSIS — RPD vs Gurobi-MILP, tabla comparación topologías, topology_stats

_df_runs_all = load_existing_runs(FILEPATH, SHEET_RUNS)
_df_stats_all = load_existing_runs(FILEPATH, SHEET_STATS)

if not _df_runs_all.empty:
    # Referencia Gurobi-MILP por (instance_label, topology)
    _milp_ref = (
        _df_runs_all[_df_runs_all["solver"] == "Gurobi-MILP"]
        .groupby(["instance_label", "topology"])
        .agg(milp_obj=("obj_value", "first"))
        .reset_index()
    )

    # RPD vs Gurobi-MILP por (solver, instance, topology)
    _df_analysis = _df_runs_all.merge(_milp_ref, on=["instance_label", "topology"], how="left")

    def _rpd(row: pd.Series) -> float:
        if (
            row["solver"] == "Gurobi-MILP"
            or not row.get("feasible", False)
            or np.isnan(row.get("obj_value", float("nan")))
            or np.isnan(row.get("milp_obj", float("nan")))
            or row["milp_obj"] <= 0
        ):
            return float("nan")
        return 100.0 * (row["obj_value"] - row["milp_obj"]) / row["milp_obj"]

    _df_analysis["rpd_vs_milp"] = _df_analysis.apply(_rpd, axis=1)

    # Tabla resumen por (instance_label, topology, solver)
    _summary = (
        _df_analysis.groupby(["instance_label", "topology", "solver"])
        .agg(
            n_runs       = ("run_id",       "count"),
            feas_rate    = ("feasible",     lambda x: x.astype(bool).mean()),
            obj_mean     = ("obj_value",    "mean"),
            obj_std      = ("obj_value",    "std"),
            rpd_mean     = ("rpd_vs_milp",  "mean"),
            wall_mean_s  = ("wall_time_s",  "mean"),
        )
        .reset_index()
    )

    print("\n=== EXP 6 — RESUMEN POR TOPOLOGÍA ===")
    for _topo in ("shared", "independent"):
        _sub = _summary[_summary["topology"] == _topo]
        print(f"\n--- Topología: {_topo.upper()} ---")
        print(_sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if not _df_stats_all.empty:
        print("\n=== EXP 6 — TOPOLOGY STATS (BQM) ===")
        print(_df_stats_all.to_string(index=False))
else:
    logger.warning("solver_runs vacío — ejecutar Celdas 4–7 primero.")

save_metadata(FILEPATH, {
    "exp_version":   "v1.0",
    "run_uuid_last": RUN_UUID,
    "timestamp":     datetime.datetime.now().isoformat(),
    "alpha_star":    alpha_star,
    "beta_star":     beta_star,
    "n_runs_sa":     N_RUNS_SA,
    "n_runs_lh":     EXP6_LH_RUNS,
    "instances":     str(list(INSTANCE_MAP.keys())),
})
logger.info("Metadata guardada en %s", FILEPATH)


# CELDA 9: PLOT comparación topologías
# Barras agrupadas por (topología, solver) — obj_mean con barras de error (obj_std).
# Un subplot por instancia.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

_df_plot = load_existing_runs(FILEPATH, SHEET_RUNS)
if _df_plot.empty:
    logger.warning("Sin datos para plot (SHEET_RUNS vacío).")
else:
    _milp_ref_p = (
        _df_plot[_df_plot["solver"] == "Gurobi-MILP"]
        .groupby(["instance_label", "topology"])
        .agg(milp_obj=("obj_value", "first"))
        .reset_index()
    )
    _df_plot = _df_plot.merge(_milp_ref_p, on=["instance_label", "topology"], how="left")
    _df_feas_p = _df_plot[_df_plot["feasible"].astype(bool)].copy()

    _SOLVER_COLORS = {
        "SA":          "#4878D0",
        "LeapHybrid":  "#EE854A",
        "Gurobi-MILP": "#6ACC65",
        "Gurobi-QUBO": "#D65F5F",
    }
    _SOLVER_ORDER = ["Gurobi-MILP", "Gurobi-QUBO", "SA", "LeapHybrid"]
    _TOPO_ORDER   = ["shared", "independent"]

    _inst_labels = sorted(_df_plot["instance_label"].unique())
    _n_inst = len(_inst_labels)

    _fig, _axes = plt.subplots(1, _n_inst, figsize=(8 * _n_inst, 6), sharey=False)
    if _n_inst == 1:
        _axes = [_axes]

    for _ax_i, _inst_lbl in zip(_axes, _inst_labels):
        _sub_inst = _df_feas_p[_df_feas_p["instance_label"] == _inst_lbl]
        _agg_inst = (
            _sub_inst.groupby(["topology", "solver"])
            .agg(obj_mean=("obj_value", "mean"), obj_std=("obj_value", "std"))
            .reset_index()
        )

        _n_topos   = len(_TOPO_ORDER)
        _n_solvers = len(_SOLVER_ORDER)
        _bar_w     = 0.18
        _x_base    = np.arange(_n_topos)

        for _si, _solver in enumerate(_SOLVER_ORDER):
            _ys   = []
            _errs = []
            for _topo in _TOPO_ORDER:
                _r = _agg_inst[
                    (_agg_inst["topology"] == _topo) &
                    (_agg_inst["solver"]   == _solver)
                ]
                _ys.append(
                    float(_r["obj_mean"].iloc[0])
                    if not _r.empty and not np.isnan(float(_r["obj_mean"].iloc[0]))
                    else 0.0
                )
                _errs.append(
                    float(_r["obj_std"].iloc[0])
                    if not _r.empty and not np.isnan(float(_r["obj_std"].iloc[0]))
                    else 0.0
                )
            _offset = (_si - (_n_solvers - 1) / 2) * _bar_w
            _ax_i.bar(
                _x_base + _offset, _ys, width=_bar_w,
                color=_SOLVER_COLORS[_solver], label=_solver, alpha=0.85,
            )
            _ax_i.errorbar(
                _x_base + _offset, _ys, yerr=_errs,
                fmt="none", color="black", capsize=3, linewidth=0.8,
            )

        _ax_i.set_xticks(_x_base)
        _ax_i.set_xticklabels(_TOPO_ORDER, fontsize=10)
        _ax_i.set_xlabel("Topología de pipeline")
        _ax_i.set_ylabel("Tardiness ponderado (Σwⱼtⱼ)")
        _ax_i.set_title(f"Instancia: {_inst_lbl}", fontsize=11)
        _ax_i.grid(True, linestyle=":", alpha=0.4)
        sns.despine(ax=_ax_i)

    _patches = [
        mpatches.Patch(color=_SOLVER_COLORS[s], label=s)
        for s in _SOLVER_ORDER
    ]
    _fig.suptitle("Exp 6 — Comparación de topologías: shared vs independent pipeline", fontsize=13)
    _fig.legend(handles=_patches, loc="lower center", ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    plt.tight_layout()

    _plot_path = RESULTS_DIR / "exp06_topology_comparison.png"
    _fig.savefig(_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info("Guardado: %s", _plot_path)

logger.info("Exp 6 completo. Resultados en: %s", FILEPATH)
