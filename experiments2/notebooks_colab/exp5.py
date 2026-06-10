"""
exp05_hybrid_convergence.py — Experimento 5: Convergencia del Bucle Híbrido

Pregunta: ¿Cuántas iteraciones necesita el bucle híbrido para producir un
          schedule volumétricamente factible? ¿Converge monotónicamente o
          exhibe oscilaciones? ¿La densidad de Q se mantiene estable?

Prerequisito: Exp 2 completado (α*, β* en exp02_lagrange_calibration.xlsx).

Outputs:
  results/exp05_hybrid_convergence.xlsx
    hoja: per_iteration  (una fila por solver × instance × seed × run × k)
    hoja: metadata       (alpha_star, beta_star, run_uuid_last)

Ejecución:
  Celda 1: SETUP + cargar α*, β* de Exp 2
  Celda 2: RUN SA — bucle iterativo en Size_1, Size_2, Dens_1 (5 runs × 1 seed)
  Celda 3: RUN LeapHybrid — bucle iterativo en Size_1, Size_2, Dens_1 (15 runs)
  Celda 4: CALCULAR métricas derivadas (k_conv, oscillating, q_density_delta)
  Celda 5: PLOT convergencia (n_violations y tardiness por iteración)
  Celda 6: PLOT estabilidad de densidad QUBO
  Celda 7: PLOT k_conv vs instancia

Diseño (metrics.md §Exp 5):
  - Instancias: Size_1 (N=8), Size_2 (N=12), Dens_1 (= Size_1 — referencia)
  - Seed: 0 (única; capturar comportamiento del bucle, no variabilidad de instancia)
  - SA: N_RUNS_SA_EXP5=5 runs por instancia → sin costo QPU
  - LH: EXP5_LH_RUNS runs por instancia → llamadas QPU
  - K_max: EXP5_K_MAX=10 iteraciones máximas — el loop para al converger (break en k_conv)
  - Bucle propio (no usar run_iterative_loop de src/) para registrar métricas por k
"""

# CELDA 0: INSTALL — ejecutar una sola vez por sesion de Colab
# dwave-system incluye LeapHybridSampler; requiere DWAVE_API_TOKEN configurado
# %pip install -q dimod dwave-samplers dwave-system openpyxl seaborn

# CELDA SETUP (Colab)
# ---- EDITAR SI TU CARPETA TIENE OTRO NOMBRE ---
DRIVE_TESIS_PATH = "MyDrive/TESIS"
# -----------------------------------------------

import os, sys, subprocess
import importlib.util as _ilu
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_TESIS      = f"/content/drive/{DRIVE_TESIS_PATH}"
REPO_ROOT        = Path(DRIVE_TESIS)
EXPERIMENTS2_DIR = REPO_ROOT / "experiments2"

from experiments2.shared.io_utils import (
    RESULTS_DIR,
)

for p in [str(REPO_ROOT / "src"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Cargar DWAVE_API_TOKEN: primero Colab Secrets, luego .env
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

print(f"  ok  REPO_ROOT: {REPO_ROOT}")


# CELDA 1: SETUP + cargar α*, β* de Exp 2

import sys
import time
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments2.shared.run_id import new_run_uuid
from experiments2.shared.experiment_config import EXP5_K_MAX
from config import (
    SLOT_HOURS,
    MIN_ULLAGE_DAYS,
    INITIAL_TERMINAL_STOCK_M3,
    N_TANKS,
    TANK_CAPACITY_M3,
    DAILY_INFLOW_M3,
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
logger = logging.getLogger("exp05")

ensure_directories()

RUN_UUID  = new_run_uuid()
FILEPATH  = RESULTS_DIR / "exp05_hybrid_convergence.xlsx"
EXP2_PATH = RESULTS_DIR / "exp02_lagrange_calibration.xlsx"
SHEET     = "per_iteration"

logger.info("Exp 5 SETUP  run_uuid=%s", RUN_UUID)

# Cargar α*, β* de Exp 2
_meta2 = load_metadata(EXP2_PATH)
if not _meta2:
    raise FileNotFoundError(
        f"No se encontró metadata en {EXP2_PATH}. Ejecutar Exp 2 primero."
    )
alpha_star = float(_meta2["alpha_star"])
beta_star  = float(_meta2["beta_star"])
logger.info("α*=%.1f  β*=%.1f  (cargados de Exp 2)", alpha_star, beta_star)


def _ensure_volume_m3(noms: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que el DataFrame de nominaciones tenga la columna volume_m3.
    Instancias generadas con versiones antiguas del generador no la incluyen;
    en ese caso se usa stock_acumulado_m3 como proxy conservador
    (stock_acumulado_m3 >= volume_m3 por construcción).
    """
    if "volume_m3" not in noms.columns:
        logger.warning(
            "Columna 'volume_m3' ausente en nominaciones — "
            "usando stock_acumulado_m3 como proxy (instancias generadas con versión antigua)."
        )
        noms = noms.copy()
        noms["volume_m3"] = noms["stock_acumulado_m3"]
    return noms


# Cargar instancias
# Fallback: Excel viejo usa 'congestion', nuevo usa 'dens'
_size_dict = load_instances_from_excel("size")
try:
    _dens_dict = load_instances_from_excel("dens")
    _dens_key  = "Dens_1"
except (ValueError, KeyError):
    _dens_dict = load_instances_from_excel("congestion")
    _dens_key  = "Cong_1"

# Asegurar volume_m3 en todas las instancias cargadas
for _d in [_size_dict, _dens_dict]:
    for _inst in _d.values():
        _inst["nominations"] = _ensure_volume_m3(_inst["nominations"])

# Instancias del experimento (metrics.md §Exp 5)
# Dens_1/Cong_1 = Size_1 — misma instancia base referenciada por dos ejes
EXP5_INSTANCES = {
    "Size_1": {"inst": _size_dict["Size_1"], "axis": "size"},
    "Size_2": {"inst": _size_dict["Size_2"], "axis": "size"},
    "Dens_1": {"inst": _dens_dict[_dens_key], "axis": "dens"},
}

# Parámetros de inventario explícitos para reproducibilidad
EXP5_INVENTORY_PARAMS = {
    "slot_duration_hours":       SLOT_HOURS,
    "min_ullage_days":           MIN_ULLAGE_DAYS,
    "initial_terminal_stock_m3": INITIAL_TERMINAL_STOCK_M3,
    "n_tanks":                   N_TANKS,
    "tank_capacity_m3":          TANK_CAPACITY_M3,
    "daily_inflow_m3":           DAILY_INFLOW_M3,
}

_TANK_CAPACITY_TOTAL = N_TANKS * TANK_CAPACITY_M3
_SAFE_THRESHOLD      = _TANK_CAPACITY_TOTAL - MIN_ULLAGE_DAYS * DAILY_INFLOW_M3

EXP5_TANK_LEVELS = {
    "low":     round(_TANK_CAPACITY_TOTAL * 0.35),
    "nominal": INITIAL_TERMINAL_STOCK_M3,
    # Exp5b/Gurobi IIS: margen de 1 día (anterior) es estructuralmente infactible —
    # el stock cruza safe_threshold en t=2-3 antes de que cualquier buque complete
    # su carga (p_j mínimo = 4 slots). 5 días validado como factible para Size_1/Size_2.
    "high":    round(_SAFE_THRESHOLD - 5 * DAILY_INFLOW_M3),
}

# Parámetros del experimento
EXP5_SEED         = 0
N_RUNS_SA_EXP5    = 5
EXP5_LH_RUNS      = 1    # 1 run × 3 instancias × K_max iteraciones (~10 min QPU)
# EXP5_K_MAX = 5  # descomentar si SA confirma k_conv_mean <= 4

logger.info(
    "Instancias: %s  seed=%d  K_max=%d  N_SA=%d  N_LH=%d",
    list(EXP5_INSTANCES), EXP5_SEED, EXP5_K_MAX, N_RUNS_SA_EXP5, EXP5_LH_RUNS,
)


# CELDA 2: RUN SA — bucle iterativo
#
# Por cada instancia × run_id, corre EXP5_K_MAX iteraciones del bucle híbrido
# usando SA como solver. Registra una fila por (instance, run_id, k).
# Append-safe: detecta runs ya completados por (instance_label, solver, run_id).
#
# Nota: el bucle se implementa aquí explícitamente (no usa run_iterative_loop de src/)
# para poder registrar métricas intermedias por iteración.

from preprocessing import compute_feasible_slots
from qubo_builder import build_qubo, calibrate_penalties
from solver import decode_schedule, check_feasibility
from inventory import check_worst_case_overlaps
from dwave.samplers import SimulatedAnnealingSampler


def _q_density(bqm) -> float:
    n = len(bqm.variables)
    return round(len(bqm.quadratic) / (n * (n - 1) / 2), 6) if n > 1 else 0.0


def _run_hybrid_loop_sa(
    label: str,
    axis: str,
    inst: dict,
    run_id: int,
    seed: int,
    alpha: float,
    beta: float,
    k_max: int,
    num_reads: int,
    num_sweeps: int,
    run_uuid: str,
    inv_params: dict,
    tank_level: str,
) -> list[dict]:
    """
    Corre el bucle iterativo completo con SA para (label, tank_level, run_id).
    El loop para en la primera k donde n_violations = 0 (break en convergencia).
    """
    noms = inst["nominations"].copy()
    T    = int(inst["T"])
    N    = int(inst["N"])
    rho  = float(inst["rho_effective"])
    rdist = str(inst["r_j_distribution"])

    vdf    = compute_feasible_slots(noms, horizon_slots=T)
    P1, _, _ = calibrate_penalties(vdf, beta=beta)
    beta_range = (1.0 / (P1 * 2.0), 10.0)

    sampler    = SimulatedAnnealingSampler()
    all_cuts   = set()
    q_density_0 = None
    rows: list[dict] = []

    n_violations_prev = None

    for k in range(1, k_max + 1):
        bqm, _, _, _, _ = build_qubo(vdf, alpha=alpha, beta=beta,
                                     cuts=all_cuts if all_cuts else None)
        q_dens_k = _q_density(bqm)
        if q_density_0 is None:
            q_density_0 = q_dens_k

        t0 = time.perf_counter()
        ss = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            beta_range=beta_range,
            seed=seed * 10_000 + run_id * 100 + k,
        )
        wall_s = time.perf_counter() - t0

        best_sample = ss.first.sample
        best_energy = float(ss.first.energy)

        sched = decode_schedule(best_sample, vdf)
        fres  = check_feasibility(sched, vdf)
        is_feas = bool(fres["is_feasible"])
        obj_val = float(fres["total_weighted_tardiness"]) if is_feas else float("nan")

        new_cuts   = check_worst_case_overlaps(sched, vdf, noms, **inv_params)
        n_viol_k   = len(new_cuts)
        converged  = (n_viol_k == 0)

        # DEPRECATED: con break en k_conv, oscillating siempre es False.
        # Mantenida para compatibilidad con datos de corridas anteriores.
        n_new_k = len(new_cuts - all_cuts)
        oscillating = (n_new_k > 0 and n_violations_prev == 0)

        # C4: verificar que los cuts se aplicaron efectivamente (k=2)
        if k == 1:
            _energy_k1 = best_energy
        elif k == 2 and all_cuts and abs(best_energy - _energy_k1) < 1e-6:
            logger.warning(
                "  SA %s level=%s run_id=%d k=2: energía igual a k=1 (%.6f) "
                "— los cuts pueden no haberse aplicado al BQM.",
                label, tank_level, run_id, best_energy,
            )

        rows.append({
            "exp_id":                   "exp05",
            "run_uuid":                 run_uuid,
            "solver":                   "SA",
            "instance_label":           label,
            "axis":                     axis,
            "N":                        N,
            "T":                        T,
            "rho_effective":            rho,
            "r_j_distribution":         rdist,
            "seed":                     seed,
            "run_id":                   run_id,
            "k":                        k,
            "alpha":                    alpha,
            "beta":                     beta,
            "feasible":                 is_feas,
            "total_weighted_tardiness": obj_val,
            "n_violations":             n_viol_k,
            "n_new_violations_k":       n_new_k,
            "n_cuts_active":            len(all_cuts),
            "n_slots_violated_k":       len({t for (_, _, t) in new_cuts}),
            "n_unique_slots_total":     len({t for (_, _, t) in all_cuts}),
            "q_density_k":              q_dens_k,
            "q_density_delta":          round(q_dens_k - q_density_0, 6),
            "converged":                converged,
            "oscillating":              oscillating,
            "best_energy":              best_energy,
            "wall_time_s":              round(wall_s, 3),
            "lh_time_s":                float("nan"),
            "lh_run_time_us":           float("nan"),
            "n_solver_calls":           float("nan"),
            "n_vars_qubo":              len(bqm.variables),
            "initial_terminal_stock_m3": inv_params["initial_terminal_stock_m3"],
            "tank_level":               tank_level,
            "run_timestamp":            datetime.datetime.now().isoformat(),
        })

        if converged:
            logger.info("  SA %s level=%s run_id=%d convergió en k=%d",
                        label, tank_level, run_id, k)
            break  # F^(k) = ∅ — no hay cuts que acumular

        all_cuts |= new_cuts
        n_violations_prev = n_viol_k

    return rows


# ── Verificación de factibilidad ullage con Gurobi ────────────────────────────
# Una sola verificación por (instance, tank_level) antes de correr SA.
# Si Gurobi detecta INFEASIBLE, se salta esa combinación.
# Si gurobipy no está instalado, se omite el check y se continúa con SA.

import time as _time

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False
    logger.warning(
        "gurobipy no disponible — verificación de factibilidad Gurobi deshabilitada. "
        "Instalar con: pip install gurobipy"
    )

_GUROBI_TIME_LIMIT_S = 30
_SLOTS_PER_DAY_EXP5  = 24 / EXP5_INVENTORY_PARAMS["slot_duration_hours"]


def _gurobi_feasibility_check(label: str, tank_level: str, inst: dict, inv_params: dict) -> str:
    """
    Verifica factibilidad ullage exacta con Gurobi. Retorna 'FEASIBLE', 'INFEASIBLE',
    'TIMEOUT', 'ERROR' o 'SKIPPED' (si gurobipy no está disponible).
    """
    if not _GUROBI_AVAILABLE:
        return "SKIPPED"

    from preprocessing import compute_feasible_slots
    from config import CONFLICT_SET_R

    noms = inst["nominations"].copy()
    T_h  = int(inst["T"])
    vdf  = compute_feasible_slots(noms, horizon_slots=T_h)

    init_stock   = float(inv_params["initial_terminal_stock_m3"])
    daily_inflow = float(inv_params["daily_inflow_m3"])
    inflow_slot  = daily_inflow / _SLOTS_PER_DAY_EXP5
    n_tanks      = int(inv_params["n_tanks"])
    cap_per_tank = float(inv_params["tank_capacity_m3"])
    min_ullage   = int(inv_params["min_ullage_days"])
    safe_thresh  = n_tanks * cap_per_tank - min_ullage * daily_inflow

    vessel_pslots = {
        str(row["vessel_id"]): int(row["p_j"])
        for _, row in vdf.drop_duplicates("vessel_id").iterrows()
    }
    vessel_volumes = {
        str(row["vessel_id"]): float(row["volume_m3"])
        for _, row in noms.iterrows()
        if str(row["vessel_id"]) in vessel_pslots
    }

    try:
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
        model.setParam("TimeLimit", _GUROBI_TIME_LIMIT_S)
        model.setParam("OutputFlag", 0)

        x = {}
        for _, row in vdf.iterrows():
            j = str(row["vessel_id"])
            m = int(row["machine"])
            t = int(row["slot"])
            x[(j, m, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}_{t}")
        model.update()

        vessels = vdf["vessel_id"].unique().tolist()
        for j in vessels:
            vars_j = [x[(jj, m, t)] for (jj, m, t) in x if jj == j]
            if vars_j:
                model.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")

        vessel_list = list(vessels)
        for idx1 in range(len(vessel_list)):
            for idx2 in range(idx1 + 1, len(vessel_list)):
                j1, j2 = vessel_list[idx1], vessel_list[idx2]
                p1_slots = vessel_pslots[j1]
                p2_slots = vessel_pslots[j2]
                for (m1, m2) in CONFLICT_SET_R:
                    slots_j1 = [t for (jj, mm, t) in x if jj == j1 and mm == m1]
                    slots_j2 = [t for (jj, mm, t) in x if jj == j2 and mm == m2]
                    for t1 in slots_j1:
                        for t2 in slots_j2:
                            lo = max(t1, t2)
                            hi = min(t1 + p1_slots, t2 + p2_slots)
                            if lo < hi:
                                model.addConstr(
                                    x[(j1, m1, t1)] + x[(j2, m2, t2)] <= 1,
                                    name=f"overlap_{j1}_{m1}_{t1}_{j2}_{m2}_{t2}",
                                )

        for t_slot in range(T_h + 1):
            inflow_at_t = init_stock + inflow_slot * t_slot
            completed_terms = []
            for (j, m, ts) in x:
                p_j = vessel_pslots[j]
                if ts + p_j <= t_slot:
                    vol_j = vessel_volumes[j]
                    completed_terms.append(vol_j * x[(j, m, ts)])
            if completed_terms:
                model.addConstr(
                    inflow_at_t - gp.quicksum(completed_terms) <= safe_thresh,
                    name=f"ullage_t{t_slot}",
                )
            else:
                if inflow_at_t > safe_thresh:
                    model.addConstr(
                        gp.LinExpr() + inflow_at_t <= safe_thresh,
                        name=f"ullage_t{t_slot}_trivial",
                    )

        model.setObjective(0, GRB.MINIMIZE)
        model.optimize()

        status = model.Status
        model.dispose()
        env.dispose()

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return "FEASIBLE"
        elif status == GRB.INFEASIBLE:
            return "INFEASIBLE"
        elif status == GRB.TIME_LIMIT:
            return "TIMEOUT"
        else:
            return f"STATUS_{status}"

    except Exception as exc:
        logger.error("  Gurobi check falló para %s/%s: %s", label, tank_level, exc)
        return "ERROR"


# Ejecutar checks de factibilidad antes del bucle SA — UNA VEZ por (instance, tank_level)
_feasibility_cache: dict[tuple, str] = {}
if _GUROBI_AVAILABLE:
    logger.info("Verificando factibilidad Gurobi por (instancia, tank_level) ...")
    for _lbl, _meta in EXP5_INSTANCES.items():
        for _tlvl, _stk in EXP5_TANK_LEVELS.items():
            _inv_check = {**EXP5_INVENTORY_PARAMS, "initial_terminal_stock_m3": _stk}
            _feas_status = _gurobi_feasibility_check(
                label=_lbl, tank_level=_tlvl,
                inst=_meta["inst"], inv_params=_inv_check,
            )
            _feasibility_cache[(_lbl, _tlvl)] = _feas_status
            logger.info("  %s / %s → %s", _lbl, _tlvl, _feas_status)
else:
    logger.warning("Verificación Gurobi omitida (gurobipy no disponible).")

# Detectar runs SA ya completados.
# Clave: (instance_label, tank_level, run_id) — incluye nivel para reiniciar correctamente.
# Si el Excel existe pero no tiene columna tank_level (corridas anteriores), se loguea
# un warning y se trata esa columna como ausente (esas filas quedarán con NaN en tank_level).
_existing_sa = load_existing_runs(FILEPATH, SHEET)
if not _existing_sa.empty:
    _sa_rows = _existing_sa[_existing_sa["solver"] == "SA"]
    if "tank_level" in _sa_rows.columns:
        _done_sa = set(zip(
            _sa_rows["instance_label"],
            _sa_rows["tank_level"].fillna("__missing__"),
            _sa_rows["run_id"].astype(int),
        ))
    else:
        logger.warning(
            "Columna 'tank_level' ausente en el Excel existente — "
            "filas previas quedarán con NaN en esa columna."
        )
        _done_sa = set()
else:
    _done_sa = set()

for label, meta in EXP5_INSTANCES.items():
    inst  = meta["inst"]
    axis  = meta["axis"]
    label_noms = inst["nominations"].copy()
    T_inst = int(inst["T"])

    vdf_check = compute_feasible_slots(label_noms, horizon_slots=T_inst)
    n_vars_check = len(build_qubo(vdf_check, alpha=alpha_star, beta=beta_star)[0].variables)
    logger.info("SA %s (N=%d, T=%d, n_vars=%d)", label, inst["N"], T_inst, n_vars_check)

    for tank_level, stock_m3 in EXP5_TANK_LEVELS.items():
        # Dens_1/high: r_j mínimo=26 slots hace infactible el ullage en cualquier
        # nivel no trivial — confirmado por Gurobi IIS (stock cruza safe_threshold
        # en t=23 antes de que cualquier buque pueda completar su carga).
        if label == "Dens_1" and tank_level == "high":
            logger.warning("  skip %s / %s — infactible estructural (Gurobi IIS)", label, tank_level)
            continue

        # Verificación Gurobi previa (si disponible): saltar si INFEASIBLE
        _cached_feas = _feasibility_cache.get((label, tank_level))
        if _cached_feas == "INFEASIBLE":
            logger.warning(
                "  skip %s / %s — Gurobi confirmó INFEASIBLE (no correr SA)",
                label, tank_level,
            )
            continue

        # Construir inv_params para este nivel: copia base con stock sobreescrito
        _inv_sa = {**EXP5_INVENTORY_PARAMS, "initial_terminal_stock_m3": stock_m3}

        for run_id in range(N_RUNS_SA_EXP5):
            if (label, tank_level, run_id) in _done_sa:
                logger.info("  skip %s level=%s run_id=%d", label, tank_level, run_id)
                continue

            logger.info("  SA %s level=%s run_id=%d ...", label, tank_level, run_id)
            try:
                rows = _run_hybrid_loop_sa(
                    label=label, axis=axis, inst=inst,
                    run_id=run_id, seed=EXP5_SEED,
                    alpha=alpha_star, beta=beta_star,
                    k_max=EXP5_K_MAX,
                    num_reads=1000, num_sweeps=1000,
                    run_uuid=RUN_UUID,
                    inv_params=_inv_sa,
                    tank_level=tank_level,
                )
                append_rows(FILEPATH, SHEET, rows)
                k_conv = next((r["k"] for r in rows if r["converged"]), None)
                logger.info("  done  k_conv=%s  iterations=%d", k_conv, len(rows))
            except Exception as exc:
                logger.error("  SA %s level=%s run_id=%d falló: %s",
                             label, tank_level, run_id, exc)

logger.info("SA completo.")

# ── Verificación pre-LH ───────────────────────────────────────────────────────
_df_sa_check = load_existing_runs(FILEPATH, SHEET)
if not _df_sa_check.empty and "solver" in _df_sa_check.columns:
    _df_sa_check = _df_sa_check[_df_sa_check["solver"] == "SA"].copy()
    if "tank_level" in _df_sa_check.columns:
        _df_sa_check["tank_level"] = _df_sa_check["tank_level"].fillna("unknown")
    else:
        _df_sa_check["tank_level"] = "unknown"

    _expected_sa = sum(
        1
        for lbl in EXP5_INSTANCES
        for tlvl in EXP5_TANK_LEVELS
        if not (lbl == "Dens_1" and tlvl == "high")
        and _feasibility_cache.get((lbl, tlvl), "FEASIBLE") != "INFEASIBLE"
        for _ in range(N_RUNS_SA_EXP5)
    )
    _completed_sa = _df_sa_check.groupby(
        ["instance_label", "tank_level", "run_id"]
    ).ngroups if not _df_sa_check.empty else 0

    _sa_conv = _df_sa_check.groupby(
        ["instance_label", "tank_level", "run_id"]
    ).apply(lambda g: (g["n_violations"] == 0).any()).reset_index(name="converged")
    _n_conv = int(_sa_conv["converged"].sum())
    _n_total_runs = len(_sa_conv)
    _n_never_conv = _n_total_runs - _n_conv

    _kconv_vals = []
    for (_, tlvl, rid), g in _df_sa_check.groupby(["instance_label", "tank_level", "run_id"]):
        g_s = g.sort_values("k")
        first_conv = g_s[g_s["n_violations"] == 0]
        if not first_conv.empty:
            _kconv_vals.append(int(first_conv.iloc[0]["k"]))

    _k1_nonzero = (
        _df_sa_check[_df_sa_check["k"] == 1]["n_cuts_active"] > 0
    ).sum() if "n_cuts_active" in _df_sa_check.columns else 0

    # Verificar consistencia: ningún run convergido debe tener más filas que su k_conv
    _inconsistent = 0
    for (lbl, tlvl, rid), g in _df_sa_check.groupby(["instance_label", "tank_level", "run_id"]):
        g_s = g.sort_values("k")
        conv_rows = g_s[g_s["n_violations"] == 0]
        if not conv_rows.empty:
            k_c = int(conv_rows.iloc[0]["k"])
            n_rows_after = int((g_s["k"] > k_c).sum())
            if n_rows_after > 0:
                _inconsistent += 1

    _problems = []
    if _k1_nonzero > 0:
        _problems.append(f"n_cuts_active > 0 en k=1 para {_k1_nonzero} run(s) — posible bug de append")
    if _inconsistent > 0:
        _problems.append(f"{_inconsistent} run(s) convergidos tienen filas tras k_conv — break no aplicado")

    print("\n══ Verificación pre-LH ══")
    print(f"  SA completados:    {_completed_sa} runs de {_expected_sa} esperados")
    print(f"  Convergieron:      {_n_conv} / {_n_total_runs} runs")
    print(f"  Nunca convergió:   {_n_never_conv} / {_n_total_runs} runs")
    if _kconv_vals:
        print(f"  k_conv (convergidos): media={float(np.mean(_kconv_vals)):.2f}  std={float(np.std(_kconv_vals)):.2f}  rango=[{min(_kconv_vals)},{max(_kconv_vals)}]")
    if _problems:
        print(f"\n  REVISAR antes de ejecutar Celda 3:")
        for _p in _problems:
            print(f"    ✗ {_p}")
    else:
        print("\n  Listo para ejecutar Celda 3 (LH)")
else:
    print("\n══ Verificación pre-LH ══")
    print("  No hay datos SA en el Excel todavía.")


# CELDA 3: RUN LeapHybrid — bucle iterativo
#
# Mismo bucle que SA pero usando LeapHybridSampler. Append-safe.
# Registra todos los campos de timing disponibles por iteración.
#
# Configuración de presupuesto QPU — ajustar aquí antes de ejecutar:
#   LH_TANK_LEVEL_CELDA3: nivel de tanque a usar ("low", "nominal" o "high").
#     Elegir "nominal" para la primera corrida; cambiar a "high" para el stress test
#     una vez que SA confirme convergencia. Cada nivel es una corrida independiente.
#   LH_N_RUNS_CELDA3: número de runs LH para esta celda.
#     Estimación de consumo: runs × instancias × K_max × ~12s/llamada.
#     Ej: 5 × 3 × 10 × 12s ≈ 30 min QPU — distribuir en dos ciclos de facturación.
#     Reducir a 3 si el presupuesto del ciclo actual es ≤ 20 min.

LH_TANK_LEVEL_CELDA3 = "nominal"  # nivel activo para esta celda
LH_N_RUNS_CELDA3     = 3          # 3 × 3 inst × 10 iter × ~12s ≈ 18 min QPU

assert LH_TANK_LEVEL_CELDA3 in EXP5_TANK_LEVELS, (
    f"LH_TANK_LEVEL_CELDA3='{LH_TANK_LEVEL_CELDA3}' no es un nivel válido. "
    f"Opciones: {list(EXP5_TANK_LEVELS)}"
)

from solver import run_solver


def _run_hybrid_loop_lh(
    label: str,
    axis: str,
    inst: dict,
    run_id: int,
    seed: int,
    alpha: float,
    beta: float,
    k_max: int,
    run_uuid: str,
    inv_params: dict,
    tank_level: str,
) -> list[dict]:
    """
    Corre el bucle iterativo completo con LeapHybrid para (label, tank_level, run_id).
    Devuelve una fila por iteración k con todos los campos de timing disponibles.
    El loop para en la primera k donde n_violations = 0 (break en convergencia).
    """
    noms  = inst["nominations"].copy()
    T     = int(inst["T"])
    N     = int(inst["N"])
    rho   = float(inst["rho_effective"])
    rdist = str(inst["r_j_distribution"])

    vdf = compute_feasible_slots(noms, horizon_slots=T)

    all_cuts    = set()
    q_density_0 = None
    rows: list[dict] = []
    n_violations_prev = None

    for k in range(1, k_max + 1):
        bqm, _, _, _, _ = build_qubo(vdf, alpha=alpha, beta=beta,
                                     cuts=all_cuts if all_cuts else None)
        q_dens_k = _q_density(bqm)
        if q_density_0 is None:
            q_density_0 = q_dens_k

        t0 = time.perf_counter()
        ss_lh, _ = run_solver(bqm, requested_sampler="leaphybrid")
        lh_wall = time.perf_counter() - t0

        dw_timing = extract_solver_timing(ss_lh)

        best_sample = ss_lh.first.sample
        best_energy = float(ss_lh.first.energy)

        sched = decode_schedule(best_sample, vdf)
        fres  = check_feasibility(sched, vdf)
        is_feas = bool(fres["is_feasible"])
        obj_val = float(fres["total_weighted_tardiness"]) if is_feas else float("nan")

        new_cuts  = check_worst_case_overlaps(sched, vdf, noms, **inv_params)
        n_viol_k  = len(new_cuts)
        converged = (n_viol_k == 0)

        # DEPRECATED: con break en k_conv, oscillating siempre es False.
        # Mantenida para compatibilidad con datos de corridas anteriores.
        n_new_k     = len(new_cuts - all_cuts)
        oscillating = (n_new_k > 0 and n_violations_prev == 0)

        # C4: verificar que los cuts se aplicaron efectivamente (k=2)
        if k == 1:
            _energy_k1_lh = best_energy
        elif k == 2 and all_cuts and abs(best_energy - _energy_k1_lh) < 1e-6:
            logger.warning(
                "  LH %s level=%s run_id=%d k=2: energía igual a k=1 (%.6f) "
                "— los cuts pueden no haberse aplicado al BQM.",
                label, tank_level, run_id, best_energy,
            )

        rows.append({
            "exp_id":                   "exp05",
            "run_uuid":                 run_uuid,
            "solver":                   "LeapHybrid",
            "instance_label":           label,
            "axis":                     axis,
            "N":                        N,
            "T":                        T,
            "rho_effective":            rho,
            "r_j_distribution":         rdist,
            "seed":                     seed,
            "run_id":                   run_id,
            "k":                        k,
            "alpha":                    alpha,
            "beta":                     beta,
            "feasible":                 is_feas,
            "total_weighted_tardiness": obj_val,
            "n_violations":             n_viol_k,
            "n_new_violations_k":       n_new_k,
            "n_cuts_active":            len(all_cuts),
            "n_slots_violated_k":       len({t for (_, _, t) in new_cuts}),
            "n_unique_slots_total":     len({t for (_, _, t) in all_cuts}),
            "q_density_k":              q_dens_k,
            "q_density_delta":          round(q_dens_k - q_density_0, 6),
            "converged":                converged,
            "oscillating":              oscillating,
            "best_energy":              best_energy,
            "wall_time_s":              round(lh_wall, 3),
            # LeapHybrid timing (desde sampleset.info)
            "lh_time_s":                dw_timing["lh_run_time_s"],
            "lh_run_time_us":           dw_timing["lh_run_time_us"],
            "n_solver_calls":           dw_timing["n_solver_calls"],
            # Columnas compartidas con SA — deben mantener el mismo orden para
            # que append_rows no desalinee el schema de la hoja (bug: las cols QPU
            # estaban aquí, empujando n_vars_qubo/tank_level a Unnamed: 34-39).
            "n_vars_qubo":              len(bqm.variables),
            "initial_terminal_stock_m3": inv_params["initial_terminal_stock_m3"],
            "tank_level":               tank_level,
            "run_timestamp":            datetime.datetime.now().isoformat(),
            # QPU-direct timing al final — columnas extras de LH que SA no tiene.
            # Deben ir DESPUÉS de run_timestamp para no desplazar columnas compartidas.
            "qpu_sampling_time_us":              dw_timing["qpu_sampling_time_us"],
            "qpu_anneal_time_per_sample_us":     dw_timing["qpu_anneal_time_per_sample_us"],
            "qpu_readout_time_per_sample_us":    dw_timing["qpu_readout_time_per_sample_us"],
            "qpu_access_time_us":                dw_timing["qpu_access_time_us"],
            "qpu_access_overhead_time_us":       dw_timing["qpu_access_overhead_time_us"],
            "total_post_processing_time_us":     dw_timing["total_post_processing_time_us"],
        })

        if converged:
            logger.info("  LH %s level=%s run_id=%d convergió en k=%d",
                        label, tank_level, run_id, k)
            break  # F^(k) = ∅ — no hay cuts que acumular

        all_cuts |= new_cuts
        n_violations_prev = n_viol_k

    return rows


# Detectar runs LH ya completados para el nivel activo en esta celda.
# Clave: (instance_label, tank_level, run_id).
_existing_lh = load_existing_runs(FILEPATH, SHEET)
if not _existing_lh.empty:
    _lh_df = _existing_lh[_existing_lh["solver"] == "LeapHybrid"]
    if "tank_level" in _lh_df.columns:
        _done_lh = set(zip(
            _lh_df["instance_label"],
            _lh_df["tank_level"].fillna("__missing__"),
            _lh_df["run_id"].astype(int),
        ))
    else:
        logger.warning(
            "Columna 'tank_level' ausente en el Excel existente — "
            "filas previas quedarán con NaN en esa columna."
        )
        _done_lh = set()
else:
    _done_lh = set()

# Construir inv_params para el nivel activo de esta celda
_inv_lh = {**EXP5_INVENTORY_PARAMS,
           "initial_terminal_stock_m3": EXP5_TANK_LEVELS[LH_TANK_LEVEL_CELDA3]}

for label, meta in EXP5_INSTANCES.items():
    inst  = meta["inst"]
    axis  = meta["axis"]

    # Dens_1/high: infactible estructural (ver C1). Mismo criterio que SA.
    if label == "Dens_1" and LH_TANK_LEVEL_CELDA3 == "high":
        logger.warning("  skip %s / %s — infactible estructural (Gurobi IIS)", label, LH_TANK_LEVEL_CELDA3)
        continue

    logger.info("LH %s level=%s (N=%d, T=%d)",
                label, LH_TANK_LEVEL_CELDA3, inst["N"], inst["T"])

    for run_id in range(LH_N_RUNS_CELDA3):
        if (label, LH_TANK_LEVEL_CELDA3, run_id) in _done_lh:
            logger.info("  skip %s level=%s run_id=%d",
                        label, LH_TANK_LEVEL_CELDA3, run_id)
            continue

        logger.info("  LH %s level=%s run_id=%d ...",
                    label, LH_TANK_LEVEL_CELDA3, run_id)
        try:
            rows = _run_hybrid_loop_lh(
                label=label, axis=axis, inst=inst,
                run_id=run_id, seed=EXP5_SEED,
                alpha=alpha_star, beta=beta_star,
                k_max=EXP5_K_MAX,
                run_uuid=RUN_UUID,
                inv_params=_inv_lh,
                tank_level=LH_TANK_LEVEL_CELDA3,
            )
            append_rows(FILEPATH, SHEET, rows)
            k_conv = next((r["k"] for r in rows if r["converged"]), None)
            logger.info(
                "  done  k_conv=%s  iterations=%d  total_lh_wall=%.1fs",
                k_conv, len(rows),
                sum(r["wall_time_s"] for r in rows),
            )
        except Exception as exc:
            logger.error("  LH %s level=%s run_id=%d falló: %s",
                         label, LH_TANK_LEVEL_CELDA3, run_id, exc)

logger.info("LeapHybrid completo.")

# Guardar metadata
save_metadata(FILEPATH, {
    "alpha_star":   alpha_star,
    "beta_star":    beta_star,
    "k_max":        EXP5_K_MAX,
    "n_sa_runs":    N_RUNS_SA_EXP5,
    "n_lh_runs":    EXP5_LH_RUNS,
    "instances":    str(list(EXP5_INSTANCES)),
    "run_uuid_last": RUN_UUID,
})


# CELDA 4: CALCULAR métricas derivadas
#
# Lee el Excel completo y calcula:
# - k_conv: primera k donde n_violations == 0 por (solver, instance, run_id)
# - oscillating_run: True si n_new_violations_k > 0 en ≥3 iteraciones post-convergencia
# - mean_k_conv, std_k_conv por (solver, instance)
# - fracción de runs convergentes

df_raw = pd.read_excel(FILEPATH, sheet_name=SHEET)

def _k_conv(g: pd.DataFrame) -> int | None:
    conv = g[g["n_violations"] == 0].sort_values("k")
    return int(conv.iloc[0]["k"]) if not conv.empty else None


def _oscillating_run(g: pd.DataFrame) -> bool:
    # DEPRECATED: con break en k_conv, esta métrica siempre es False.
    # Mantenida para compatibilidad con datos de corridas anteriores.
    conv_rows = g[g["n_violations"] == 0]["k"]
    if conv_rows.empty:
        return False
    k_first = int(conv_rows.min())
    post = g[g["k"] > k_first]
    return int((post["n_new_violations_k"] > 0).sum()) >= 3


# tank_level puede estar ausente o nulo en corridas grabadas con versiones
# anteriores del script (antes de que se añadiera la columna).
# Diagnóstico: mostrar valores crudos antes de cualquier fillna.
if "tank_level" not in df_raw.columns:
    logger.warning("Columna 'tank_level' ausente en el Excel — usando 'unknown' para filas antiguas.")
    df_raw["tank_level"] = "unknown"
else:
    # Verificar si hay nulos y de qué solvers provienen — los nulos son filas
    # grabadas por versiones anteriores del script, no un bug del código actual.
    _null_by_solver = df_raw[df_raw["tank_level"].isna()].groupby("solver").size()
    if not _null_by_solver.empty:
        logger.warning(
            "tank_level nulo en %d fila(s) del Excel (grabadas por versión anterior "
            "del script sin esta columna):\n%s\n"
            "→ Se asigna 'unknown'. El nivel real puede inferirse de "
            "initial_terminal_stock_m3 si está bien registrado.",
            df_raw["tank_level"].isna().sum(),
            _null_by_solver.to_string(),
        )
        # Diagnóstico: valores únicos crudos por solver antes del fillna
        print("\n── tank_level crudo en Excel (antes de fillna) ──")
        print(df_raw.groupby("solver")["tank_level"].apply(
            lambda s: s.value_counts(dropna=False).to_dict()
        ).to_string())
    df_raw["tank_level"] = df_raw["tank_level"].fillna("unknown")

# Filas con tank_level="unknown": cruzar con initial_terminal_stock_m3 para
# recuperar el nivel real. Útil para corridas LH grabadas sin la columna.
_unknown_rows = df_raw[df_raw["tank_level"] == "unknown"]
if not _unknown_rows.empty and "initial_terminal_stock_m3" in df_raw.columns:
    _stock_to_level = {v: k for k, v in EXP5_TANK_LEVELS.items()}
    _recovered = _unknown_rows["initial_terminal_stock_m3"].map(
        lambda s: _stock_to_level.get(round(float(s)) if pd.notna(s) else None, "unknown")
    )
    _n_recovered = (_recovered != "unknown").sum()
    if _n_recovered > 0:
        df_raw.loc[_unknown_rows.index, "tank_level"] = _recovered
        logger.warning(
            "Recuperado tank_level desde initial_terminal_stock_m3 para %d fila(s) "
            "con 'unknown'. Niveles recuperados: %s",
            _n_recovered,
            _recovered[_recovered != "unknown"].value_counts().to_dict(),
        )

summary_rows = []
for (solver, label, tank_level, run_id), grp in df_raw.groupby(
    ["solver", "instance_label", "tank_level", "run_id"]
):
    grp_s = grp.sort_values("k")
    kc = _k_conv(grp_s)
    osc = _oscillating_run(grp_s)
    summary_rows.append({
        "solver":          solver,
        "instance_label":  label,
        "tank_level":      tank_level,
        "run_id":          run_id,
        "k_conv":          kc,
        "converged":       kc is not None,
        "never_converged": kc is None,
        "oscillating_run": osc,  # DEPRECATED — ver _oscillating_run
    })

df_summary = pd.DataFrame(summary_rows)

print("\n── Resumen de convergencia por (solver, instancia, tank_level) ──")
agg = (
    df_summary.groupby(["solver", "instance_label", "tank_level"])
    .agg(
        n_runs=("run_id", "count"),
        converged_rate=("converged", "mean"),
        never_converged_rate=("never_converged", "mean"),
        k_conv_mean=("k_conv", lambda s: s.dropna().mean()),
        k_conv_std=("k_conv", lambda s: s.dropna().std()),
        k_conv_min=("k_conv", "min"),
        k_conv_max=("k_conv", "max"),
        oscillating_any=("oscillating_run", "any"),  # DEPRECATED — ver nota
    )
    .reset_index()
)
print(agg.to_string(index=False))

# Verificación: n_vars_qubo debe ser constante por (solver, instance, run_id)
vars_check = (
    df_raw.dropna(subset=["n_vars_qubo"])
    .groupby(["solver", "instance_label", "run_id"])["n_vars_qubo"]
    .nunique()
)
_violators = vars_check[vars_check > 1]
if not _violators.empty:
    logger.warning(
        "n_vars_qubo varía entre iteraciones en %d grupo(s) — "
        "los cortes podrían estar agregando variables:\n%s",
        len(_violators), _violators.to_string(),
    )
else:
    print("\n✓ n_vars_qubo constante en todas las iteraciones (cortes son puramente diagonales)")

# Alerta de oscilación
osc_cases = df_summary[df_summary["oscillating_run"]]
if not osc_cases.empty:
    print(f"\n⚠  Oscilación detectada en {len(osc_cases)} run(s):")
    print(osc_cases[["solver", "instance_label", "tank_level", "run_id"]].to_string(index=False))
else:
    print("\n✓ Sin oscilaciones detectadas.")

# Tasa de reconvergencia (v9.1)
_reconverge_flags: list[dict] = []
for (solver_r, label_r, tank_r, run_id_r), grp_r in df_raw.groupby(
    ["solver", "instance_label", "tank_level", "run_id"]
):
    grp_r = grp_r.sort_values("k")
    conv_series = grp_r["converged"].astype(bool).reset_index(drop=True)
    had_reconvergence = False
    for _idx in range(1, len(conv_series)):
        if conv_series[_idx - 1] and not conv_series[_idx]:
            had_reconvergence = True
            break
    _reconverge_flags.append({
        "solver":           solver_r,
        "instance_label":   label_r,
        "tank_level":       tank_r,
        "run_id":           run_id_r,
        "had_reconvergence": had_reconvergence,
    })

df_reconverge = pd.DataFrame(_reconverge_flags)
reconverge_rate_agg = (
    df_reconverge.groupby(["solver", "instance_label", "tank_level"])
    .agg(
        n_runs_total=("run_id", "count"),
        n_reconvergence=("had_reconvergence", "sum"),
        reconvergence_rate=("had_reconvergence", "mean"),
    )
    .reset_index()
)
print("\n── Tasa de reconvergencia por (solver, instancia, tank_level) ──")
print(reconverge_rate_agg.to_string(index=False))

_n_reconverge_total = int(df_reconverge["had_reconvergence"].sum())
if _n_reconverge_total > 0:
    logger.warning(
        "Reconvergencia detectada en %d run(s) — "
        "la convergencia del bucle NO es monotónica para estos casos.",
        _n_reconverge_total,
    )
else:
    print("\n✓ Sin reconvergencia — todos los runs muestran convergencia monotónica.")


# CELDA 5: PLOT convergencia — n_violations y tardiness por iteración

import matplotlib.pyplot as plt
plt.switch_backend("inline")

FIGURES_DIR = EXPERIMENTS2_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES_PLOT = ["Size_1", "Size_2"]
SOLVERS_PLOT   = ["SA", "LeapHybrid"]
COLORS = {"SA": "#2166ac", "LeapHybrid": "#d6604d"}
ALPHA_SA = 0.35

fig, axes = plt.subplots(
    nrows=len(INSTANCES_PLOT), ncols=2,
    figsize=(12, 4 * len(INSTANCES_PLOT)),
    sharex=True,
)
if len(INSTANCES_PLOT) == 1:
    axes = np.array([axes])

for row_i, inst_label in enumerate(INSTANCES_PLOT):
    ax_viol = axes[row_i, 0]
    ax_tard = axes[row_i, 1]

    for solver in SOLVERS_PLOT:
        sub = df_raw[(df_raw["instance_label"] == inst_label) & (df_raw["solver"] == solver)]
        if sub.empty:
            continue

        color = COLORS[solver]
        for (tank_lvl, run_id), grp in sub.groupby(["tank_level", "run_id"]):
            grp_s = grp.sort_values("k")
            alpha_line = 1.0 if solver == "LeapHybrid" else ALPHA_SA
            ax_viol.plot(grp_s["k"], grp_s["n_violations"],
                         color=color, alpha=alpha_line, linewidth=1.0)
            ax_tard.plot(grp_s["k"],
                         grp_s["total_weighted_tardiness"].fillna(np.nan),
                         color=color, alpha=alpha_line, linewidth=1.0)

        mean_viol = sub.groupby("k")["n_violations"].mean()
        mean_tard = sub.groupby("k")["total_weighted_tardiness"].mean()
        ax_viol.plot(mean_viol.index, mean_viol.values,
                     color=color, linewidth=2.2, label=solver)
        ax_tard.plot(mean_tard.index, mean_tard.values,
                     color=color, linewidth=2.2, label=solver)

    ax_viol.axhline(0, color="black", linestyle="--", linewidth=0.8)

    # Línea vertical en k_conv_mean de SA (si está disponible)
    _sa_kconv_mean = (
        agg.loc[
            (agg["instance_label"] == inst_label) & (agg["solver"] == "SA"),
            "k_conv_mean",
        ].dropna().values
    )
    if len(_sa_kconv_mean) > 0 and not np.isnan(_sa_kconv_mean[0]):
        ax_viol.axvline(
            _sa_kconv_mean[0], color=COLORS["SA"], linestyle=":",
            linewidth=1.2, alpha=0.7, label=f"mean k_conv (SA)={_sa_kconv_mean[0]:.1f}",
        )

    ax_viol.set_xlim(1, EXP5_K_MAX)
    ax_viol.set_title(f"{inst_label} — Ullage violations |F(k)|")
    ax_viol.set_ylabel("|F(k)|")
    ax_viol.legend(fontsize=8)

    ax_tard.set_xlim(1, EXP5_K_MAX)
    ax_tard.set_title(f"{inst_label} — Weighted tardiness")
    ax_tard.set_ylabel("Σ wⱼtⱼ")
    ax_tard.legend(fontsize=8)

for ax in axes[-1]:
    ax.set_xlabel("Iteration k")

fig.suptitle("Exp 5 — Hybrid Loop Convergence", fontsize=13, y=1.01)
fig.tight_layout()
fig_path = RESULTS_DIR / "exp05_convergence.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
logger.info("Figura guardada: %s", fig_path)


# CELDA 6: PLOT estabilidad de densidad QUBO
#
# q_density_k debe permanecer constante ±2% respecto a k=0.
# Confirma que los cortes lineales no densifican el grafo de couplers.

import matplotlib.pyplot as plt

fig2, axes2 = plt.subplots(
    nrows=1, ncols=len(INSTANCES_PLOT),
    figsize=(6 * len(INSTANCES_PLOT), 4),
    sharey=False,
)
if len(INSTANCES_PLOT) == 1:
    axes2 = [axes2]

for col_i, inst_label in enumerate(INSTANCES_PLOT):
    ax = axes2[col_i]

    for solver in SOLVERS_PLOT:
        sub = df_raw[(df_raw["instance_label"] == inst_label) & (df_raw["solver"] == solver)]
        if sub.empty:
            continue

        color = COLORS[solver]
        mean_dens = sub.groupby("k")["q_density_k"].mean()
        q0 = float(sub[sub["k"] == 1]["q_density_k"].mean())

        ax.plot(mean_dens.index, mean_dens.values,
                color=color, linewidth=2.0, label=solver)
        ax.axhline(q0,        color=color, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(q0 * 1.02, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(q0 * 0.98, color="gray", linestyle=":", linewidth=0.8)

    ax.set_title(f"{inst_label} — Q-matrix density")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("q_density")
    ax.legend(fontsize=8)

fig2.suptitle("Exp 5 — QUBO Density Stability (±2% band)", fontsize=12)
fig2.tight_layout()
fig2_path = RESULTS_DIR / "exp05_qubo_density.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig2)
logger.info("Figura guardada: %s", fig2_path)


# CELDA 7: PLOT k_conv vs instancia
#
# Eje X: instancia (Size_1, Size_2, Dens_1)
# Eje Y: k_conv — iteraciones hasta convergencia
# SA:  una serie por tank_level (colores distintos) — media ± std.
# LH:  todos los tank_levels disponibles — puntos individuales + media.

import matplotlib.pyplot as plt

SA_LEVEL_COLORS  = {"low": "#4dac26", "nominal": "#2166ac", "high": "#d01c8b", "unknown": "#888888"}
SA_LEVEL_OFFSETS = {"low": -0.20, "nominal": 0.0, "high": 0.20, "unknown": 0.0}

fig3, ax3 = plt.subplots(figsize=(9, 4))

_x_labels = [l for l in ["Size_1", "Size_2", "Dens_1"]
              if l in df_summary["instance_label"].unique()]
x_pos = {l: i for i, l in enumerate(_x_labels)}

# SA — una serie por nivel de tanque disponible
sa_summary = df_summary[df_summary["solver"] == "SA"]
for tank_lvl in sorted(sa_summary["tank_level"].unique()):
    color  = SA_LEVEL_COLORS.get(tank_lvl, "#888888")
    offset = SA_LEVEL_OFFSETS.get(tank_lvl, 0.0)
    sub_lvl = sa_summary[sa_summary["tank_level"] == tank_lvl]
    for inst_label in _x_labels:
        grp = sub_lvl[sub_lvl["instance_label"] == inst_label]["k_conv"].dropna()
        if grp.empty:
            continue
        ax3.errorbar(
            x_pos[inst_label] + offset,
            grp.mean(),
            yerr=(grp.std() if len(grp) > 1 else 0),
            fmt="o", color=color, capsize=4,
            label=f"SA-{tank_lvl}" if inst_label == _x_labels[0] else "",
        )

# LH — todos los tank_levels disponibles en los datos
lh_color   = COLORS["LeapHybrid"]
lh_summary = df_summary[df_summary["solver"] == "LeapHybrid"]
_lh_levels = sorted(lh_summary["tank_level"].unique())
for inst_label in _x_labels:
    grp = lh_summary[lh_summary["instance_label"] == inst_label]["k_conv"].dropna()
    if grp.empty:
        continue
    x = x_pos[inst_label] + 0.30
    ax3.scatter([x] * len(grp), grp.values,
                color=lh_color, alpha=0.6, s=30, zorder=3)
    ax3.scatter(x, grp.mean(),
                color=lh_color, s=80, marker="D", zorder=4,
                label=f"LH ({', '.join(_lh_levels)})" if inst_label == _x_labels[0] else "")

ax3.set_xticks(list(x_pos.values()))
ax3.set_xticklabels(list(x_pos))
ax3.set_ylabel("k_conv (iterations to convergence)")
ax3.set_xlabel("Instance")
ax3.set_title("Exp 5 — Iterations to Convergence by Tank Level")
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.4)

fig3.tight_layout()
fig3_path = RESULTS_DIR / "exp05_kconv_vs_instance.png"
fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig3)
logger.info("Figura guardada: %s", fig3_path)

logger.info("Exp 5 completo.")
