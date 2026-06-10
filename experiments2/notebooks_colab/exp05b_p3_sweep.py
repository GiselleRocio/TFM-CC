"""
exp05b_p3_sweep.py — Experimento 5b: Sweep de Calibración del Parámetro P3

Pregunta: ¿Existe un valor de P3 que logre convergencia en los casos donde el
          Exp 5 falló? ¿Cuál es el tradeoff entre P3 y la calidad del scheduling?

Motivación: En Exp 5, todos los runs con violaciones en k=1 nunca convergieron
en K=10. El desplazamiento de energía por corte (~47 unidades) es ~32x menor
que el ancho del basin de la solución violada (~1500 unidades). La hipótesis
es que P3 es demasiado pequeño para redirigir al solver.

Solo SA — cero costo QPU.

Instancias: únicamente las que fallaron en Exp 5:
  - Size_1, tank_level=high
  - Size_2, tank_level=high
  - Dens_1, tank_level=nominal
  - Dens_1, tank_level=high

Grid de P3_multiplier: [1, 5, 10, 30, 100]
  (relativo al P3 de Exp5 — el multiplicador 1 replica Exp5 como baseline)

  P3_base = P2 / beta  = (alpha * n * c_max) / beta   [qubo_builder.py Eq. 13c]
  P3_eff  = P3_base * p3_multiplier

WARNING de jerarquía: si P3_eff > P1 / 2 se emite WARNING en el log.
  P1/2 es el umbral donde el bias de un corte compensa la mitad de la
  penalidad de asignación de un buque completo. Por encima de ese umbral,
  los cortes de ullage pueden interferir con la semántica de H_assign.
  (P1 = alpha^2 * n * c_max  [qubo_builder.py Eq. 13a])

Prerequisito: Exp 2 completado (α*, β* en exp02_lagrange_calibration.xlsx).

Outputs (modo validación — P3_multiplier=[1]):
  results/exp05b_fix_validation.xlsx
    hoja: per_iteration  (una fila por instance × tank_level × p3_multiplier × run × k)
    hoja: metadata       (p3_grid, alpha_star, beta_star, instances, run_uuid)

  Cuando el fix esté validado, restaurar P3_multiplier=[1, 5, 10, 30, 100] y
  cambiar FILEPATH a results/exp05b_p3_sweep.xlsx para el sweep completo.

Ejecución (modo validación):
  Celda 0: INSTALL (solo una vez por sesión de Colab)
  Celda SETUP: configurar Drive y paths
  Celda 1: SETUP + cargar α*, β* de Exp 2
  Celda 2: RUN SA — P3×1 sobre las 4 instancias fallidas de Exp5
  Celda 3: RESULTADOS — tabla instance | tank_level | run_id | k_conv | n_cuts_at_kconv | converged
"""

# CELDA 0: INSTALL — ejecutar una sola vez por sesion de Colab
# %pip install -q dimod dwave-samplers openpyxl seaborn

# CELDA SETUP (Colab)
# ---- EDITAR SI TU CARPETA TIENE OTRO NOMBRE ---
DRIVE_TESIS_PATH = "MyDrive/TESIS"
# -----------------------------------------------

import os, sys
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_TESIS      = f"/content/drive/{DRIVE_TESIS_PATH}"
REPO_ROOT        = Path(DRIVE_TESIS)
EXPERIMENTS2_DIR = REPO_ROOT / "experiments2"

for p in [str(REPO_ROOT / "src"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments2.shared.io_utils import RESULTS_DIR
print(f"  ok  REPO_ROOT: {REPO_ROOT}")

# Cargar credenciales desde TESIS/.env
# (GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID para Gurobi WLS)
_dotenv_path = REPO_ROOT / ".env"
if _dotenv_path.exists():
    for _ln in _dotenv_path.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
    print("  ok  credenciales cargadas desde TESIS/.env")
else:
    print("  AVISO: TESIS/.env no encontrado. Credenciales Gurobi no cargadas.")


# CELDA 1: SETUP + cargar α*, β* de Exp 2

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
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("exp05b")

ensure_directories()

RUN_UUID  = new_run_uuid()
FILEPATH  = RESULTS_DIR / "exp05b_fix_validation.xlsx"
EXP2_PATH = RESULTS_DIR / "exp02_lagrange_calibration.xlsx"
SHEET     = "per_iteration"

logger.info("Exp 5b SETUP  run_uuid=%s", RUN_UUID)

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
    if "volume_m3" not in noms.columns:
        logger.warning(
            "Columna 'volume_m3' ausente — usando stock_acumulado_m3 como proxy."
        )
        noms = noms.copy()
        noms["volume_m3"] = noms["stock_acumulado_m3"]
    return noms


# Cargar instancias que fallaron en Exp 5
_size_dict = load_instances_from_excel("size")
for _inst in _size_dict.values():
    _inst["nominations"] = _ensure_volume_m3(_inst["nominations"])

# Dens_1 excluida — r_j mínimo=26 hace infactible el ullage en cualquier nivel
# no trivial porque el inflow acumula 260k m³ antes de que cualquier buque pueda
# completar su carga. Confirmado por Gurobi IIS (slot t=23 ya viola sin ningún
# buque activo). No hay schedule posible que satisfaga la restricción de ullage.
EXP5B_INSTANCES = [
    {"label": "Size_1", "axis": "size", "inst": _size_dict["Size_1"], "tank_level": "high"},
    {"label": "Size_2", "axis": "size", "inst": _size_dict["Size_2"], "tank_level": "high"},
]

# Parámetros físicos de inventario
EXP5B_INVENTORY_PARAMS = {
    "slot_duration_hours":       SLOT_HOURS,
    "min_ullage_days":           MIN_ULLAGE_DAYS,
    "initial_terminal_stock_m3": INITIAL_TERMINAL_STOCK_M3,
    "n_tanks":                   N_TANKS,
    "tank_capacity_m3":          TANK_CAPACITY_M3,
    "daily_inflow_m3":           DAILY_INFLOW_M3,
}

_TANK_CAPACITY_TOTAL = N_TANKS * TANK_CAPACITY_M3
_SAFE_THRESHOLD      = _TANK_CAPACITY_TOTAL - MIN_ULLAGE_DAYS * DAILY_INFLOW_M3

EXP5B_TANK_LEVELS = {
    "low":     round(_TANK_CAPACITY_TOTAL * 0.35),
    "nominal": INITIAL_TERMINAL_STOCK_M3,
    "high":    round(_SAFE_THRESHOLD - DAILY_INFLOW_M3),
}

# Parámetros del experimento (mismos que Exp 5 salvo el grid de P3)
EXP5B_SEED           = 0
N_RUNS_SA_EXP5B      = 5
EXP5B_K_MAX          = EXP5_K_MAX       # 10 iteraciones
# Grid reducido a [1]: validar que el fix de granularidad de cuts restaura
# convergencia con el mismo P3 de Exp5. Si converge, el problema estaba en
# los cuts, no en P3. Re-expandir a [1, 5, 10, 30, 100] tras validar.
EXP5B_P3_MULTIPLIERS = [1]

logger.info(
    "Instancias fallidas de Exp5: %d  seed=%d  K_max=%d  N_SA=%d  P3_grid=%s",
    len(EXP5B_INSTANCES), EXP5B_SEED, EXP5B_K_MAX,
    N_RUNS_SA_EXP5B, EXP5B_P3_MULTIPLIERS,
)


# CELDA 2: RUN SA — sweep de P3 sobre instancias fallidas de Exp 5
#
# Por cada (instancia, tank_level, p3_multiplier, run_id) corre EXP5B_K_MAX
# iteraciones del bucle híbrido con SA. Registra una fila por k.
# Append-safe: detecta runs ya completados por (instance_label, tank_level,
# p3_multiplier, run_id).
#
# P3_base se obtiene de calibrate_penalties (qubo_builder.py, Eq. 13c):
#   P3_base = P2 / beta = (alpha * n * c_max) / beta
#   P3_eff  = P3_base * p3_multiplier
#
# WARNING de jerarquía: si P3_eff > P1 / 2, los cortes pueden empezar a
# compensar la mitad del costo de dejar un buque sin asignar, rompiendo la
# semántica de H_assign. Se loguea como WARNING al inicio de cada run.
#   P1 = alpha^2 * n * c_max  (qubo_builder.py, Eq. 13a)

from preprocessing import compute_feasible_slots
from qubo_builder import build_qubo, calibrate_penalties
from solver import decode_schedule, check_feasibility
from inventory import check_worst_case_overlaps
from dwave.samplers import SimulatedAnnealingSampler


def _q_density(bqm) -> float:
    n = len(bqm.variables)
    return round(len(bqm.quadratic) / (n * (n - 1) / 2), 6) if n > 1 else 0.0


def _run_hybrid_loop_sa_p3(
    label: str,
    axis: str,
    inst: dict,
    run_id: int,
    seed: int,
    alpha: float,
    beta: float,
    p3_multiplier: float,
    k_max: int,
    num_reads: int,
    num_sweeps: int,
    run_uuid: str,
    inv_params: dict,
    tank_level: str,
) -> list[dict]:
    """
    Corre el bucle iterativo completo con SA y P3 escalado por p3_multiplier.

    P3_base se calcula mediante calibrate_penalties (mismo código que Exp5,
    qubo_builder.py Eq. 13c: P3 = P2 / beta). El BQM se construye sin cuts
    vía build_qubo, luego se inyectan los bias de corte con P3_eff directamente
    sobre las variables correspondientes (equivalente a _add_h_cuts con P3_eff).

    WARNING si P3_eff > P1 / 2: los cortes podrían compensar parte del costo
    de no asignar un buque, interfiriendo con H_assign.

    Devuelve una fila por iteración k con columnas estándar de Exp5 más
    p3_multiplier, p3_absolute, p3_base y p1_absolute.
    """
    noms  = inst["nominations"].copy()
    T     = int(inst["T"])
    N     = int(inst["N"])
    rho   = float(inst["rho_effective"])
    rdist = str(inst["r_j_distribution"])

    vdf = compute_feasible_slots(noms, horizon_slots=T)

    # P1, P2, P3_base de qubo_builder.calibrate_penalties (Eqs. 13a–13c)
    P1, P2, P3_base = calibrate_penalties(vdf, alpha=alpha, beta=beta)
    P3_eff     = P3_base * p3_multiplier
    P1_half    = P1 / 2.0
    beta_range = (1.0 / (P1 * 2.0), 10.0)

    # WARNING: umbral P1/2 — por encima, los cortes pueden interferir con H_assign
    if P3_eff > P1_half:
        logger.warning(
            "  WARNING: P3_eff (%.4f) > P1/2 (%.4f) para %s level=%s p3_mult=%.0f — "
            "los cortes de ullage pueden compensar parte del costo de asignación. "
            "Resultados en este punto del grid pueden no ser confiables.",
            P3_eff, P1_half, label, tank_level, p3_multiplier,
        )

    sampler           = SimulatedAnnealingSampler()
    all_cuts          = set()
    q_density_0       = None
    rows: list[dict]  = []
    n_violations_prev = None

    for k in range(1, k_max + 1):
        # Construir BQM base sin cuts escalados (build_qubo con cuts=None)
        bqm, _, _, _, _ = build_qubo(vdf, alpha=alpha, beta=beta, cuts=None)

        # Inyectar bias de corte con P3_eff para cada cut acumulado
        # (equivalente a _add_h_cuts de qubo_builder con P3_eff en lugar de P3_base)
        if all_cuts:
            applied = 0
            for (vessel_id, machine, slot) in all_cuts:
                var = f"x_{vessel_id}_{machine}_{slot}"
                if var in bqm.variables:
                    bqm.add_variable(var, P3_eff)
                    applied += 1
            if applied == 0:
                logger.error(
                    "  NINGÚN cut fue aplicado al BQM — verificar naming de variables en qubo_builder. "
                    "Primeros cuts: %s  |  Primeras vars BQM: %s",
                    list(all_cuts)[:3],
                    list(bqm.variables)[:3],
                )

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

        sched   = decode_schedule(best_sample, vdf)
        fres    = check_feasibility(sched, vdf)
        is_feas = bool(fres["is_feasible"])
        obj_val = float(fres["total_weighted_tardiness"]) if is_feas else float("nan")

        new_cuts  = check_worst_case_overlaps(sched, vdf, noms, **inv_params)
        n_viol_k  = len(new_cuts)
        converged = (n_viol_k == 0)

        n_new_k     = len(new_cuts - all_cuts)
        oscillating = (n_new_k > 0 and n_violations_prev == 0)

        rows.append({
            "exp_id":                    "exp05b",
            "run_uuid":                  run_uuid,
            "solver":                    "SA",
            "instance_label":            label,
            "axis":                      axis,
            "N":                         N,
            "T":                         T,
            "rho_effective":             rho,
            "r_j_distribution":          rdist,
            "seed":                      seed,
            "run_id":                    run_id,
            "k":                         k,
            "alpha":                     alpha,
            "beta":                      beta,
            "p3_multiplier":             p3_multiplier,
            "p3_absolute":               round(P3_eff, 6),
            "p3_base":                   round(P3_base, 6),
            "p1_absolute":               round(P1, 6),
            "p1_half":                   round(P1_half, 6),
            "feasible":                  is_feas,
            "total_weighted_tardiness":  obj_val,
            "n_violations":              n_viol_k,
            "n_new_violations_k":        n_new_k,
            "n_cuts_active":             len(all_cuts),
            "q_density_k":               q_dens_k,
            "q_density_delta":           round(q_dens_k - q_density_0, 6),
            "converged":                 converged,
            "oscillating":               oscillating,
            "best_energy":               best_energy,
            "wall_time_s":               round(wall_s, 3),
            "n_vars_qubo":               len(bqm.variables),
            "initial_terminal_stock_m3": inv_params["initial_terminal_stock_m3"],
            "tank_level":                tank_level,
            "run_timestamp":             datetime.datetime.now().isoformat(),
        })

        if converged:
            logger.info(
                "  SA %s level=%s p3_mult=%.0f run_id=%d convergió en k=%d",
                label, tank_level, p3_multiplier, run_id, k,
            )
            break  # F^(k) = ∅ — no hay cuts que acumular

        all_cuts |= new_cuts
        n_violations_prev = n_viol_k

    return rows


# Detectar runs SA ya completados.
# Clave: (instance_label, tank_level, p3_multiplier, run_id)
_existing = load_existing_runs(FILEPATH, SHEET)
if not _existing.empty and all(
    c in _existing.columns
    for c in ["instance_label", "tank_level", "p3_multiplier", "run_id"]
):
    _done_sa = set(zip(
        _existing["instance_label"],
        _existing["tank_level"].fillna("__missing__"),
        _existing["p3_multiplier"].astype(float),
        _existing["run_id"].astype(int),
    ))
else:
    _done_sa = set()

for entry in EXP5B_INSTANCES:
    label      = entry["label"]
    axis       = entry["axis"]
    inst       = entry["inst"]
    tank_level = entry["tank_level"]
    stock_m3   = EXP5B_TANK_LEVELS[tank_level]
    inv_params = {**EXP5B_INVENTORY_PARAMS, "initial_terminal_stock_m3": stock_m3}

    noms_check = inst["nominations"].copy()
    T_check    = int(inst["T"])
    vdf_check  = compute_feasible_slots(noms_check, horizon_slots=T_check)
    n_vars_chk = len(build_qubo(vdf_check, alpha=alpha_star, beta=beta_star)[0].variables)
    logger.info(
        "  Instancia %s level=%s (N=%d, T=%d, n_vars=%d)",
        label, tank_level, inst["N"], T_check, n_vars_chk,
    )

    for p3_mult in EXP5B_P3_MULTIPLIERS:
        for run_id in range(N_RUNS_SA_EXP5B):
            if (label, tank_level, float(p3_mult), run_id) in _done_sa:
                logger.info(
                    "  skip %s level=%s p3_mult=%.0f run_id=%d",
                    label, tank_level, p3_mult, run_id,
                )
                continue

            logger.info(
                "  SA %s level=%s p3_mult=%.0f run_id=%d ...",
                label, tank_level, p3_mult, run_id,
            )
            try:
                rows = _run_hybrid_loop_sa_p3(
                    label=label, axis=axis, inst=inst,
                    run_id=run_id, seed=EXP5B_SEED,
                    alpha=alpha_star, beta=beta_star,
                    p3_multiplier=float(p3_mult),
                    k_max=EXP5B_K_MAX,
                    num_reads=200, num_sweeps=1000,
                    run_uuid=RUN_UUID,
                    inv_params=inv_params,
                    tank_level=tank_level,
                )
                append_rows(FILEPATH, SHEET, rows)
                k_conv = next((r["k"] for r in rows if r["converged"]), None)
                logger.info(
                    "  done  k_conv=%s  iterations=%d",
                    k_conv, len(rows),
                )
            except Exception as exc:
                logger.error(
                    "  SA %s level=%s p3_mult=%.0f run_id=%d falló: %s",
                    label, tank_level, p3_mult, run_id, exc,
                )

logger.info("SA sweep P3 completo.")

# Guardar metadata
save_metadata(FILEPATH, {
    "alpha_star":    alpha_star,
    "beta_star":     beta_star,
    "k_max":         EXP5B_K_MAX,
    "n_sa_runs":     N_RUNS_SA_EXP5B,
    "p3_grid":       str(EXP5B_P3_MULTIPLIERS),
    "instances":     str([(e["label"], e["tank_level"]) for e in EXP5B_INSTANCES]),
    "run_uuid_last": RUN_UUID,
})


# CELDA 3: RESULTADOS — tabla de validación del fix de granularidad de cuts
#
# Pregunta: ¿el fix de cuts temporales restaura convergencia con P3×1 (el mismo
# P3 de Exp5)? Si la respuesta es sí, el problema era la granularidad de cuts,
# no el valor de P3.
#
# Tabla: instance | tank_level | run_id | k_conv | n_cuts_at_kconv | converged

df_raw = pd.read_excel(FILEPATH, sheet_name=SHEET)

if "tank_level" not in df_raw.columns:
    df_raw["tank_level"] = "unknown"
else:
    df_raw["tank_level"] = df_raw["tank_level"].fillna("unknown")


def _k_conv_run(g: pd.DataFrame) -> int | None:
    conv = g[g["n_violations"] == 0].sort_values("k")
    return int(conv.iloc[0]["k"]) if not conv.empty else None


result_rows = []
for (label, tank_level, p3_mult, run_id), grp in df_raw.groupby(
    ["instance_label", "tank_level", "p3_multiplier", "run_id"]
):
    grp_s = grp.sort_values("k")
    kc    = _k_conv_run(grp_s)

    if kc is not None:
        n_cuts = int(grp_s[grp_s["k"] == kc]["n_cuts_active"].iloc[0])
    else:
        # No convergió: mostrar n_cuts en la última iteración
        n_cuts = int(grp_s.iloc[-1]["n_cuts_active"])

    result_rows.append({
        "instance":        label,
        "tank_level":      tank_level,
        "run_id":          int(run_id),
        "k_conv":          kc if kc is not None else "-",
        "n_cuts_at_kconv": n_cuts,
        "converged":       "YES" if kc is not None else "NO",
    })

df_results = pd.DataFrame(result_rows).sort_values(
    ["instance", "tank_level", "run_id"]
)

# Resumen por instancia
_n_conv = sum(1 for r in result_rows if r["converged"] == "YES")
_n_total = len(result_rows)

print("\n══ Exp 5b — Validación del fix de granularidad de cuts (P3×1) ══")
print(f"   Output: {FILEPATH}")
print(f"   Runs que convergieron: {_n_conv}/{_n_total}\n")
print(df_results.to_string(index=False))

if _n_conv == _n_total:
    print("\n✓ FIX VALIDADO: todos los runs convergieron. El problema era la granularidad de cuts.")
    print("  Próximo paso: re-ejecutar sweep completo P3 = [1, 5, 10, 30, 100].")
elif _n_conv > 0:
    print(f"\n~ FIX PARCIAL: {_n_conv}/{_n_total} runs convergieron.")
    print("  Revisar qué instancias siguen fallando antes de ampliar el sweep.")
else:
    print("\n✗ FIX NO EFECTIVO: ningún run convergió con P3×1.")
    print("  El problema puede ser estructural (instancias sobre-restringidas) o el P3 sigue siendo insuficiente.")

logger.info("Exp 5b validación completa. %d/%d runs convergieron.", _n_conv, _n_total)


# CELDA 4: DIAGNÓSTICO DE FACTIBILIDAD ULLAGE CON GUROBI
#
# Pregunta: ¿existe solución ullage-feasible para cada instancia fallida?
# Gurobi resuelve el problema de factibilidad pura (MIP binario) con un
# límite de 30 s. Si la instancia es INFEASIBLE, calcula el IIS para
# identificar qué restricción hace imposible el problema.
#
# Modelo:
#   Variables: x[j, m, t] ∈ {0,1}  —  mismas que el QUBO (vía variables_df)
#   H_assign:  Σ_{m,t} x[j,m,t] = 1  para cada buque j
#   H_overlap: para cada par (j1,j2) y par de máquinas (m1,m2) en conflict_set,
#              las asignaciones solapadas no pueden coexistir (misma lógica que
#              qubo_builder._add_h_overlap, clamped bounds)
#   Ullage:    para cada slot t ∈ [0, T]:
#                stock(t) = init_stock + inflow_rate * (t / slots_per_day)
#                           - Σ_{j,m,t'} volume_j * x[j,m,t']  donde t' + p_j ≤ t
#                stock(t) ≤ safe_threshold
#   Objetivo:  minimizar 0  (factibilidad pura)
#
# Salida hoja "feasibility_gurobi" en exp05b_p3_sweep.xlsx:
#   instance_label, tank_level, gurobi_status, solve_time_s,
#   n_vessels_placed, min_ullage_margin_m3, note

import time as _time

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False
    logger.warning("gurobipy no disponible — instalar con: pip install gurobipy")

GUROBI_TIME_LIMIT_S  = 30
GUROBI_SHEET         = "feasibility_gurobi"
GUROBI_OUTPUT_PATH   = RESULTS_DIR / "exp05b_p3_sweep.xlsx"

# slots_per_day derivado de SLOT_HOURS (config.py: SLOT_HOURS=12 → 2 slots/día)
_SLOTS_PER_DAY = 24 / EXP5B_INVENTORY_PARAMS["slot_duration_hours"]


def _gurobi_feasibility(
    label: str,
    tank_level: str,
    inst: dict,
    inv_params: dict,
    time_limit_s: int = GUROBI_TIME_LIMIT_S,
) -> dict:
    """
    Resuelve la factibilidad ullage exacta con Gurobi para una instancia.

    Construye un MIP binario con las mismas variables que el QUBO y añade:
      - Restricciones de asignación única (H_assign)
      - Restricciones de no-overlap sobre conflict_set completo (H_overlap)
      - Restricciones de ullage para cada slot t del horizonte

    Retorna un dict con los campos de la hoja feasibility_gurobi.
    """
    from preprocessing import compute_feasible_slots
    from config import CONFLICT_SET_R

    noms = inst["nominations"].copy()
    T_h  = int(inst["T"])
    vdf  = compute_feasible_slots(noms, horizon_slots=T_h)

    init_stock    = float(inv_params["initial_terminal_stock_m3"])
    daily_inflow  = float(inv_params["daily_inflow_m3"])
    inflow_slot   = daily_inflow / _SLOTS_PER_DAY  # m³ por slot
    n_tanks       = int(inv_params["n_tanks"])
    cap_per_tank  = float(inv_params["tank_capacity_m3"])
    min_ullage    = int(inv_params["min_ullage_days"])
    safe_thresh   = n_tanks * cap_per_tank - min_ullage * daily_inflow

    vessels = vdf["vessel_id"].unique().tolist()
    machines = sorted(vdf["machine"].unique().tolist())

    # p_j: processing slots per vessel (columna 'p_j' en vdf, no 'processing_slots')
    vessel_pslots = {
        str(row["vessel_id"]): int(row["p_j"])
        for _, row in vdf.drop_duplicates("vessel_id").iterrows()
    }
    # volume_m3: viene de nominations, no de vdf
    vessel_volumes = {
        str(row["vessel_id"]): float(row["volume_m3"])
        for _, row in noms.iterrows()
        if str(row["vessel_id"]) in vessel_pslots
    }

    t0_wall = _time.perf_counter()

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
    model.setParam("TimeLimit", time_limit_s)
    model.setParam("OutputFlag", 0)

    # Variables x[j, m, t] ∈ {0,1} — solo slots factibles según vdf
    x = {}
    for _, row in vdf.iterrows():
        j = str(row["vessel_id"])
        m = int(row["machine"])
        t = int(row["slot"])
        x[(j, m, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{m}_{t}")

    model.update()

    # H_assign: cada buque debe ser asignado exactamente una vez
    for j in vessels:
        vars_j = [x[(jj, m, t)] for (jj, m, t) in x if jj == j]
        if vars_j:
            model.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")

    # H_overlap: no puede haber dos buques activos simultáneamente en ningún
    # par de máquinas del conflict_set (pipeline compartido — serialización total)
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
                        # solapamiento si [t1, t1+p1) ∩ [t2, t2+p2) ≠ ∅
                        lo = max(t1, t2)
                        hi = min(t1 + p1_slots, t2 + p2_slots)
                        if lo < hi:
                            model.addConstr(
                                x[(j1, m1, t1)] + x[(j2, m2, t2)] <= 1,
                                name=f"overlap_{j1}_{m1}_{t1}_{j2}_{m2}_{t2}",
                            )

    # Ullage: para cada slot t ∈ [0, T_h]:
    #   init_stock + inflow_slot * t
    #   - Σ_{j,m,t': t'+p_j ≤ t} volume_j * x[j,m,t'] ≤ safe_thresh
    for t_slot in range(T_h + 1):
        inflow_at_t = init_stock + inflow_slot * t_slot
        # buques completados en o antes de t_slot
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
            # sin buques completados aún: stock = init + inflow
            if inflow_at_t > safe_thresh:
                # restricción trivialmente violada independientemente de x —
                # se añade como ineq infactible para que Gurobi lo detecte
                model.addConstr(
                    gp.LinExpr() + inflow_at_t <= safe_thresh,
                    name=f"ullage_t{t_slot}_trivial",
                )

    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    wall_s = _time.perf_counter() - t0_wall
    status = model.Status

    result = {
        "instance_label":      label,
        "tank_level":          tank_level,
        "solve_time_s":        round(wall_s, 3),
        "n_vessels_placed":    None,
        "min_ullage_margin_m3": None,
        "note":                "",
    }

    if status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL:
        result["gurobi_status"] = "FEASIBLE"
        placed = 0
        for (j, m, t), var in x.items():
            if var.X > 0.5:
                placed += 1
        result["n_vessels_placed"] = placed

        # Calcular stock mínimo de ullage (margen = safe_thresh - stock_max)
        stocks = []
        sol = {key: var.X for key, var in x.items()}
        for t_slot in range(T_h + 1):
            stock_t = init_stock + inflow_slot * t_slot
            for (j, m, ts), val in sol.items():
                if val > 0.5 and ts + vessel_pslots[j] <= t_slot:
                    stock_t -= vessel_volumes[j]
            stocks.append(stock_t)
        stock_max = max(stocks)
        result["min_ullage_margin_m3"] = round(safe_thresh - stock_max, 1)
        result["n_vessels_placed"] = placed

    elif status == GRB.INFEASIBLE:
        result["gurobi_status"] = "INFEASIBLE"
        # Calcular IIS para identificar restricción conflictiva
        try:
            model.computeIIS()
            iis_constrs = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
            # Resumir: slot más temprano de ullage en IIS, si aplica
            ullage_iis = [c for c in iis_constrs if c.startswith("ullage_t")]
            if ullage_iis:
                slots_iis = sorted(int(c.split("ullage_t")[1].split("_")[0]) for c in ullage_iis)
                result["note"] = (
                    f"IIS: ullage violada desde slot t={slots_iis[0]} "
                    f"({len(ullage_iis)} restricciones de ullage en IIS)"
                )
            else:
                result["note"] = f"IIS: {iis_constrs[:3]}"
        except Exception as iis_exc:
            result["note"] = f"IIS no disponible: {iis_exc}"

    elif status == GRB.TIME_LIMIT:
        result["gurobi_status"] = "TIMEOUT"
        obj_bound = model.ObjBound if hasattr(model, "ObjBound") else float("nan")
        result["note"] = (
            f"Límite de {time_limit_s}s alcanzado. "
            f"Best bound: {obj_bound:.4f}"
        )
    else:
        result["gurobi_status"] = f"STATUS_{status}"
        result["note"] = "Estado Gurobi inesperado"

    model.dispose()
    env.dispose()
    return result


if not _GUROBI_AVAILABLE:
    print("gurobipy no instalado. Instalar con:  pip install gurobipy")
    print("(La licencia académica de Gurobi es gratuita para uso en tesis.)")
else:
    gurobi_rows = []
    for entry in EXP5B_INSTANCES:
        label      = entry["label"]
        tank_level = entry["tank_level"]
        inst       = entry["inst"]
        stock_m3   = EXP5B_TANK_LEVELS[tank_level]
        inv_params = {**EXP5B_INVENTORY_PARAMS, "initial_terminal_stock_m3": stock_m3}

        logger.info(
            "  Gurobi  %s / %s  (stock_init=%.0f m³, safe_thresh=%.0f m³) ...",
            label, tank_level, stock_m3,
            inv_params["n_tanks"] * inv_params["tank_capacity_m3"]
            - inv_params["min_ullage_days"] * inv_params["daily_inflow_m3"],
        )
        try:
            row = _gurobi_feasibility(
                label=label,
                tank_level=tank_level,
                inst=inst,
                inv_params=inv_params,
                time_limit_s=GUROBI_TIME_LIMIT_S,
            )
        except Exception as exc:
            logger.error("  Gurobi falló para %s/%s: %s", label, tank_level, exc)
            row = {
                "instance_label":       label,
                "tank_level":           tank_level,
                "gurobi_status":        "ERROR",
                "solve_time_s":         None,
                "n_vessels_placed":     None,
                "min_ullage_margin_m3": None,
                "note":                 str(exc),
            }
        gurobi_rows.append(row)
        logger.info(
            "  → %s  (%.2f s)  %s",
            row["gurobi_status"], row.get("solve_time_s") or 0.0, row.get("note", ""),
        )

    df_gurobi = pd.DataFrame(gurobi_rows)[
        [
            "instance_label", "tank_level", "gurobi_status",
            "solve_time_s", "n_vessels_placed", "min_ullage_margin_m3", "note",
        ]
    ]

    print("\n══ Diagnóstico de factibilidad ullage — Gurobi ══")
    print(f"   Time limit: {GUROBI_TIME_LIMIT_S} s por instancia\n")
    print(df_gurobi.to_string(index=False))

    # Guardar en hoja "feasibility_gurobi" de exp05b_p3_sweep.xlsx
    try:
        if GUROBI_OUTPUT_PATH.exists():
            with pd.ExcelWriter(
                GUROBI_OUTPUT_PATH,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:
                df_gurobi.to_excel(writer, sheet_name=GUROBI_SHEET, index=False)
        else:
            with pd.ExcelWriter(
                GUROBI_OUTPUT_PATH,
                engine="openpyxl",
                mode="w",
            ) as writer:
                df_gurobi.to_excel(writer, sheet_name=GUROBI_SHEET, index=False)
        logger.info(
            "Hoja '%s' guardada en %s", GUROBI_SHEET, GUROBI_OUTPUT_PATH
        )
    except Exception as exc:
        logger.error("No se pudo guardar la hoja Gurobi: %s", exc)
        print(f"\nERROR al guardar: {exc}")


# CELDA 5: REDISEÑO DE TANK_LEVEL + SA REDESIGNED
#
# Motivación: el diagnóstico Gurobi (Celda 4) confirmó que:
#   - "high" original = safe_thresh - 1×daily_inflow es infactible (IIS en t=3)
#   - Se necesita un margen mayor para que exista al menos una solución
#
# Paso A — Búsqueda del menor margen feasible para Size_1 y Size_2:
#   Iterar márgenes [3, 4, 5, 6, 7] días de inflow:
#     high_candidate = round(_SAFE_THRESHOLD - n_days * DAILY_INFLOW_M3)
#   Usar el menor n_days donde AMBAS instancias sean FEASIBLE según Gurobi.
#
# Paso B — Definir nivel "medium":
#   medium = round(_SAFE_THRESHOLD - 10 * DAILY_INFLOW_M3)
#   Verificar factibilidad. Si INFEASIBLE, aumentar en pasos de 1 día hasta feasible.
#
# Paso C — SA loop (P3×1, 5 runs, K_max=10, slot-level cuts):
#   Size_1/medium, Size_1/high_new, Size_2/medium, Size_2/high_new
#   Guardar en results/exp05b_redesigned.xlsx con columna margin_days.
#
# Paso D — Tabla final:
#   instance | tank_level | margin_days | tank_stock_m3 | run_id | k_conv | converged

REDESIGNED_PATH  = RESULTS_DIR / "exp05b_redesigned.xlsx"
REDESIGNED_SHEET = "per_iteration"


def _gurobi_feasibility_quick(label, inst, stock_m3, inv_params_base):
    """Versión ligera: solo retorna 'FEASIBLE' / 'INFEASIBLE' / 'TIMEOUT' / 'ERROR'."""
    inv_params = {**inv_params_base, "initial_terminal_stock_m3": float(stock_m3)}
    try:
        row = _gurobi_feasibility(
            label=label,
            tank_level="candidate",
            inst=inst,
            inv_params=inv_params,
            time_limit_s=30,
        )
        return row["gurobi_status"]
    except Exception as exc:
        logger.error("  _gurobi_feasibility_quick %s stock=%.0f: %s", label, stock_m3, exc)
        return "ERROR"


# ── Paso A: búsqueda del menor margen feasible para "high_new" ──────────────
print("\n══ Paso A — Búsqueda de margen mínimo feasible para high_new ══")
print(f"   _SAFE_THRESHOLD = {_SAFE_THRESHOLD:,.0f} m³  |  DAILY_INFLOW = {DAILY_INFLOW_M3:,.0f} m³/día\n")

_HIGH_MARGIN_DAYS_CANDIDATES = [3, 4, 5, 6, 7]
_high_new_margin_days = None
_high_new_stock       = None

for _n_days in _HIGH_MARGIN_DAYS_CANDIDATES:
    _candidate_stock = round(_SAFE_THRESHOLD - _n_days * DAILY_INFLOW_M3)
    _s1_status = _gurobi_feasibility_quick(
        "Size_1", _size_dict["Size_1"], _candidate_stock, EXP5B_INVENTORY_PARAMS
    )
    _s2_status = _gurobi_feasibility_quick(
        "Size_2", _size_dict["Size_2"], _candidate_stock, EXP5B_INVENTORY_PARAMS
    )
    _both_ok = (_s1_status == "FEASIBLE" and _s2_status == "FEASIBLE")
    print(
        f"  margen={_n_days}d  stock={_candidate_stock:,.0f} m³  "
        f"Size_1={_s1_status}  Size_2={_s2_status}  "
        f"{'← ELEGIDO' if _both_ok and _high_new_margin_days is None else ''}"
    )
    if _both_ok and _high_new_margin_days is None:
        _high_new_margin_days = _n_days
        _high_new_stock       = _candidate_stock

if _high_new_margin_days is None:
    print("\n  AVISO: ningún candidato resultó feasible para ambas instancias.")
    print("  Usando margen=7 días como fallback.")
    _high_new_margin_days = 7
    _high_new_stock       = round(_SAFE_THRESHOLD - 7 * DAILY_INFLOW_M3)

print(f"\n  → high_new: margen={_high_new_margin_days}d  stock={_high_new_stock:,.0f} m³")

# ── Paso B: nivel "medium" ───────────────────────────────────────────────────
print("\n══ Paso B — Verificación de nivel medium ══")
_MEDIUM_BASE_DAYS = 10
_medium_margin_days = _MEDIUM_BASE_DAYS
_medium_stock       = round(_SAFE_THRESHOLD - _medium_margin_days * DAILY_INFLOW_M3)

while True:
    _s1_med = _gurobi_feasibility_quick(
        "Size_1", _size_dict["Size_1"], _medium_stock, EXP5B_INVENTORY_PARAMS
    )
    _s2_med = _gurobi_feasibility_quick(
        "Size_2", _size_dict["Size_2"], _medium_stock, EXP5B_INVENTORY_PARAMS
    )
    _both_med = (_s1_med == "FEASIBLE" and _s2_med == "FEASIBLE")
    print(
        f"  margen={_medium_margin_days}d  stock={_medium_stock:,.0f} m³  "
        f"Size_1={_s1_med}  Size_2={_s2_med}  "
        f"{'← OK' if _both_med else '← ajustando...'}"
    )
    if _both_med:
        break
    _medium_margin_days += 1
    _medium_stock = round(_SAFE_THRESHOLD - _medium_margin_days * DAILY_INFLOW_M3)
    if _medium_margin_days > 20:
        print("  AVISO: medium no converge — usando margen=20d como fallback.")
        break

print(f"\n  → medium: margen={_medium_margin_days}d  stock={_medium_stock:,.0f} m³")

# Resumen de niveles validados
_REDESIGNED_LEVELS = {
    "medium":   {"stock": _medium_stock,   "margin_days": _medium_margin_days},
    "high_new": {"stock": _high_new_stock, "margin_days": _high_new_margin_days},
}
print("\n  Niveles validados:")
for _lvl, _info in _REDESIGNED_LEVELS.items():
    print(f"    {_lvl:10s}  stock={_info['stock']:,.0f} m³  margen={_info['margin_days']}d")

# ── Paso C: SA loop sobre los 4 combos validados ─────────────────────────────
print("\n══ Paso C — SA loop (P3×1, 5 runs, K_max=10) ══")

_REDESIGNED_INSTANCES = []
for _lvl, _info in _REDESIGNED_LEVELS.items():
    for _lbl in ["Size_1", "Size_2"]:
        _REDESIGNED_INSTANCES.append({
            "label":       _lbl,
            "axis":        "size",
            "inst":        _size_dict[_lbl],
            "tank_level":  _lvl,
            "stock_m3":    _info["stock"],
            "margin_days": _info["margin_days"],
        })

_existing_rd = load_existing_runs(REDESIGNED_PATH, REDESIGNED_SHEET)
if not _existing_rd.empty and all(
    c in _existing_rd.columns
    for c in ["instance_label", "tank_level", "p3_multiplier", "run_id"]
):
    _done_rd = set(zip(
        _existing_rd["instance_label"],
        _existing_rd["tank_level"],
        _existing_rd["p3_multiplier"].astype(float),
        _existing_rd["run_id"].astype(int),
    ))
else:
    _done_rd = set()

for _entry in _REDESIGNED_INSTANCES:
    _lbl       = _entry["label"]
    _axis      = _entry["axis"]
    _inst      = _entry["inst"]
    _tank_lvl  = _entry["tank_level"]
    _stock_m3  = _entry["stock_m3"]
    _marg_days = _entry["margin_days"]
    _inv_p     = {**EXP5B_INVENTORY_PARAMS, "initial_terminal_stock_m3": _stock_m3}

    for _run_id in range(N_RUNS_SA_EXP5B):
        if (_lbl, _tank_lvl, 1.0, _run_id) in _done_rd:
            logger.info("  skip %s/%s run_id=%d", _lbl, _tank_lvl, _run_id)
            continue

        logger.info("  SA %s/%s run_id=%d (stock=%.0f, margen=%dd) ...",
                    _lbl, _tank_lvl, _run_id, _stock_m3, _marg_days)
        try:
            _rows = _run_hybrid_loop_sa_p3(
                label=_lbl, axis=_axis, inst=_inst,
                run_id=_run_id, seed=EXP5B_SEED,
                alpha=alpha_star, beta=beta_star,
                p3_multiplier=1.0,
                k_max=EXP5B_K_MAX,
                num_reads=200, num_sweeps=1000,
                run_uuid=RUN_UUID,
                inv_params=_inv_p,
                tank_level=_tank_lvl,
            )
            # Añadir margin_days a cada fila
            for _r in _rows:
                _r["margin_days"]    = _marg_days
                _r["tank_stock_m3"]  = _stock_m3
            append_rows(REDESIGNED_PATH, REDESIGNED_SHEET, _rows)
            _kc = next((_r["k"] for _r in _rows if _r["converged"]), None)
            logger.info("  done  k_conv=%s", _kc)
        except Exception as _exc:
            logger.error("  SA %s/%s run_id=%d falló: %s", _lbl, _tank_lvl, _run_id, _exc)

logger.info("SA redesigned completo.")

save_metadata(REDESIGNED_PATH, {
    "alpha_star":        alpha_star,
    "beta_star":         beta_star,
    "k_max":             EXP5B_K_MAX,
    "n_sa_runs":         N_RUNS_SA_EXP5B,
    "high_new_days":     _high_new_margin_days,
    "high_new_stock_m3": _high_new_stock,
    "medium_days":       _medium_margin_days,
    "medium_stock_m3":   _medium_stock,
    "run_uuid_last":     RUN_UUID,
})

# ── Paso D: tabla final ───────────────────────────────────────────────────────
print("\n══ Paso D — Tabla final de convergencia ══\n")

_df_rd = pd.read_excel(REDESIGNED_PATH, sheet_name=REDESIGNED_SHEET)

_final_rows = []
for (_lbl, _tlvl, _p3m, _run_id), _grp in _df_rd.groupby(
    ["instance_label", "tank_level", "p3_multiplier", "run_id"]
):
    _grp_s = _grp.sort_values("k")
    _kc    = next(
        (int(r["k"]) for _, r in _grp_s.iterrows() if r["n_violations"] == 0),
        None,
    )
    _marg  = int(_grp_s["margin_days"].iloc[0]) if "margin_days" in _grp_s.columns else "-"
    _stock = int(_grp_s["tank_stock_m3"].iloc[0]) if "tank_stock_m3" in _grp_s.columns else "-"
    _final_rows.append({
        "instance":      _lbl,
        "tank_level":    _tlvl,
        "margin_days":   _marg,
        "tank_stock_m3": _stock,
        "run_id":        int(_run_id),
        "k_conv":        _kc if _kc is not None else "-",
        "converged":     "YES" if _kc is not None else "NO",
    })

_df_final = pd.DataFrame(_final_rows).sort_values(
    ["instance", "tank_level", "run_id"]
)
print(_df_final.to_string(index=False))

_n_conv_rd = sum(1 for r in _final_rows if r["converged"] == "YES")
_n_tot_rd  = len(_final_rows)
print(f"\n  Convergencia: {_n_conv_rd}/{_n_tot_rd} runs")
logger.info("Exp 5b redesigned completo. %d/%d runs convergieron.", _n_conv_rd, _n_tot_rd)
