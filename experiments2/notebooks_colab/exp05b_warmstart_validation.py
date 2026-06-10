"""
exp05b_warmstart_validation.py — Warm-start SA desde solución greedy (Dens_1/nominal)

Pregunta: ¿arrancando SA desde un schedule greedy válido converge el bucle iterativo
          donde sin warm-start falló?

Instancia: Dens_1, tank_level=nominal únicamente.
  - Es la única de las 4 instancias fallidas de Exp5 que tiene solución factible
    (greedy la encuentra; Size_1/high, Size_2/high y Dens_1/high son infactibles
    por construcción: el stock inicial supera safe_threshold antes de que termine
    el buque más rápido).

Método:
  1. Greedy (vol desc) construye un schedule inicial válido con n_violations=0.
  2. El schedule greedy se codifica como initial_state = {x_j_m_t: 1|0}.
  3. SA arranca desde ese estado en cada iteración k del loop (mismo BQM + cuts).
  4. 5 runs × K_max=10, P3×1, seed=0. Cada run usa el mismo estado greedy.

Output: results/exp05b_warmstart_validation.xlsx
  Mismas columnas que exp05b_fix_validation.xlsx + columna warm_start=True.

Prerequisito: Exp 2 completado (α*, β* en exp02_lagrange_calibration.xlsx).

Ejecución:
  Celda 0: INSTALL (solo una vez)
  Celda SETUP: configurar Drive y paths
  Celda 1: SETUP + cargar α*, β* + construir estado greedy
  Celda 2: RUN SA warm-start — loop iterativo Dens_1/nominal
  Celda 3: RESULTADOS — tabla + comparación con fix_validation
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


# CELDA 1: SETUP + cargar α*, β* + construir estado greedy

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
from preprocessing import compute_feasible_slots
from qubo_builder import build_qubo, calibrate_penalties
from solver import decode_schedule, check_feasibility
from inventory import check_worst_case_overlaps
from dwave.samplers import SimulatedAnnealingSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("exp05b_ws")

ensure_directories()

RUN_UUID  = new_run_uuid()
FILEPATH  = RESULTS_DIR / "exp05b_warmstart_validation.xlsx"
EXP2_PATH = RESULTS_DIR / "exp02_lagrange_calibration.xlsx"
SHEET     = "per_iteration"

logger.info("Exp 5b warm-start SETUP  run_uuid=%s", RUN_UUID)

# Cargar α*, β* de Exp 2
_meta2 = load_metadata(EXP2_PATH)
if not _meta2:
    raise FileNotFoundError(
        f"No se encontró metadata en {EXP2_PATH}. Ejecutar Exp 2 primero."
    )
alpha_star = float(_meta2["alpha_star"])
beta_star  = float(_meta2["beta_star"])
logger.info("α*=%.1f  β*=%.1f  (cargados de Exp 2)", alpha_star, beta_star)

# Parámetros físicos
_TANK_CAPACITY_TOTAL = N_TANKS * TANK_CAPACITY_M3          # 600_000 m³
_SAFE_THRESHOLD      = _TANK_CAPACITY_TOTAL - MIN_ULLAGE_DAYS * DAILY_INFLOW_M3  # 520_000 m³
_INFLOW_PER_SLOT     = DAILY_INFLOW_M3 / (24.0 / SLOT_HOURS)  # 10_000 m³/slot

STOCK_NOMINAL = INITIAL_TERMINAL_STOCK_M3  # 300_000 m³

WS_SEED     = 0
N_RUNS_WS   = 5
WS_K_MAX    = EXP5_K_MAX     # 10
WS_P3_MULT  = 1.0            # mismo P3 que Exp5 baseline

# Cargar Dens_1
def _ensure_volume_m3(noms: pd.DataFrame) -> pd.DataFrame:
    if "volume_m3" not in noms.columns:
        noms = noms.copy()
        noms["volume_m3"] = noms["stock_acumulado_m3"]
    return noms

try:
    _dens_dict = load_instances_from_excel("dens")
    _dens_key  = "Dens_1"
except (ValueError, KeyError):
    _dens_dict = load_instances_from_excel("congestion")
    _dens_key  = "Cong_1"

_dens_inst = _dens_dict[_dens_key]
_dens_inst["nominations"] = _ensure_volume_m3(_dens_inst["nominations"])

INST_LABEL     = "Dens_1"
INST_AXIS      = "dens"
INST_TANK_LEVEL = "nominal"
INST           = _dens_inst

INV_PARAMS = {
    "slot_duration_hours":       SLOT_HOURS,
    "min_ullage_days":           MIN_ULLAGE_DAYS,
    "initial_terminal_stock_m3": STOCK_NOMINAL,
    "n_tanks":                   N_TANKS,
    "tank_capacity_m3":          TANK_CAPACITY_M3,
    "daily_inflow_m3":           DAILY_INFLOW_M3,
}

logger.info(
    "Instancia: %s  tank_level=%s  stock=%.0f  seed=%d  K_max=%d  N_runs=%d",
    INST_LABEL, INST_TANK_LEVEL, STOCK_NOMINAL, WS_SEED, WS_K_MAX, N_RUNS_WS,
)

# ── Construir estado greedy ──────────────────────────────────────────────────
#
# Greedy: ordena buques por volume_m3 desc, asigna cada uno al primer slot
# donde stock <= safe_threshold en todo t ∈ [start, start+p_j].
# El pipeline es de un solo canal efectivo (shared pipeline) — no hay
# paralelismo real, así que el greedy verifica overlaps secuenciales.
#
# El estado greedy se convierte en initial_state = {x_j_m_t: 1|0}
# usando el mismo naming que build_qubo: x_{vessel_id}_{machine}_{slot}.
# Las variables asignadas se ponen a 1; el resto a 0.

def _build_greedy_schedule(
    noms: pd.DataFrame,
    horizon_slots: int,
    inflow_per_slot: float,
    initial_stock: float,
    safe_threshold: float,
) -> list[dict] | None:
    """
    Construye un schedule greedy válido ordenando por volume_m3 desc.

    Devuelve lista de dicts con vessel_id, machine, start_slot, p_j, volume_m3,
    o None si no encuentra solución factible.

    Restricciones modeladas:
    - stock(t) <= safe_threshold para todo t en [start, start+p_j]
    - Un solo slot activo por vez (pipeline serializado): no overlaps
    """
    vessels = sorted(
        [
            {
                "vessel_id": str(row["vessel_id"]),
                "volume_m3": float(row["volume_m3"]),
                "p_j":       int(row["p_j"]),
            }
            for _, row in noms.iterrows()
        ],
        key=lambda v: v["volume_m3"],
        reverse=True,
    )

    schedule: list[dict] = []

    def stock_at(t: int) -> float:
        s = initial_stock + inflow_per_slot * t
        for item in schedule:
            if item["start_slot"] + item["p_j"] <= t:
                s -= item["volume_m3"]
        return s

    def slot_blocked(t: int, p_j: int) -> bool:
        """True si el slot [t, t+p_j) se solapa con algún buque ya asignado."""
        for item in schedule:
            s0, s1 = item["start_slot"], item["start_slot"] + item["p_j"]
            if not (t + p_j <= s0 or t >= s1):
                return True
        return False

    for v in vessels:
        vid, pj, vol = v["vessel_id"], v["p_j"], v["volume_m3"]
        placed = False
        for start in range(0, horizon_slots - pj + 1):
            if slot_blocked(start, pj):
                continue
            ok = all(
                stock_at(t) <= safe_threshold + 0.5
                for t in range(start, start + pj + 1)
            )
            if ok:
                # Asignar a monobuoy 1 por defecto (pipeline serializado —
                # ambas monobuoys comparten pipeline, da igual cuál)
                schedule.append({
                    "vessel_id":  vid,
                    "machine":    1,
                    "start_slot": start,
                    "p_j":        pj,
                    "volume_m3":  vol,
                })
                placed = True
                break
        if not placed:
            return None

    return schedule


def _greedy_to_initial_state(
    greedy_schedule: list[dict],
    vdf: pd.DataFrame,
) -> dict[str, int]:
    """
    Convierte un schedule greedy en un initial_state para SA.

    Construye {x_{vessel_id}_{machine}_{slot}: 1|0} para todas las variables
    en vdf. Las variables que corresponden a una asignación del greedy se ponen
    a 1; el resto a 0.

    El naming x_{vessel_id}_{machine}_{slot} debe coincidir con el que genera
    build_qubo. Si hay desajuste, initial_state tendrá todos 0 para esa variable
    (SA arrancaría igual que sin warm-start para esas vars).
    """
    assigned: set[tuple[str, int, int]] = {
        (item["vessel_id"], item["machine"], item["start_slot"])
        for item in greedy_schedule
    }

    initial_state: dict[str, int] = {}
    for _, row in vdf.iterrows():
        var = f"x_{row['vessel_id']}_{int(row['machine'])}_{int(row['slot'])}"
        key = (str(row["vessel_id"]), int(row["machine"]), int(row["slot"]))
        initial_state[var] = 1 if key in assigned else 0

    assigned_in_vdf = sum(1 for v in initial_state.values() if v == 1)
    logger.info(
        "Greedy → initial_state: %d variables totales, %d asignadas a 1",
        len(initial_state), assigned_in_vdf,
    )
    if assigned_in_vdf != len(greedy_schedule):
        logger.warning(
            "Solo %d/%d asignaciones greedy encontradas en vdf — "
            "verificar que los start_slots greedy están en el espacio factible.",
            assigned_in_vdf, len(greedy_schedule),
        )

    return initial_state


# Construir el estado greedy una sola vez (igual para todos los runs)
_noms   = INST["nominations"].copy()
_T      = int(INST["T"])
_vdf    = compute_feasible_slots(_noms, horizon_slots=_T)

_greedy = _build_greedy_schedule(
    noms=_noms,
    horizon_slots=_T,
    inflow_per_slot=_INFLOW_PER_SLOT,
    initial_stock=STOCK_NOMINAL,
    safe_threshold=_SAFE_THRESHOLD,
)

if _greedy is None:
    raise RuntimeError(
        "Greedy no encontró solución factible para Dens_1/nominal. "
        "Revisar parámetros físicos."
    )

logger.info("Greedy encontró schedule válido con %d buques:", len(_greedy))
for item in _greedy:
    logger.info(
        "  %s: machine=%d  slot %d–%d  vol=%.0f m³",
        item["vessel_id"], item["machine"],
        item["start_slot"], item["start_slot"] + item["p_j"],
        item["volume_m3"],
    )

GREEDY_INITIAL_STATE = _greedy_to_initial_state(_greedy, _vdf)

# Verificar con check_worst_case_overlaps que el greedy es realmente válido
_greedy_df = pd.DataFrame(_greedy).rename(columns={"start_slot": "start_slot"})
# decode_schedule devuelve un df con vessel_id, machine, start_slot, p_j
# Construimos uno compatible directamente
_greedy_sched_df = pd.DataFrame({
    "vessel_id":  [i["vessel_id"]  for i in _greedy],
    "machine":    [i["machine"]    for i in _greedy],
    "start_slot": [i["start_slot"] for i in _greedy],
    "p_j":        [i["p_j"]        for i in _greedy],
})
_greedy_viol = check_worst_case_overlaps(_greedy_sched_df, _vdf, _noms, **INV_PARAMS)
logger.info(
    "Violaciones ullage del schedule greedy: %d  (esperado: 0)",
    len(_greedy_viol),
)
if _greedy_viol:
    logger.warning(
        "El schedule greedy tiene violaciones — el warm-start no arranca desde "
        "un estado válido. Revisar la lógica del greedy."
    )


# CELDA 2: RUN SA warm-start — loop iterativo Dens_1/nominal

def _q_density(bqm) -> float:
    n = len(bqm.variables)
    return round(len(bqm.quadratic) / (n * (n - 1) / 2), 6) if n > 1 else 0.0


def _run_hybrid_loop_sa_warmstart(
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
    initial_state: dict[str, int],
) -> list[dict]:
    """
    Bucle iterativo SA con warm-start desde initial_state en cada iteración k.

    El initial_state se pasa como initial_states=[initial_state] a SA.sample().
    SA parte desde ese estado y puede alejarse de él — es un punto de inicio,
    no una restricción dura. En k>1, el mismo estado greedy se reutiliza como
    punto de partida (el warm-start no acumula el estado de la iteración anterior,
    ya que la hipótesis es que el greedy conoce la región factible y SA debería
    explorar desde allí en cada iteración con el BQM actualizado con cuts).

    Todas las demás decisiones son idénticas a _run_hybrid_loop_sa_p3 de exp05b.
    """
    noms  = inst["nominations"].copy()
    T     = int(inst["T"])
    N     = int(inst["N"])
    rho   = float(inst["rho_effective"])
    rdist = str(inst["r_j_distribution"])

    vdf = compute_feasible_slots(noms, horizon_slots=T)

    P1, P2, P3_base = calibrate_penalties(vdf, alpha=alpha, beta=beta)
    P3_eff  = P3_base * p3_multiplier
    P1_half = P1 / 2.0
    beta_range = (1.0 / (P1 * 2.0), 10.0)

    if P3_eff > P1_half:
        logger.warning(
            "  WARNING: P3_eff (%.4f) > P1/2 (%.4f) para %s level=%s — "
            "cortes pueden interferir con H_assign.",
            P3_eff, P1_half, label, tank_level,
        )

    # Filtrar initial_state a las variables que existen en este BQM concreto.
    # Se reconstruye vdf aquí, así que las variables coinciden con las del BQM.
    bqm_vars_check, _, _, _, _ = build_qubo(vdf, alpha=alpha, beta=beta, cuts=None)
    bqm_var_set = set(bqm_vars_check.variables)
    filtered_state = {k: v for k, v in initial_state.items() if k in bqm_var_set}
    n_matched = sum(1 for v in filtered_state.values() if v == 1)
    logger.info(
        "  run_id=%d: initial_state filtrado — %d/%d vars en BQM, %d asignadas a 1",
        run_id, len(filtered_state), len(initial_state), n_matched,
    )

    sampler           = SimulatedAnnealingSampler()
    all_cuts          = set()
    q_density_0       = None
    rows: list[dict]  = []
    n_violations_prev = None

    for k in range(1, k_max + 1):
        bqm, _, _, _, _ = build_qubo(vdf, alpha=alpha, beta=beta, cuts=None)

        if all_cuts:
            applied = 0
            for (vessel_id, machine, slot) in all_cuts:
                var = f"x_{vessel_id}_{machine}_{slot}"
                if var in bqm.variables:
                    bqm.add_variable(var, P3_eff)
                    applied += 1
            if applied == 0:
                logger.error(
                    "  NINGÚN cut aplicado al BQM — verificar naming. "
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
            initial_states=[filtered_state],
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
            "exp_id":                    "exp05b_ws",
            "run_uuid":                  run_uuid,
            "solver":                    "SA",
            "warm_start":                True,
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
            "oscillating":              oscillating,
            "best_energy":               best_energy,
            "wall_time_s":               round(wall_s, 3),
            "n_vars_qubo":               len(bqm.variables),
            "initial_terminal_stock_m3": inv_params["initial_terminal_stock_m3"],
            "tank_level":                tank_level,
            "run_timestamp":             datetime.datetime.now().isoformat(),
        })

        all_cuts |= new_cuts
        n_violations_prev = n_viol_k

        if converged:
            logger.info(
                "  SA_ws %s level=%s run_id=%d convergió en k=%d",
                label, tank_level, run_id, k,
            )

    return rows


# Append-safe: clave (instance_label, tank_level, run_id)
_existing = load_existing_runs(FILEPATH, SHEET)
if not _existing.empty and all(
    c in _existing.columns for c in ["instance_label", "tank_level", "run_id"]
):
    _done = set(zip(
        _existing["instance_label"],
        _existing["tank_level"].fillna("__missing__"),
        _existing["run_id"].astype(int),
    ))
else:
    _done = set()

stock_m3 = STOCK_NOMINAL
logger.info(
    "SA warm-start %s level=%s  N_runs=%d  K_max=%d  P3_mult=%.0f",
    INST_LABEL, INST_TANK_LEVEL, N_RUNS_WS, WS_K_MAX, WS_P3_MULT,
)

for run_id in range(N_RUNS_WS):
    if (INST_LABEL, INST_TANK_LEVEL, run_id) in _done:
        logger.info("  skip run_id=%d", run_id)
        continue

    logger.info("  SA_ws run_id=%d ...", run_id)
    try:
        rows = _run_hybrid_loop_sa_warmstart(
            label=INST_LABEL,
            axis=INST_AXIS,
            inst=INST,
            run_id=run_id,
            seed=WS_SEED,
            alpha=alpha_star,
            beta=beta_star,
            p3_multiplier=WS_P3_MULT,
            k_max=WS_K_MAX,
            num_reads=200,
            num_sweeps=1000,
            run_uuid=RUN_UUID,
            inv_params=INV_PARAMS,
            tank_level=INST_TANK_LEVEL,
            initial_state=GREEDY_INITIAL_STATE,
        )
        append_rows(FILEPATH, SHEET, rows)
        k_conv = next((r["k"] for r in rows if r["converged"]), None)
        logger.info("  done  k_conv=%s  iterations=%d", k_conv, len(rows))
    except Exception as exc:
        logger.error("  run_id=%d falló: %s", run_id, exc)

logger.info("SA warm-start sweep completo.")

save_metadata(FILEPATH, {
    "alpha_star":    alpha_star,
    "beta_star":     beta_star,
    "k_max":         WS_K_MAX,
    "n_sa_runs":     N_RUNS_WS,
    "p3_multiplier": WS_P3_MULT,
    "instance":      f"{INST_LABEL}/{INST_TANK_LEVEL}",
    "warm_start":    "greedy_vol_desc",
    "run_uuid_last": RUN_UUID,
})


# CELDA 3: RESULTADOS — tabla warm-start + comparación con fix_validation

df_ws = pd.read_excel(FILEPATH, sheet_name=SHEET)

if "tank_level" not in df_ws.columns:
    df_ws["tank_level"] = "unknown"
else:
    df_ws["tank_level"] = df_ws["tank_level"].fillna("unknown")


def _k_conv_run(g: pd.DataFrame) -> int | None:
    conv = g[g["n_violations"] == 0].sort_values("k")
    return int(conv.iloc[0]["k"]) if not conv.empty else None


# Tabla warm-start
ws_rows = []
for (label, tank_level, run_id), grp in df_ws.groupby(
    ["instance_label", "tank_level", "run_id"]
):
    grp_s  = grp.sort_values("k")
    kc     = _k_conv_run(grp_s)
    n_cuts = int(grp_s[grp_s["k"] == kc]["n_cuts_active"].iloc[0]) if kc is not None \
             else int(grp_s.iloc[-1]["n_cuts_active"])
    ws_rows.append({
        "instance":        label,
        "tank_level":      tank_level,
        "run_id":          int(run_id),
        "k_conv":          kc if kc is not None else "-",
        "n_cuts_at_kconv": n_cuts,
        "converged":       "YES" if kc is not None else "NO",
        "warm_start":      "YES",
    })

df_ws_table = pd.DataFrame(ws_rows).sort_values(["instance", "tank_level", "run_id"])
n_conv_ws   = sum(1 for r in ws_rows if r["converged"] == "YES")

print("\n══ Exp 5b Warm-Start — Resultados (Dens_1/nominal) ══")
print(f"   Output: {FILEPATH}")
print(f"   Runs convergidos: {n_conv_ws}/{N_RUNS_WS}\n")
print(df_ws_table.to_string(index=False))

# Comparación con fix_validation (sin warm-start)
FIX_PATH = RESULTS_DIR / "exp05b_fix_validation.xlsx"
if FIX_PATH.exists():
    try:
        df_fix = pd.read_excel(FIX_PATH, sheet_name=SHEET)
        df_fix = df_fix[
            (df_fix["instance_label"] == INST_LABEL)
            & (df_fix["tank_level"].fillna("") == INST_TANK_LEVEL)
        ]
        fix_rows = []
        for (label, tank_level, run_id), grp in df_fix.groupby(
            ["instance_label", "tank_level", "run_id"]
        ):
            grp_s  = grp.sort_values("k")
            kc     = _k_conv_run(grp_s)
            n_cuts = int(grp_s[grp_s["k"] == kc]["n_cuts_active"].iloc[0]) if kc is not None \
                     else int(grp_s.iloc[-1]["n_cuts_active"])
            fix_rows.append({
                "instance":        label,
                "tank_level":      tank_level,
                "run_id":          int(run_id),
                "k_conv":          kc if kc is not None else "-",
                "n_cuts_at_kconv": n_cuts,
                "converged":       "YES" if kc is not None else "NO",
                "warm_start":      "NO",
            })
        df_fix_table = pd.DataFrame(fix_rows).sort_values("run_id")
        n_conv_fix   = sum(1 for r in fix_rows if r["converged"] == "YES")

        print(f"\n── Comparación: sin warm-start (fix_validation) ──")
        print(f"   Runs convergidos sin warm-start: {n_conv_fix}/{len(fix_rows)}\n")
        print(df_fix_table.to_string(index=False))

        print(f"\n── Resumen ──")
        print(f"   Sin warm-start:  {n_conv_fix}/{len(fix_rows)} convergidos")
        print(f"   Con warm-start:  {n_conv_ws}/{N_RUNS_WS} convergidos")
        if n_conv_ws > n_conv_fix:
            print("   MEJORA: el warm-start greedy aumenta la tasa de convergencia.")
        elif n_conv_ws == n_conv_fix and n_conv_ws == N_RUNS_WS:
            print("   EMPATE TOTAL: ambos convergen en todos los runs.")
        elif n_conv_ws == n_conv_fix:
            print("   SIN DIFERENCIA: warm-start no mejora la convergencia.")
        else:
            print("   REGRESIÓN: warm-start convergió menos que sin warm-start.")
            print("   Posible causa: el estado greedy orienta SA hacia una región")
            print("   que no es el óptimo QUBO, y el annealing no escapa de allí.")
    except Exception as e:
        print(f"\n[WARN] No se pudo cargar fix_validation para comparar: {e}")
else:
    print(f"\n[INFO] {FIX_PATH} no encontrado — ejecutar exp05b_fix_validation primero para comparar.")

logger.info("Exp 5b warm-start completo. %d/%d runs convergieron.", n_conv_ws, N_RUNS_WS)
