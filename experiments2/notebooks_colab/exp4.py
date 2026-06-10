"""
exp04_scaling_crossover.py — Scaling & Crossover Analysis

Post-procesamiento puro de tiempos de Gurobi (Exp 1) y LeapHybrid (Exp 3).
Ajusta dos modelos de regresión y proyecta el crossover N** donde LH iguala a Gurobi.

PREREQUISITO: Exp 1 y Exp 3 completos.
QPU: 0 (post-procesamiento puro).

Celdas:
  1: SETUP — imports, paths, logging
  2: LOAD tiempos Gurobi de Exp 1
  3: LOAD tiempos LeapHybrid de Exp 3
  4: FIT modelo log-lineal y log-log por eje
  5: CALCULAR N** proyectado + IC 95%
  6: SAVE resultados → results/exp04_crossover.xlsx hoja computed
  7: PLOT curva de scaling (figura central del paper)
"""

# CELDA 0: INSTALL — ejecutar una sola vez por sesión de Colab
# Exp 4 es post-procesamiento puro: no requiere gurobipy ni dwave
# %pip install -q scipy openpyxl seaborn matplotlib pandas numpy

# CELDA 1: SETUP (Colab)
# ---- EDITAR SI TU CARPETA TIENE OTRO NOMBRE ---
DRIVE_TESIS_PATH = "MyDrive/TESIS"
# -----------------------------------------------

import os, sys, logging, datetime, subprocess
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

print("  ok  paths configurados")
print(f"  REPO_ROOT: {REPO_ROOT}")

# =============================================================================
# CELDA 1 — SETUP (imports principales)
# =============================================================================

import sys
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- path setup ---
# Paths ya configurados en CELDA 1 (SETUP Colab)
# REPO_ROOT y EXPERIMENTS2_DIR están definidos en la celda anterior

from experiments2.shared.io_utils import (
    load_existing_runs,
    append_rows,
    save_metadata,
    load_metadata,
    RESULTS_DIR,
)
from experiments2.shared.run_id import new_run_uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exp04")

# --- file paths ---
EXP01_FILE = RESULTS_DIR / "exp01_gurobi_baseline.xlsx"
EXP03_FILE = RESULTS_DIR / "exp03_solution_quality.xlsx"
EXP04_FILE = RESULTS_DIR / "exp04_crossover.xlsx"
EXP07_FILE = RESULTS_DIR / "exp07_gurobi_scalability.xlsx"

AXES = ["size", "dens", "slack"]

RUN_UUID = new_run_uuid()
logger.info("exp04 run_uuid=%s", RUN_UUID)


# =============================================================================
# CELDA 2 — LOAD tiempos Gurobi
#   load_gurobi_times_exp7() → fuente primaria (13 puntos N=8..200, 5 seeds)
#   load_gurobi_times()      → fuente de validación (puntos de Exp 1 para overlay)
# =============================================================================

def load_gurobi_times_exp7() -> pd.DataFrame:
    """
    Carga raw_runs de Exp 7 (barrido denso N=8..200, 13 puntos, 5 seeds).
    Fuente primaria para el ajuste de regresión de Gurobi en Exp 4.
    Si Exp 7 no existe, devuelve DataFrame vacío y la regresión cae back a Exp 1.
    """
    if not EXP07_FILE.exists():
        logger.warning(
            "Exp 7 Excel no encontrado: %s — regresión usará Exp 1 como fallback.",
            EXP07_FILE,
        )
        return pd.DataFrame()

    try:
        df = load_existing_runs(EXP07_FILE, "raw_runs")
    except Exception as exc:
        logger.warning("No se pudo leer Exp 7: %s — fallback a Exp 1.", exc)
        return pd.DataFrame()

    if df.empty:
        logger.warning("Exp 7 raw_runs vacío — fallback a Exp 1.")
        return pd.DataFrame()

    # Exp 7 usa el eje size por construcción (barrido de N fijo)
    if "axis" not in df.columns:
        df["axis"] = "size"
    df["axis"] = df["axis"].str.lower()

    # Asegurar que tiene las columnas esperadas por _aggregate_gurobi
    needed = {"axis", "instance_label", "N", "n_vars_qubo", "wall_time_s"}
    missing = needed - set(df.columns)
    if missing:
        # Exp 7 puede tener n_milp_vars en lugar de n_vars_qubo
        if "n_milp_vars" in df.columns and "n_vars_qubo" not in df.columns:
            df["n_vars_qubo"] = df["n_milp_vars"]
        still_missing = needed - set(df.columns)
        if still_missing:
            logger.warning("Exp 7 faltan columnas %s — fallback a Exp 1.", still_missing)
            return pd.DataFrame()

    logger.info(
        "Gurobi Exp 7 data cargada (regresión primaria): %d runs, N=%s",
        len(df), sorted(df["N"].unique()),
    )
    return df


def load_gurobi_times() -> pd.DataFrame:
    """
    Carga raw_runs de Exp 1.
    Devuelve un DataFrame con columnas relevantes para el ajuste de regresión.
    Filtra solo runs Gurobi no TimeLimit (status Optimal) para el ajuste principal;
    incluye TimeLimit runs con wall_time_s como dato de degradación.
    """
    if not EXP01_FILE.exists():
        raise FileNotFoundError(
            f"Exp 1 Excel no encontrado: {EXP01_FILE}\n"
            "Ejecutar exp01_gurobi_baseline.py primero."
        )

    df = load_existing_runs(EXP01_FILE, "raw_runs")
    if df.empty:
        raise ValueError("Exp 1 raw_runs está vacío.")

    needed = {"axis", "instance_label", "N", "T", "n_vars_qubo",
              "rho_effective", "gurobi_status", "wall_time_s", "seed"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en Exp 1: {missing}")

    df["axis"] = df["axis"].str.lower()
    logger.info(
        "Gurobi data cargada: %d runs, instancias=%s",
        len(df), sorted(df["instance_label"].unique()),
    )
    return df


# =============================================================================
# CELDA 3 — LOAD tiempos LeapHybrid (Exp 3)
# =============================================================================

def load_gurobi_qubo_times() -> pd.DataFrame:
    """
    Carga tiempos Gurobi-QUBO (BQP) de la hoja milp_qubo_equiv de Exp 1.
    Devuelve DataFrame con qubo_wall_time_s, instance_label, axis, qubo_n_vars.
    """
    if not EXP01_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(EXP01_FILE, sheet_name="milp_qubo_equiv")
    except ValueError:
        logger.warning("Hoja milp_qubo_equiv no encontrada en Exp 1 — se omite Gurobi-QUBO.")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df["axis"] = df["axis"].str.lower()
    logger.info(
        "Gurobi-QUBO data cargada: %d instancias, instances=%s",
        len(df), sorted(df["instance_label"].unique()),
    )
    return df


def load_lh_times() -> pd.DataFrame:
    """
    Carga las tres hojas de Exp 3 y filtra solo filas LeapHybrid.
    Devuelve DataFrame con lh_time_s y n_vars por eje/instancia.
    """
    if not EXP03_FILE.exists():
        logger.warning(
            "Exp 3 Excel no encontrado: %s — se usará t_LH=NaN (solo regresión Gurobi).",
            EXP03_FILE,
        )
        return pd.DataFrame()

    frames = []
    for sheet in ("size_axis", "dens_axis", "slack_axis"):
        try:
            df_s = pd.read_excel(EXP03_FILE, sheet_name=sheet)
            if df_s.empty:
                continue
            axis_name = sheet.replace("_axis", "")
            df_s["axis"] = axis_name
            frames.append(df_s)
        except ValueError:
            logger.warning("Hoja %s no encontrada en Exp 3.", sheet)

    if not frames:
        logger.warning("Exp 3 sin datos — se usará t_LH=NaN.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    lh = df[df["solver"] == "LeapHybrid"].copy()

    if lh.empty:
        logger.warning("Ninguna fila LeapHybrid en Exp 3 — se usará t_LH=NaN.")
        return pd.DataFrame()

    logger.info(
        "LeapHybrid data cargada: %d runs, instancias=%s",
        len(lh), sorted(lh["instance_label"].unique()),
    )
    return lh


# =============================================================================
# CELDA 4 — FIT modelos de regresión
# =============================================================================

def _aggregate_gurobi(df_g: pd.DataFrame, axis: str) -> pd.DataFrame:
    """
    Agrega wall_time_s de Gurobi por instancia (media sobre seeds).
    Usa N (número de buques) como variable independiente para el ajuste.
    Incluye todas las instancias del eje, independientemente del status.
    """
    sub = df_g[df_g["axis"] == axis].copy()
    if sub.empty:
        return pd.DataFrame()

    agg = (
        sub.groupby(["instance_label", "N", "n_vars_qubo"], as_index=False)
        .agg(
            wall_time_mean=("wall_time_s", "mean"),
            wall_time_std=("wall_time_s", "std"),
            n_seeds=("wall_time_s", "count"),
        )
    )
    agg = agg.sort_values("N").reset_index(drop=True)
    return agg


def _fit_log_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Ajuste log-lineal: log(t) = a*N + b.
    x = N (valores del eje X), y = wall_time_s.
    """
    log_y = np.log(y)
    result = stats.linregress(x, log_y)
    se_a = result.stderr
    t_crit = stats.t.ppf(0.975, df=len(x) - 2)
    ci_low  = result.slope - t_crit * se_a
    ci_high = result.slope + t_crit * se_a
    return {
        "model_type":   "log_linear",
        "slope_a":      result.slope,
        "intercept_b":  result.intercept,
        "r_squared":    result.rvalue ** 2,
        "slope_ci_95":  f"[{ci_low:.6f}, {ci_high:.6f}]",
        "slope_ci_low":  ci_low,
        "slope_ci_high": ci_high,
        "stderr_a":     se_a,
        "n_points_fit": len(x),
    }


def _fit_log_log(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Ajuste log-log: log(t) = a*log(N) + b.
    x = N, y = wall_time_s.
    """
    log_x = np.log(x)
    log_y = np.log(y)
    result = stats.linregress(log_x, log_y)
    se_a = result.stderr
    t_crit = stats.t.ppf(0.975, df=len(x) - 2)
    ci_low  = result.slope - t_crit * se_a
    ci_high = result.slope + t_crit * se_a
    return {
        "model_type":   "log_log",
        "slope_a":      result.slope,
        "intercept_b":  result.intercept,
        "r_squared":    result.rvalue ** 2,
        "slope_ci_95":  f"[{ci_low:.6f}, {ci_high:.6f}]",
        "slope_ci_low":  ci_low,
        "slope_ci_high": ci_high,
        "stderr_a":     se_a,
        "n_points_fit": len(x),
    }


def fit_regressions(
    df_g: pd.DataFrame,
    df_lh: pd.DataFrame,
) -> list[dict]:
    """
    Ajusta log-lineal y log-log por eje. Calcula N** proyectado.
    Devuelve lista de dicts para la hoja 'computed'.
    """
    rows = []

    for axis in AXES:
        agg = _aggregate_gurobi(df_g, axis)
        if agg.empty or len(agg) < 2:
            logger.warning("Eje '%s': insuficientes puntos Gurobi (%d) — omitiendo.", axis, len(agg))
            continue

        x = agg["N"].values.astype(float)
        y = agg["wall_time_mean"].values.astype(float)

        # Sanidad: tiempos positivos para el log
        mask = y > 0
        if mask.sum() < 2:
            logger.warning("Eje '%s': <2 tiempos positivos — omitiendo.", axis)
            continue
        x_fit, y_fit = x[mask], y[mask]

        # Tiempo medio LeapHybrid para este eje
        t_lh_mean = np.nan
        t_lh_std  = np.nan
        if not df_lh.empty:
            lh_ax = df_lh[df_lh["axis"] == axis]
            if not lh_ax.empty and "lh_time_s" in lh_ax.columns:
                lh_times = lh_ax["lh_time_s"].dropna()
                if not lh_times.empty:
                    t_lh_mean = float(lh_times.mean())
                    t_lh_std  = float(lh_times.std())

        x_constant = np.all(x_fit == x_fit[0])
        for fit_fn, label in [(_fit_log_linear, "log_linear"), (_fit_log_log, "log_log")]:
            if x_constant:
                logger.warning("Eje '%s': N constante (%g) — %s no aplicable.", axis, x_fit[0], label)
                continue
            fit = fit_fn(x_fit, y_fit)

            # Proyección N**
            nstar = _project_nstar(fit, t_lh_mean)
            nstar_ci = _project_nstar_ci(fit, t_lh_mean)
            crossover_observed = _is_crossover_observed(nstar, x_fit) if not np.isnan(nstar) else False

            row = {
                "run_uuid":            RUN_UUID,
                "axis":                axis,
                "model_type":          fit["model_type"],
                "r_squared":           round(fit["r_squared"], 6),
                "slope_a":             round(fit["slope_a"], 8),
                "intercept_b":         round(fit["intercept_b"], 6),
                "slope_ci_95":         fit["slope_ci_95"],
                "t_LH_mean_s":         round(t_lh_mean, 4) if not np.isnan(t_lh_mean) else None,
                "t_LH_std_s":          round(t_lh_std, 4)  if not np.isnan(t_lh_std)  else None,
                "nstar_crossover":     round(nstar, 2) if not np.isnan(nstar) else None,
                "nstar_crossover_ci":  nstar_ci,
                "crossover_observed":  crossover_observed,
                "n_points_fit":        fit["n_points_fit"],
                "run_timestamp":       datetime.datetime.now().isoformat(),
            }
            rows.append(row)
            logger.info(
                "Eje %s | %s: R²=%.4f  a=%.5f  N**=%s  observado=%s",
                axis, label, fit["r_squared"], fit["slope_a"],
                f"{nstar:.1f}" if not np.isnan(nstar) else "NaN",
                crossover_observed,
            )

    return rows


def _project_nstar(fit: dict, t_lh: float) -> float:
    """N** proyectado según modelo."""
    if np.isnan(t_lh) or t_lh <= 0:
        return np.nan
    a = fit["slope_a"]
    b = fit["intercept_b"]
    if a == 0:
        return np.nan
    if fit["model_type"] == "log_linear":
        # log(t_LH) = a*N + b  →  N** = (log(t_LH) - b) / a
        return (np.log(t_lh) - b) / a
    else:  # log_log
        # log(t_LH) = a*log(N) + b  →  N** = exp((log(t_LH) - b) / a)
        val = (np.log(t_lh) - b) / a
        return float(np.exp(val))


def _project_nstar_ci(fit: dict, t_lh: float) -> str:
    """IC 95% del crossover propagando la incertidumbre en la pendiente."""
    if np.isnan(t_lh) or t_lh <= 0:
        return "[NaN, NaN]"

    b  = fit["intercept_b"]
    se = fit["stderr_a"]

    a_low  = fit["slope_ci_low"]
    a_high = fit["slope_ci_high"]

    def _nstar_from_a(a: float) -> float:
        if a == 0:
            return np.nan
        if fit["model_type"] == "log_linear":
            return (np.log(t_lh) - b) / a
        else:
            val = (np.log(t_lh) - b) / a
            return float(np.exp(val))

    n_low  = _nstar_from_a(a_high)  # pendiente alta → N** más bajo (log-lineal)
    n_high = _nstar_from_a(a_low)

    # Ordenar porque el signo de la inversión depende del modelo
    lo = min(n_low, n_high)
    hi = max(n_low, n_high)
    return f"[{lo:.1f}, {hi:.1f}]"


def _is_crossover_observed(nstar: float, x_fit: np.ndarray) -> bool:
    """True si N** cae dentro del rango experimental."""
    return float(x_fit.min()) <= nstar <= float(x_fit.max())


# =============================================================================
# CELDA 5 ya integrada en fit_regressions (N** + CI calculados allí)
# =============================================================================


# =============================================================================
# CELDA 6 — SAVE resultados
# =============================================================================

def save_results(rows: list[dict]) -> None:
    """Guarda hoja 'computed' y 'metadata' en exp04_crossover.xlsx."""
    if not rows:
        logger.warning("No hay filas para guardar.")
        return

    append_rows(EXP04_FILE, "computed", rows)
    logger.info("Guardadas %d filas en %s hoja=computed", len(rows), EXP04_FILE.name)

    save_metadata(EXP04_FILE, {
        "exp_version":    "v7.0",
        "run_uuid_last":  RUN_UUID,
        "exp01_source":   str(EXP01_FILE),
        "exp03_source":   str(EXP03_FILE),
        "timestamp":      datetime.datetime.now().isoformat(),
    })


# =============================================================================
# CELDA 7 — PLOT curva de scaling (figura central)
# =============================================================================

def plot_scaling_curves(
    df_g: pd.DataFrame,
    df_lh: pd.DataFrame,
    rows: list[dict],
    df_gqubo: pd.DataFrame | None = None,
    df_g_val: pd.DataFrame | None = None,
) -> None:
    """
    Genera la figura de scaling: 3 subplots (uno por eje), escala Y logarítmica.

    Para cada eje:
      - Puntos Gurobi Exp 7 (df_g): media ± std — fuente de la regresión
      - Puntos Gurobi Exp 1 (df_g_val): overlay de validación (deben caer sobre curva)
      - Línea horizontal LeapHybrid: media ± banda 1σ
      - Curva de regresión seleccionada (mejor R²): sólida en rango, punteada fuera
      - Marcador vertical en N** (si proyectable)
    """
    matplotlib.rcParams.update({
        "font.size":      11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi":     150,
    })

    fig, axes_arr = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    axis_titles = {"size": "Eje 1 — Escalabilidad (Size)", "dens": "Eje 2 — Congestión (Dens)", "slack": "Eje 3 — Clustering (Slack)"}

    computed_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    for ax, axis in zip(axes_arr, AXES):
        # --- Gurobi points ---
        agg = _aggregate_gurobi(df_g, axis)
        if agg.empty:
            ax.set_title(f"{axis_titles.get(axis, axis)}\n(sin datos Gurobi)")
            ax.set_visible(False)
            continue

        x_pts = agg["N"].values.astype(float)
        y_mean = agg["wall_time_mean"].values.astype(float)
        y_std  = agg["wall_time_std"].fillna(0).values.astype(float)
        x_constant = np.all(x_pts == x_pts[0])

        ax.errorbar(
            x_pts, y_mean, yerr=y_std,
            fmt="o", color="#2c5f8a", capsize=4, linewidth=1.2,
            markersize=6, label="Gurobi Exp 7 (regresión, media ± std)",
        )

        # --- Validation overlay: Exp 1 points (should lie on regression curve) ---
        if df_g_val is not None and not df_g_val.empty:
            _agg_val = _aggregate_gurobi(df_g_val, axis)
            if not _agg_val.empty:
                ax.errorbar(
                    _agg_val["N"].values.astype(float),
                    _agg_val["wall_time_mean"].values.astype(float),
                    yerr=_agg_val["wall_time_std"].fillna(0).values.astype(float),
                    fmt="s", color="#2c5f8a", capsize=3, linewidth=1.0,
                    markersize=5, alpha=0.55, linestyle="",
                    label="Gurobi Exp 1 (validación)",
                )

        # --- Regression curve (mejor R²) — omitir si N es constante ---
        if not x_constant and not computed_df.empty:
            sub = computed_df[computed_df["axis"] == axis]
            if not sub.empty:
                best_row = sub.loc[sub["r_squared"].idxmax()]
                a = best_row["slope_a"]
                b = best_row["intercept_b"]
                model = best_row["model_type"]

                x_range  = np.linspace(x_pts.min(), x_pts.max(), 200)
                x_extrap = np.linspace(x_pts.max(), x_pts.max() * 3, 200)

                def _predict(xv):
                    if model == "log_linear":
                        return np.exp(a * xv + b)
                    else:
                        return np.exp(b) * xv ** a

                ax.plot(x_range,  _predict(x_range),  "-",  color="#e07b39", linewidth=1.8, label=f"Regresión {model.replace('_', '-')} (R²={best_row['r_squared']:.3f})")
                ax.plot(x_extrap, _predict(x_extrap), "--", color="#e07b39", linewidth=1.4, alpha=0.7)

                # N** marker
                nstar = best_row.get("nstar_crossover")
                if nstar is not None and not np.isnan(float(nstar)):
                    nstar_f = float(nstar)
                    ax.axvline(nstar_f, color="#7b2d8b", linestyle=":", linewidth=1.5, label=f"N** = {nstar_f:.0f}")
                    ax.annotate(
                        f"N** ≈ {nstar_f:.0f}",
                        xy=(nstar_f, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="#7b2d8b",
                    )

        # --- LeapHybrid horizontal band ---
        if not df_lh.empty:
            lh_ax = df_lh[df_lh["axis"] == axis]
            if not lh_ax.empty and "lh_time_s" in lh_ax.columns:
                lh_times = lh_ax["lh_time_s"].dropna()
                if not lh_times.empty:
                    t_mean = float(lh_times.mean())
                    t_std  = float(lh_times.std())
                    ax.axhline(t_mean, color="#2da34e", linewidth=1.8, linestyle="-", label=f"LeapHybrid (μ={t_mean:.1f}s)")
                    ax.axhspan(t_mean - t_std, t_mean + t_std, alpha=0.15, color="#2da34e", label="LH ± 1σ")

        # --- Gurobi-QUBO scaling curve ---
        if df_gqubo is not None and not df_gqubo.empty:
            gq_ax = df_gqubo[df_gqubo["axis"] == axis].copy()
            if not gq_ax.empty and "qubo_wall_time_s" in gq_ax.columns:
                gq_agg = (
                    gq_ax.groupby("instance_label", as_index=False)
                    .agg(
                        N=("qubo_n_vars", "first"),
                        t_mean=("qubo_wall_time_s", "mean"),
                        t_std=("qubo_wall_time_s", "std"),
                    )
                    .sort_values("N")
                )
                if len(gq_agg) >= 2:
                    ax.errorbar(
                        gq_agg["N"].values, gq_agg["t_mean"].values,
                        yerr=gq_agg["t_std"].fillna(0).values,
                        fmt="D--", color="#9b59b6", capsize=4, linewidth=1.4,
                        markersize=5, label="Gurobi-QUBO (media ± std)",
                    )
                elif len(gq_agg) == 1:
                    ax.errorbar(
                        gq_agg["N"].values, gq_agg["t_mean"].values,
                        fmt="D", color="#9b59b6", capsize=4, markersize=5,
                        label="Gurobi-QUBO",
                    )

        ax.set_yscale("log")
        ax.set_xlabel("N (número de buques)")
        ax.set_ylabel("wall_time_s")
        title = axis_titles.get(axis, axis)
        if x_constant:
            title += f"\n(N={int(x_pts[0])} constante — regresión no aplicable)"
        ax.set_title(title)
        ax.legend(loc="upper left", framealpha=0.85)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        if x_constant:
            ax.set_xlim(x_pts[0] - 1, x_pts[0] + 1)
            ax.xaxis.set_major_locator(mticker.FixedLocator([x_pts[0]]))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"N={int(x_pts[0])}"]))

    fig.suptitle(
        "Exp 4 — Scaling & Crossover: Gurobi vs LeapHybrid",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    fig_path = RESULTS_DIR / "exp04_scaling_crossover.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    logger.info("Plot guardado: %s", fig_path)
    plt.show()


# =============================================================================
# MAIN — ejecutar todas las celdas en secuencia
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("EXP 4 — Scaling & Crossover  (run_uuid=%s)", RUN_UUID)
    logger.info("=" * 60)

    # Celda 2 — cargar Gurobi
    # Fuente primaria: Exp 7 (13 puntos densos, 5 seeds). Fallback: Exp 1.
    df_g7 = load_gurobi_times_exp7()
    df_g1 = load_gurobi_times()   # Exp 1 — validación overlay

    if not df_g7.empty:
        logger.info("Usando Exp 7 como fuente primaria de regresión Gurobi.")
        df_g_primary = df_g7
    else:
        logger.warning("Exp 7 no disponible — usando Exp 1 como fuente primaria (fallback).")
        df_g_primary = df_g1
        df_g1 = pd.DataFrame()  # no repetir puntos en validación

    # Celda 3 — LH + Gurobi-QUBO
    df_lh    = load_lh_times()
    df_gqubo = load_gurobi_qubo_times()

    # Celdas 4 + 5
    rows = fit_regressions(df_g_primary, df_lh)

    if not rows:
        logger.error("No se generaron filas de resultados. Verificar Exp 1/7 y Exp 3.")
        return

    # Celda 6
    save_results(rows)

    # Celda 7
    try:
        plot_scaling_curves(
            df_g_primary, df_lh, rows,
            df_gqubo=df_gqubo,
            df_g_val=df_g1 if not df_g1.empty else None,
        )
    except Exception as exc:
        logger.warning("Plot falló: %s — continuar sin figura.", exc)

    # Resumen
    df_res = pd.DataFrame(rows)
    logger.info("\n%s\n", df_res[["axis", "model_type", "r_squared", "slope_a", "nstar_crossover", "crossover_observed"]].to_string(index=False))
    logger.info("Exp 4 completado.")


# =============================================================================
# CELDA 8 — COMPARACIÓN DE TIEMPOS POR INSTANCIA Y SOLVER
# =============================================================================

def plot_solver_times_comparison() -> None:
    """
    Carga tiempos de Gurobi (Exp 1), SA y LeapHybrid (Exp 3) y genera
    una figura con un subplot por eje comparando wall_time_s de cada solver
    por instancia, ordenadas por N creciente.
    """
    # --- Cargar Gurobi (Exp 1) ---
    if not EXP01_FILE.exists():
        logger.error("Exp 1 Excel no encontrado: %s", EXP01_FILE)
        return
    df_g_raw = load_existing_runs(EXP01_FILE, "raw_runs")
    if df_g_raw.empty:
        logger.error("Exp 1 raw_runs vacío.")
        return

    # Agregar Gurobi por (axis, instance_label, N): media sobre seeds
    df_g_raw["axis"] = df_g_raw["axis"].str.lower()
    gurobi_agg = (
        df_g_raw
        .groupby(["axis", "instance_label", "N"], as_index=False)
        .agg(wall_time_mean=("wall_time_s", "mean"), wall_time_std=("wall_time_s", "std"))
    )
    gurobi_agg["solver"] = "Gurobi"

    # --- Cargar SA + LeapHybrid (Exp 3) ---
    if not EXP03_FILE.exists():
        logger.warning("Exp 3 Excel no encontrado: %s — se omiten SA y LeapHybrid.", EXP03_FILE)
        df_exp3 = pd.DataFrame()
    else:
        frames = []
        for sheet in ("size_axis", "dens_axis", "slack_axis"):
            try:
                df_s = pd.read_excel(EXP03_FILE, sheet_name=sheet)
                if df_s.empty:
                    continue
                df_s["axis"] = sheet.replace("_axis", "")
                frames.append(df_s)
            except ValueError:
                logger.warning("Hoja %s no encontrada en Exp 3.", sheet)
        df_exp3 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Para SA: wall_time_s; para LH: lh_time_s (cuando no es NaN)
    exp3_rows = []
    if not df_exp3.empty:
        for solver, time_col in [("SA", "wall_time_s"), ("LeapHybrid", "lh_time_s")]:
            sub = df_exp3[df_exp3["solver"] == solver].copy()
            if sub.empty or time_col not in sub.columns:
                continue
            sub = sub[sub[time_col].notna()]
            if sub.empty:
                continue
            agg = (
                sub
                .groupby(["axis", "instance_label", "N"], as_index=False)
                .agg(wall_time_mean=(time_col, "mean"), wall_time_std=(time_col, "std"))
            )
            agg["solver"] = solver
            exp3_rows.append(agg)

    # Gurobi-QUBO from Exp 1 milp_qubo_equiv sheet
    df_gqubo_comp = load_gurobi_qubo_times()
    gqubo_rows = []
    if not df_gqubo_comp.empty and "qubo_wall_time_s" in df_gqubo_comp.columns:
        gq_agg = (
            df_gqubo_comp
            .groupby(["axis", "instance_label", "qubo_n_vars"], as_index=False)
            .agg(wall_time_mean=("qubo_wall_time_s", "mean"), wall_time_std=("qubo_wall_time_s", "std"))
            .rename(columns={"qubo_n_vars": "N"})
        )
        gq_agg["solver"] = "Gurobi-QUBO"
        gqubo_rows.append(gq_agg)

    all_frames = [gurobi_agg] + exp3_rows + gqubo_rows
    df_all = pd.concat(all_frames, ignore_index=True)

    # --- Figura: 3 subplots (uno por eje) ---
    matplotlib.rcParams.update({
        "font.size":       11,
        "axes.titlesize":  12,
        "axes.labelsize":  11,
        "legend.fontsize":  9,
        "figure.dpi":      150,
    })

    solver_styles = {
        "Gurobi":       {"color": "#2c5f8a", "marker": "o",  "linestyle": "-"},
        "SA":           {"color": "#e07b39", "marker": "s",  "linestyle": "--"},
        "LeapHybrid":   {"color": "#2da34e", "marker": "^",  "linestyle": "-."},
        "Gurobi-QUBO":  {"color": "#9b59b6", "marker": "D",  "linestyle": ":"},
    }

    axis_titles = {
        "size":  "Eje 1 — Escalabilidad (Size)",
        "dens":  "Eje 2 — Congestión (Dens)",
        "slack": "Eje 3 — Clustering (Slack)",
    }

    fig, axes_arr = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, axis in zip(axes_arr, AXES):
        sub_axis = df_all[df_all["axis"] == axis].copy()
        if sub_axis.empty:
            ax.set_title(f"{axis_titles.get(axis, axis)}\n(sin datos)")
            ax.set_visible(False)
            continue

        # Ordenar instancias por N
        inst_order = (
            sub_axis[["instance_label", "N"]]
            .drop_duplicates()
            .sort_values("N")["instance_label"]
            .tolist()
        )

        x_ticks = range(len(inst_order))
        inst_idx = {label: i for i, label in enumerate(inst_order)}

        any_line = False
        for solver in ["Gurobi", "SA", "LeapHybrid", "Gurobi-QUBO"]:
            s = sub_axis[sub_axis["solver"] == solver].copy()
            if s.empty:
                continue
            s = s.sort_values("N")
            xs = [inst_idx[lbl] for lbl in s["instance_label"] if lbl in inst_idx]
            ys = s.loc[s["instance_label"].isin(inst_idx), "wall_time_mean"].values
            yerr = s.loc[s["instance_label"].isin(inst_idx), "wall_time_std"].fillna(0).values

            style = solver_styles[solver]
            ax.errorbar(
                xs, ys, yerr=yerr,
                fmt=style["marker"] + style["linestyle"],
                color=style["color"],
                capsize=4, linewidth=1.4, markersize=6,
                label=solver,
            )
            any_line = True

        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels(inst_order, rotation=35, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("Instancia (N creciente →)")
        ax.set_ylabel("Tiempo de resolución (s)")
        ax.set_title(axis_titles.get(axis, axis))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        if any_line:
            ax.legend(loc="upper left", framealpha=0.85)

    fig.suptitle(
        "Exp 4 — Comparación de Tiempos por Instancia y Solver",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    fig_path = RESULTS_DIR / "exp04_solver_times_comparison.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    logger.info("Plot comparación de tiempos guardado: %s", fig_path)
    plt.show()


if __name__ == "__main__":
    main()
    plot_solver_times_comparison()
