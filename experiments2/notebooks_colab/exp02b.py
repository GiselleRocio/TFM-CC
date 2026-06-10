"""
exp02b_lagrange_calibration_sa — Experimento 2b: Calibración Lagrange SA-only (grilla ampliada)

Pregunta: ¿Cómo varía la factibilidad QUBO en una grilla más fina de (α, β)?
          ¿Cuál es el α* robusto para distintos niveles de β?
          ¿El paisaje Lagrangiano es monótono o presenta anomalías?

Este experimento extiende exp02 con una grilla más fina de α (paso 0.2, rango 1.2–2.8)
y 5 valores de β (1.0, 1.5, 2.0, 2.5, 3.0), sin validación LeapHybrid.
Objetivo: caracterizar con precisión el paisaje de calibración Lagrangiana para la tesis.

Outputs:
  results/exp02b_lagrange_calibration_sa.xlsx
    hoja: sa_sweep   (una fila por (alpha, beta, instance_label, seed, run_id))
    hoja: metadata   (alpha_star, beta_star, timestamp)

Ejecución:
  Celda 0: INSTALL (comentada)
  Celda 1: SETUP
  Celda 2: LOAD instancias (Size_1, Cong_3)
  Celda 3: RUN SA sweep — grid α × β × instancias × seeds × runs
  Celda 4: DETECT α*, β*
  Celda 5: SAVE metadata
  Celda 6: PLOT heatmap de factibilidad (Size_1, Cong_3)
  Celda 7: PLOT curva de factibilidad vs α (múltiples líneas β)
  Celda 8: PLOT distribución de energías por α
"""

# CELDA 0: INSTALL — ejecutar una sola vez por sesión de Colab
# dwave-samplers incluye SimulatedAnnealingSampler; no requiere token QPU
# %pip install -q dimod dwave-samplers openpyxl seaborn

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
    ("openpyxl",       "openpyxl"),
    ("seaborn",        "seaborn"),
]
for _pip, _mod in _PKGS:
    if _ilu.find_spec(_mod.split(".")[0]) is None:
        print(f"  instalando {_pip}...", end=" ", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("listo")
    else:
        print(f"  ok  {_pip}")

import numpy as np
import pandas as pd

from experiments2.shared.run_id import new_run_uuid
from experiments2.shared.io_utils import (
    ensure_directories,
    load_instances_from_excel,
    load_existing_runs,
    append_rows,
    save_metadata,
    INSTANCES_EXCEL,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("exp02b")

ensure_directories()

RUN_UUID = new_run_uuid()
FILEPATH = RESULTS_DIR / "exp02b_lagrange_calibration_sa.xlsx"
SHEET_SA = "sa_sweep"

logger.info("Exp 2b SETUP  run_uuid=%s", RUN_UUID)
logger.info("REPO_ROOT: %s", REPO_ROOT)
logger.info("Output: %s", FILEPATH)

# CELDA 2: LOAD instancias Size_1 y Cong_3
# Size_1 (ρ≈74%, N=8) y Cong_3 (ρ≈65%, N=10, T=62 fijo) representan distintos niveles
# de congestión. La calibración sobre dos ρ distintos garantiza que α* sea robusto.

if not INSTANCES_EXCEL.exists():
    raise FileNotFoundError(
        f"No se encontró {INSTANCES_EXCEL}. Ejecutar setup.py primero."
    )

_size_dict = load_instances_from_excel("size")
_cong_dict = load_instances_from_excel("congestion")

SWEEP_INSTANCES = {
    "Size_1": _size_dict["Size_1"],
    "Cong_3": _cong_dict["Cong_3"],
}

for label, inst in SWEEP_INSTANCES.items():
    noms = inst["nominations"]
    rho  = noms["p_j"].sum() / inst["T"]
    logger.info("  %s: N=%d  T=%d  ρ_eff=%.3f", label, inst["N"], inst["T"], rho)

# CELDA 3: RUN SA sweep — paralelo por bloque (alpha, beta, instance_label)
# Cada bloque es independiente: mismo BQM, distintos seeds/runs.
# ThreadPoolExecutor paraleliza los bloques (más fiable en Colab que ProcessPoolExecutor).
# Append-safe: saltar (alpha, beta, instance_label, seed, run_id) ya en el Excel.
# Grilla propia de exp02b: α más fina (paso 0.2, rango 1.2–2.8) y 5 valores de β.

import os
import concurrent.futures
from preprocessing import compute_feasible_slots
from qubo_builder import build_qubo
from solver import decode_schedule, check_feasibility

# Grilla propia de exp02b — NO importar de experiment_config (grillas distintas a exp02)
alpha_grid = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
beta_grid  = [1.2, 1.5, 2.0, 2.5, 3.0]

# Parámetros SA idénticos a exp02
n_seeds    = 5
n_runs_sa  = 20
num_reads  = 200
num_sweeps = 1000

total_runs = len(alpha_grid) * len(beta_grid) * len(SWEEP_INSTANCES) * n_seeds * n_runs_sa
logger.info(
    "SA sweep: %d α × %d β × %d instancias × %d seeds × %d runs = %d runs",
    len(alpha_grid), len(beta_grid), len(SWEEP_INSTANCES), n_seeds, n_runs_sa, total_runs,
)

existing = load_existing_runs(FILEPATH, SHEET_SA)
if not existing.empty:
    done = set(zip(
        existing["alpha"], existing["beta"],
        existing["instance_label"], existing["seed"], existing["run_id"],
    ))
else:
    done = set()
logger.info("  Runs ya completados: %d", len(done))

# Precomputar variables_df y BQM por (label, alpha, beta) — no en el worker
_precomputed: dict = {}
for label, inst in SWEEP_INSTANCES.items():
    noms = inst["nominations"].copy()
    T    = int(inst["T"])
    N    = int(inst["N"])
    vdf  = compute_feasible_slots(noms, horizon_slots=T)
    for alpha in alpha_grid:
        for beta in beta_grid:
            bqm, P1, P2, P3_val, _ = build_qubo(vdf, alpha=alpha, beta=beta)
            n_vars    = len(bqm.variables)
            n_edges   = len(bqm.quadratic)
            max_edges = n_vars * (n_vars - 1) / 2 if n_vars > 1 else 1
            _precomputed[(label, alpha, beta)] = {
                "bqm": bqm, "vdf": vdf,
                "N": N, "T": T,
                "P1": P1, "P2": P2, "P3": P3_val,
                "n_vars": n_vars,
                "q_density": round(n_edges / max_edges, 6),
            }


def _run_block(
    label: str, alpha: float, beta: float,
    precomp: dict, n_seeds: int, n_runs: int, num_reads: int, num_sweeps: int,
    done_set: set, run_uuid: str,
) -> list[dict]:
    """Ejecuta todos los seeds×runs de un bloque (alpha, beta, label). Sin escritura a disco."""
    import datetime as _dt

    from dwave.samplers import SimulatedAnnealingSampler as SA
    from solver import decode_schedule, check_feasibility

    sampler = SA()
    bqm     = precomp["bqm"]
    vdf     = precomp["vdf"]
    rows: list[dict] = []

    # Escalar temperatura con P1 para que SA pueda cruzar barreras de penalización.
    P1 = precomp["P1"]
    beta_min   = max(1.0 / (P1 * 2.0), 1e-4)
    beta_range = (beta_min, 10.0)

    for seed in range(n_seeds):
        for run_id in range(n_runs):
            if (alpha, beta, label, seed, run_id) in done_set:
                continue
            try:
                ss          = sampler.sample(bqm, num_reads=num_reads,
                                             num_sweeps=num_sweeps,
                                             beta_range=beta_range,
                                             seed=seed * 1000 + run_id)
                best_sample = ss.first.sample
                best_energy = float(ss.first.energy)
                sched       = decode_schedule(best_sample, vdf)
                fres        = check_feasibility(sched, vdf)
                is_feas     = bool(fres["is_feasible"])
                obj_val     = float(fres["total_weighted_tardiness"]) if is_feas else float("nan")

                rows.append({
                    "exp_id":         "exp02b_sa",
                    "run_uuid":       run_uuid,
                    "alpha":          alpha,
                    "beta":           beta,
                    "instance_label": label,
                    "N":              precomp["N"],
                    "T":              precomp["T"],
                    "seed":           seed,
                    "run_id":         run_id,
                    "feasible":       is_feas,
                    "obj_value":      obj_val,
                    "energy":         best_energy,
                    "energy_gap":     float("nan"),
                    "q_density":      precomp["q_density"],
                    "n_vars":         precomp["n_vars"],
                    "p1":             float(precomp["P1"]),
                    "p2":             float(precomp["P2"]),
                    "p3":             float(precomp["P3"]),
                    "eps_eff":        float("nan"),
                    "run_timestamp":  _dt.datetime.now().isoformat(),
                })
            except Exception:
                pass  # bloque individual no frena el sweep
    return rows


# Armar lista de tareas pendientes
tasks = [
    (label, alpha, beta)
    for label in SWEEP_INSTANCES
    for alpha in alpha_grid
    for beta in beta_grid
    if any(
        (alpha, beta, label, seed, run_id) not in done
        for seed in range(n_seeds)
        for run_id in range(n_runs_sa)
    )
]
logger.info("  Bloques a ejecutar: %d / %d", len(tasks),
            len(alpha_grid) * len(beta_grid) * len(SWEEP_INSTANCES))

if not tasks:
    logger.info("SA sweep: nada pendiente — todos los runs ya completados.")
else:
    N_WORKERS = min(len(tasks), os.cpu_count() or 4)
    logger.info("  Workers: %d", N_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                _run_block,
                label, alpha, beta,
                _precomputed[(label, alpha, beta)],
                n_seeds, n_runs_sa, num_reads, num_sweeps,
                done, RUN_UUID,
            ): (label, alpha, beta)
            for label, alpha, beta in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            label, alpha, beta = futures[future]
            try:
                block_rows = future.result()
                if block_rows:
                    append_rows(FILEPATH, SHEET_SA, block_rows)
                    logger.info("  α=%.1f β=%.1f %s — %d runs guardados",
                                alpha, beta, label, len(block_rows))
            except Exception as exc:
                logger.error("  bloque α=%.1f β=%.1f %s falló: %s", alpha, beta, label, exc)

logger.info("SA sweep completo.")

# CELDA 4: DETECT α*, β*
# Criterio α*: mínimo α con feasibility_rate ≥ 0.95 en AMBAS instancias simultáneamente.
# El patrón debe ser creciente: más α = penalización más fuerte = más soluciones factibles.

df_sa = pd.read_excel(FILEPATH, sheet_name=SHEET_SA)

feas_agg = (
    df_sa.groupby(["alpha", "beta", "instance_label"])["feasible"]
    .mean()
    .reset_index()
    .rename(columns={"feasible": "feasibility_rate"})
)

print("\nFeasibility rate por (α, β, instancia):")
print(feas_agg.to_string(index=False))

alpha_star = None
beta_star  = None
TARGET_FEAS = 0.95
instances_labels = list(SWEEP_INSTANCES.keys())

for alpha in sorted(df_sa["alpha"].unique()):
    for beta in sorted(df_sa["beta"].unique()):
        sub = feas_agg[
            (feas_agg["alpha"] == alpha) &
            (feas_agg["beta"]  == beta)  &
            (feas_agg["instance_label"].isin(instances_labels))
        ]
        if len(sub) < len(instances_labels):
            continue
        if (sub["feasibility_rate"] >= TARGET_FEAS).all():
            alpha_star = alpha
            beta_star  = beta
            break
    if alpha_star is not None:
        break

if alpha_star is None:
    # Fallback data-driven: ningún (α, β) alcanzó el umbral en todas las instancias.
    # Criterio max-min: seleccionar el (α, β) que maximiza la feasibility_rate mínima
    # entre instancias (el más robusto). Desempate: α menor.
    _fallback_scores = (
        feas_agg
        .groupby(["alpha", "beta"])["feasibility_rate"]
        .min()
        .reset_index()
        .rename(columns={"feasibility_rate": "min_feas"})
        .sort_values(["min_feas", "alpha"], ascending=[False, True])
    )
    _best = _fallback_scores.iloc[0]
    alpha_star = float(_best["alpha"])
    beta_star  = float(_best["beta"])
    logger.warning(
        "No se encontró (α*, β*) con feasibility_rate ≥ %.0f%% en todas las instancias. "
        "Fallback max-min: α*=%.1f, β*=%.1f (min_feas_entre_instancias=%.2f).",
        TARGET_FEAS * 100, alpha_star, beta_star, float(_best["min_feas"]),
    )

print(f"\nα* = {alpha_star}  β* = {beta_star}")

# Verificar ε efectivo
from config import EPSILON

_inst  = SWEEP_INSTANCES["Size_1"]
_noms  = _inst["nominations"].copy()
p_max  = int(_noms["p_j"].max())
w_min  = float(_noms["w_j"].min()) if "w_j" in _noms.columns else 1.0
eps_eff = EPSILON * (int(_inst["T"]) + p_max)
print(f"\nε efectivo = {eps_eff:.4f}  w_min = {w_min:.4f}")
if eps_eff < w_min * 0.01:
    logger.warning("ε efectivo << w_min — symmetry-breaking invisible. Ajustar EPSILON.")

# CELDA 5: SAVE metadata

save_metadata(FILEPATH, {
    "exp_version":    "v1.0",
    "run_uuid_last":  RUN_UUID,
    "timestamp":      datetime.datetime.now().isoformat(),
    "alpha_star":     alpha_star,
    "beta_star":      beta_star,
    "target_feas":    TARGET_FEAS,
    "eps_eff":        round(eps_eff, 4),
    "alpha_grid":     str(alpha_grid),
    "beta_grid":      str(beta_grid),
    "n_seeds":        n_seeds,
    "n_runs_sa":      n_runs_sa,
    "sa_num_reads":   num_reads,
    "sa_num_sweeps":  num_sweeps,
})
logger.info("Exp 2b completo. α*=%s  β*=%s  Resultados: %s", alpha_star, beta_star, FILEPATH)

# CELDA 6: PLOT heatmap de factibilidad (uno por instancia)
# Con la grilla ampliada (9 α × 5 β) el heatmap muestra con más detalle
# cómo evoluciona la factibilidad en el espacio de parámetros Lagrangianos.

import matplotlib.pyplot as plt
import seaborn as sns

df_sa = pd.read_excel(FILEPATH, sheet_name=SHEET_SA)

feas_agg = (
    df_sa.groupby(["alpha", "beta", "instance_label"])["feasible"]
    .mean()
    .reset_index()
    .rename(columns={"feasible": "feasibility_rate"})
)

fig, axes_plot = plt.subplots(1, len(SWEEP_INSTANCES), figsize=(18, 6), sharey=True)
if len(SWEEP_INSTANCES) == 1:
    axes_plot = [axes_plot]

for ax, label in zip(axes_plot, SWEEP_INSTANCES.keys()):
    pivot = (
        feas_agg[feas_agg["instance_label"] == label]
        .pivot(index="beta", columns="alpha", values="feasibility_rate")
    )
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".2f", vmin=0, vmax=1,
        cmap="RdYlGn", linewidths=0.5, linecolor="gray",
        cbar_kws={"label": "Feasibility rate"},
    )
    if alpha_star is not None and beta_star is not None:
        alpha_cols = list(pivot.columns)
        beta_rows  = list(pivot.index)
        if alpha_star in alpha_cols and beta_star in beta_rows:
            col_idx = alpha_cols.index(alpha_star)
            row_idx = beta_rows.index(beta_star)
            ax.add_patch(plt.Rectangle(
                (col_idx, row_idx), 1, 1,
                fill=False, edgecolor="blue", lw=2.5, label=f"α*={alpha_star}, β*={beta_star}"
            ))
    ax.set_title(f"{label}  (ρ_eff={feas_agg[feas_agg['instance_label']==label]['feasibility_rate'].mean():.2f})")
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    sns.despine(ax=ax)

fig.suptitle("Exp 2b — Heatmap feasibility rate por (α, β)  [grilla ampliada]", fontsize=13)
plt.tight_layout()
plot_path = RESULTS_DIR / "exp02b_heatmap_feasibility.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
logger.info("Guardado: %s", plot_path)

# CELDA 7: PLOT curva de factibilidad vs α (múltiples líneas β)
# Con 5 valores de β se pueden ver claramente las diferencias entre regímenes de β.

fig, ax = plt.subplots(figsize=(11, 6))

linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
beta_vals  = sorted(df_sa["beta"].unique())

for label in SWEEP_INSTANCES.keys():
    for i, beta in enumerate(beta_vals):
        sub = (
            df_sa[(df_sa["instance_label"] == label) & (df_sa["beta"] == beta)]
            .groupby("alpha")["feasible"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "feas_mean", "std": "feas_std"})
        )
        if sub.empty:
            continue
        ls = "-" if label == "Size_1" else "--"
        ax.errorbar(
            sub["alpha"], sub["feas_mean"], yerr=sub["feas_std"].fillna(0),
            fmt=f"o{ls}", capsize=3, linestyle=linestyles[i % len(linestyles)],
            label=f"{label} β={beta}",
        )

ax.axhline(TARGET_FEAS, color="red", linestyle="--", linewidth=1.2,
           label=f"Target ({TARGET_FEAS:.0%})")
if alpha_star is not None:
    ax.axvline(alpha_star, color="blue", linestyle=":", linewidth=1.5,
               label=f"α*={alpha_star}")

ax.set_xlabel("α (penalización)")
ax.set_ylabel("Feasibility rate")
ax.set_ylim(0, 1.05)
ax.set_title("Exp 2b — Curva de factibilidad vs α  [múltiples β]")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, linestyle=":", alpha=0.5)
sns.despine(ax=ax)

plt.tight_layout()
plot_path = RESULTS_DIR / "exp02b_feasibility_vs_alpha.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
logger.info("Guardado: %s", plot_path)
logger.info("Exp 2b VISUALIZE completo.")

# CELDA 8: PLOT distribución de energías por α (violinplot)
#
# Violin / stripplot de las energías QUBO faceteado por α.
# Colorear muestras feasibles vs infeasibles.
# Con la grilla fina se puede observar si la transición de energía es gradual o abrupta.

df_sa9 = pd.read_excel(FILEPATH, sheet_name=SHEET_SA)

if "energy" not in df_sa9.columns:
    logger.warning("Columna 'energy' no encontrada en sa_sweep — Celda 8 saltada.")
else:
    alpha_vals9 = sorted(df_sa9["alpha"].unique())
    n_alphas9   = len(alpha_vals9)

    for inst_label9 in df_sa9["instance_label"].unique():
        df_inst9 = df_sa9[df_sa9["instance_label"] == inst_label9].copy()
        df_inst9["feasible_str"] = df_inst9["feasible"].apply(
            lambda v: "Feasible" if bool(v) else "Infeasible"
        )

        fig9, axes9 = plt.subplots(
            1, n_alphas9, figsize=(3 * n_alphas9, 5), sharey=False
        )
        if n_alphas9 == 1:
            axes9 = [axes9]

        palette9 = {"Feasible": "#2da34e", "Infeasible": "#e07b39"}

        for ax9, alpha_v9 in zip(axes9, alpha_vals9):
            sub9 = df_inst9[df_inst9["alpha"] == alpha_v9]

            if sub9["energy"].dropna().empty:
                ax9.set_title(f"α={alpha_v9}")
                ax9.text(0.5, 0.5, "(sin datos)", ha="center", va="center",
                         transform=ax9.transAxes, fontsize=9, color="gray")
                continue

            try:
                sns.violinplot(
                    data=sub9, y="energy", hue="feasible_str",
                    palette=palette9, split=True, inner="quart",
                    linewidth=0.8, ax=ax9,
                )
            except TypeError:
                # versiones antiguas de seaborn no tienen split en violinplot
                sns.violinplot(
                    data=sub9, y="energy", hue="feasible_str",
                    palette=palette9, inner="quart",
                    linewidth=0.8, ax=ax9,
                )

            n_feas9   = int(sub9["feasible"].astype(bool).sum())
            n_infeas9 = len(sub9) - n_feas9
            ax9.set_title(f"α={alpha_v9}\n(F={n_feas9}, I={n_infeas9})", fontsize=9)
            ax9.set_xlabel("")
            ax9.set_ylabel("Energía QUBO" if ax9 == axes9[0] else "")
            if ax9.get_legend():
                ax9.get_legend().remove()
            sns.despine(ax=ax9)

        from matplotlib.patches import Patch
        _handles9 = [
            Patch(color="#2da34e", label="Feasible"),
            Patch(color="#e07b39", label="Infeasible"),
        ]
        fig9.legend(handles=_handles9, loc="lower center", ncol=2,
                    fontsize=9, bbox_to_anchor=(0.5, -0.06))
        fig9.suptitle(
            f"Exp 2b — Distribución de energías QUBO por α  |  {inst_label9}",
            fontsize=12,
        )
        plt.tight_layout()
        plot9_path = RESULTS_DIR / f"exp02b_energy_dist_by_alpha_{inst_label9}.png"
        fig9.savefig(plot9_path, dpi=300, bbox_inches="tight")
        plt.show()
        logger.info("Guardado: %s", plot9_path)

    logger.info("Celda 8 completo — distribución de energías por α.")
