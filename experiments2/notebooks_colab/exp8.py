"""
exp08_qpu_direct.py — Experimento 8: QPU Directo (Exploratorio)

Pregunta: ¿Es posible embeber el QUBO de este problema de scheduling en el hardware
          Pegasus de D-Wave? ¿Qué calidad de solución produce el QPU si el embedding
          tiene éxito? Los resultados negativos (fallo de embedding) también son datos
          valiosos para la tesis.

Instancias (en orden):
  1. Tiny_3  — N=3 buques, generado inline con generate_nominations(n=3, seed=DEFAULT_SEED)
  2. Size_1  — N=8 buques, cargado de data/instances.xlsx eje "size"

Por cada instancia:
  1. Construir BQM con build_qubo + CONFLICT_SET_R
  2. Registrar diagnósticos BQM: n_vars, n_interactions, q_density, max_degree
  3. Intentar embedding: minorminer.find_embedding(Q_dict, target_graph)
     - Para Size_1: timeout controlado con max_no_improvement=10
  4. Si éxito: someter al QPU (num_reads=1000, N_RUNS runs para Tiny_3), decodificar, verificar factibilidad
  5. Si fallo: registrar y continuar a la siguiente instancia

Para Tiny_3:
  - N_QPU_RUNS runs independientes (en lugar de 1) para obtener distribución estadística
  - Se prueban múltiples chain_strength values: "scaled" (default), un valor fijo basado en max bias
  - Se guardan energías individuales de cada sampleset en hoja "individual_energies"
  - Se corre LeapHybrid sobre Tiny_3 como referencia (en hoja "lh_reference")
  - Se resuelve Tiny_3 con Gurobi (MILP + QUBO) para certificar el óptimo (en hoja "gurobi_ground_truth")

Prerequisito: DWAVE_API_TOKEN en .env (o en el entorno).
  Si no está configurado, los bloques QPU/LH se saltan con un warning.
  La celda Gurobi corre igualmente sin token D-Wave.

Prerequisito opcional: Exp 2 completado (α*, β* en metadata).
  Si no está disponible, usa PENALTY_ALPHA y PENALTY_BETA como fallback.

Outputs:
  results/exp08_qpu_direct.xlsx
    hoja: embedding_results    (una fila por instancia — diagnósticos + resultado embedding)
    hoja: qpu_runs             (una fila por run QPU — solo si embedding tuvo éxito)
    hoja: individual_energies  (una fila por read individual del sampleset)
    hoja: lh_reference         (runs LeapHybrid sobre Tiny_3)
    hoja: gurobi_ground_truth  (MILP obj + QUBO ground energy + match flag para Tiny_3)
    hoja: metadata

Ejecución:
  Celda 1: SETUP
  Celda 2: LOAD instancias (genera Tiny_3 inline + carga Size_1 de instances.xlsx)
  Celda 3: BQM DIAGNOSTICS — n_vars, n_interactions, q_density, max_degree por instancia
  Celda 4: EMBEDDING ATTEMPTS — minorminer por instancia, registrar outcome
  Celda 5: QPU RUNS — someter al QPU instancias con embedding exitoso (N_QPU_RUNS para Tiny_3)
  Celda 6: LEAPHYBRID REFERENCE — correr LH sobre Tiny_3 directamente
  Celda 7: GUROBI GROUND TRUTH — MILP directo + QUBO linealizado (McCormick) sobre Tiny_3
  Celda 8: ANALYSIS — tabla resumen, comparación QPU vs LH vs óptimo, histograma de energías
"""

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
    ("dimod",           "dimod"),
    ("dwave-samplers",  "dwave.samplers"),
    ("dwave-system",    "dwave.system"),
    ("dwave-ocean-sdk", "minorminer"),
    ("dwave-cloud-client", "dwave.cloud"),
    ("openpyxl",        "openpyxl"),
    ("seaborn",         "seaborn"),
]
for _pip, _mod in _PKGS:
    _needs_install = _ilu.find_spec(_mod.split(".")[0]) is None
    if not _needs_install and _pip == "dwave-system":
        try:
            from dwave.system import DWaveSampler as _DWS  # noqa: F401
        except Exception:
            _needs_install = True
    if _needs_install:
        print(f"  instalando {_pip}...", end=" ", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("listo")
    else:
        print(f"  ok  {_pip}")

# Forzar import de DWaveSampler despues de instalacion
try:
    import importlib, dwave.system as _dws
    importlib.reload(_dws)
    from dwave.system import DWaveSampler as _DWS_CHECK  # noqa: F401
    print("  ok  DWaveSampler importable")
except Exception as _e:
    print(f"  AVISO: DWaveSampler no importable tras instalacion: {_e}")

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
            print("  AVISO: DWAVE_API_TOKEN no configurado — bloques QPU seran saltados.")
else:
    print("  ok  DWAVE_API_TOKEN ya en el entorno")

import numpy as np
import pandas as pd

from config import DEFAULT_SEED, PENALTY_ALPHA, PENALTY_BETA
from config import CONFLICT_SET_R

from experiments2.shared.run_id import new_run_uuid
from experiments2.shared.experiment_config import EXP8_QPU_INSTANCES
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
logger = logging.getLogger("exp08")

ensure_directories()

RUN_UUID  = new_run_uuid()
FILEPATH  = RESULTS_DIR / "exp08_qpu_direct.xlsx"
EXP2_PATH = RESULTS_DIR / "exp02_lagrange_calibration.xlsx"
EXP3_PATH = RESULTS_DIR / "exp03_solution_quality.xlsx"

SHEET_EMBED      = "embedding_results"
SHEET_QPU        = "qpu_runs"
SHEET_ENERGIES   = "individual_energies"
SHEET_LH_REF     = "lh_reference"

QPU_NUM_READS     = 1000
QPU_NUM_READS_MIN = 100

# Número de runs independientes para Tiny_3 (mod 1: múltiples runs)
N_QPU_RUNS = 10

# chain_strength values a probar para Tiny_3 (mod 3: tuning de chain strength)
# "scaled" = default D-Wave automático
# "auto_max_bias" = se calculará como max(|Q_ij|) en tiempo de ejecución
# valor fijo adicional para exploración
CHAIN_STRENGTH_VALUES = ["scaled", "auto_max_bias", 2.0]

logger.info("Exp 8 SETUP  run_uuid=%s  num_reads=%d  n_qpu_runs=%d", RUN_UUID, QPU_NUM_READS, N_QPU_RUNS)
logger.info("REPO_ROOT: %s", REPO_ROOT)
logger.info("Output: %s", FILEPATH)
logger.info("chain_strength values a probar: %s", CHAIN_STRENGTH_VALUES)

# alpha*, beta* con fallback
_meta2_8 = load_metadata(EXP2_PATH)
if _meta2_8 and "alpha_star" in _meta2_8 and "beta_star" in _meta2_8:
    alpha_star = float(_meta2_8["alpha_star"])
    beta_star  = float(_meta2_8["beta_star"])
    logger.info("alpha*=%.1f  beta*=%.1f  (cargados de Exp 2)", alpha_star, beta_star)
else:
    alpha_star = float(PENALTY_ALPHA)
    beta_star  = float(PENALTY_BETA)
    logger.warning(
        "Exp 2 metadata no disponible — usando fallback: alpha=%.1f beta=%.1f",
        alpha_star, beta_star,
    )

# Verificar token D-Wave
_dwave_token = os.environ.get("DWAVE_API_TOKEN", "")
_qpu_available = bool(_dwave_token)
if not _qpu_available:
    logger.warning("DWAVE_API_TOKEN no configurado — bloques QPU seran saltados.")
else:
    logger.info("DWAVE_API_TOKEN encontrado — QPU habilitado.")


# CELDA 2: LOAD instancias
# Tiny_3: generado inline con generate_nominations(n=3, seed=DEFAULT_SEED)
# Size_1: cargado de data/instances.xlsx eje "size"

from preprocessing import compute_feasible_slots
from qubo_builder import build_qubo
from config import generate_nominations

# ---- Tiny_3 (generado inline) ----
_tiny3_noms = generate_nominations(n=3, seed=DEFAULT_SEED)
# Determinar horizonte mínimo viable: max(r_j + p_j) + margen
_tiny3_T = int(_tiny3_noms["r_j"].max() + _tiny3_noms["p_j"].max() + 4)
_tiny3_T = max(_tiny3_T, int(_tiny3_noms["p_j"].sum()) + 2)

TINY3_INST: dict = {
    "instance_label": "Tiny_3",
    "N":              3,
    "T":              _tiny3_T,
    "rho_effective":  float("nan"),
    "r_j_distribution": "generated",
    "nominations":    _tiny3_noms,
}

logger.info(
    "Tiny_3 generado: N=%d  T=%d  vessels=%s",
    3, _tiny3_T, list(_tiny3_noms["vessel_id"]),
)

# ---- Size_1 (desde instances.xlsx) ----
try:
    _size_dict8 = load_instances_from_excel("size")
    SIZE1_INST  = _size_dict8["Size_1"]
    logger.info(
        "Size_1 cargado: N=%d  T=%d  rho=%.3f",
        SIZE1_INST["N"], SIZE1_INST["T"], SIZE1_INST["rho_effective"],
    )
except (FileNotFoundError, KeyError) as _exc_load:
    logger.warning("Size_1 no disponible: %s — se omitirá.", _exc_load)
    SIZE1_INST = None

# Lista de instancias en el orden de EXP8_QPU_INSTANCES = ["Tiny_3", "Size_1"]
INSTANCES_8: list[dict] = []
for _lbl8 in EXP8_QPU_INSTANCES:
    if _lbl8 == "Tiny_3":
        INSTANCES_8.append(TINY3_INST)
    elif _lbl8 == "Size_1" and SIZE1_INST is not None:
        INSTANCES_8.append(SIZE1_INST)
    else:
        logger.warning("Instancia %s no disponible — se omitirá.", _lbl8)

logger.info("Instancias para Exp 8: %s", [i["instance_label"] for i in INSTANCES_8])


# CELDA 3: BQM DIAGNOSTICS
# Para cada instancia: construir QUBO, registrar n_vars, n_interactions, q_density, max_degree.
# Almacena en _bqm_cache8 para reutilizar en Celdas 4-6.

_bqm_cache8: dict[str, dict] = {}

print("\n=== BQM DIAGNOSTICS ===")
print(f"{'Instancia':<12} {'N':>4} {'T':>4} {'n_vars':>8} {'n_inter':>9} {'q_dens':>8} {'max_deg':>8} {'max_bias':>10}")
print("-" * 68)

for _inst8 in INSTANCES_8:
    _lbl8    = _inst8["instance_label"]
    _noms8   = _inst8["nominations"].copy()
    _T8      = int(_inst8["T"])
    _N8      = int(_inst8["N"])

    _vdf8 = compute_feasible_slots(_noms8, horizon_slots=_T8)
    _bqm8, _, _, _, _ = build_qubo(_vdf8, alpha=alpha_star, beta=beta_star)

    _nv8  = len(_bqm8.variables)
    _ni8  = len(_bqm8.quadratic)
    _me8  = _nv8 * (_nv8 - 1) / 2 if _nv8 > 1 else 1
    _qd8  = round(_ni8 / _me8, 6)
    _md8  = (
        max(len(_bqm8.adj[_v]) for _v in _bqm8.variables)
        if _bqm8.variables else 0
    )

    # max_bias para cálculo de chain_strength (mod 3)
    _Q_dict8_diag, _ = _bqm8.to_qubo()
    _max_bias8 = float(max(abs(v) for v in _Q_dict8_diag.values())) if _Q_dict8_diag else 1.0

    _bqm_cache8[_lbl8] = {
        "bqm":         _bqm8,
        "vdf":         _vdf8,
        "n_vars":      _nv8,
        "n_inter":     _ni8,
        "q_density":   _qd8,
        "max_degree":  _md8,
        "max_bias":    _max_bias8,
        "inst":        _inst8,
    }

    print(f"{_lbl8:<12} {_N8:>4} {_T8:>4} {_nv8:>8} {_ni8:>9} {_qd8:>8.4f} {_md8:>8} {_max_bias8:>10.4f}")

logger.info("BQM cache8 construido: %d instancias.", len(_bqm_cache8))


# CELDA 4: EMBEDDING ATTEMPTS
# Para cada instancia: intentar minorminer.find_embedding sobre el grafo del QPU.
# Registra outcome en embedding_results sheet (append-safe por instance_label).
# Si DWAVE_API_TOKEN no está configurado, registra como embedding_attempted=False.
# Para Size_1: timeout controlado con max_no_improvement=10 (mod 4).

_existing_embed = load_existing_runs(FILEPATH, SHEET_EMBED)
_done_embed: set[str] = set()
if not _existing_embed.empty:
    _done_embed = set(_existing_embed["instance_label"].astype(str).tolist())

# Almacenar embeddings exitosos para Celda 5
_embeddings8: dict[str, object] = {}

# Parámetros de timeout por instancia (mod 4: timeout controlado para Size_1)
_EMBED_PARAMS: dict[str, dict] = {
    "Tiny_3":  {},                          # sin timeout — instancia pequeña
    "Size_1":  {"max_no_improvement": 10},  # falla rápido (~segundos) en vez de ~17 min
}

for _inst8 in INSTANCES_8:
    _lbl8   = _inst8["instance_label"]
    _N8     = int(_inst8["N"])
    _T8     = int(_inst8["T"])
    _cache8 = _bqm_cache8[_lbl8]
    _bqm8   = _cache8["bqm"]
    _nv8    = _cache8["n_vars"]
    _ni8    = _cache8["n_inter"]
    _qd8    = _cache8["q_density"]
    _md8    = _cache8["max_degree"]

    if _lbl8 in _done_embed:
        # Si la corrida previa tuvo éxito, re-correr minorminer para repoblar _embeddings8
        # (el objeto embedding no se persiste en xlsx).
        _prev_row = _existing_embed[_existing_embed["instance_label"] == _lbl8]
        _prev_success = (
            bool(_prev_row["embedding_success"].iloc[0])
            if not _prev_row.empty and "embedding_success" in _prev_row.columns
            else False
        )
        if _prev_success and _qpu_available:
            logger.info("  embed %s: ya registrado con éxito — re-corriendo minorminer para repoblar cache.", _lbl8)
            try:
                import minorminer
                from dwave.system import DWaveSampler as _DWS
                _dws_tmp = _DWS()
                _target_edgelist = _dws_tmp.edgelist
                _Q_dict, _ = _bqm8.to_qubo()
                _emb_kwargs = _EMBED_PARAMS.get(_lbl8, {})
                _embedding = minorminer.find_embedding(_Q_dict, _target_edgelist, **_emb_kwargs)
                if len(_embedding) > 0:
                    _embeddings8[_lbl8] = _embedding
                    logger.info("  embed %s: re-embedding exitoso.", _lbl8)
                else:
                    logger.warning("  embed %s: re-embedding falló — instancia será saltada en QPU.", _lbl8)
            except Exception as _exc_re:
                logger.error("  embed %s: excepción en re-embedding: %s", _lbl8, _exc_re)
        else:
            logger.info("  embed skip %s (ya registrado, sin éxito previo o QPU no disponible)", _lbl8)
        continue

    _embed_row: dict = {
        "exp_id":             "exp08",
        "run_uuid":           RUN_UUID,
        "instance_label":     _lbl8,
        "N":                  _N8,
        "T":                  _T8,
        "n_vars":             _nv8,
        "n_interactions":     _ni8,
        "q_density":          _qd8,
        "max_degree":         _md8,
        "embedding_attempted": False,
        "embedding_success":  False,
        "mean_chain_length":  float("nan"),
        "max_chain_length":   float("nan"),
        "embedding_time_s":   float("nan"),
        "embedding_timeout_param": str(_EMBED_PARAMS.get(_lbl8, {})),
        "qpu_attempted":      False,
        "n_qpu_reads":        0,
        "run_timestamp":      datetime.datetime.now().isoformat(),
    }

    if not _qpu_available:
        logger.warning("  embed %s: QPU no disponible (sin token) — registrando sin intento.", _lbl8)
        append_rows(FILEPATH, SHEET_EMBED, [_embed_row])
        _done_embed.add(_lbl8)
        continue

    # Intentar embedding
    _embed_row["embedding_attempted"] = True
    try:
        import minorminer
        from dwave.system import DWaveSampler as _DWS

        logger.info("  embed %s: obteniendo grafo QPU ...", _lbl8)
        _dws_tmp = _DWS()
        _target_edgelist = _dws_tmp.edgelist
        _target_graph    = _dws_tmp.adjacency

        # BQM → QUBO dict: {(v1, v2): bias}
        _Q_dict, _qubo_offset = _bqm8.to_qubo()

        _emb_kwargs = _EMBED_PARAMS.get(_lbl8, {})
        logger.info(
            "  embed %s: minorminer.find_embedding (n_vars=%d, n_qubo_terms=%d, params=%s) ...",
            _lbl8, _nv8, len(_Q_dict), _emb_kwargs,
        )
        _t0_emb = time.perf_counter()
        _embedding = minorminer.find_embedding(_Q_dict, _target_edgelist, **_emb_kwargs)
        _emb_time  = time.perf_counter() - _t0_emb

        _emb_success = len(_embedding) > 0

        if _emb_success:
            _chain_lengths = [len(chain) for chain in _embedding.values()]
            _mean_chain    = float(np.mean(_chain_lengths))
            _max_chain     = float(np.max(_chain_lengths))
            _embeddings8[_lbl8] = _embedding
            logger.info(
                "  embed %s: ÉXITO  mean_chain=%.1f  max_chain=%.0f  t=%.2fs",
                _lbl8, _mean_chain, _max_chain, _emb_time,
            )
        else:
            _mean_chain = float("nan")
            _max_chain  = float("nan")
            logger.warning("  embed %s: FALLO (embedding vacío)  t=%.2fs  — resultado registrado.", _lbl8, _emb_time)

        _embed_row.update({
            "embedding_success":  _emb_success,
            "mean_chain_length":  _mean_chain,
            "max_chain_length":   _max_chain,
            "embedding_time_s":   round(_emb_time, 3),
        })

    except Exception as _exc_emb:
        logger.error("  embed %s: excepción durante embedding: %s", _lbl8, _exc_emb)
        _embed_row.update({
            "embedding_success": False,
            "embedding_time_s":  float("nan"),
        })

    append_rows(FILEPATH, SHEET_EMBED, [_embed_row])
    _done_embed.add(_lbl8)

logger.info("Embedding attempts completo.")

# Resumen de embeddings
_df_embed_summary = load_existing_runs(FILEPATH, SHEET_EMBED)
if not _df_embed_summary.empty:
    print("\n=== EMBEDDING RESULTS ===")
    _cols_show = [
        "instance_label", "n_vars", "q_density", "max_degree",
        "embedding_attempted", "embedding_success",
        "mean_chain_length", "max_chain_length", "embedding_time_s",
        "embedding_timeout_param",
    ]
    _cols_avail = [c for c in _cols_show if c in _df_embed_summary.columns]
    print(_df_embed_summary[_cols_avail].to_string(index=False))


# CELDA 5: QPU RUNS
# Solo para instancias con embedding exitoso en Celda 4.
# Append-safe por (instance_label, run_id, chain_strength_label).
# Para Tiny_3: N_QPU_RUNS runs × len(CHAIN_STRENGTH_VALUES) combinaciones (mod 1 + mod 3).
# Para otras instancias: 1 run con chain_strength="scaled".
# Se guardan las energías individuales de cada sampleset (mod 5).
# Si DWAVE_API_TOKEN no disponible, saltar.

_existing_qpu = load_existing_runs(FILEPATH, SHEET_QPU)
_done_qpu8: set[tuple[str, int, str]] = set()
if not _existing_qpu.empty:
    _cs_col = "chain_strength_label" if "chain_strength_label" in _existing_qpu.columns else None
    for _, _row_qpu in _existing_qpu.iterrows():
        _cs_lbl = str(_row_qpu[_cs_col]) if _cs_col else "scaled"
        _done_qpu8.add((
            str(_row_qpu["instance_label"]),
            int(_row_qpu["run_id"]),
            _cs_lbl,
        ))

if not _qpu_available:
    logger.warning("QPU RUNS: DWAVE_API_TOKEN no configurado — sección saltada.")
elif not _embeddings8:
    logger.warning("QPU RUNS: ningún embedding exitoso — sección saltada.")
else:
    from solver import decode_schedule, check_feasibility

    try:
        from dwave.system import DWaveSampler as _DWaveSampler8
        from dwave.system import FixedEmbeddingComposite as _FixedEmbComp8
        import dimod as _dimod8

        _dws8 = _DWaveSampler8()

        for _inst8 in INSTANCES_8:
            _lbl8 = _inst8["instance_label"]
            if _lbl8 not in _embeddings8:
                logger.info("  QPU skip %s (sin embedding exitoso)", _lbl8)
                continue

            _cache8_q  = _bqm_cache8[_lbl8]
            _bqm8_q    = _cache8_q["bqm"]
            _vdf8_q    = _cache8_q["vdf"]
            _nv8_q     = _cache8_q["n_vars"]
            _ni8_q     = _cache8_q["n_inter"]
            _qd8_q     = _cache8_q["q_density"]
            _max_bias8_q = _cache8_q["max_bias"]
            _N8_q      = int(_inst8["N"])
            _T8_q      = int(_inst8["T"])
            _emb8_q    = _embeddings8[_lbl8]

            # Determinar runs y chain_strength values según instancia
            if _lbl8 == "Tiny_3":
                _n_runs_inst   = N_QPU_RUNS
                _cs_values_raw = CHAIN_STRENGTH_VALUES
            else:
                _n_runs_inst   = 1
                _cs_values_raw = ["scaled"]

            # Resolver "auto_max_bias" a valor numérico (mod 3)
            _cs_resolved: list[tuple[str, object]] = []
            for _cs_raw in _cs_values_raw:
                if _cs_raw == "scaled":
                    _cs_resolved.append(("scaled", None))  # None → omitir kwarg → default scaled
                elif _cs_raw == "auto_max_bias":
                    _cs_val = round(_max_bias8_q, 4)
                    _cs_resolved.append((f"auto_max_bias={_cs_val:.4f}", _cs_val))
                elif isinstance(_cs_raw, (int, float)):
                    _cs_resolved.append((f"fixed={_cs_raw}", float(_cs_raw)))
                else:
                    _cs_resolved.append((str(_cs_raw), None))

            for _cs_label, _cs_value in _cs_resolved:
                logger.info(
                    "  QPU %s  chain_strength=%s  n_runs=%d  n_vars=%d",
                    _lbl8, _cs_label, _n_runs_inst, _nv8_q,
                )

                # FixedEmbeddingComposite usa el embedding precomputado de Celda 4
                _sampler_qpu8 = _FixedEmbComp8(_dws8, embedding=_emb8_q)

                for _run_id8 in range(_n_runs_inst):
                    if (_lbl8, _run_id8, _cs_label) in _done_qpu8:
                        logger.info("  QPU skip %s run_id=%d cs=%s (ya completado)", _lbl8, _run_id8, _cs_label)
                        continue

                    _reads_this_run = QPU_NUM_READS
                    logger.info(
                        "  QPU %s run_id=%d/%d cs=%s (num_reads=%d) ...",
                        _lbl8, _run_id8, _n_runs_inst - 1, _cs_label, _reads_this_run,
                    )
                    try:
                        _t0_qpu8 = time.perf_counter()

                        # Construir kwargs de chain_strength (mod 3)
                        _sample_kwargs: dict = {"num_reads": _reads_this_run, "return_embedding": True}
                        if _cs_value is not None:
                            _sample_kwargs["chain_strength"] = _cs_value

                        _ss8 = _sampler_qpu8.sample(_bqm8_q, **_sample_kwargs)
                        _wall_qpu8 = time.perf_counter() - _t0_qpu8

                        _dw8 = extract_solver_timing(_ss8)

                        # chain_break_fraction (por read)
                        _cbf8_vals = (
                            _ss8.record["chain_break_fraction"]
                            if "chain_break_fraction" in _ss8.record.dtype.names
                            else []
                        )
                        _cbf8_mean = float(np.mean(_cbf8_vals)) if len(_cbf8_vals) > 0 else float("nan")
                        _cbf8_max  = float(np.max(_cbf8_vals))  if len(_cbf8_vals) > 0 else float("nan")

                        _best_sample8 = _ss8.first.sample
                        _best_energy8 = float(_ss8.first.energy)

                        _sched8   = decode_schedule(_best_sample8, _vdf8_q)
                        _fres8    = check_feasibility(_sched8, _vdf8_q)
                        _is_feas8 = bool(_fres8["is_feasible"])
                        _obj8     = float(_fres8["total_weighted_tardiness"]) if _is_feas8 else float("nan")

                        _qpu_row8 = {
                            "exp_id":               "exp08",
                            "run_uuid":             RUN_UUID,
                            "instance_label":       _lbl8,
                            "run_id":               _run_id8,
                            "chain_strength_label": _cs_label,
                            "chain_strength_value": _cs_value if _cs_value is not None else "scaled",
                            "feasible":             _is_feas8,
                            "obj_value":            _obj8,
                            "best_energy":          _best_energy8,
                            "chain_break_fraction_mean": _cbf8_mean,
                            "chain_break_fraction_max":  _cbf8_max,
                            "qpu_sampling_time_us": _dw8["qpu_sampling_time_us"],
                            "qpu_access_time_us":   _dw8["qpu_access_time_us"],
                            "wall_time_s":          round(_wall_qpu8, 3),
                            "n_reads":              _reads_this_run,
                            "run_timestamp":        datetime.datetime.now().isoformat(),
                        }
                        append_rows(FILEPATH, SHEET_QPU, [_qpu_row8])
                        _done_qpu8.add((_lbl8, _run_id8, _cs_label))

                        # Guardar energías individuales del sampleset (mod 5)
                        _energy_rows8: list[dict] = []
                        for _read_idx, (_sample_r, _energy_r, _num_oc_r) in enumerate(
                            zip(_ss8.record["sample"], _ss8.record["energy"], _ss8.record["num_occurrences"])
                        ):
                            _cbf_r = (
                                float(_ss8.record["chain_break_fraction"][_read_idx])
                                if "chain_break_fraction" in _ss8.record.dtype.names
                                else float("nan")
                            )
                            # Decodificar factibilidad del read individual
                            try:
                                _sample_dict_r = dict(zip(_ss8.variables, _sample_r))
                                _sched_r   = decode_schedule(_sample_dict_r, _vdf8_q)
                                _fres_r    = check_feasibility(_sched_r, _vdf8_q)
                                _feas_r    = bool(_fres_r["is_feasible"])
                                _obj_r     = float(_fres_r["total_weighted_tardiness"]) if _feas_r else float("nan")
                            except Exception:
                                _feas_r = False
                                _obj_r  = float("nan")

                            _energy_rows8.append({
                                "exp_id":               "exp08",
                                "run_uuid":             RUN_UUID,
                                "instance_label":       _lbl8,
                                "run_id":               _run_id8,
                                "chain_strength_label": _cs_label,
                                "read_index":           _read_idx,
                                "energy":               float(_energy_r),
                                "num_occurrences":      int(_num_oc_r),
                                "chain_break_fraction": _cbf_r,
                                "feasible":             _feas_r,
                                "obj_value":            _obj_r,
                            })
                        if _energy_rows8:
                            append_rows(FILEPATH, SHEET_ENERGIES, _energy_rows8)
                            logger.info(
                                "    %d energías individuales guardadas para %s run_id=%d cs=%s",
                                len(_energy_rows8), _lbl8, _run_id8, _cs_label,
                            )

                        # Actualizar embedding_results: marcar qpu_attempted=True, n_qpu_reads
                        _df_embed_upd = load_existing_runs(FILEPATH, SHEET_EMBED)
                        if not _df_embed_upd.empty:
                            _mask_upd = _df_embed_upd["instance_label"] == _lbl8
                            if _mask_upd.any():
                                _df_embed_upd.loc[_mask_upd, "qpu_attempted"] = True
                                _df_embed_upd.loc[_mask_upd, "n_qpu_reads"]   = _reads_this_run
                                import openpyxl as _opxl8
                                _wb8 = _opxl8.load_workbook(FILEPATH)
                                if SHEET_EMBED in _wb8.sheetnames:
                                    del _wb8[SHEET_EMBED]
                                _wb8.save(FILEPATH)
                                append_rows(FILEPATH, SHEET_EMBED,
                                            _df_embed_upd.to_dict(orient="records"))

                        logger.info(
                            "    QPU %s run_id=%d cs=%s  feasible=%s  obj=%.2f  "
                            "cbf_mean=%.3f  cbf_max=%.3f  wall=%.1fs",
                            _lbl8, _run_id8, _cs_label, _is_feas8,
                            _obj8 if not np.isnan(_obj8) else -1,
                            _cbf8_mean if not np.isnan(_cbf8_mean) else -1,
                            _cbf8_max  if not np.isnan(_cbf8_max)  else -1,
                            _wall_qpu8,
                        )

                    except Exception as _exc_qpu8:
                        logger.error("  QPU %s run_id=%d cs=%s falló: %s", _lbl8, _run_id8, _cs_label, _exc_qpu8)

    except ImportError as _imp_err8:
        logger.warning("  QPU RUNS: no se pudo importar dwave.system: %s", _imp_err8)

logger.info("QPU runs completo.")


# CELDA 6: LEAPHYBRID REFERENCE sobre Tiny_3
# Corre LeapHybrid directamente sobre Tiny_3 para tener referencia QPU vs LH (mod 2).
# Append-safe por run_id; N_QPU_RUNS runs.

_existing_lh_ref = load_existing_runs(FILEPATH, SHEET_LH_REF)
_done_lh_ref8: set[int] = set()
if not _existing_lh_ref.empty:
    _done_lh_ref8 = set(_existing_lh_ref["run_id"].astype(int).tolist())

if not _qpu_available:
    logger.warning("LH REFERENCE: DWAVE_API_TOKEN no configurado — sección saltada.")
elif "Tiny_3" not in _bqm_cache8:
    logger.warning("LH REFERENCE: Tiny_3 no disponible en BQM cache — sección saltada.")
else:
    try:
        from dwave.system import LeapHybridSampler as _LeapHS8
        from solver import decode_schedule as _decode8_lh, check_feasibility as _check8_lh

        _lh_sampler8 = _LeapHS8()
        _cache8_lh   = _bqm_cache8["Tiny_3"]
        _bqm8_lh     = _cache8_lh["bqm"]
        _vdf8_lh     = _cache8_lh["vdf"]

        print(f"\n=== LEAPHYBRID REFERENCE (Tiny_3, {N_QPU_RUNS} runs) ===")

        for _lh_run_id in range(N_QPU_RUNS):
            if _lh_run_id in _done_lh_ref8:
                logger.info("  LH skip Tiny_3 run_id=%d (ya completado)", _lh_run_id)
                continue

            logger.info("  LH Tiny_3 run_id=%d/%d ...", _lh_run_id, N_QPU_RUNS - 1)
            try:
                _t0_lh8 = time.perf_counter()
                _ss8_lh = _lh_sampler8.sample(_bqm8_lh)
                _wall_lh8 = time.perf_counter() - _t0_lh8

                _best_sample_lh = _ss8_lh.first.sample
                _best_energy_lh = float(_ss8_lh.first.energy)

                _sched_lh   = _decode8_lh(_best_sample_lh, _vdf8_lh)
                _fres_lh    = _check8_lh(_sched_lh, _vdf8_lh)
                _is_feas_lh = bool(_fres_lh["is_feasible"])
                _obj_lh     = float(_fres_lh["total_weighted_tardiness"]) if _is_feas_lh else float("nan")

                _dw_lh = extract_solver_timing(_ss8_lh)

                _lh_row8 = {
                    "exp_id":           "exp08",
                    "run_uuid":         RUN_UUID,
                    "instance_label":   "Tiny_3",
                    "solver":           "LeapHybrid",
                    "run_id":           _lh_run_id,
                    "feasible":         _is_feas_lh,
                    "obj_value":        _obj_lh,
                    "best_energy":      _best_energy_lh,
                    "qpu_access_time_us": _dw_lh.get("qpu_access_time_us", float("nan")),
                    "wall_time_s":      round(_wall_lh8, 3),
                    "run_timestamp":    datetime.datetime.now().isoformat(),
                }
                append_rows(FILEPATH, SHEET_LH_REF, [_lh_row8])
                _done_lh_ref8.add(_lh_run_id)

                logger.info(
                    "    LH Tiny_3 run_id=%d  feasible=%s  obj=%.2f  best_energy=%.4f  wall=%.1fs",
                    _lh_run_id, _is_feas_lh,
                    _obj_lh if not np.isnan(_obj_lh) else -1,
                    _best_energy_lh, _wall_lh8,
                )

            except Exception as _exc_lh8:
                logger.error("  LH Tiny_3 run_id=%d falló: %s", _lh_run_id, _exc_lh8)

        # Resumen LH
        _df_lh_ref_final = load_existing_runs(FILEPATH, SHEET_LH_REF)
        if not _df_lh_ref_final.empty:
            _tiny3_lh = _df_lh_ref_final[_df_lh_ref_final["instance_label"] == "Tiny_3"]
            if not _tiny3_lh.empty:
                _lh_feas_rate_8 = _tiny3_lh["feasible"].astype(bool).mean()
                _lh_obj_mean_8  = _tiny3_lh.loc[_tiny3_lh["feasible"].astype(bool), "obj_value"].mean()
                _lh_obj_std_8   = _tiny3_lh.loc[_tiny3_lh["feasible"].astype(bool), "obj_value"].std()
                print(f"  LeapHybrid Tiny_3: feasibility_rate={_lh_feas_rate_8:.2f}  "
                      f"obj_mean={_lh_obj_mean_8:.2f}  obj_std={_lh_obj_std_8:.2f}  "
                      f"n_runs={len(_tiny3_lh)}")

    except ImportError as _imp_lh8:
        logger.warning("  LH REFERENCE: no se pudo importar LeapHybridSampler: %s", _imp_lh8)

logger.info("LeapHybrid reference completo.")


# CELDA 7: GUROBI GROUND TRUTH para Tiny_3
# Resuelve Tiny_3 con (a) MILP de scheduling directo y (b) QUBO linealizado con Gurobi.
# Ambos producen el óptimo certificado. (a) verifica que el modelo de scheduling es correcto;
# (b) establece el ground state energético real del QUBO, para calcular RPD del QPU y confirmar
# si LeapHybrid encontró el óptimo.
# Append-safe: si ya existe la fila de Tiny_3 en gurobi_ground_truth, se saltea.
# Sin token D-Wave: esta celda corre igualmente (solo necesita Gurobi).

SHEET_GUROBI_GT = "gurobi_ground_truth"

_existing_gt = load_existing_runs(FILEPATH, SHEET_GUROBI_GT)
_done_gt: set[str] = set()
if not _existing_gt.empty and "instance_label" in _existing_gt.columns:
    _done_gt = set(_existing_gt["instance_label"].astype(str).tolist())

if "Tiny_3" in _done_gt:
    logger.info("Gurobi GT: Tiny_3 ya registrado — saltando.")
    _df_gt_final = _existing_gt
else:
    try:
        import gurobipy as _gp8
        from gurobipy import GRB as _GRB8
        from solver import decode_schedule as _decode8_gt, check_feasibility as _check8_gt

        _cache8_gt = _bqm_cache8["Tiny_3"]
        _vdf8_gt   = _cache8_gt["vdf"]
        _bqm8_gt   = _cache8_gt["bqm"]
        _noms8_gt  = TINY3_INST["nominations"].copy()
        _T8_gt     = int(TINY3_INST["T"])
        _N8_gt     = int(TINY3_INST["N"])

        # ----------------------------------------------------------------
        # (a) MILP directo de scheduling (misma formulación que Exp 1)
        # ----------------------------------------------------------------
        print("\n=== GUROBI GROUND TRUTH (Tiny_3) ===")
        print("(a) MILP de scheduling directo ...")

        _env_a = _gp8.Env(empty=True)
        _env_a.setParam("OutputFlag", 0)
        for _grb_k, _grb_env in [
            ("WLSAccessID", os.environ.get("GRB_WLSACCESSID")),
            ("WLSSecret",   os.environ.get("GRB_WLSSECRET")),
            ("LicenseID",   os.environ.get("GRB_LICENSEID")),
        ]:
            if _grb_env:
                _env_a.setParam(_grb_k, int(_grb_env) if _grb_k == "LicenseID" else _grb_env)
        _env_a.start()

        _m_a = _gp8.Model(env=_env_a)
        _m_a.Params.Threads   = 2
        _m_a.Params.TimeLimit = 600
        _m_a.Params.MIPGap    = 0.0
        _m_a.Params.Seed      = 0

        # Variables x[j, t] ∈ {0,1}: buque j asignado a slot t
        _vdf_rows = _vdf8_gt.to_dict(orient="records")
        _x_a = {}
        for _row_a in _vdf_rows:
            _j_a = _row_a["vessel_id"]
            _t_a = int(_row_a["slot"])
            _x_a[(_j_a, _t_a)] = _m_a.addVar(vtype=_GRB8.BINARY, name=f"x_{_j_a}_{_t_a}")
        _m_a.update()

        # Restricción 1: cada buque asignado a exactamente un slot
        for _jj_a in _vdf8_gt["vessel_id"].unique():
            _slots_j = [_t for (_jj2, _t) in _x_a if _jj2 == _jj_a]
            _m_a.addConstr(
                _gp8.quicksum(_x_a[(_jj_a, _t)] for _t in _slots_j) == 1,
                name=f"assign_{_jj_a}",
            )

        # Restricción 2: no solapamiento — cada slot usa máximo 1 buque
        _all_slots = sorted(_vdf8_gt["slot"].unique())
        for _tt_a in _all_slots:
            _keys_t = [(_jj2, _tt_a) for (_jj2, _tt2) in _x_a if _tt2 == _tt_a]
            if len(_keys_t) > 1:
                _m_a.addConstr(
                    _gp8.quicksum(_x_a[k] for k in _keys_t) <= 1,
                    name=f"no_overlap_{_tt_a}",
                )

        # Objetivo: minimizar tardanza ponderada Σ w_j * max(0, C_j - d_j)
        _obj_a = _gp8.LinExpr()
        for _row_a in _vdf_rows:
            _j_a  = _row_a["vessel_id"]
            _t_a  = int(_row_a["slot"])
            _pj_a = int(_row_a.get("p_j", 1))
            _dj_a = float(_row_a.get("d_j", _t_a))
            _wj_a = float(_row_a.get("w_j", 1.0))
            _Cj_a = _t_a + _pj_a  # completion time = start + processing
            _tard_a = max(0.0, _Cj_a - _dj_a)
            _obj_a.add(_x_a[(_j_a, _t_a)], _wj_a * _tard_a)
        _m_a.setObjective(_obj_a, _GRB8.MINIMIZE)
        _m_a.update()

        _t0_a = time.perf_counter()
        _m_a.optimize()
        _wall_a = time.perf_counter() - _t0_a

        _status_a = {_GRB8.OPTIMAL: "Optimal", _GRB8.TIME_LIMIT: "TimeLimit",
                     _GRB8.INFEASIBLE: "Infeasible"}.get(_m_a.Status, f"Unknown_{_m_a.Status}")
        _milp_obj_gt   = float(_m_a.ObjVal) if _m_a.SolCount > 0 else float("nan")
        _milp_bound_gt = float(_m_a.ObjBound) if _m_a.SolCount > 0 else float("nan")
        _milp_gap_gt   = float(_m_a.MIPGap) * 100.0 if _m_a.SolCount > 0 else float("nan")
        _milp_n_vars_gt = len(_x_a)

        print(f"  MILP status={_status_a}  obj={_milp_obj_gt:.4f}  "
              f"bound={_milp_bound_gt:.4f}  gap={_milp_gap_gt:.4f}%  "
              f"n_vars={_milp_n_vars_gt}  wall={_wall_a:.2f}s")

        _m_a.dispose()
        _env_a.dispose()

        # ----------------------------------------------------------------
        # (b) QUBO linealizado con Gurobi — ground state energético + obj
        # ----------------------------------------------------------------
        print("(b) QUBO como BQP linealizado (McCormick) ...")

        import gc as _gc8
        _var_list_gt = list(_bqm8_gt.variables)
        _n_vars_qubo_gt = len(_var_list_gt)
        _var_idx_gt = {v: i for i, v in enumerate(_var_list_gt)}

        _env_b = _gp8.Env(empty=True)
        _env_b.setParam("OutputFlag", 0)
        for _grb_k, _grb_env in [
            ("WLSAccessID", os.environ.get("GRB_WLSACCESSID")),
            ("WLSSecret",   os.environ.get("GRB_WLSSECRET")),
            ("LicenseID",   os.environ.get("GRB_LICENSEID")),
        ]:
            if _grb_env:
                _env_b.setParam(_grb_k, int(_grb_env) if _grb_k == "LicenseID" else _grb_env)
        _env_b.start()

        _m_b = _gp8.Model(env=_env_b)
        _m_b.Params.Threads   = 2
        _m_b.Params.TimeLimit = 600
        _m_b.Params.MIPGap    = 0.0
        _m_b.Params.Seed      = 0

        _x_b = [_m_b.addVar(vtype=_GRB8.BINARY, name=f"x{_i}") for _i in range(_n_vars_qubo_gt)]
        _m_b.update()

        _obj_b = _gp8.LinExpr()
        for _v_b, _bias_b in _bqm8_gt.linear.items():
            _obj_b.add(_x_b[_var_idx_gt[_v_b]], float(_bias_b))

        # Linealización McCormick exacta para variables binarias: z_ij = x_i * x_j
        for (_v1_b, _v2_b), _bias_q in _bqm8_gt.quadratic.items():
            _i_b = _var_idx_gt[_v1_b]
            _j_b = _var_idx_gt[_v2_b]
            _z_b = _m_b.addVar(vtype=_GRB8.BINARY, name=f"z{_i_b}_{_j_b}")
            _m_b.addConstr(_z_b <= _x_b[_i_b])
            _m_b.addConstr(_z_b <= _x_b[_j_b])
            _m_b.addConstr(_z_b >= _x_b[_i_b] + _x_b[_j_b] - 1)
            _obj_b.add(_z_b, float(_bias_q))

        _m_b.setObjective(_obj_b, _GRB8.MINIMIZE)
        _m_b.update()

        _t0_b = time.perf_counter()
        _m_b.optimize()
        _wall_b = time.perf_counter() - _t0_b

        _status_b = {_GRB8.OPTIMAL: "Optimal", _GRB8.TIME_LIMIT: "TimeLimit",
                     _GRB8.INFEASIBLE: "Infeasible"}.get(_m_b.Status, f"Unknown_{_m_b.Status}")

        _qubo_ground_energy_gt = float(_m_b.ObjVal)  if _m_b.SolCount > 0 else float("nan")
        _qubo_bound_gt         = float(_m_b.ObjBound) if _m_b.SolCount > 0 else float("nan")
        _qubo_gap_gt           = float(_m_b.MIPGap) * 100.0 if _m_b.SolCount > 0 else float("nan")
        _n_vars_bqp_gt         = _n_vars_qubo_gt + len(_bqm8_gt.quadratic)

        # Decodificar la solución BQP para verificar factibilidad y obj de scheduling
        _qubo_gt_obj    = float("nan")
        _qubo_gt_feas   = False
        if _m_b.SolCount > 0:
            try:
                _sol_b = {v: round(_x_b[_var_idx_gt[v]].X) for v in _var_list_gt}
                _sched_b = _decode8_gt(_sol_b, _vdf8_gt)
                _fres_b  = _check8_gt(_sched_b, _vdf8_gt)
                _qubo_gt_feas = bool(_fres_b["is_feasible"])
                if _qubo_gt_feas:
                    _qubo_gt_obj = float(_fres_b["total_weighted_tardiness"])
            except Exception as _exc_dec_b:
                logger.warning("  Gurobi BQP: error al decodificar solución: %s", _exc_dec_b)

        _match_gt = (
            _status_a == "Optimal" and _status_b == "Optimal"
            and not np.isnan(_milp_obj_gt) and not np.isnan(_qubo_gt_obj)
            and abs(_qubo_gt_obj - _milp_obj_gt) <= 1e-4
        )

        print(f"  BQP  status={_status_b}  ground_energy={_qubo_ground_energy_gt:.4f}  "
              f"scheduling_obj={_qubo_gt_obj:.4f}  feasible={_qubo_gt_feas}  "
              f"bound={_qubo_bound_gt:.4f}  n_vars={_n_vars_bqp_gt}  wall={_wall_b:.2f}s")
        print(f"  MILP == QUBO match: {_match_gt}  "
              f"(MILP obj={_milp_obj_gt:.4f}, QUBO sched obj={_qubo_gt_obj:.4f})")

        _m_b.dispose()
        _env_b.dispose()
        _gc8.collect()

        _gt_row = {
            "exp_id":                 "exp08",
            "run_uuid":               RUN_UUID,
            "instance_label":         "Tiny_3",
            "N":                      _N8_gt,
            "T":                      _T8_gt,
            # MILP directo
            "milp_status":            _status_a,
            "milp_obj":               _milp_obj_gt,
            "milp_best_bound":        _milp_bound_gt,
            "milp_mip_gap_pct":       round(_milp_gap_gt, 6) if not np.isnan(_milp_gap_gt) else float("nan"),
            "milp_n_vars":            _milp_n_vars_gt,
            "milp_wall_time_s":       round(_wall_a, 3),
            # QUBO linealizado
            "qubo_bqp_status":        _status_b,
            "qubo_ground_energy":     _qubo_ground_energy_gt,
            "qubo_best_bound":        _qubo_bound_gt,
            "qubo_mip_gap_pct":       round(_qubo_gap_gt, 6) if not np.isnan(_qubo_gap_gt) else float("nan"),
            "qubo_scheduling_obj":    _qubo_gt_obj,
            "qubo_feasible":          _qubo_gt_feas,
            "qubo_n_vars_bqp":        _n_vars_bqp_gt,
            "qubo_n_vars_original":   _n_vars_qubo_gt,
            "qubo_wall_time_s":       round(_wall_b, 3),
            # Equivalencia
            "milp_qubo_match":        _match_gt,
            "alpha":                  alpha_star,
            "beta":                   beta_star,
            "run_timestamp":          datetime.datetime.now().isoformat(),
        }
        append_rows(FILEPATH, SHEET_GUROBI_GT, [_gt_row])
        _df_gt_final = load_existing_runs(FILEPATH, SHEET_GUROBI_GT)

        logger.info(
            "Gurobi GT Tiny_3: MILP obj=%.4f  QUBO ground_energy=%.4f  "
            "QUBO sched_obj=%.4f  match=%s",
            _milp_obj_gt, _qubo_ground_energy_gt, _qubo_gt_obj, _match_gt,
        )

    except ImportError:
        logger.warning("Gurobi GT: gurobipy no disponible — celda saltada. "
                       "Instalar con: pip install gurobipy")
        _df_gt_final = _existing_gt
    except Exception as _exc_gt:
        logger.error("Gurobi GT: excepción inesperada: %s", _exc_gt)
        _df_gt_final = _existing_gt

logger.info("Gurobi Ground Truth completo.")


# CELDA 8: ANALYSIS — tabla resumen + comparación QPU vs LH vs óptimo + histograma energías individuales

_df_embed_final    = load_existing_runs(FILEPATH, SHEET_EMBED)
_df_qpu_final      = load_existing_runs(FILEPATH, SHEET_QPU)
_df_energies_final = load_existing_runs(FILEPATH, SHEET_ENERGIES)
_df_lh_ref_final   = load_existing_runs(FILEPATH, SHEET_LH_REF)
_df_gt_final       = load_existing_runs(FILEPATH, SHEET_GUROBI_GT)

print("\n" + "=" * 70)
print("  EXP 8 — QPU DIRECTO: RESUMEN")
print("=" * 70)

if not _df_embed_final.empty:
    print("\n--- Embedding Results ---")
    _embed_cols = [
        "instance_label", "N", "n_vars", "q_density", "max_degree",
        "embedding_attempted", "embedding_success",
        "mean_chain_length", "max_chain_length", "embedding_time_s",
        "embedding_timeout_param", "qpu_attempted", "n_qpu_reads",
    ]
    _show_cols = [c for c in _embed_cols if c in _df_embed_final.columns]
    print(_df_embed_final[_show_cols].to_string(index=False))
else:
    print("  (sin datos de embedding)")

# --- Gurobi Ground Truth (Tiny_3) ---
# Extraer milp_obj y qubo_ground_energy para usar en RPD más abajo
_gt_milp_obj_8:          float = float("nan")
_gt_qubo_ground_energy_8: float = float("nan")
_gt_qubo_sched_obj_8:     float = float("nan")

if not _df_gt_final.empty:
    print("\n--- Gurobi Ground Truth (Tiny_3) ---")
    _gt_cols = [
        "instance_label",
        "milp_status", "milp_obj", "milp_best_bound", "milp_mip_gap_pct", "milp_wall_time_s",
        "qubo_bqp_status", "qubo_ground_energy", "qubo_scheduling_obj", "qubo_feasible",
        "qubo_mip_gap_pct", "qubo_wall_time_s", "milp_qubo_match",
    ]
    _show_gt = [c for c in _gt_cols if c in _df_gt_final.columns]
    print(_df_gt_final[_show_gt].to_string(index=False))

    _gt_tiny3 = _df_gt_final[_df_gt_final["instance_label"] == "Tiny_3"]
    if not _gt_tiny3.empty:
        _gt_milp_obj_8           = float(_gt_tiny3["milp_obj"].iloc[0])           if "milp_obj" in _gt_tiny3.columns else float("nan")
        _gt_qubo_ground_energy_8 = float(_gt_tiny3["qubo_ground_energy"].iloc[0]) if "qubo_ground_energy" in _gt_tiny3.columns else float("nan")
        _gt_qubo_sched_obj_8     = float(_gt_tiny3["qubo_scheduling_obj"].iloc[0]) if "qubo_scheduling_obj" in _gt_tiny3.columns else float("nan")
        _gt_match_8              = bool(_gt_tiny3["milp_qubo_match"].iloc[0])      if "milp_qubo_match" in _gt_tiny3.columns else False

        print(f"\n  ► Óptimo certificado (MILP):  obj = {_gt_milp_obj_8:.4f}")
        print(f"  ► QUBO ground state (Gurobi): energy = {_gt_qubo_ground_energy_8:.4f}  "
              f"sched_obj = {_gt_qubo_sched_obj_8:.4f}  match = {_gt_match_8}")
else:
    print("\n  (Gurobi GT no disponible — ejecutar Celda 7)")

if not _df_qpu_final.empty:
    print("\n--- QPU Run Results (resumen por instancia × chain_strength) ---")
    _cs_col_exists = "chain_break_fraction_mean" in _df_qpu_final.columns
    _group_cols_q = (
        ["instance_label", "chain_strength_label"]
        if "chain_strength_label" in _df_qpu_final.columns
        else ["instance_label"]
    )
    _qpu_summary = (
        _df_qpu_final
        .groupby(_group_cols_q)
        .agg(
            n_runs           = ("run_id",                    "count"),
            feasibility_rate = ("feasible",                  lambda x: x.astype(bool).mean()),
            obj_mean         = ("obj_value",                  "mean"),
            obj_std          = ("obj_value",                  "std"),
            best_energy_min  = ("best_energy",                "min"),
            best_energy_mean = ("best_energy",                "mean"),
            cbf_mean         = ("chain_break_fraction_mean",  "mean"),
        )
        .reset_index()
    )
    # Añadir RPD vs óptimo MILP y vs ground energy del QUBO si disponibles
    if not np.isnan(_gt_milp_obj_8) and _gt_milp_obj_8 > 0:
        _qpu_summary["rpd_vs_milp_pct"] = _qpu_summary["obj_mean"].apply(
            lambda o: round(100.0 * (o - _gt_milp_obj_8) / _gt_milp_obj_8, 2) if not np.isnan(o) else float("nan")
        )
    if not np.isnan(_gt_qubo_ground_energy_8):
        _qpu_summary["energy_gap_vs_ground"] = _qpu_summary["best_energy_mean"].apply(
            lambda e: round(e - _gt_qubo_ground_energy_8, 4) if not np.isnan(e) else float("nan")
        )
    print(_qpu_summary.to_string(index=False))

    print("\n--- QPU Run Results (detalle) ---")
    _qpu_cols = [
        "instance_label", "run_id", "chain_strength_label", "feasible",
        "obj_value", "best_energy", "chain_break_fraction_mean", "chain_break_fraction_max",
        "qpu_sampling_time_us", "qpu_access_time_us", "wall_time_s",
    ]
    _show_qpu = [c for c in _qpu_cols if c in _df_qpu_final.columns]
    print(_df_qpu_final[_show_qpu].to_string(index=False))
else:
    print("  (sin runs QPU — embedding falló o QPU no disponible)")

# LeapHybrid Tiny_3 reference
if not _df_lh_ref_final.empty:
    print("\n--- LeapHybrid Reference (Tiny_3) ---")
    _lh_cols = ["instance_label", "run_id", "feasible", "obj_value", "best_energy", "wall_time_s"]
    _show_lh = [c for c in _lh_cols if c in _df_lh_ref_final.columns]
    print(_df_lh_ref_final[_show_lh].to_string(index=False))

    # Resumen estadístico LH vs óptimo
    _lh_tiny3_an = _df_lh_ref_final[_df_lh_ref_final["instance_label"] == "Tiny_3"]
    if not _lh_tiny3_an.empty:
        _lh_feas_an   = _lh_tiny3_an["feasible"].astype(bool).mean()
        _lh_feas_rows = _lh_tiny3_an.loc[_lh_tiny3_an["feasible"].astype(bool), "obj_value"]
        _lh_obj_mean_an = float(_lh_feas_rows.mean()) if not _lh_feas_rows.empty else float("nan")
        _lh_obj_std_an  = float(_lh_feas_rows.std())  if not _lh_feas_rows.empty else float("nan")
        _lh_rpd_milp = (
            100.0 * (_lh_obj_mean_an - _gt_milp_obj_8) / _gt_milp_obj_8
            if not np.isnan(_lh_obj_mean_an) and not np.isnan(_gt_milp_obj_8) and _gt_milp_obj_8 > 0
            else float("nan")
        )
        _lh_found_opt = (
            not np.isnan(_lh_obj_mean_an) and not np.isnan(_gt_milp_obj_8)
            and abs(_lh_obj_mean_an - _gt_milp_obj_8) <= 1e-4
        )
        print(f"\n  ► LH Tiny_3: feasibility_rate={_lh_feas_an:.2f}  "
              f"obj_mean={_lh_obj_mean_an:.4f}  obj_std={_lh_obj_std_an:.4f}  "
              f"rpd_vs_milp={_lh_rpd_milp:.2f}%  found_optimal={_lh_found_opt}")

# Comparación con LeapHybrid externo (Exp 3) si disponible
_lh_ext_available = False
_lh_ext8 = pd.DataFrame()
try:
    _df_lh_ext8 = pd.concat(
        [load_existing_runs(EXP3_PATH, sh) for sh in ("size_axis", "dens_axis", "sa_baseline")],
        ignore_index=True,
    )
    if not _df_lh_ext8.empty:
        _lh_ext8 = (
            _df_lh_ext8[
                (_df_lh_ext8["solver"] == "LeapHybrid") &
                (_df_lh_ext8["instance_label"].isin([i["instance_label"] for i in INSTANCES_8]))
            ]
            .groupby("instance_label")
            .agg(
                lh_feas_rate = ("feasible", lambda x: x.astype(bool).mean()),
                lh_obj_mean  = ("obj_value", "mean"),
                lh_obj_std   = ("obj_value", "std"),
                lh_n_runs    = ("run_id", "count"),
            )
            .reset_index()
        )
        if not _lh_ext8.empty:
            _lh_ext_available = True
            print("\n--- Referencia LeapHybrid externo (Exp 3) ---")
            print(_lh_ext8.to_string(index=False))
except Exception as _exc_lh_ext:
    logger.warning("No se pudo cargar referencia LH externa de Exp 3: %s", _exc_lh_ext)

# Tabla comparativa QPU vs LH vs Óptimo
_any_lh_available = not _df_lh_ref_final.empty or _lh_ext_available
if not _df_qpu_final.empty and (_any_lh_available or not np.isnan(_gt_milp_obj_8)):
    print("\n--- Comparación QPU vs LeapHybrid vs Óptimo (Tiny_3) ---")
    _comp_rows8: list[dict] = []
    for _inst8 in INSTANCES_8:
        _lbl8_c  = _inst8["instance_label"]
        _qpu_sub = _df_qpu_final[_df_qpu_final["instance_label"] == _lbl8_c]

        # LH: interno (Celda 6) tiene prioridad; fallback Exp 3
        if not _df_lh_ref_final.empty:
            _lh_sub_int   = _df_lh_ref_final[_df_lh_ref_final["instance_label"] == _lbl8_c]
            _lh_feas_c    = float(_lh_sub_int["feasible"].astype(bool).mean()) if not _lh_sub_int.empty else float("nan")
            _lh_fv        = _lh_sub_int.loc[_lh_sub_int["feasible"].astype(bool), "obj_value"]
            _lh_obj_c     = float(_lh_fv.mean()) if not _lh_fv.empty else float("nan")
            _lh_source    = "internal_lh"
        elif _lh_ext_available and not _lh_ext8.empty:
            _lh_sub_ext = _lh_ext8[_lh_ext8["instance_label"] == _lbl8_c]
            _lh_feas_c  = float(_lh_sub_ext["lh_feas_rate"].iloc[0]) if not _lh_sub_ext.empty else float("nan")
            _lh_obj_c   = float(_lh_sub_ext["lh_obj_mean"].iloc[0])  if not _lh_sub_ext.empty else float("nan")
            _lh_source  = "exp3"
        else:
            _lh_feas_c, _lh_obj_c, _lh_source = float("nan"), float("nan"), "none"

        # Óptimo: MILP certificado de Celda 7
        _opt_c = _gt_milp_obj_8 if _lbl8_c == "Tiny_3" else float("nan")

        # RPD del LH vs óptimo
        _lh_rpd_opt_c = (
            100.0 * (_lh_obj_c - _opt_c) / _opt_c
            if not np.isnan(_lh_obj_c) and not np.isnan(_opt_c) and _opt_c > 0
            else float("nan")
        )

        # Una fila por chain_strength
        _cs_groups = (
            _qpu_sub.groupby("chain_strength_label")
            if "chain_strength_label" in _qpu_sub.columns
            else [("scaled", _qpu_sub)]
        )
        for _cs_lbl_c, _qpu_cs_grp in _cs_groups:
            _qpu_feas_c  = float(_qpu_cs_grp["feasible"].astype(bool).mean())
            _qpu_fv      = _qpu_cs_grp.loc[_qpu_cs_grp["feasible"].astype(bool), "obj_value"]
            _qpu_obj_c   = float(_qpu_fv.mean()) if not _qpu_fv.empty else float("nan")

            _rpd_lh  = (
                100.0 * (_qpu_obj_c - _lh_obj_c) / _lh_obj_c
                if not np.isnan(_qpu_obj_c) and not np.isnan(_lh_obj_c) and _lh_obj_c > 0
                else float("nan")
            )
            _rpd_opt = (
                100.0 * (_qpu_obj_c - _opt_c) / _opt_c
                if not np.isnan(_qpu_obj_c) and not np.isnan(_opt_c) and _opt_c > 0
                else float("nan")
            )
            _comp_rows8.append({
                "instance_label":      _lbl8_c,
                "chain_strength":      _cs_lbl_c,
                "optimal_obj (MILP)":  round(_opt_c, 4)       if not np.isnan(_opt_c)       else float("nan"),
                "lh_feas_rate":        round(_lh_feas_c, 3)   if not np.isnan(_lh_feas_c)   else float("nan"),
                "lh_obj_mean":         round(_lh_obj_c, 4)    if not np.isnan(_lh_obj_c)    else float("nan"),
                "lh_rpd_vs_opt_%":     round(_lh_rpd_opt_c,2) if not np.isnan(_lh_rpd_opt_c) else float("nan"),
                "lh_source":           _lh_source,
                "qpu_feas_rate":       round(_qpu_feas_c, 3),
                "qpu_obj_mean":        round(_qpu_obj_c, 4)   if not np.isnan(_qpu_obj_c)   else float("nan"),
                "rpd_qpu_vs_lh_%":     round(_rpd_lh, 2)      if not np.isnan(_rpd_lh)      else float("nan"),
                "rpd_qpu_vs_opt_%":    round(_rpd_opt, 2)      if not np.isnan(_rpd_opt)     else float("nan"),
            })

    _df_comp8 = pd.DataFrame(_comp_rows8)
    print(_df_comp8.to_string(index=False))

# Histograma de energías — con energías individuales del sampleset + líneas de referencia
if not _df_qpu_final.empty:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        _instances_qpu_done = _df_qpu_final["instance_label"].unique()
        for _lbl8_h in _instances_qpu_done:
            _sub_qpu_h = _df_qpu_final[_df_qpu_final["instance_label"] == _lbl8_h]
            _cs_labels_h = (
                _sub_qpu_h["chain_strength_label"].unique().tolist()
                if "chain_strength_label" in _sub_qpu_h.columns
                else ["scaled"]
            )

            for _cs_lbl_h in _cs_labels_h:
                # Energías individuales del sampleset si disponibles
                if (
                    not _df_energies_final.empty
                    and "instance_label" in _df_energies_final.columns
                    and "chain_strength_label" in _df_energies_final.columns
                ):
                    _sub_e = _df_energies_final[
                        (_df_energies_final["instance_label"] == _lbl8_h) &
                        (_df_energies_final["chain_strength_label"] == _cs_lbl_h)
                    ]
                    if not _sub_e.empty:
                        _energies_h = _sub_e["energy"].dropna()
                        _feasible_h = _sub_e["feasible"].astype(bool)
                        _source_lbl = f"energías individuales (n={len(_energies_h)} reads)"
                    else:
                        _sub_r = (_sub_qpu_h[_sub_qpu_h["chain_strength_label"] == _cs_lbl_h]
                                  if "chain_strength_label" in _sub_qpu_h.columns else _sub_qpu_h)
                        _energies_h = _sub_r["best_energy"].dropna()
                        _feasible_h = _sub_r["feasible"].astype(bool)
                        _source_lbl = f"best_energy por run (n={len(_energies_h)} runs)"
                else:
                    _energies_h = _sub_qpu_h["best_energy"].dropna()
                    _feasible_h = _sub_qpu_h["feasible"].astype(bool)
                    _source_lbl = f"best_energy por run (n={len(_energies_h)} runs)"

                if _energies_h.empty:
                    continue

                fig_h, ax_h = plt.subplots(figsize=(7, 4))
                _palette_h = {True: "#2da34e", False: "#e07b39"}
                _df_plot_h = pd.DataFrame({
                    "energy":   _energies_h.values,
                    "feasible": _feasible_h.values[:len(_energies_h)],
                })
                for _feas_val, _grp_h in _df_plot_h.groupby("feasible"):
                    ax_h.hist(
                        _grp_h["energy"], bins=30,
                        color=_palette_h[_feas_val], alpha=0.75,
                        label=f"{'Feasible' if _feas_val else 'Infeasible'} (n={len(_grp_h)})",
                        edgecolor="white",
                    )

                # Línea 1: QUBO ground state certificado por Gurobi (Celda 7)
                if _lbl8_h == "Tiny_3" and not np.isnan(_gt_qubo_ground_energy_8):
                    ax_h.axvline(
                        _gt_qubo_ground_energy_8, color="red", linestyle="--", linewidth=1.8,
                        label=f"QUBO ground state (Gurobi) = {_gt_qubo_ground_energy_8:.2f}",
                    )

                # Línea 2: LH best_energy media (referencia interna Celda 6)
                if not _df_lh_ref_final.empty:
                    _lh_h = _df_lh_ref_final[_df_lh_ref_final["instance_label"] == _lbl8_h]
                    if not _lh_h.empty:
                        _lh_be_mean = _lh_h["best_energy"].mean()
                        ax_h.axvline(
                            _lh_be_mean, color="steelblue", linestyle=":", linewidth=1.5,
                            label=f"LH best_energy mean = {_lh_be_mean:.2f}",
                        )

                ax_h.set_xlabel("Energía QUBO")
                ax_h.set_ylabel("Frecuencia")
                ax_h.set_title(
                    f"Exp 8 — Distribución energías QPU  |  {_lbl8_h}  |  cs={_cs_lbl_h}\n"
                    f"({_source_lbl})"
                )
                ax_h.legend(fontsize=9)
                ax_h.grid(True, linestyle=":", alpha=0.4)
                sns.despine(ax=ax_h)
                plt.tight_layout()

                _hist_path = RESULTS_DIR / f"exp08_energy_histogram_{_lbl8_h}_{_cs_lbl_h}.png"
                fig_h.savefig(_hist_path, dpi=300, bbox_inches="tight")
                plt.show()
                logger.info("Histograma QPU guardado: %s", _hist_path)

    except Exception as _exc_hist8:
        logger.warning("No se pudo generar histograma de energías QPU: %s", _exc_hist8)

# Diagnósticos BQM: tabla condensada
print("\n--- Diagnósticos BQM (referencia para tesis) ---")
_diag_rows8: list[dict] = []
for _lbl8_d, _cache8_d in _bqm_cache8.items():
    _diag_rows8.append({
        "instance_label": _lbl8_d,
        "N":              int(_cache8_d["inst"]["N"]),
        "T":              int(_cache8_d["inst"]["T"]),
        "n_vars":         _cache8_d["n_vars"],
        "n_interactions": _cache8_d["n_inter"],
        "q_density":      round(_cache8_d["q_density"], 4),
        "max_degree":     _cache8_d["max_degree"],
        "max_bias":       round(_cache8_d["max_bias"], 4),
        "alpha":          alpha_star,
        "beta":           beta_star,
    })
_df_diag8 = pd.DataFrame(_diag_rows8)
print(_df_diag8.to_string(index=False))

save_metadata(FILEPATH, {
    "exp_version":              "v3.0",
    "run_uuid_last":            RUN_UUID,
    "timestamp":                datetime.datetime.now().isoformat(),
    "alpha_star":               alpha_star,
    "beta_star":                beta_star,
    "qpu_available":            _qpu_available,
    "qpu_num_reads":            QPU_NUM_READS,
    "n_qpu_runs_tiny3":         N_QPU_RUNS,
    "chain_strength_values":    str(CHAIN_STRENGTH_VALUES),
    "instances_planned":        str(list(EXP8_QPU_INSTANCES)),
    "instances_run":            str([i["instance_label"] for i in INSTANCES_8]),
    "n_embeddings_ok":          len(_embeddings8),
    "n_qpu_runs":               len(_df_qpu_final)      if not _df_qpu_final.empty      else 0,
    "n_lh_ref_runs":            len(_df_lh_ref_final)   if not _df_lh_ref_final.empty   else 0,
    "n_individual_energies":    len(_df_energies_final) if not _df_energies_final.empty else 0,
    "gurobi_gt_milp_obj":       _gt_milp_obj_8,
    "gurobi_gt_qubo_energy":    _gt_qubo_ground_energy_8,
    "gurobi_gt_qubo_sched_obj": _gt_qubo_sched_obj_8,
})
logger.info("Metadata guardada en %s", FILEPATH)
logger.info("Exp 8 completo. Resultados en: %s", FILEPATH)
