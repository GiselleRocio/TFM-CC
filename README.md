# TFM Terminal Scheduling

Hybrid classical-quantum optimization system for the Berth Allocation Problem (BAP)
at crude oil terminal. Uses D-Wave Quantum Annealing (QUBO) and Gurobi
as the classical solver backend.

## Services

| Service | URL                                            | Description                           |
| ------- | ---------------------------------------------- | ------------------------------------- |
| Web UI  | [http://localhost:3000](http://localhost:3000) | Next.js admin panel (main UI)         |
| API     | [http://localhost:8000](http://localhost:8000) | FastAPI backend (`/docs` for Swagger) |

## Quick Start

1. **Copy `.env.example` to `.env` and fill in your credentials:**

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and set the following:

   ```env
   # D-Wave (optional — falls back to SimulatedAnnealingSampler if empty)
   DWAVE_API_TOKEN=

   # Gurobi WLS license (required)
   GRB_WLSACCESSID=
   GRB_WLSSECRET=
   GRB_LICENSEID=
   ```

2. **Build and start all services:**

   ```bash
   docker compose up --build
   ```

3. **Open the admin panel:** [http://localhost:3000](http://localhost:3000)

## Experiments (Jupyter)

The thesis metrics run as Jupyter notebooks under `experiments2/notebooks_colab/`.
Each notebook has a corresponding `.py` script of the same name.
Pre-computed results from the thesis runs are available in `experiments2/results/`.

**Option A — Docker (recommended)**

```bash
docker compose up jupyter
# Open http://localhost:8888  (copy the token from the terminal output)
```

**Option B — local venv**

```bash
pip install -r experiments2/requirements.txt
jupyter lab --notebook-dir=experiments2/notebooks_colab
```

Run notebooks in this order:

```
setup.ipynb                        ← required first
exp1.ipynb
exp2.ipynb
exp02b.ipynb
exp3.ipynb
exp4.ipynb
exp5.ipynb
exp05b_p3_sweep.ipynb
exp05b_warmstart_validation.ipynb
exp6.ipynb
exp7.ipynb
exp8.ipynb
```

`setup.ipynb` is mandatory. The rest can run independently but are best run in order —
some plots depend on outputs from earlier experiments.

## Architecture

```
browser → Next.js (3000) → FastAPI (8000) → QUBO solver → D-Wave / SimulatedAnnealing
```

The web frontend communicates with the core exclusively via REST API.
