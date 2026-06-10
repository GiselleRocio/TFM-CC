"""FastAPI wrapper exposing the scheduling core as HTTP endpoints.

Thin API layer — no business logic. All scheduling computation lives in src/.
"""

import json
import logging
import os
import pathlib
import pickle
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import src.config as cfg
from src.config import (
    DEFAULT_SEED,
    EPSILON,
    MAX_ITERATIONS,
    generate_nominations,
)
from src.inventory import check_worst_case_overlaps
from src.preprocessing import compute_feasible_slots, validate_nominations
from src.qubo_builder import build_qubo
from src.solver import check_feasibility, decode_schedule, post_process_schedule, run_solver
from src.config import slot_to_datetime

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="TFM Terminal Scheduler API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# In-memory job store: job_id → job dict
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# History file — serialised access to prevent concurrent write corruption
_history_lock = threading.Lock()
_DEFAULT_HISTORY_PATH = "data/history.json"


def _history_path() -> pathlib.Path:
    """Resolve the history file path from the environment (read at call time)."""
    return pathlib.Path(os.environ.get("HISTORY_PATH", _DEFAULT_HISTORY_PATH))


def _read_history() -> list[dict[str, Any]]:
    """Read and return the history array. Returns [] on missing/empty/corrupt file."""
    path = _history_path()
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return json.loads(text)  # type: ignore[return-value]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read history file %s: %s — returning [].", path, exc)
        return []


def _write_history(entries: list[dict[str, Any]]) -> None:
    """Persist the history array, creating parent directories as needed."""
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InventoryCurveEntry(BaseModel):
    slot: int
    date: str
    stock_m3: float


class ScheduleEntry(BaseModel):
    vessel_id: str
    monobuoy: int
    start_slot: int
    end_slot: int
    priority_weight: float
    tardiness_slots: int
    within_window: bool


class KPIs(BaseModel):
    total_weighted_tardiness: float
    missing_vessels: int
    pipeline_violations: int
    buffer_cuts_applied: int
    tardy_vessels: int
    total_vessels: int
    iterations_used: int
    converged: bool
    oversaturated: bool


class QUBOStats(BaseModel):
    n_vars: int
    n_interactions: int
    q_matrix_density: float
    iterations_run: int
    sampler_used: str
    penalty_alpha: float
    n_vessels: int
    bqm_variables: int
    buffer_cuts_triples: int
    p1: float
    p2: float
    p3: float
    c_max: float
    best_energy: float


class ResultDone(BaseModel):
    status: str  # "done"
    iteration: int
    max_iterations: int
    converged: bool
    solve_time_seconds: float
    schedule: list[ScheduleEntry]
    kpis: KPIs
    qubo_stats: QUBOStats
    inventory_curve: list[InventoryCurveEntry] = []


class ResultError(BaseModel):
    status: str  # "error"
    message: str
    iteration: int


class VesselInput(BaseModel):
    vessel_id: str
    volume_m3: float        # stock_acumulado_m3 — used for ESD priority weight
    cargo_m3: float | None = None  # volume loaded onto this vessel (m³); falls back to volume_m3 if absent
    daily_inflow_m3: float
    release_slot: int
    due_slot: int
    processing_slots: int   # p_{j,m}: full pipeline-blocking duration in slots


class RunConfig(BaseModel):
    n_machines: int
    start_date: str
    end_date: str
    horizon_days: int
    slot_duration_hours: int
    min_ullage_days: int | None = None
    min_stock_buffer_days: int | None = None  # Legacy field
    n_tanks: int | None = None
    tank_capacity_m3: float | None = None
    initial_terminal_stock_m3: float | None = None
    daily_inflow_m3: float | None = None  # Daily crude inflow rate from upstream
    shared_pipeline_groups: list[list[int]] | None = None
    shared_pipeline: bool | None = None
    alpha: float
    beta: float = 2.0
    sampler: str
    blocked_slots: dict[str, list[int]]

    def model_post_init(self, __context) -> None:
        # Handle legacy data: min_stock_buffer_days -> min_ullage_days
        if self.min_stock_buffer_days is not None and self.min_ullage_days is None:
            self.min_ullage_days = self.min_stock_buffer_days
        # Set defaults for missing infrastructure fields
        if self.min_ullage_days is None:
            self.min_ullage_days = 4
        if self.n_tanks is None:
            self.n_tanks = 6
        if self.tank_capacity_m3 is None:
            self.tank_capacity_m3 = 100000.0
        if self.initial_terminal_stock_m3 is None:
            self.initial_terminal_stock_m3 = 300000.0
        if self.daily_inflow_m3 is None:
            self.daily_inflow_m3 = 20000.0  # Default daily inflow

    def get_conflict_set(self, machines: list[int]) -> frozenset[tuple[int, int]]:
        if self.shared_pipeline_groups is not None:
            pairs: set[tuple[int, int]] = set()
            for m in machines:
                pairs.add((m, m))
            for group in self.shared_pipeline_groups:
                for m1 in group:
                    for m2 in group:
                        pairs.add((m1, m2))
            return frozenset(pairs)
        if self.shared_pipeline:
            pairs = set()
            for m in machines:
                pairs.add((m, m))
            for m1 in machines:
                for m2 in machines:
                    pairs.add((m1, m2))
            return frozenset(pairs)
        return frozenset((m, m) for m in machines)

    def effective_capacity(self, machines: list[int], t_effective: int) -> int:
        if self.shared_pipeline_groups is not None:
            grouped: set[int] = set()
            n_shared_groups = 0
            for group in self.shared_pipeline_groups:
                if len(group) >= 2:
                    n_shared_groups += 1
                    grouped.update(group)
            n_independent = sum(1 for m in machines if m not in grouped)
            return (n_shared_groups + n_independent) * t_effective
        if self.shared_pipeline:
            return t_effective
        return len(machines) * t_effective


class SolveRequest(BaseModel):
    vessels: list[VesselInput]
    config: RunConfig


class JobAccepted(BaseModel):
    job_id: str


class ResultRunning(BaseModel):
    status: str  # "running"
    iteration: int
    max_iterations: int
    best_tardiness: float | None = None
    converged: bool = False


# --- config defaults ---


class ConfigDefaults(BaseModel):
    n_machines: int
    horizon_days: int
    slot_duration_hours: int
    min_ullage_days: int
    n_tanks: int
    tank_capacity_m3: float
    initial_terminal_stock_m3: float
    daily_inflow_m3: float
    shared_pipeline_groups: list[list[int]]
    alpha: float
    alpha_min: float
    alpha_max: float
    blocked_slots: dict[str, list[int]]
    max_iterations: int
    sampler_options: list[str]


# --- generate ---


class GenerateRequest(BaseModel):
    n_vessels: int
    slot_duration_hours: int = cfg.SLOT_HOURS
    n_machines: int = len(cfg.MACHINES)
    seed: int | None = None


class VesselGenerated(BaseModel):
    vessel_id: str
    volume_m3: float
    daily_inflow_m3: float
    cargo_m3: float
    release_slot: int
    due_slot: int
    processing_slots: int   # p_{j,m}: full pipeline-blocking duration in slots
    priority_weight: float


# --- history ---


class HistoryEntry(BaseModel):
    job_id: str
    timestamp: str
    n_vessels: int
    converged: bool
    total_weighted_tardiness: float
    solve_time_seconds: float
    sampler: str
    iterations_used: int


class HistoryPayload(BaseModel):
    vessels: list[VesselInput]
    config: RunConfig
    result: ResultDone


class HistoryEntryFull(BaseModel):
    job_id: str
    timestamp: str
    n_vessels: int
    converged: bool
    total_weighted_tardiness: float
    solve_time_seconds: float
    sampler: str
    iterations_used: int
    vessels: list[dict[str, Any]] | None = None
    config: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Background scheduling worker
# ---------------------------------------------------------------------------




def _run_job(job_id: str, body: SolveRequest) -> None:
    """Background worker that runs the iterative hybrid QUBO scheduling loop.

    Mirrors the logic of src.inventory.run_iterative_loop, but updates the
    in-memory job store at each iteration so the polling endpoint can surface
    real-time progress.

    Args:
        job_id: UUID string key in _jobs.
        body:   Validated POST /solve request.
    """

    def _set_error(msg: str) -> None:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "message": msg,
                "iteration": _jobs[job_id].get("iteration", 0),
            })

    start_time = time.monotonic()

    try:
        # --- 1. Convert API vessel list → nominations DataFrame ---
        # volume_m3 in VesselInput is stock_acumulado_m3 (drives ESD priority weight).
        # cargo_m3 is the actual volume loaded onto the vessel (drives inventory curve).
        # If cargo_m3 is absent (e.g. manual Excel upload), fall back to volume_m3.
        # w_j = stock_acumulado_m3 / daily_inflow_m3  (Equivalent Stock in Days).
        records = [
            {
                "vessel_id": v.vessel_id,
                "r_j": v.release_slot,
                "d_j": v.due_slot,
                "p_j": v.processing_slots,
                "stock_acumulado_m3": v.volume_m3,
                "volume_m3": v.cargo_m3 if v.cargo_m3 is not None else v.volume_m3,
                "daily_inflow_m3": v.daily_inflow_m3,
                "w_j": v.volume_m3 / v.daily_inflow_m3,
            }
            for v in body.vessels
        ]
        nominations_df = pd.DataFrame(records)

        # --- 2. Derive horizon and config before validation so the check uses
        #        the correct T for this request (not the module-level default).
        t_effective: int = body.config.horizon_days * (24 // body.config.slot_duration_hours)
        machines: list[int] = list(range(1, body.config.n_machines + 1))
        blocked_slots_map: dict[int, set[int]] = {
            int(k): set(v) for k, v in body.config.blocked_slots.items()
        }

        # --- 3. Validate (logs warnings; KeyError = hard schema failure) ---
        try:
            validate_nominations(
                nominations_df,
                horizon_slots=t_effective,
                min_ullage_days=body.config.min_ullage_days,
            )
        except KeyError as exc:
            _set_error(f"Nomination validation failed — missing columns: {exc}")
            return

        # --- 4. Feasible slot table ---
        variables_df = compute_feasible_slots(
            nominations_df,
            machines=machines,
            blocked_slots_map=blocked_slots_map,
            horizon_slots=t_effective,
        )
        if variables_df.empty:
            _set_error("No feasible (vessel, machine, slot) triples found.")
            return

        # --- 5. Over-saturation guard ---
        total_p_j = int(nominations_df["p_j"].sum())
        effective_capacity = body.config.effective_capacity(machines, t_effective)
        if total_p_j > effective_capacity:
            _set_error(
                f"Nomination set is over-saturated: "
                f"Σ p_j = {total_p_j} > capacity = {effective_capacity} "
                f"(slot_duration_hours={body.config.slot_duration_hours}).  "
                "Reduce the nomination set or extend the horizon."
            )
            return

        n_vessels: int = variables_df["vessel_id"].nunique()
        sampler_used: str = ""  # set from run_solver on first iteration

        # BQM structure stats (captured on k=1, where no cuts are applied).
        n_vars: int = 0
        n_interactions: int = 0
        density: float = 0.0
        P1_stats: float = 0.0
        P2_stats: float = 0.0
        c_max: float = 0.0
        best_energy: float = float("inf")

        # --- 6. Build conflict set from shared_pipeline_groups ---
        conflict_set = body.config.get_conflict_set(machines)

        # --- 7. Iterative loop ---
        all_cuts: set[tuple[str, int, int]] = set()
        best_schedule_df = pd.DataFrame()
        best_feasibility: dict[str, Any] = {}
        best_wt: float = float("inf")
        P3: float = 0.0
        schedule_df = pd.DataFrame()
        feasibility: dict[str, Any] = {}
        converged = False
        final_k = MAX_ITERATIONS

        print(f"\n[_run_job] CONFIG: initial_stock={body.config.initial_terminal_stock_m3} m³, "
              f"daily_inflow={body.config.daily_inflow_m3} m³/day, "
              f"min_ullage_days={body.config.min_ullage_days}, "
              f"capacity={body.config.n_tanks * body.config.tank_capacity_m3} m³")

        for k in range(1, cfg.MAX_ITERATIONS + 1):
            print(f"\n[_run_job] ── Iteration {k}/{cfg.MAX_ITERATIONS} ──")
            # Publish iteration progress before the (potentially long) solve.
            with _jobs_lock:
                _jobs[job_id]["iteration"] = k
                _jobs[job_id]["best_tardiness"] = (
                    best_wt if best_wt != float("inf") else 0.0
                )

            bqm, P1_k, P2_k, P3, _ = build_qubo(
                variables_df,
                cuts=all_cuts if all_cuts else None,
                conflict_set=conflict_set,
                alpha=body.config.alpha,
                beta=body.config.beta,
            )

            # Capture BQM structure from the first (cuts-free) build.
            if k == 1:
                n_vars = len(bqm.variables)
                n_interactions = len(bqm.quadratic)
                max_possible = n_vars * (n_vars - 1) // 2
                density = n_interactions / max_possible if max_possible > 0 else 0.0
                P1_stats = P1_k
                P2_stats = P2_k
                # c_max derived from P2 = α·n·c_max
                c_max = P2_k / (body.config.alpha * n_vessels) if n_vessels > 0 else 0.0

            sampleset, sampler_used = run_solver(bqm, requested_sampler=body.config.sampler)
            best_energy = min(best_energy, sampleset.first.energy)
            best_sample: dict[str, int] = dict(sampleset.first.sample)
            schedule_df = decode_schedule(best_sample, variables_df)
            # Derive bool for post_process_schedule: True when any group ≥ 2
            # (i.e. at least some monobuoys must serialise their loads).
            if body.config.shared_pipeline_groups is not None:
                has_shared = any(len(g) >= 2 for g in body.config.shared_pipeline_groups)
            else:
                has_shared = bool(body.config.shared_pipeline)
            schedule_df = post_process_schedule(
                schedule_df, variables_df, has_shared
            )
            feasibility = check_feasibility(
                schedule_df, variables_df, conflict_set=conflict_set
            )

            current_wt: float = feasibility["total_weighted_tardiness"]
            if feasibility["is_feasible"] and current_wt < best_wt:
                best_schedule_df = schedule_df.copy()
                best_feasibility = feasibility.copy()
                best_wt = current_wt

            # Update best tardiness after the solve.
            with _jobs_lock:
                _jobs[job_id]["best_tardiness"] = (
                    best_wt if best_wt != float("inf") else current_wt
                )

            new_cuts: set[tuple[str, int, int]] = check_worst_case_overlaps(
                schedule_df,
                variables_df,
                nominations_df,
                slot_duration_hours=body.config.slot_duration_hours,
                min_ullage_days=body.config.min_ullage_days,
                initial_terminal_stock_m3=body.config.initial_terminal_stock_m3,
                n_tanks=body.config.n_tanks,
                tank_capacity_m3=body.config.tank_capacity_m3,
                daily_inflow_m3=body.config.daily_inflow_m3,
            )

            print(f"[_run_job] Iteration {k}: {len(new_cuts)} new cuts found. "
                  f"Total accumulated cuts: {len(all_cuts | new_cuts)}")

            if not new_cuts:
                # Converged — no ullage violations.
                print(f"[_run_job] ✅ CONVERGED at iteration {k}")
                converged = True
                final_k = k
                if best_schedule_df.empty and feasibility["is_feasible"]:
                    best_schedule_df = schedule_df.copy()
                    best_feasibility = feasibility.copy()
                    best_wt = current_wt
                break

            all_cuts |= new_cuts
        # If the loop completed without break, final_k = MAX_ITERATIONS (initialized above).

        final_schedule = (
            best_schedule_df if not best_schedule_df.empty else schedule_df
        )
        final_feasibility = best_feasibility if best_feasibility else feasibility

        # Add volume_m3 from nominations to final_schedule for inventory curve
        volume_map: dict[str, float] = {
            str(row["vessel_id"]): float(row["volume_m3"])
            for _, row in nominations_df.iterrows()
        }
        final_schedule_with_volume = final_schedule.copy()
        final_schedule_with_volume["volume_m3"] = final_schedule_with_volume["vessel_id"].map(volume_map).fillna(0)

        # --- 8. Build response payload ---
        schedule_entries: list[ScheduleEntry] = []
        tardy_count = 0
        for _, row in final_schedule_with_volume.iterrows():
            tardy_slots = int(row.get("tardiness_slots", 0))
            is_late = bool(row.get("is_late", False))
            if tardy_slots > 0:
                tardy_count += 1
            schedule_entries.append(
                ScheduleEntry(
                    vessel_id=str(row["vessel_id"]),
                    monobuoy=int(row["machine"]),
                    start_slot=int(row["start_slot"]),
                    end_slot=int(row["end_slot"]),
                    priority_weight=float(row["w_j"]),
                    tardiness_slots=tardy_slots,
                    within_window=not is_late,
                )
            )

        kpis = KPIs(
            total_weighted_tardiness=float(
                final_feasibility.get("total_weighted_tardiness", 0.0)
            ),
            missing_vessels=len(final_feasibility.get("missing_vessels", [])),
            pipeline_violations=len(final_feasibility.get("pipeline_violations", [])),
            buffer_cuts_applied=len(all_cuts),
            tardy_vessels=tardy_count,
            total_vessels=len(nominations_df),
            iterations_used=final_k,
            converged=converged,
            oversaturated=False,
        )

        qubo_stats = QUBOStats(
            n_vars=n_vars,
            n_interactions=n_interactions,
            q_matrix_density=round(density, 4),
            iterations_run=final_k,
            sampler_used=sampler_used,
            penalty_alpha=body.config.alpha,
            n_vessels=n_vessels,
            bqm_variables=n_vars,
            buffer_cuts_triples=len(all_cuts),
            p1=round(P1_stats, 4),
            p2=round(P2_stats, 4),
            p3=round(P2_stats / body.config.beta, 4),
            c_max=round(c_max, 4),
            best_energy=round(best_energy, 6),
        )

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "done",
                "iteration": final_k,
                "max_iterations": MAX_ITERATIONS,
                "converged": converged,
                "solve_time_seconds": round(time.monotonic() - start_time, 3),
                "schedule": [s.model_dump() for s in schedule_entries],
                "kpis": kpis.model_dump(),
                "qubo_stats": qubo_stats.model_dump(),
                "inventory_curve": _generate_inventory_curve(
                    final_schedule_with_volume,
                    nominations_df,
                    body.config,
                    body.config.slot_duration_hours,
                ),
            })

    except AssertionError as exc:
        # Penalty hierarchy violation from build_qubo — matches architecture doc example.
        _set_error(f"Penalty hierarchy assertion failed: {exc}")
    except Exception as exc:
        # str(exc) is empty for bare exceptions like MemoryError() — fall back to type name.
        _set_error(str(exc) or type(exc).__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _generate_inventory_curve(
    schedule_df: pd.DataFrame,
    nominations_df: pd.DataFrame,
    config: RunConfig,
    slot_duration_hours: int,
) -> list[dict[str, Any]]:
    """Generate the projected inventory curve for visualization.

    Stock model (saw-tooth):
    - Rises continuously at the terminal-level daily inflow (config.daily_inflow_m3).
    - Drops instantly when a vessel completes loading (by its cargo_m3).

    stock[t] = stock[t-1] - cargo_m3_finished_at[t] + inflow_per_slot
    """
    if schedule_df.empty:
        return []

    initial_stock: float = config.initial_terminal_stock_m3
    horizon_slots: int = config.horizon_days * (24 // slot_duration_hours)

    # Use the terminal-level daily inflow configured by the user (upstream
    # crude arriving at the tank farm), NOT the per-vessel daily_inflow_m3
    # which is a shipper-level field used only for ESD priority calculation.
    total_daily_inflow: float = config.daily_inflow_m3

    start_date_str = config.start_date.replace('/', '-')
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")

    # Map vessel_id → cargo_m3 from nominations (the actual volume loaded).
    cargo_map: dict[str, float] = {
        str(row["vessel_id"]): float(row["volume_m3"])
        for _, row in nominations_df.iterrows()
    }

    # Build cargo drop events keyed by end_slot.
    # schedule_df["volume_m3"] already holds cargo_m3 (mapped in the caller).
    # Use cargo_map as the authoritative source to avoid fillna(0) masking misses.
    cargo_drops: dict[int, float] = {}
    for _, row in schedule_df.iterrows():
        end_slot: int = int(row["end_slot"])
        vessel_id: str = str(row["vessel_id"])
        cargo_m3: float = cargo_map.get(vessel_id, float(row.get("volume_m3", 0.0)))
        cargo_drops[end_slot] = cargo_drops.get(end_slot, 0.0) + cargo_m3

    # Inflow per macro-slot period.
    inflow_per_slot: float = (total_daily_inflow * slot_duration_hours) / 24.0

    curve: list[dict[str, Any]] = []
    current_stock: float = initial_stock

    # Slot 0: initial stock before any inflow or loading.
    curve.append({
        "slot": 0,
        "date": start_dt.strftime("%Y-%m-%d"),
        "stock_m3": round(current_stock, 2),
    })

    # Slots 1..horizon: subtract cargo drops first, then add inflow.
    for slot in range(1, horizon_slots + 1):
        if slot in cargo_drops:
            current_stock -= cargo_drops[slot]
        current_stock += inflow_per_slot

        slot_date = start_dt + timedelta(hours=slot * slot_duration_hours)

        curve.append({
            "slot": slot,
            "date": slot_date.strftime("%Y-%m-%d"),
            "stock_m3": round(current_stock, 2),
        })

    return curve


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


def _build_lp_text(
    variables_df: pd.DataFrame,
    t_effective: int,
    pipeline_groups: list[list[int]],
    machines: list[int],
) -> str:
    """Build a Gurobi-compatible LP file for the terminal scheduling MILP.

    Equivalent to the QUBO at iteration k=1 (no inventory cuts):

        min  Σ_j w_j·T_j  +  ε·Σ_{j,m,t} (t+p_j)·x_{jmt}

    Subject to:
        (1) Assignment:   Σ_{m,t} x_{jmt} = 1                              ∀ j
        (2) Tardiness:    T_j ≥ (t+p_j−d_j)·x_{jmt}                       ∀ j, m, t with tardiness > 0
        (3) No-overlap:   Σ_{m∈group, covering τ} x_{jmt} ≤ 1             ∀ group, ∀ τ
                          (per-machine constraint for independent monobuoys)

    The no-overlap constraints mirror the QUBO conflict set derived from
    pipeline_groups: monobuoys in the same group share one physical pipeline
    and generate a joint per-slot constraint; independent monobuoys each get
    their own per-slot constraint.

    Args:
        variables_df:    Feasible-slot DataFrame from preprocessing.
        t_effective:     Planning horizon in slots.
        pipeline_groups: List of monobuoy-index groups sharing a pipeline.
                         E.g. [[1,2]] = one shared pipeline; [[3,6],[2,8,9]] = two groups.
                         Monobuoys not in any group are treated as independent.
        machines:        Full list of machine IDs.

    Returns:
        LP file text (UTF-8 string).
    """
    # Derive a human-readable pipeline topology label for the file header
    grouped: set[int] = set(m for g in pipeline_groups for m in g)
    independent_machines = [m for m in machines if m not in grouped]
    group_labels = [f"M{'+M'.join(str(m) for m in g)}" for g in pipeline_groups if len(g) >= 2]
    indep_labels = [f"M{m}" for m in independent_machines]
    if group_labels and not indep_labels:
        pipeline_label = "shared pipeline: " + ", ".join(group_labels)
    elif group_labels:
        pipeline_label = "groups: " + ", ".join(group_labels) + "; independent: " + ", ".join(indep_labels)
    else:
        pipeline_label = "independent pipelines"

    lines: list[str] = [
        r"\ MILP — TFM Terminal Scheduling",
        r"\ Equivalent to QUBO H_obj + H_assign + H_overlap (k=1, no inventory cuts)",
        f"\\ Notation: Rm | rj | sum wj*Tj  with {pipeline_label}",
        r"\ Generated by TFM API — POST /milp/export",
        "",
    ]

    vessel_ids: list[str] = sorted(variables_df["vessel_id"].unique().tolist())

    # Per-vessel parameters and variable lists
    vessel_data: dict[str, dict] = {}
    for vid in vessel_ids:
        grp = variables_df[variables_df["vessel_id"] == vid]
        vessel_data[vid] = {
            "w_j": float(grp.iloc[0]["w_j"]),
            "p_j": int(grp.iloc[0]["p_j"]),
            "d_j": int(grp.iloc[0]["d_j"]),
            "rows": grp,
        }

    # tau → list of variable names that cover that slot (for no-overlap)
    # Indexed by machine so we can enforce per-machine or global constraints.
    tau_to_vars_by_machine: dict[int, dict[int, list[str]]] = {m: defaultdict(list) for m in machines}
    for _, row in variables_df.iterrows():
        t = int(row["slot"])
        p = int(row["p_j"])
        m = int(row["machine"])
        var = f"x_{row['vessel_id']}_{m}_{t}"
        for tau in range(t, min(t + p, t_effective)):
            tau_to_vars_by_machine[m][tau].append(var)

    # ── Objective ────────────────────────────────────────────────
    lines.append("Minimize")
    obj_parts: list[str] = []
    for vid in vessel_ids:
        obj_parts.append(f"{vessel_data[vid]['w_j']:.6f} T_{vid}")
    for _, row in variables_df.iterrows():
        coeff = EPSILON * (int(row["slot"]) + int(row["p_j"]))
        obj_parts.append(f"{coeff:.6f} x_{row['vessel_id']}_{int(row['machine'])}_{int(row['slot'])}")

    obj_str = f"  obj: {obj_parts[0]}"
    for part in obj_parts[1:]:
        obj_str += f"\n    + {part}"
    lines.append(obj_str)
    lines.append("")

    # ── Constraints ──────────────────────────────────────────────
    lines.append("Subject To")

    # (1) Assignment: Σ x_{jmt} = 1  ∀ j
    for vid in vessel_ids:
        grp = vessel_data[vid]["rows"]
        terms = " + ".join(
            f"x_{vid}_{int(r['machine'])}_{int(r['slot'])}" for _, r in grp.iterrows()
        )
        lines.append(f"  assign_{vid}: {terms} = 1")

    # (2) Tardiness linearisation: T_j >= (t+p_j-d_j)·x_{jmt}  ∀ j, m, t
    #
    # One constraint per (vessel, machine, slot) triple — not one aggregate.
    # This is the correct big-M-free linearisation of T_j = max(0, C_j - d_j):
    # when x_{jmt}=1, it forces T_j >= t+p_j-d_j (which is >=0 only if late).
    # When x_{jmt}=0 the RHS is 0 and the T_j>=0 bound covers it.
    # The aggregate form (T_j - Σ coeff·x >= 0) is incorrect: the solver can
    # cancel positive and negative terms and set T_j=0 even for late slots.
    for vid in vessel_ids:
        p_j = vessel_data[vid]["p_j"]
        d_j = vessel_data[vid]["d_j"]
        grp = vessel_data[vid]["rows"]
        for idx, (_, row) in enumerate(grp.iterrows()):
            tardiness = int(row["slot"]) + p_j - d_j
            if tardiness <= 0:
                # Slot finishes on time: T_j >= 0 bound already covers this
                continue
            var = f"x_{vid}_{int(row['machine'])}_{int(row['slot'])}"
            # T_j >= tardiness·x_{jmt}  ↔  T_j - tardiness·x_{jmt} >= 0
            lines.append(f"  tard_{vid}_{idx}: T_{vid} - {tardiness} {var} >= 0")

    # (3) No-overlap — one constraint per (pipeline-group, slot τ).
    #
    # For each group of monobuoys sharing a physical pipeline: at most one
    # vessel may be loading across all group members at any slot τ.
    # Independent monobuoys (not in any group) each get their own per-slot
    # constraint (equivalent to a singleton group).
    #
    # Build the effective constraint groups: shared groups + singleton groups
    # for independent machines.
    constraint_groups: list[tuple[str, list[int]]] = []
    grouped_machines: set[int] = set()
    for idx, group in enumerate(pipeline_groups):
        if len(group) >= 2:
            constraint_groups.append((f"g{idx}", group))
            grouped_machines.update(group)
    for m in machines:
        if m not in grouped_machines:
            constraint_groups.append((f"m{m}", [m]))

    for group_label, group_machines in constraint_groups:
        for tau in range(t_effective):
            covering = [
                var
                for m in group_machines
                for var in tau_to_vars_by_machine[m].get(tau, [])
            ]
            if len(covering) >= 2:
                lines.append(f"  no_overlap_{group_label}_{tau}: {' + '.join(covering)} <= 1")

    lines.append("")

    # ── Bounds ───────────────────────────────────────────────────
    lines.append("Bounds")
    for vid in vessel_ids:
        lines.append(f"  T_{vid} >= 0")
    lines.append("")

    # ── Binary section ───────────────────────────────────────────
    lines.append("Binary")
    for _, row in variables_df.iterrows():
        lines.append(f"  x_{row['vessel_id']}_{int(row['machine'])}_{int(row['slot'])}")
    lines.append("")
    lines.append("End")

    return "\n".join(lines)


@app.post("/milp/export")
def post_milp_export(body: SolveRequest) -> Response:
    """Build the equivalent MILP and return it as a Gurobi-compatible .lp file.

    Runs the same pre-solve pipeline as POST /qubo/export (validation, feasible
    slot computation, conflict set construction) but instead of assembling a BQM
    returns the original Mixed Integer Linear Program in LP format.

    The MILP is equivalent to the QUBO at iteration k=1 (no inventory cuts):

        min  Σ_j w_j·T_j  +  ε·Σ (t+p_j)·x_{jmt}
        s.t. assignment, tardiness linearisation, no-overlap (shared pipeline)

    The file can be loaded directly in Gurobi:
        import gurobipy as gp
        m = gp.read("milp_<jobId>.lp")
        m.optimize()
    """
    if not body.vessels:
        raise HTTPException(status_code=422, detail="vessels list must not be empty.")

    # -- 1. Build nominations DataFrame ------------------------------------
    records = [
        {
            "vessel_id": v.vessel_id,
            "r_j": v.release_slot,
            "d_j": v.due_slot,
            "p_j": v.processing_slots,
            "stock_acumulado_m3": v.volume_m3,
            "volume_m3": v.cargo_m3 if v.cargo_m3 is not None else v.volume_m3,
            "daily_inflow_m3": v.daily_inflow_m3,
            "w_j": v.volume_m3 / v.daily_inflow_m3,
        }
        for v in body.vessels
    ]
    nominations_df = pd.DataFrame(records)

    # -- 2. Horizon and machine list --------------------------------------
    t_effective: int = body.config.horizon_days * (24 // body.config.slot_duration_hours)
    machines: list[int] = list(range(1, body.config.n_machines + 1))
    blocked_slots_map: dict[int, set[int]] = {
        int(k): set(v) for k, v in body.config.blocked_slots.items()
    }

    # -- 3. Validate nominations ------------------------------------------
    try:
        validate_nominations(
            nominations_df,
            horizon_slots=t_effective,
            min_ullage_days=body.config.min_ullage_days,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Nomination validation failed — missing columns: {exc}",
        ) from exc

    # -- 4. Feasible slot table -------------------------------------------
    variables_df = compute_feasible_slots(
        nominations_df,
        machines=machines,
        blocked_slots_map=blocked_slots_map,
        horizon_slots=t_effective,
    )
    if variables_df.empty:
        raise HTTPException(
            status_code=422,
            detail="No feasible (vessel, machine, slot) triples found.",
        )

    # -- 5. Over-saturation guard -----------------------------------------
    total_p_j = int(nominations_df["p_j"].sum())
    effective_capacity = body.config.effective_capacity(machines, t_effective)
    if total_p_j > effective_capacity:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nomination set is over-saturated: "
                f"Σ p_j = {total_p_j} > capacity = {effective_capacity}."
            ),
        )

    # -- 6. Build LP text and return as downloadable file -----------------
    pipeline_groups = body.config.shared_pipeline_groups or []
    lp_text = _build_lp_text(variables_df, t_effective, pipeline_groups, machines)

    return Response(
        content=lp_text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=milp.lp"},
    )


@app.post("/qubo/export")
def post_qubo_export(body: SolveRequest) -> Response:
    """Build the QUBO matrix and return it as a pickled file for external benchmarking.

    Executes the same pre-solve pipeline as POST /solve (validation, feasible
    slot computation, conflict set construction, BQM assembly) but stops before
    the solver runs.  Returns a binary .pkl file containing the full Q matrix
    and supporting metadata needed to reconstruct the problem in CPLEX or any
    other MIP/QUBO solver.

    Pickle payload keys:
        Q (dict): QUBO matrix as {(var_i, var_j): coeff}.  Diagonal entries
            use repeated keys (var, var) for linear terms; off-diagonal entries
            are in upper-triangular form.
        Q_indexed (dict): Same matrix indexed by integers {(i, j): coeff} where
            the integer mapping is given by ``variables``.
        variables (list[str]): Ordered list of BQM variable names.  Index i in
            this list corresponds to integer i in Q_indexed.
        offset (float): Constant energy offset from the H_assign expansion.
        P1 (float): Assignment penalty.
        P2 (float): Overlap penalty.
        n_vars (int): Number of binary variables.
        n_interactions (int): Number of off-diagonal Q entries.
        density (float): Q-matrix density (interactions / max_possible).
        n_vessels (int): Number of vessels in the nomination set.
    """
    if not body.vessels:
        raise HTTPException(status_code=422, detail="vessels list must not be empty.")

    # -- 1. Build nominations DataFrame -------------------------------------
    records = [
        {
            "vessel_id": v.vessel_id,
            "r_j": v.release_slot,
            "d_j": v.due_slot,
            "p_j": v.processing_slots,
            "stock_acumulado_m3": v.volume_m3,
            "volume_m3": v.cargo_m3 if v.cargo_m3 is not None else v.volume_m3,
            "daily_inflow_m3": v.daily_inflow_m3,
            "w_j": v.volume_m3 / v.daily_inflow_m3,
        }
        for v in body.vessels
    ]
    nominations_df = pd.DataFrame(records)

    # -- 2. Derive horizon and machine list ---------------------------------
    t_effective: int = body.config.horizon_days * (24 // body.config.slot_duration_hours)
    machines: list[int] = list(range(1, body.config.n_machines + 1))
    blocked_slots_map: dict[int, set[int]] = {
        int(k): set(v) for k, v in body.config.blocked_slots.items()
    }

    # -- 3. Validate nominations --------------------------------------------
    try:
        validate_nominations(
            nominations_df,
            horizon_slots=t_effective,
            min_ullage_days=body.config.min_ullage_days,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Nomination validation failed — missing columns: {exc}",
        ) from exc

    # -- 4. Feasible slot table ---------------------------------------------
    variables_df = compute_feasible_slots(
        nominations_df,
        machines=machines,
        blocked_slots_map=blocked_slots_map,
        horizon_slots=t_effective,
    )
    if variables_df.empty:
        raise HTTPException(
            status_code=422,
            detail="No feasible (vessel, machine, slot) triples found.",
        )

    # -- 5. Over-saturation guard -------------------------------------------
    total_p_j = int(nominations_df["p_j"].sum())
    effective_capacity = body.config.effective_capacity(machines, t_effective)
    if total_p_j > effective_capacity:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nomination set is over-saturated: "
                f"Σ p_j = {total_p_j} > capacity = {effective_capacity}."
            ),
        )

    # -- 6. Conflict set ----------------------------------------------------
    conflict_set = body.config.get_conflict_set(machines)

    # -- 7. Build QUBO (no cuts on first build — same as iteration k=1) -----
    bqm, P1, P2, _, offset = build_qubo(
        variables_df,
        cuts=None,
        conflict_set=conflict_set,
        alpha=body.config.alpha,
        beta=body.config.beta,
    )

    # -- 8. Serialise Q matrix ----------------------------------------------
    # bqm.to_qubo() returns (Q, offset_qubo) where Q is a dict
    # {(u, v): bias} with diagonal as (u, u) and upper-triangle off-diagonal.
    Q_raw, _qubo_offset = bqm.to_qubo()
    Q: dict[tuple[str, str], float] = dict(Q_raw)

    variables: list[str] = list(bqm.variables)
    var_to_idx: dict[str, int] = {v: i for i, v in enumerate(variables)}
    Q_indexed: dict[tuple[int, int], float] = {
        (var_to_idx[u], var_to_idx[v]): c for (u, v), c in Q.items()
    }

    n_vars: int = len(bqm.variables)
    n_interactions: int = len(bqm.quadratic)
    max_possible: int = n_vars * (n_vars - 1) // 2
    density: float = n_interactions / max_possible if max_possible > 0 else 0.0

    payload: dict[str, Any] = {
        "Q": Q,
        "Q_indexed": Q_indexed,
        "variables": variables,
        "offset": offset + _qubo_offset,
        "P1": P1,
        "P2": P2,
        "n_vars": n_vars,
        "n_interactions": n_interactions,
        "density": density,
        "n_vessels": variables_df["vessel_id"].nunique(),
    }

    return Response(
        content=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=qubo_matrix.pkl"},
    )


@app.post("/solve", status_code=202, response_model=JobAccepted)
def post_solve(body: SolveRequest) -> JobAccepted:
    """Launch an async scheduling job. Returns job_id immediately (202 Accepted)."""
    if not body.vessels:
        raise HTTPException(status_code=422, detail="vessels list must not be empty.")

    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "iteration": 0,
            "max_iterations": MAX_ITERATIONS,
            "best_tardiness": 0.0,
            "converged": False,
            "config": body.config.model_dump(),
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, body),
        daemon=True,
        name=f"job-{job_id[:8]}",
    )
    thread.start()

    return JobAccepted(job_id=job_id)


@app.get("/results/{job_id}")
def get_results(job_id: str) -> ResultRunning | ResultDone | ResultError:
    """Poll job status. Returns running / done / error state."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    status = job["status"]

    if status == "running":
        return ResultRunning(
            status="running",
            iteration=job["iteration"],
            max_iterations=job["max_iterations"],
            best_tardiness=job["best_tardiness"],
            converged=job["converged"],
        )

    if status == "done":
        # Use the inventory_curve stored when the job completed — it was already
        # computed with the correct nominations_df and config at solve time.
        stored_curve = job.get("inventory_curve", [])
        return ResultDone(
            status="done",
            iteration=job["iteration"],
            max_iterations=job["max_iterations"],
            converged=job["converged"],
            solve_time_seconds=job["solve_time_seconds"],
            schedule=[ScheduleEntry(**e) for e in job["schedule"]],
            kpis=KPIs(**job["kpis"]),
            qubo_stats=QUBOStats(**job["qubo_stats"]),
            inventory_curve=[InventoryCurveEntry(**e) for e in stored_curve],
        )

    # status == "error"
    return ResultError(
        status="error",
        message=job.get("message", "Unknown error."),
        iteration=job.get("iteration", 0),
    )


@app.get("/config/defaults", response_model=ConfigDefaults)
def get_config_defaults() -> ConfigDefaults:
    """Return default terminal configuration from src/config.py."""
    machines = list(range(1, len(cfg.MACHINES) + 1))
    return ConfigDefaults(
        n_machines=len(cfg.MACHINES),
        horizon_days=cfg.HORIZON_DAYS,
        slot_duration_hours=cfg.SLOT_HOURS,
        min_ullage_days=cfg.MIN_ULLAGE_DAYS,
        n_tanks=cfg.N_TANKS,
        tank_capacity_m3=cfg.TANK_CAPACITY_M3,
        initial_terminal_stock_m3=cfg.INITIAL_TERMINAL_STOCK_M3,
        daily_inflow_m3=cfg.DAILY_INFLOW_M3,
        # Default: all monobuoys share one pipeline ( physical setup)
        shared_pipeline_groups=[machines],
        alpha=cfg.PENALTY_ALPHA,
        alpha_min=1.0,
        alpha_max=10.0,
        blocked_slots={str(k): sorted(v) for k, v in cfg.BLOCKED_SLOTS.items()},
        max_iterations=cfg.MAX_ITERATIONS,
        sampler_options=["leap_hybrid", "simulated_annealing"],
    )


@app.post("/generate", response_model=list[VesselGenerated])
def post_generate(body: GenerateRequest) -> list[VesselGenerated]:
    """Generate synthetic vessel data via config.py generator.

    Nominations are always generated using the native SLOT_HOURS from config.py,
    then rescaled to ``body.slot_duration_hours`` so that every slot-relative
    value (r_j, d_j, p_j) is expressed in the requested slot unit.
    """
    effective_seed = body.seed if body.seed is not None else DEFAULT_SEED
    try:
        df = generate_nominations(n=body.n_vessels, seed=effective_seed, n_machines=body.n_machines)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Rescale slot-relative columns when the requested slot size differs from
    # the native SLOT_HOURS.  factor = native / requested (e.g. 12/24 = 0.5).
    # p_j values (48 h and 96 h expressed in native slots) rescale exactly.
    # r_j / d_j are floored; the user can adjust them in the table.
    if body.slot_duration_hours != cfg.SLOT_HOURS:
        factor = cfg.SLOT_HOURS / body.slot_duration_hours
        df["p_j"] = (df["p_j"] * factor).astype(int)
        df["r_j"] = (df["r_j"] * factor).astype(int)
        df["d_j"] = (df["d_j"] * factor).astype(int)
        # Ensure d_j > r_j + p_j after rounding
        df["d_j"] = df.apply(
            lambda row: max(int(row["d_j"]), int(row["r_j"]) + int(row["p_j"]) + 1),
            axis=1,
        )

    return [
        VesselGenerated(
            vessel_id=row["vessel_id"],
            volume_m3=row["stock_acumulado_m3"],
            cargo_m3=row["volume_m3"],
            daily_inflow_m3=row["daily_inflow_m3"],
            release_slot=int(row["r_j"]),
            due_slot=int(row["d_j"]),
            processing_slots=int(row["p_j"]),
            priority_weight=float(row["w_j"]),
        )
        for row in df.to_dict(orient="records")
    ]


@app.get("/history", response_model=list[HistoryEntry])
def get_history() -> list[HistoryEntry]:
    """Return list of past runs persisted in the history file."""
    with _history_lock:
        raw = _read_history()
    return [HistoryEntry(**e) for e in raw]


@app.post("/history", status_code=201)
def post_history(job_id: str, body: HistoryPayload) -> dict[str, str]:
    """Persist a completed result to the history file.

    Args:
        job_id: The job identifier from POST /solve (passed as query parameter).
        body:   HistoryPayload with vessels, config, and full ResultDone.
    """
    summary = HistoryEntry(
        job_id=job_id,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_vessels=body.result.kpis.total_vessels,
        converged=body.result.converged,
        total_weighted_tardiness=body.result.kpis.total_weighted_tardiness,
        solve_time_seconds=body.result.solve_time_seconds,
        sampler=body.result.qubo_stats.sampler_used,
        iterations_used=body.result.kpis.iterations_used,
    )
    entry = {
        **summary.model_dump(),
        "vessels": [v.model_dump() for v in body.vessels],
        "config": body.config.model_dump(),
        "result": body.result.model_dump(),
    }
    with _history_lock:
        entries = _read_history()
        entries.append(entry)
        _write_history(entries)
    return {"status": "ok"}


@app.get("/history/{job_id}", response_model=HistoryEntryFull)
def get_history_entry(job_id: str) -> HistoryEntryFull:
    """Return full persisted data for a single history entry.

    Args:
        job_id: The job identifier used when saving the result.
    """
    with _history_lock:
        entries = _read_history()
    for e in entries:
        if e.get("job_id") == job_id:
            return HistoryEntryFull(**e)
    raise HTTPException(status_code=404, detail=f"History entry '{job_id}' not found.")
