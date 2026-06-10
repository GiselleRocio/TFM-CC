// ============================================================
// TFM — Scheduling TypeScript interfaces
// All shapes derived from docs/architecture.md API contracts.
// ============================================================

// ------------------------------------------------------------
// Vessel
// ------------------------------------------------------------

/** Vessel nomination data — input to POST /solve and output of POST /generate */
export interface Vessel {
  vessel_id: string;
  volume_m3: number;
  daily_inflow_m3: number;
  /** Computed by API — cargo loaded in this call */
  cargo_m3?: number;
  release_slot: number;
  due_slot: number;
  /** Macro-slots — p_{j,m}: full pipeline-blocking duration (48h or 96h expressed in slots) */
  processing_slots: number;
  /** ESD — Equivalent Stock in Days; computed as volume_m3 / daily_inflow_m3 */
  priority_weight?: number;
}

// ------------------------------------------------------------
// Terminal configuration
// ------------------------------------------------------------

export interface TerminalConfig {
  n_machines: number;
  start_date: string;
  end_date: string;
  horizon_days: number;
  slot_duration_hours: 12 | 24 | 48;
  min_ullage_days: number;
  n_tanks: number;
  tank_capacity_m3: number;
  initial_terminal_stock_m3: number;
  daily_inflow_m3: number;
  /**
   * Groups of monobuoy indices that share a single submarine pipeline.
   * E.g. [[1,2]] = all share one; [[1],[2]] = fully independent;
   * [[3,6],[2,8,9]] = two separate groups.
   * Monobuoys not listed in any group are treated as independent.
   */
  shared_pipeline_groups?: number[][];
  /**
   * @deprecated use shared_pipeline_groups instead.
   * Kept for backward compatibility with older history entries.
   */
  shared_pipeline?: boolean;
  alpha: number;
  sampler: SamplerOption;
  /** Keys are monobuoy indices ("1", "2", …); values are blocked slot numbers */
  blocked_slots: Record<string, number[]>;
}

export type SamplerOption = "leap_hybrid" | "simulated_annealing";

// ------------------------------------------------------------
// POST /solve
// ------------------------------------------------------------

export interface SolveRequest {
  vessels: Vessel[];
  config: TerminalConfig;
}

/** 202 Accepted response from POST /solve */
export interface SolveResponse {
  job_id: string;
}

// ------------------------------------------------------------
// GET /results/{job_id}
// ------------------------------------------------------------

export interface JobRunning {
  status: "running";
  iteration: number;
  max_iterations: number;
  best_tardiness: number;
  converged: false;
}

export interface ScheduleEntry {
  vessel_id: string;
  /** Monobuoy index (1 = M1, 2 = M2) */
  monobuoy: number;
  start_slot: number;
  end_slot: number;
  priority_weight: number;
  tardiness_slots: number;
  within_window: boolean;
}

export interface KPIs {
  total_weighted_tardiness: number;
  missing_vessels: number;
  pipeline_violations: number;
  buffer_cuts_applied: number;
  tardy_vessels: number;
  total_vessels: number;
  iterations_used: number;
  converged: boolean;
  oversaturated: boolean;
}

export interface QUBOStats {
  n_vars: number;
  n_interactions: number;
  q_matrix_density: number;
  iterations_run: number;
  sampler_used: string;
  penalty_alpha: number;
  n_vessels: number;
  bqm_variables: number;
  buffer_cuts_triples: number;
  /** P₁ = α²·n·c_max */
  p1: number;
  /** P₂ = α·n·c_max */
  p2: number;
  /** P₃ = P₂/2 */
  p3: number;
  /** Maximum cost coefficient */
  c_max: number;
  /** Minimum QUBO energy found by the sampler across all iterations */
  best_energy?: number;
}

export interface InventoryCurveEntry {
  slot: number;
  date: string;
  stock_m3: number;
}

export interface JobDone {
  status: "done";
  iteration: number;
  max_iterations: number;
  converged: boolean;
  solve_time_seconds: number;
  schedule: ScheduleEntry[];
  kpis: KPIs;
  qubo_stats: QUBOStats;
  inventory_curve: InventoryCurveEntry[];
}

export interface JobError {
  status: "error";
  message: string;
  iteration: number;
}

export type JobResult = JobRunning | JobDone | JobError;

// ------------------------------------------------------------
// GET /config/defaults
// ------------------------------------------------------------

export interface ConfigDefaults {
  n_machines: number;
  horizon_days: number;
  slot_duration_hours: 12 | 24 | 48;
  min_ullage_days: number;
  n_tanks: number;
  tank_capacity_m3: number;
  initial_terminal_stock_m3: number;
  daily_inflow_m3: number;
  /** @deprecated use shared_pipeline_groups */
  shared_pipeline?: boolean;
  shared_pipeline_groups: number[][];
  alpha: number;
  alpha_min: number;
  alpha_max: number;
  blocked_slots: Record<string, number[]>;
  max_iterations: number;
  sampler_options: string[];
}

// ------------------------------------------------------------
// POST /generate
// ------------------------------------------------------------

export interface GenerateRequest {
  n_vessels: number;
  slot_duration_hours: 12 | 24 | 48;
  n_machines?: number;
  seed?: number;
}

// ------------------------------------------------------------
// GET /history  /  POST /history
// ------------------------------------------------------------

export interface HistoryEntry {
  job_id: string;
  /** ISO 8601 timestamp */
  timestamp: string;
  n_vessels: number;
  converged: boolean;
  total_weighted_tardiness: number;
  solve_time_seconds: number;
  sampler: string;
  iterations_used: number;
}

export interface HistoryEntryFull extends HistoryEntry {
  vessels: Vessel[] | null;
  config: TerminalConfig | null;
  result: JobDone | null;
}

export interface HistoryPayload {
  vessels: Vessel[];
  config: TerminalConfig;
  result: JobDone;
}

// ------------------------------------------------------------
// Utility types
// ------------------------------------------------------------

/** Priority weight colour tier — used for table cell backgrounds */
export type PriorityTier = "none" | "yellow" | "amber" | "red";

export function getPriorityTier(weight: number): PriorityTier {
  if (weight < 10) return "none";
  if (weight < 15) return "yellow";
  if (weight < 25) return "amber";
  return "red";
}
