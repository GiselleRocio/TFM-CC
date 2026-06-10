// ============================================================
// TFM — Typed API client
// All fetch() calls live here. Components never call fetch directly.
// Base URL from NEXT_PUBLIC_API_URL env var.
// ============================================================

import type {
  ConfigDefaults,
  GenerateRequest,
  HistoryEntry,
  HistoryEntryFull,
  HistoryPayload,
  JobDone,
  JobResult,
  SolveRequest,
  SolveResponse,
  Vessel,
} from "../types/scheduling";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ------------------------------------------------------------
// Internal helper
// ------------------------------------------------------------

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} ${path}: ${body}`);
  }

  // 204 No Content — return undefined cast to T
  if (res.status === 204) return undefined as T;

  return res.json() as Promise<T>;
}

// ------------------------------------------------------------
// Endpoints
// ------------------------------------------------------------

/**
 * POST /solve
 * Launches an async scheduling job. Returns job_id immediately (202).
 */
export function postSolve(request: SolveRequest): Promise<SolveResponse> {
  return apiFetch<SolveResponse>("/solve", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * GET /results/{jobId}
 * Polls job status. Returns running / done / error shape.
 */
export function getResults(jobId: string): Promise<JobResult> {
  return apiFetch<JobResult>(`/results/${jobId}`);
}

/**
 * GET /config/defaults
 * Returns default terminal configuration from src/config.py.
 */
export function getConfigDefaults(): Promise<ConfigDefaults> {
  return apiFetch<ConfigDefaults>("/config/defaults");
}

/**
 * POST /generate
 * Generates synthetic vessel nominations.
 */
export function postGenerate(request: GenerateRequest): Promise<Vessel[]> {
  return apiFetch<Vessel[]>("/generate", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * GET /history
 * Returns list of persisted past runs.
 */
export function getHistory(): Promise<HistoryEntry[]> {
  return apiFetch<HistoryEntry[]>("/history");
}

/**
 * POST /history
 * Persists a completed job result with full input/output data.
 * job_id is required as a query parameter by the API.
 */
export function postHistory(
  jobId: string,
  payload: HistoryPayload,
): Promise<void> {
  return apiFetch<void>(`/history?job_id=${encodeURIComponent(jobId)}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * GET /history/{jobId}
 * Returns the full persisted data for a single history entry.
 */
export function getHistoryEntry(jobId: string): Promise<HistoryEntryFull> {
  return apiFetch<HistoryEntryFull>(`/history/${encodeURIComponent(jobId)}`);
}

/**
 * POST /milp/export
 * Builds the equivalent MILP model and triggers a .lp file download.
 * The file is Gurobi-compatible (LP format).
 *
 * The filename embeds the first 8 chars of jobId: milp_<jobId[:8]>.lp
 */
export async function exportMilp(
  request: SolveRequest,
  jobId: string,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/milp/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} /milp/export: ${body}`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `milp_${jobId.slice(0, 8)}.lp`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * POST /qubo/export
 * Builds the QUBO matrix (identical to iteration k=1 of the solve job) and
 * triggers a .pkl file download in the browser.
 *
 * The filename embeds the first 8 chars of jobId so the file can be traced
 * back to its solve run: qubo_<jobId[:8]>.pkl
 */
export async function exportQubo(
  request: SolveRequest,
  jobId: string,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/qubo/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} /qubo/export: ${body}`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `qubo_${jobId.slice(0, 8)}.pkl`;
  a.click();
  URL.revokeObjectURL(url);
}
