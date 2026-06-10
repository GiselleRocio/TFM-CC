"use client";

// ============================================================
// TFM — useJobPolling
// Polls GET /results/{jobId} every 2 s until done/error/timeout.
// Pass jobId=null to keep the hook idle (no requests).
// ============================================================

import { useEffect, useRef, useState } from "react";
import { getResults } from "@/lib/api";
import type { JobDone } from "@/types/scheduling";

const POLL_INTERVAL_MS = 2_000;
const TIMEOUT_MS = 35 * 60 * 1_000; // 15 minutes

// ------------------------------------------------------------
// Public types
// ------------------------------------------------------------

export type PollingStatus = "idle" | "polling" | "done" | "error" | "timeout";

export interface JobPollingState {
  status: PollingStatus;
  /** Current solver iteration (0 while idle) */
  iteration: number;
  /** Max iterations reported by the API */
  maxIterations: number;
  /** Best weighted tardiness seen so far — null until first running response */
  bestTardiness: number | null;
  /** Human-readable error or timeout message */
  errorMessage: string | null;
  /** Full result payload — only set when status === 'done' */
  result: JobDone | null;
}

// ------------------------------------------------------------
// Hook
// ------------------------------------------------------------

export function useJobPolling(
  jobId: string | null,
): JobPollingState & { stop: () => void } {
  const [state, setState] = useState<JobPollingState>({
    status: "idle",
    iteration: 0,
    maxIterations: 0,
    bestTardiness: null,
    errorMessage: null,
    result: null,
  });

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startTimeRef = useRef<number>(0);
  const stoppedRef = useRef(false);

  const clearTimer = () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const stop = () => {
    stoppedRef.current = true;
    clearTimer();
  };

  useEffect(() => {
    if (!jobId) {
      // No active job — reset to idle without clearing a previous result
      setState((prev) =>
        prev.status === "idle"
          ? prev
          : {
              status: "idle",
              iteration: 0,
              maxIterations: 0,
              bestTardiness: null,
              errorMessage: null,
              result: null,
            },
      );
      return;
    }

    stoppedRef.current = false;
    startTimeRef.current = Date.now();

    setState({
      status: "polling",
      iteration: 0,
      maxIterations: 0,
      bestTardiness: null,
      errorMessage: null,
      result: null,
    });

    const poll = async () => {
      if (stoppedRef.current) return;

      // ── Timeout guard ──────────────────────────────────────
      if (Date.now() - startTimeRef.current > TIMEOUT_MS) {
        setState((prev) => ({
          ...prev,
          status: "timeout",
          errorMessage:
            "El solver tardó más de 5 minutos sin responder. Verifique el servidor e intente nuevamente.",
        }));
        return;
      }

      // ── Poll ───────────────────────────────────────────────
      try {
        const res = await getResults(jobId);
        if (stoppedRef.current) return;

        if (res.status === "running") {
          setState((prev) => ({
            ...prev,
            status: "polling",
            iteration: res.iteration,
            maxIterations: res.max_iterations,
            bestTardiness: res.best_tardiness,
          }));
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
        } else if (res.status === "done") {
          setState({
            status: "done",
            iteration: res.iteration,
            maxIterations: res.max_iterations,
            bestTardiness: null,
            errorMessage: null,
            result: res,
          });
          // No next tick — terminal state
        } else if (res.status === "error") {
          setState((prev) => ({
            ...prev,
            status: "error",
            errorMessage: res.message,
          }));
        }
      } catch (err) {
        if (stoppedRef.current) return;
        setState((prev) => ({
          ...prev,
          status: "error",
          errorMessage:
            err instanceof Error
              ? err.message
              : "Error de conexión al servidor. Verifique que la API esté activa.",
        }));
      }
    };

    // Start first poll immediately
    poll();

    return () => {
      stoppedRef.current = true;
      clearTimer();
    };
  }, [jobId]);

  return { ...state, stop };
}
