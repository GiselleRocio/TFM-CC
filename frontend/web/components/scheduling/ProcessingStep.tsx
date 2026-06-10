'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { AlertTriangle, ChevronLeft, RefreshCw } from 'lucide-react'
import { postSolve } from '@/lib/api'
import { useJobPolling } from '@/hooks/useJobPolling'
import type { Vessel, TerminalConfig, JobDone } from '@/types/scheduling'
import Button from '@/components/ui/Button'
import Spinner from '@/components/ui/Spinner'
import styles from './ProcessingStep.module.css'

// ------------------------------------------------------------
// Props
// ------------------------------------------------------------

interface ProcessingStepProps {
  vessels: Vessel[]
  config: TerminalConfig
  onBack: () => void
  onDone: (result: JobDone, jobId: string) => void
}

// ------------------------------------------------------------
// QUBO pre-solve estimate (computed client-side)
// ------------------------------------------------------------

interface QuboEstimate {
  T: number
  nVarsUpper: number
  densityEstimate: number
  avgP: number
  cMaxEstimate: number
  p1: number
  p2: number
  p3: number
}

function computeQuboEstimate(vessels: Vessel[], config: TerminalConfig): QuboEstimate {
  const T = Math.floor((config.horizon_days * 24) / config.slot_duration_hours)
  const n = vessels.length
  const m = config.n_machines
  const nVarsUpper = n * m * T

  // Density estimate: (assign interactions + overlap interactions) / max possible
  // - Assign: for each vessel, all pairs among its m×T variables
  // - Overlap: for each vessel pair × machine pair, the overlap window per t1 is
  //   (P_A + P_B - 1) slots, giving T × (P_A + P_B - 1) / 2 unordered (t1,t2) pairs
  const avgP = vessels.length > 0
    ? vessels.reduce((sum, v) => sum + v.processing_slots, 0) / vessels.length
    : 6
  const avgOverlapWindow = 2 * avgP - 1   // P_A + P_B - 1 when P_A = P_B = avgP
  const mT = m * T
  const assignInteractions = n * (mT * (mT - 1)) / 2
  const overlapInteractions = (n * (n - 1)) / 2 * m * m * T * avgOverlapWindow
  const maxInteractions = nVarsUpper > 1 ? (nVarsUpper * (nVarsUpper - 1)) / 2 : 1
  const densityEstimate = Math.min(1, (assignInteractions + overlapInteractions) / maxInteractions)

  const cMaxEstimate = vessels.reduce((acc, v) => {
    const w = v.priority_weight ?? (v.volume_m3 / v.daily_inflow_m3)
    const maxTardiness = Math.max(0, T - v.due_slot)
    return Math.max(acc, w * maxTardiness)
  }, 0)
  const p2 = config.alpha * n * cMaxEstimate
  const p1 = config.alpha * p2
  const p3 = p2 / 2
  return { T, nVarsUpper, densityEstimate, avgP, cMaxEstimate, p1, p2, p3 }
}

// ------------------------------------------------------------
// Internal phase machine
// ------------------------------------------------------------

type Phase = 'idle' | 'submitting' | 'polling' | 'done' | 'error'

// ------------------------------------------------------------
// Component
// ------------------------------------------------------------

export default function ProcessingStep({ vessels, config, onBack, onDone }: ProcessingStepProps) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [pollingError, setPollingError] = useState<string | null>(null)
  const [wasTimeout, setWasTimeout] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  const startTimeRef = useRef<number>(0)
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Stable refs for callbacks to avoid effect dep issues
  const onDoneRef = useRef(onDone)
  useEffect(() => {
    onDoneRef.current = onDone
  }, [onDone])

  // Only activate polling while in polling phase
  const polling = useJobPolling(phase === 'polling' ? jobId : null)

  // ── Elapsed timer ─────────────────────────────────────────
  useEffect(() => {
    if (phase === 'polling') {
      startTimeRef.current = Date.now()
      setElapsedSeconds(0)
      elapsedTimerRef.current = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)
    } else {
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current)
        elapsedTimerRef.current = null
      }
    }
    return () => {
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current)
    }
  }, [phase])

  // ── React to polling terminal states ─────────────────────
  useEffect(() => {
    if (phase !== 'polling') return

    if (polling.status === 'done' && polling.result) {
      setPhase('done')
      onDoneRef.current(polling.result, jobId!)
    } else if (polling.status === 'error' || polling.status === 'timeout') {
      // Save the message NOW — setPhase('error') will flip phase away from
      // 'polling', causing useJobPolling to receive null and reset its state
      // (including errorMessage back to null) before the next render.
      setPollingError(polling.errorMessage)
      setWasTimeout(polling.status === 'timeout')
      setPhase('error')
    }
  }, [polling.status, polling.result, phase])

  // ── Handlers ──────────────────────────────────────────────
  const handleProcesar = useCallback(async () => {
    setPhase('submitting')
    setSubmitError(null)
    try {
      const { job_id } = await postSolve({ vessels, config })
      setJobId(job_id)
      setPhase('polling')
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Error al enviar la solicitud al solver.')
      setPhase('error')
    }
  }, [vessels, config])

  const handleRetry = useCallback(() => {
    polling.stop()
    setPhase('idle')
    setJobId(null)
    setSubmitError(null)
    setPollingError(null)
    setWasTimeout(false)
    setElapsedSeconds(0)
  }, [polling])

  // ── Helpers ───────────────────────────────────────────────
  const formatElapsed = (secs: number): string => {
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return m > 0 ? `${m}m ${s.toString().padStart(2, '0')}s` : `${s}s`
  }

  const progressPct =
    polling.maxIterations > 0
      ? Math.min(100, Math.round((polling.iteration / polling.maxIterations) * 100))
      : 0

  const errorMessage =
    submitError ??
    pollingError ??
    'Error desconocido. Revise la consola del servidor.'

  const samplerLabel =
    config.sampler === 'leap_hybrid' ? 'LeapHybridSampler' : 'SimulatedAnnealingSampler'

  const isActive = phase === 'submitting' || phase === 'polling'

  const quboEstimate = computeQuboEstimate(vessels, config)

  const fmtNum = (n: number) =>
    n >= 1000 ? n.toLocaleString('es-AR', { maximumFractionDigits: 2 }) : n.toFixed(2)

  // ── Render ────────────────────────────────────────────────
  return (
    <div className={styles.root}>
      {/* Header */}
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Procesamiento</h1>
        <p className={styles.subtitle}>
          Optimización híbrida cuántica-clásica · {vessels.length} buque
          {vessels.length !== 1 ? 's' : ''} · {config.n_machines} monoboya
          {config.n_machines !== 1 ? 's' : ''} · Horizonte {config.horizon_days} días
        </p>
      </div>

      {/* Config summary card */}
      <div className={styles.summaryCard}>
        <p className={styles.summaryTitle}>Parámetros de optimización</p>
        <div className={styles.summaryGrid}>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Buques</span>
            <span className={styles.summaryValue}>{vessels.length}</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Monoboyas</span>
            <span className={styles.summaryValue}>{config.n_machines}</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Horizonte</span>
            <span className={styles.summaryValue}>{config.horizon_days} días</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Slots/día</span>
            <span className={styles.summaryValue}>{24 / config.slot_duration_hours}</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Alpha (α)</span>
            <span className={styles.summaryValue}>{config.alpha.toFixed(1)}</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Solver</span>
            <span className={styles.summaryValueMono}>{samplerLabel}</span>
          </div>
        </div>
      </div>

      {/* ── QUBO Estimate (idle + polling) ───────────────── */}
      {(phase === 'idle' || phase === 'polling' || phase === 'submitting') && (
        <div className={styles.quboCard}>
          <p className={styles.quboTitle}>Estimación QUBO</p>

          <div className={styles.quboKpis}>
            <div className={styles.quboKpi}>
              <span className={styles.quboKpiValue}>{quboEstimate.T.toLocaleString('es-AR')}</span>
              <span className={styles.quboKpiLabel}>T (slots)</span>
            </div>
            <div className={styles.quboKpi}>
              <span className={styles.quboKpiValue}>{quboEstimate.nVarsUpper.toLocaleString('es-AR')}</span>
              <span className={styles.quboKpiLabel}>Variables (n×m×T)</span>
            </div>
            <div className={styles.quboKpi}>
              <span className={styles.quboKpiValue}>{quboEstimate.densityEstimate.toFixed(4)}</span>
              <span className={styles.quboKpiLabel}>Densidad Q est.</span>
            </div>
            <div className={styles.quboKpi}>
              <span className={styles.quboKpiValue}>{fmtNum(quboEstimate.cMaxEstimate)}</span>
              <span className={styles.quboKpiLabel}>c_max estimado</span>
            </div>
          </div>

          <div className={styles.quboTableWrap}>
            <table className={styles.quboTable}>
              <thead>
                <tr>
                  <th className={styles.quboTh}>Penalidad</th>
                  <th className={styles.quboTh}>Fórmula</th>
                  <th className={`${styles.quboTh} ${styles.quboThNum}`}>Valor est.</th>
                  <th className={styles.quboTh}>Condición</th>
                </tr>
              </thead>
              <tbody>
                <tr className={styles.quboTr}>
                  <td className={styles.quboTdPenalty}>P₁</td>
                  <td className={styles.quboTdFormula}>α²·n·c_max</td>
                  <td className={`${styles.quboTd} ${styles.quboTdNum}`}>{fmtNum(quboEstimate.p1)}</td>
                  <td className={styles.quboTd}>
                    <span className={quboEstimate.p1 > quboEstimate.p2 ? styles.checkPass : styles.checkFail}>
                      {quboEstimate.p1 > quboEstimate.p2 ? '✓' : '✗'} P₁ &gt; P₂
                    </span>
                  </td>
                </tr>
                <tr className={styles.quboTr}>
                  <td className={styles.quboTdPenalty}>P₂</td>
                  <td className={styles.quboTdFormula}>α·n·c_max</td>
                  <td className={`${styles.quboTd} ${styles.quboTdNum}`}>{fmtNum(quboEstimate.p2)}</td>
                  <td className={styles.quboTd}>
                    <span className={quboEstimate.p2 > quboEstimate.p3 ? styles.checkPass : styles.checkFail}>
                      {quboEstimate.p2 > quboEstimate.p3 ? '✓' : '✗'} P₂ &gt; P₃
                    </span>
                  </td>
                </tr>
                <tr className={styles.quboTr}>
                  <td className={styles.quboTdPenalty}>P₃</td>
                  <td className={styles.quboTdFormula}>P₂ / 2</td>
                  <td className={`${styles.quboTd} ${styles.quboTdNum}`}>{fmtNum(quboEstimate.p3)}</td>
                  <td className={styles.quboTd}>
                    <span className={quboEstimate.p3 > quboEstimate.cMaxEstimate ? styles.checkPass : styles.checkFail}>
                      {quboEstimate.p3 > quboEstimate.cMaxEstimate ? '✓' : '✗'} P₃ &gt; c_max
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {quboEstimate.cMaxEstimate === 0 && (
            <p className={styles.quboWarn}>
              c_max = 0: todos los buques tienen slot límite ≥ T. El QUBO no tiene costo de tardanza;
              la jerarquía de penalidades no aplica para esta configuración.
            </p>
          )}
          <p className={styles.quboCaption}>
            Variables y densidad = cotas aproximadas (sin filtrar slots bloqueados ni ventana de factibilidad).
            Densidad de solapamiento calculada con ventana promedio {(2 * quboEstimate.avgP - 1).toFixed(1)} slots (P_avg = {quboEstimate.avgP.toFixed(1)}).
            Penalidades calculadas con α = {config.alpha}, n = {vessels.length}.
          </p>
        </div>
      )}

      {/* ── IDLE ─────────────────────────────────────────── */}
      {phase === 'idle' && (
        <div className={styles.idleCard}>
          <p className={styles.idleText}>
            Revise los parámetros y presione <strong>Procesar</strong> para iniciar la
            optimización. El solver corre de forma asíncrona; puede tardar varios minutos según el
            tamaño del problema y el sampler seleccionado.
          </p>
        </div>
      )}

      {/* ── SUBMITTING ───────────────────────────────────── */}
      {phase === 'submitting' && (
        <div className={styles.progressCard}>
          <div className={styles.spinnerRow}>
            <Spinner size="sm" label="Enviando solicitud al solver…" />
            <span className={styles.spinnerLabel}>Enviando solicitud al solver…</span>
          </div>
        </div>
      )}

      {/* ── POLLING ──────────────────────────────────────── */}
      {phase === 'polling' && (
        <div className={styles.progressCard}>
          <div className={styles.spinnerRow}>
            <Spinner size="sm" label="Optimizando" />
            <span className={styles.pollingTitle}>Optimizando asignación de atraques…</span>
          </div>

          {/* Progress bar */}
          <div className={styles.progressSection}>
            <div className={styles.progressHeader}>
              <span className={styles.progressLabel}>
                Iteración&nbsp;
                <span className={styles.progressCount}>
                  {polling.iteration}
                </span>
                &nbsp;/&nbsp;
                <span className={styles.progressCount}>
                  {polling.maxIterations > 0 ? polling.maxIterations : '—'}
                </span>
              </span>
              {polling.maxIterations > 0 && (
                <span className={styles.progressPct}>{progressPct}%</span>
              )}
            </div>
            <div className={styles.progressTrack} role="progressbar" aria-valuenow={progressPct} aria-valuemin={0} aria-valuemax={100}>
              <div className={styles.progressFill} style={{ width: `${progressPct}%` }} />
            </div>
          </div>

          {/* Live metrics */}
          <div className={styles.metricsRow}>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Mejor tardanza ponderada</span>
              <span className={styles.metricValue}>
                {polling.bestTardiness !== null ? polling.bestTardiness.toFixed(2) : '—'}
              </span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Tiempo transcurrido</span>
              <span className={styles.metricValue}>{formatElapsed(elapsedSeconds)}</span>
            </div>
          </div>

          <p className={styles.pollingNote}>
            No cierre esta pestaña. El resultado se cargará automáticamente al finalizar.
          </p>
        </div>
      )}

      {/* ── ERROR / TIMEOUT ──────────────────────────────── */}
      {phase === 'error' && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={16} className={styles.errorIcon} aria-hidden="true" />
          <div className={styles.errorContent}>
            <span className={styles.errorTitle}>
              {wasTimeout ? 'Tiempo de espera agotado' : 'Error del solver'}
            </span>
            <span className={styles.errorMessage}>{errorMessage}</span>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className={styles.footer}>
        <Button variant="ghost" onClick={onBack} disabled={isActive}>
          <ChevronLeft size={16} />
          ← Volver
        </Button>

        <div className={styles.footerRight}>
          {phase === 'error' && (
            <Button variant="ghost" onClick={handleRetry}>
              <RefreshCw size={15} />
              Reintentar
            </Button>
          )}

          {(phase === 'idle' || phase === 'error') && (
            <Button variant="primary" onClick={handleProcesar}>
              Procesar
            </Button>
          )}

          {isActive && (
            <Button variant="primary" disabled loading>
              Procesando…
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
