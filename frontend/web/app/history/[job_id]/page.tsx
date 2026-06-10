'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ChevronLeft, AlertTriangle, Download, Package } from 'lucide-react'
import { getHistoryEntry, exportQubo, exportMilp } from '@/lib/api'
import { downloadResultsExcel } from '@/lib/excel'
import type { HistoryEntryFull, SolveRequest } from '@/types/scheduling'
import Spinner from '@/components/ui/Spinner'
import GanttChart from '@/components/scheduling/GanttChart'
import InventoryChart from '@/components/scheduling/InventoryChart'
import KPIPanel from '@/components/scheduling/KPIPanel'
import QUBOStatsPanel from '@/components/scheduling/QUBOStatsPanel'
import { getPriorityTier } from '@/types/scheduling'
import styles from './page.module.css'

// ── Helpers ──────────────────────────────────────────────────

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString('es-AR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const PRIORITY_TIER_CLASS: Record<string, string> = {
  none: '',
  yellow: styles.tierYellow,
  amber: styles.tierAmber,
  red: styles.tierRed,
}

function getPipelineGroups(
  groups: number[][] | undefined,
  shared: boolean | undefined,
  nMachines: number,
): string {
  if (groups && groups.length > 0) {
    return groups.map((g) => g.map((m) => `M${m}`).join('+')).join(', ')
  }
  if (shared === true) {
    const all = Array.from({ length: nMachines }, (_, i) => i + 1)
    return `M${all.join('+M')} (compartido)`
  }
  if (shared === false) {
    return 'Independientes'
  }
  return '—'
}

// ── Component ─────────────────────────────────────────────────

export default function HistoryDetailPage() {
  const params = useParams()
  const router = useRouter()
  const jobId = typeof params.job_id === 'string' ? params.job_id : ''

  const [entry, setEntry] = useState<HistoryEntryFull | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [exportingQubo, setExportingQubo] = useState(false)
  const [exportingMilp, setExportingMilp] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    getHistoryEntry(jobId)
      .then(setEntry)
      .catch((err) => setError(err instanceof Error ? err.message : 'Error al cargar el registro.'))
      .finally(() => setLoading(false))
  }, [jobId])

  const handleExportQubo = async () => {
    if (!entry?.vessels || !entry?.config) return
    setExportingQubo(true)
    setExportError(null)
    try {
      const request: SolveRequest = { vessels: entry.vessels, config: entry.config }
      await exportQubo(request, jobId)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Error al exportar la matriz QUBO.')
    } finally {
      setExportingQubo(false)
    }
  }

  const handleExportMilp = async () => {
    if (!entry?.vessels || !entry?.config) return
    setExportingMilp(true)
    setExportError(null)
    try {
      const request: SolveRequest = { vessels: entry.vessels, config: entry.config }
      await exportMilp(request, jobId)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Error al exportar el modelo MILP.')
    } finally {
      setExportingMilp(false)
    }
  }

  if (loading) {
    return (
      <div className={styles.center}>
        <Spinner size="md" label="Cargando registro…" />
      </div>
    )
  }

  if (error || !entry) {
    return (
      <div className={styles.page}>
        <button className={styles.backBtn} onClick={() => router.back()}>
          <ChevronLeft size={16} /> Volver al historial
        </button>
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={16} />
          <span>{error ?? 'Registro no encontrado.'}</span>
        </div>
      </div>
    )
  }

  const { result, vessels, config } = entry

  return (
    <div className={styles.page}>
      {/* ── Back + header ── */}
      <button className={styles.backBtn} onClick={() => router.back()}>
        <ChevronLeft size={16} /> Volver al historial
      </button>

      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.title}>Detalle de corrida</h1>
          <p className={styles.sub}>
            {formatTimestamp(entry.timestamp)} · {entry.n_vessels} buques ·{' '}
            <span className={entry.converged ? styles.accentText : styles.warnText}>
              {entry.converged ? 'convergido' : 'no convergido'}
            </span>
          </p>
        </div>
        <div className={styles.headerRight}>
          {result && vessels && (
            <button
              className={styles.downloadBtn}
              onClick={() => downloadResultsExcel(vessels, result.schedule, `resultado-${jobId.slice(0, 8)}.xlsx`)}
            >
              <Download size={14} aria-hidden="true" />
              Descargar Excel
            </button>
          )}
          {vessels && config && (
            <button
              className={styles.downloadBtn}
              onClick={handleExportQubo}
              disabled={exportingQubo}
            >
              <Package size={14} aria-hidden="true" />
              {exportingQubo ? 'Exportando…' : 'Exportar QUBO (.pkl)'}
            </button>
          )}
          {vessels && config && (
            <button
              className={styles.downloadBtn}
              onClick={handleExportMilp}
              disabled={exportingMilp}
            >
              <Package size={14} aria-hidden="true" />
              {exportingMilp ? 'Exportando…' : 'Exportar MILP (.lp)'}
            </button>
          )}
          <span className={styles.jobIdBadge}>{jobId.slice(0, 8).toUpperCase()}</span>
        </div>
      </div>

      {exportError && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={16} />
          <span>{exportError}</span>
        </div>
      )}

      {/* ── Config summary ── */}
      {config && (
        <div className={styles.card}>
          <p className={styles.cardTitle}>Configuración</p>
          <div className={styles.configGrid}>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Solver</span>
              <span className={styles.configValue}>{config.sampler}</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Monoboyas</span>
              <span className={styles.configValue}>{config.n_machines}</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Horizonte</span>
              <span className={styles.configValue}>{config.horizon_days} días</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Slot</span>
              <span className={styles.configValue}>{config.slot_duration_hours} h</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Alpha (α)</span>
              <span className={styles.configValue}>{config.alpha.toFixed(1)}</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Buffer mín.</span>
              <span className={styles.configValue}>{config.min_ullage_days} días</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Tanques</span>
              <span className={styles.configValue}>{config.n_tanks}</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Cap. tanque</span>
              <span className={styles.configValue}>{config.tank_capacity_m3?.toLocaleString() ?? 'N/A'} m³</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Stock inicial</span>
              <span className={styles.configValue}>{config.initial_terminal_stock_m3?.toLocaleString() ?? 'N/A'} m³</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Inicio</span>
              <span className={styles.configValue}>{config.start_date}</span>
            </div>
            <div className={styles.configItem}>
              <span className={styles.configLabel}>Oleoductos</span>
              <span className={styles.configValue}>
                {getPipelineGroups(
                  config.shared_pipeline_groups,
                  config.shared_pipeline,
                  config.n_machines,
                )}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Gantt + KPIs ── */}
      {result && vessels && config && (
        <div className={styles.mainPanels}>
          <div className={styles.ganttPanel}>
            <p className={styles.cardTitle}>Diagrama de Gantt — Asignación de atraques</p>
            <GanttChart
              schedule={result.schedule}
              vessels={vessels}
              startDate={config.start_date}
              slotDurationHours={config.slot_duration_hours as 12 | 24 | 48}
              nMachines={config.n_machines}
              blockedSlots={config.blocked_slots}
            />
          </div>
          <div className={styles.kpiPanel}>
            <KPIPanel
              kpis={result.kpis}
              solveTimeSeconds={result.solve_time_seconds}
              maxIterations={result.max_iterations}
              schedule={result.schedule}
            />
          </div>
        </div>
      )}

      {/* ── Inventory Curve ── */}
      {result && config && result.inventory_curve && result.inventory_curve.length > 0 && (
        <InventoryChart
          inventoryCurve={result.inventory_curve}
          config={config}
          slotDurationHours={config.slot_duration_hours}
        />
      )}

      {/* ── QUBO stats ── */}
      {result && (
        <QUBOStatsPanel
          stats={result.qubo_stats}
          kpis={result.kpis}
          solveTimeSeconds={result.solve_time_seconds}
        />
      )}

      {/* ── Vessel input table ── */}
      {vessels && vessels.length > 0 && (
        <div className={styles.card}>
          <p className={styles.cardTitle}>Buques de entrada</p>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Buque</th>
                  <th className={`${styles.th} ${styles.thNum}`}>Slot llegada</th>
                  <th className={`${styles.th} ${styles.thNum}`}>Slot límite</th>
                  <th className={`${styles.th} ${styles.thNum}`}>Slots proc.</th>
                  <th className={`${styles.th} ${styles.thNum}`}>Stock acum. (m³)</th>
                  <th className={`${styles.th} ${styles.thNum}`}>Inflow/día (m³)</th>
                  <th className={`${styles.th} ${styles.thNum}`}>wj (días)</th>
                </tr>
              </thead>
              <tbody>
                {vessels.map((v) => {
                  const wj = v.priority_weight ?? (v.volume_m3 / v.daily_inflow_m3)
                  const tier = getPriorityTier(wj)
                  return (
                    <tr key={v.vessel_id} className={styles.tr}>
                      <td className={styles.td}>
                        <span className={styles.vesselId}>{v.vessel_id}</span>
                      </td>
                      <td className={`${styles.td} ${styles.tdNum}`}>{v.release_slot}</td>
                      <td className={`${styles.td} ${styles.tdNum}`}>{v.due_slot}</td>
                      <td className={`${styles.td} ${styles.tdNum}`}>{v.processing_slots}</td>
                      <td className={`${styles.td} ${styles.tdNum}`}>{v.volume_m3.toLocaleString('es-AR')}</td>
                      <td className={`${styles.td} ${styles.tdNum}`}>{v.daily_inflow_m3.toLocaleString('es-AR')}</td>
                      <td className={`${styles.td} ${styles.tdNum} ${PRIORITY_TIER_CLASS[tier]}`}>
                        {wj.toFixed(1)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
