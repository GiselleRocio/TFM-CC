'use client'

import { useState } from 'react'
import { RotateCcw, Save, CheckCircle, AlertTriangle, Download, Package } from 'lucide-react'
import { postHistory, exportQubo, exportMilp } from '@/lib/api'
import { downloadResultsExcel } from '@/lib/excel'
import type { JobDone, Vessel, TerminalConfig, HistoryPayload, SolveRequest } from '@/types/scheduling'
import Button from '@/components/ui/Button'
import GanttChart from './GanttChart'
import InventoryChart from './InventoryChart'
import KPIPanel from './KPIPanel'
import QUBOStatsPanel from './QUBOStatsPanel'
import styles from './ResultsStep.module.css'

// ------------------------------------------------------------
// Props
// ------------------------------------------------------------

interface ResultsStepProps {
  result: JobDone
  jobId: string
  vessels: Vessel[]
  config: TerminalConfig
  onNewRun: () => void
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

function getPipelineGroupsLabel(
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

// ------------------------------------------------------------
// Component
// ------------------------------------------------------------

type SaveStatus = 'idle' | 'saving' | 'saved' | 'error'
type ExportStatus = 'idle' | 'exporting' | 'error'

export default function ResultsStep({ result, jobId, vessels, config, onNewRun }: ResultsStepProps) {
  const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle')
  const [saveError, setSaveError] = useState<string | null>(null)
  const [exportStatus, setExportStatus] = useState<ExportStatus>('idle')
  const [exportError, setExportError] = useState<string | null>(null)
  const [exportMilpStatus, setExportMilpStatus] = useState<ExportStatus>('idle')
  const [exportMilpError, setExportMilpError] = useState<string | null>(null)

  const handleExportQubo = async () => {
    setExportStatus('exporting')
    setExportError(null)
    try {
      const request: SolveRequest = { vessels, config }
      await exportQubo(request, jobId)
      setExportStatus('idle')
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Error al exportar la matriz QUBO.')
      setExportStatus('error')
    }
  }

  const handleExportMilp = async () => {
    setExportMilpStatus('exporting')
    setExportMilpError(null)
    try {
      const request: SolveRequest = { vessels, config }
      await exportMilp(request, jobId)
      setExportMilpStatus('idle')
    } catch (err) {
      setExportMilpError(err instanceof Error ? err.message : 'Error al exportar el modelo MILP.')
      setExportMilpStatus('error')
    }
  }

  const handleSave = async () => {
    setSaveStatus('saving')
    setSaveError(null)
    try {
      const payload: HistoryPayload = { vessels, config, result }
      await postHistory(jobId, payload)
      setSaveStatus('saved')
    } catch (err) {
      setSaveError(
        err instanceof Error ? err.message : 'Error al guardar el resultado.',
      )
      setSaveStatus('error')
    }
  }

  const convergeLabel = result.kpis.converged ? 'convergido' : 'no convergido'
  const tardyLabel =
    result.kpis.tardy_vessels === 0
      ? 'sin tardanza'
      : `${result.kpis.tardy_vessels} tardío${result.kpis.tardy_vessels > 1 ? 's' : ''}`

  return (
    <div className={styles.root}>
      {/* Header */}
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Resultados</h1>
        <p className={styles.subtitle}>
          {result.kpis.total_vessels} buques · {convergeLabel} · {tardyLabel} ·{' '}
          {result.solve_time_seconds.toFixed(1)} s
        </p>
      </div>

      {/* ── Config card (same style as history detail) ── */}
      <div className={styles.configCard}>
        <p className={styles.configCardTitle}>Configuración</p>
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
            <span className={styles.configValue}>{config.slot_duration_hours}h</span>
          </div>
          <div className={styles.configItem}>
            <span className={styles.configLabel}>Oleoductos</span>
            <span className={styles.configValue}>
              {getPipelineGroupsLabel(
                config.shared_pipeline_groups,
                config.shared_pipeline,
                config.n_machines,
              )}
            </span>
          </div>
          <div className={styles.configItem}>
            <span className={styles.configLabel}>Alpha</span>
            <span className={styles.configValue}>{config.alpha.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {/* ── Main panels: Gantt (65%) + KPI (35%) ── */}
      <div className={styles.mainPanels}>
        <div className={styles.ganttPanel}>
          <p className={styles.panelTitle}>Diagrama de Gantt — Asignación de atraques</p>
          <GanttChart
            schedule={result.schedule}
            vessels={vessels}
            startDate={config.start_date}
            slotDurationHours={config.slot_duration_hours}
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

      {/* ── Inventory Curve ── */}
      {result.inventory_curve && result.inventory_curve.length > 0 && (
        <InventoryChart
          inventoryCurve={result.inventory_curve}
          config={config}
          slotDurationHours={config.slot_duration_hours}
        />
      )}

      {/* ── QUBO Stats ── */}
      <QUBOStatsPanel
        stats={result.qubo_stats}
        kpis={result.kpis}
        solveTimeSeconds={result.solve_time_seconds}
      />

      {/* ── Save feedback ── */}
      {saveStatus === 'saved' && (
        <div className={styles.saveBanner}>
          <CheckCircle size={15} aria-hidden="true" />
          Resultado guardado correctamente en el historial.
        </div>
      )}

      {saveStatus === 'error' && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={15} aria-hidden="true" />
          {saveError}
        </div>
      )}

      {exportStatus === 'error' && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={15} aria-hidden="true" />
          {exportError}
        </div>
      )}

      {exportMilpStatus === 'error' && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={15} aria-hidden="true" />
          {exportMilpError}
        </div>
      )}

      {/* Footer */}
      <div className={styles.footer}>
        <Button variant="ghost" onClick={onNewRun}>
          <RotateCcw size={15} />
          Nueva corrida
        </Button>

        <Button
          variant="ghost"
          onClick={() => downloadResultsExcel(vessels, result.schedule)}
        >
          <Download size={15} />
          Descargar Excel
        </Button>

        <Button
          variant="ghost"
          onClick={handleExportQubo}
          loading={exportStatus === 'exporting'}
        >
          <Package size={15} />
          Exportar QUBO (.pkl)
        </Button>

        <Button
          variant="ghost"
          onClick={handleExportMilp}
          loading={exportMilpStatus === 'exporting'}
        >
          <Package size={15} />
          Exportar MILP (.lp)
        </Button>

        <Button
          variant="ghost-teal"
          onClick={handleSave}
          disabled={saveStatus === 'saved'}
          loading={saveStatus === 'saving'}
        >
          <Save size={15} />
          {saveStatus === 'saved' ? 'Guardado' : 'Guardar resultado'}
        </Button>
      </div>
    </div>
  )
}
