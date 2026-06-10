import type { KPIs, ScheduleEntry } from '@/types/scheduling'
import Badge from '@/components/ui/Badge'
import styles from './KPIPanel.module.css'

interface KPIPanelProps {
  kpis: KPIs
  solveTimeSeconds: number
  maxIterations: number
  schedule: ScheduleEntry[]
}

interface MetricRowProps {
  label: string
  value: string
  accent?: boolean
  warn?: boolean
}

function MetricRow({ label, value, accent, warn }: MetricRowProps) {
  const valueClass = [
    styles.metricValue,
    accent ? styles.metricValueAccent : '',
    warn ? styles.metricValueWarn : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className={styles.metricRow}>
      <span className={styles.metricLabel}>{label}</span>
      <span className={valueClass}>{value}</span>
    </div>
  )
}

export default function KPIPanel({
  kpis,
  solveTimeSeconds,
  maxIterations,
  schedule,
}: KPIPanelProps) {
  const avgWeight =
    schedule.length > 0
      ? schedule.reduce((s, e) => s + e.priority_weight, 0) / schedule.length
      : 0

  const tardyFraction = kpis.total_vessels > 0 ? kpis.tardy_vessels / kpis.total_vessels : 0

  return (
    <div className={styles.root}>
      <p className={styles.sectionTitle}>KPIs del resultado</p>

      {/* Primary KPI — weighted tardiness */}
      <div className={styles.primaryKpi}>
        <span className={styles.primaryKpiLabel}>Tardanza ponderada (Σ wj·Tj)</span>
        <span className={styles.primaryKpiValue}>
          {kpis.total_weighted_tardiness.toFixed(2)}
        </span>
      </div>

      <div className={styles.divider} />

      {/* Secondary metrics */}
      <div className={styles.metricList}>
        <MetricRow
          label="Buques tardíos"
          value={`${kpis.tardy_vessels} / ${kpis.total_vessels}`}
          warn={tardyFraction > 0}
        />
        <MetricRow
          label="Peso de prioridad promedio (w̄)"
          value={avgWeight.toFixed(2)}
        />
        <MetricRow
          label="Iteraciones usadas"
          value={`${kpis.iterations_used} / ${maxIterations}`}
        />

        <div className={styles.metricRow}>
          <span className={styles.metricLabel}>Convergido</span>
          <Badge variant={kpis.converged ? 'success' : 'warning'}>
            {kpis.converged ? 'Sí' : 'No'}
          </Badge>
        </div>

        <MetricRow
          label="Tiempo de resolución"
          value={`${solveTimeSeconds.toFixed(1)} s`}
        />

        {kpis.missing_vessels > 0 && (
          <MetricRow
            label="Buques no asignados"
            value={kpis.missing_vessels.toString()}
            warn
          />
        )}

        {kpis.pipeline_violations > 0 && (
          <MetricRow
            label="Violaciones pipeline"
            value={kpis.pipeline_violations.toString()}
            warn
          />
        )}

        {kpis.buffer_cuts_applied > 0 && (
          <MetricRow
            label="Cortes buffer aplicados"
            value={kpis.buffer_cuts_applied.toString()}
          />
        )}
      </div>
    </div>
  )
}
