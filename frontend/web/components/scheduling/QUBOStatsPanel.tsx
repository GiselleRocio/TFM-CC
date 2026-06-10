import type { KPIs, QUBOStats } from '@/types/scheduling'
import Badge from '@/components/ui/Badge'
import styles from './QUBOStatsPanel.module.css'

interface QUBOStatsPanelProps {
  stats: QUBOStats
  kpis: KPIs
  solveTimeSeconds: number
}

// Penalty hierarchy rows
interface PenaltyRow {
  name: string
  formula: string
  value: number
  checkLabel: string
  passes: boolean
}

export default function QUBOStatsPanel({ stats, kpis, solveTimeSeconds }: QUBOStatsPanelProps) {
  const penaltyRows: PenaltyRow[] = [
    {
      name: 'P₁',
      formula: 'α²·n·c_max',
      value: stats.p1,
      checkLabel: 'P₁ > P₂',
      passes: stats.p1 > stats.p2,
    },
    {
      name: 'P₂',
      formula: 'α·n·c_max',
      value: stats.p2,
      checkLabel: 'P₂ > P₃',
      passes: stats.p2 > stats.p3,
    },
    {
      name: 'P₃',
      formula: 'P₂ / 2',
      value: stats.p3,
      checkLabel: 'P₃ > c_max',
      passes: stats.p3 > stats.c_max,
    },
    {
      name: 'c_max',
      formula: 'max c_{jmt}',
      value: stats.c_max,
      checkLabel: 'Baseline',
      passes: true,
    },
  ]

  const fmt = (n: number) =>
    n >= 1000 ? n.toLocaleString('es-AR', { maximumFractionDigits: 2 }) : n.toFixed(4)

  return (
    <div className={styles.root}>
      {/* ── Top KPIs ──────────────────────────────────────── */}
      <p className={styles.sectionTitle}>Estadísticas QUBO</p>

      <div className={styles.topKpis}>
        <div className={styles.topKpi}>
          <span className={styles.topKpiValue}>{stats.bqm_variables.toLocaleString('es-AR')}</span>
          <span className={styles.topKpiLabel}>Variables BQM</span>
        </div>
        <div className={styles.topKpi}>
          <span className={styles.topKpiValue}>{stats.n_interactions.toLocaleString('es-AR')}</span>
          <span className={styles.topKpiLabel}>Interacciones</span>
        </div>
        <div className={styles.topKpi}>
          <span className={styles.topKpiValue}>{stats.q_matrix_density.toFixed(4)}</span>
          <span className={styles.topKpiLabel}>Densidad Q</span>
        </div>
        <div className={styles.topKpi}>
          <span className={styles.topKpiValue}>{stats.iterations_run}</span>
          <span className={styles.topKpiLabel}>Iteraciones</span>
        </div>
      </div>

      {/* ── Penalty hierarchy ─────────────────────────────── */}
      <div className={styles.tableSection}>
        <p className={styles.tableTitle}>Jerarquía de penalidades (Ec. 16: P₁ &gt; P₂ &gt; P₃ &gt; c_max)</p>

        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead className={styles.thead}>
              <tr>
                <th className={styles.th}>Penalidad</th>
                <th className={styles.th}>Fórmula</th>
                <th className={`${styles.th} ${styles.thNum}`}>Valor</th>
                <th className={styles.th}>Condición</th>
              </tr>
            </thead>
            <tbody>
              {penaltyRows.map((row) => (
                <tr key={row.name} className={styles.tr}>
                  <td className={`${styles.td} ${styles.tdPenalty}`}>{row.name}</td>
                  <td className={`${styles.td} ${styles.tdFormula}`}>{row.formula}</td>
                  <td className={`${styles.td} ${styles.tdNum}`}>{fmt(row.value)}</td>
                  <td className={styles.td}>
                    {row.name === 'c_max' ? (
                      <span className={styles.baseline}>—</span>
                    ) : (
                      <span className={row.passes ? styles.checkPass : styles.checkFail}>
                        {row.passes ? '✓' : '✗'} {row.checkLabel}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className={styles.tableCaption}>
          α = {stats.penalty_alpha}, n = {stats.n_vessels} &nbsp;·&nbsp;
          P₁ = α²·n·c_max &nbsp;·&nbsp; P₂ = α·n·c_max &nbsp;·&nbsp; P₃ = P₂/2
        </p>
      </div>

      {/* ── Run information ───────────────────────────────── */}
      <div className={styles.runInfo}>
        <p className={styles.tableTitle}>Información de ejecución</p>

        <div className={styles.runGrid}>
          <InfoRow label="Sampler">
            <span className={styles.samplerText}>{stats.sampler_used}</span>
          </InfoRow>
          <InfoRow label="Penalty α">{stats.penalty_alpha.toFixed(1)}</InfoRow>
          <InfoRow label="Buques (n)">{stats.n_vessels}</InfoRow>
          <InfoRow label="Variables BQM">{stats.bqm_variables}</InfoRow>
          <InfoRow label="Convergido">
            <Badge variant={kpis.converged ? 'success' : 'warning'}>
              {kpis.converged ? 'Sí' : 'No'}
            </Badge>
          </InfoRow>
          <InfoRow label="Iteraciones">{stats.iterations_run}</InfoRow>
          <InfoRow label="Sobresaturado">
            <Badge variant={kpis.oversaturated ? 'error' : 'neutral'}>
              {kpis.oversaturated ? 'Sí' : 'No'}
            </Badge>
          </InfoRow>
          <InfoRow label="Cortes buffer (triples)">{stats.buffer_cuts_triples}</InfoRow>
          <InfoRow label="Tiempo de resolución">{solveTimeSeconds.toFixed(1)} s</InfoRow>
          {stats.best_energy !== undefined && (
            <InfoRow label="Energía mínima QUBO">{stats.best_energy.toFixed(6)}</InfoRow>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Internal sub-component ────────────────────────────────────

function InfoRow({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className={styles.infoRow}>
      <span className={styles.infoLabel}>{label}</span>
      <span className={styles.infoValue}>{children}</span>
    </div>
  )
}
