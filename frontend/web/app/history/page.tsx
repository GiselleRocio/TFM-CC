'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { History, RefreshCw, AlertTriangle } from 'lucide-react'
import { getHistory } from '@/lib/api'
import type { HistoryEntry } from '@/types/scheduling'
import Spinner from '@/components/ui/Spinner'
import styles from './page.module.css'

// ── Helpers ──────────────────────────────────────────────────

function formatTimestamp(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString('es-AR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function shortId(id: string): string {
  return id.slice(0, 8).toUpperCase()
}

// ── Component ─────────────────────────────────────────────────

export default function HistoryPage() {
  const router = useRouter()
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getHistory()
      setEntries([...data].reverse()) // most recent first
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al cargar el historial.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <div className={styles.page}>
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.title}>Historial de Corridas</h1>
          <p className={styles.sub}>Resultados guardados del optimizador cuántico</p>
        </div>
        <button
          className={styles.refreshBtn}
          onClick={load}
          disabled={loading}
          aria-label="Recargar historial"
        >
          <RefreshCw size={15} className={loading ? styles.spinning : ''} />
          Recargar
        </button>
      </div>

      {loading && (
        <div className={styles.center}>
          <Spinner size="md" label="Cargando historial…" />
        </div>
      )}

      {!loading && error && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={16} className={styles.errorIcon} />
          <span>{error}</span>
        </div>
      )}

      {!loading && !error && entries.length === 0 && (
        <div className={styles.empty}>
          <History size={32} className={styles.emptyIcon} strokeWidth={1.5} />
          <p className={styles.emptyTitle}>Sin corridas guardadas</p>
          <p className={styles.emptySub}>
            Los resultados guardados desde el wizard aparecerán aquí
          </p>
        </div>
      )}

      {!loading && !error && entries.length > 0 && (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th className={styles.th}>Fecha</th>
                <th className={styles.th}>Job ID</th>
                <th className={`${styles.th} ${styles.thNum}`}>Buques</th>
                <th className={styles.th}>Solver</th>
                <th className={`${styles.th} ${styles.thNum}`}>Iteraciones</th>
                <th className={`${styles.th} ${styles.thNum}`}>Tardanza pond.</th>
                <th className={`${styles.th} ${styles.thNum}`}>Tiempo (s)</th>
                <th className={styles.th}>Convergió</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr
                  key={e.job_id}
                  className={styles.tr}
                  onClick={() => router.push(`/history/${e.job_id}`)}
                  style={{ cursor: 'pointer' }}
                >
                  <td className={styles.td}>{formatTimestamp(e.timestamp)}</td>
                  <td className={styles.td}>
                    <span className={styles.jobId}>{shortId(e.job_id)}</span>
                  </td>
                  <td className={`${styles.td} ${styles.tdNum}`}>{e.n_vessels}</td>
                  <td className={styles.td}>
                    <span className={styles.sampler}>{e.sampler}</span>
                  </td>
                  <td className={`${styles.td} ${styles.tdNum}`}>{e.iterations_used}</td>
                  <td className={`${styles.td} ${styles.tdNum}`}>
                    {e.total_weighted_tardiness.toFixed(2)}
                  </td>
                  <td className={`${styles.td} ${styles.tdNum}`}>
                    {e.solve_time_seconds.toFixed(1)}
                  </td>
                  <td className={styles.td}>
                    <span className={e.converged ? styles.badgeOk : styles.badgeWarn}>
                      {e.converged ? 'Sí' : 'No'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
