'use client'

import { useRef, useState, useEffect } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import Button from '@/components/ui/Button'
import { slotToDate, formatDateShort } from '@/lib/utils'
import { getPriorityTier } from '@/types/scheduling'
import styles from './VesselWindowPreview.module.css'

interface VesselWindowInput {
  vessel_id: string
  r_j: number
  d_j: number
  p_j: number
  w_j: number
  stock_acumulado_m3: number
  daily_inflow_m3: number
}

interface ParsedInstance {
  label: string
  vessels: VesselWindowInput[]
}

const LEFT_W     = 130
const RIGHT_PAD  = 24
const ROW_H      = 56
const AXIS_H     = 52
const WIN_PAD_V  = 14
const PROC_PAD_V = 8
const BAR_R      = 4

const TEAL   = '#00C4B4'
const RED    = '#E24B4A'
const AMBER  = '#F59E0B'
const YELLOW = '#EAB308'

function tierColor(wj: number): string {
  const tier = getPriorityTier(wj)
  if (tier === 'red')    return RED
  if (tier === 'amber')  return AMBER
  if (tier === 'yellow') return YELLOW
  return TEAL
}

function tickStepMs(spanMs: number): number {
  const days = spanMs / 86_400_000
  if (days <= 7)  return 86_400_000
  if (days <= 21) return 3 * 86_400_000
  if (days <= 50) return 5 * 86_400_000
  return 7 * 86_400_000
}

interface TooltipState {
  screenX: number
  screenY: number
  vessel: VesselWindowInput
}

const DEFAULT_DATE = new Date().toISOString().slice(0, 10)

const REQUIRED_FIELDS = ['vessel_id', 'r_j', 'd_j', 'p_j', 'w_j']

function splitIntoJsonArrayBlocks(raw: string): string[] {
  // Handles three formats:
  //   1. One JSON array per line (no separator)
  //   2. Blocks separated by --- on its own line
  //   3. A single JSON array (possibly multiline)
  const trimmed = raw.trim()

  // If there's a --- separator, use that
  if (/\n\s*---+\s*\n/.test(trimmed)) {
    return trimmed.split(/\n\s*---+\s*\n/).map(b => b.trim()).filter(Boolean)
  }

  // Try splitting by top-level array boundaries: find every line that starts with '['
  // and collect characters until the matching ']' is closed (depth tracking).
  const blocks: string[] = []
  let depth = 0
  let start = -1
  let inString = false
  let escape = false

  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i]
    if (escape) { escape = false; continue }
    if (ch === '\\' && inString) { escape = true; continue }
    if (ch === '"') { inString = !inString; continue }
    if (inString) continue

    if (ch === '[') {
      if (depth === 0) start = i
      depth++
    } else if (ch === ']') {
      depth--
      if (depth === 0 && start !== -1) {
        blocks.push(trimmed.slice(start, i + 1))
        start = -1
      }
    }
  }

  return blocks.length > 0 ? blocks : [trimmed]
}

function parseInstances(raw: string): { instances: ParsedInstance[] | null; error: string | null } {
  const trimmed = raw.trim()
  if (!trimmed) return { instances: null, error: 'Pegá el JSON de buques antes de visualizar.' }

  // Extract label lines (# or //) that precede each block
  const labeledBlocks: { label: string; json: string }[] = []
  const lines = trimmed.split('\n')
  let pendingLabel: string | null = null
  let jsonBuffer: string[] = []

  const flush = () => {
    const json = jsonBuffer.join('\n').trim()
    if (json) {
      const label = pendingLabel ?? `Instancia ${labeledBlocks.length + 1}`
      labeledBlocks.push({ label, json })
    }
    pendingLabel = null
    jsonBuffer = []
  }

  for (const line of lines) {
    const labelMatch = line.match(/^\s*(?:#|\/\/)\s*(.+)$/)
    if (labelMatch) {
      if (jsonBuffer.length > 0) flush()
      pendingLabel = labelMatch[1].trim()
    } else {
      jsonBuffer.push(line)
    }
  }
  flush()

  // Now split any json buffers that contain multiple arrays
  const expandedBlocks: { label: string; json: string }[] = []
  for (const block of labeledBlocks) {
    const subBlocks = splitIntoJsonArrayBlocks(block.json)
    if (subBlocks.length === 1) {
      expandedBlocks.push(block)
    } else {
      subBlocks.forEach((sub, si) => {
        expandedBlocks.push({ label: `${block.label} ${si + 1}`, json: sub })
      })
    }
  }

  const instances: ParsedInstance[] = []

  for (let i = 0; i < expandedBlocks.length; i++) {
    const { label, json } = expandedBlocks[i]
    if (!json.trim()) continue

    let parsed: unknown
    try {
      parsed = JSON.parse(json)
    } catch {
      return { instances: null, error: `Bloque ${i + 1} ("${label}"): JSON inválido. Verificá la sintaxis.` }
    }

    if (!Array.isArray(parsed) || parsed.length === 0) {
      return { instances: null, error: `Bloque ${i + 1}: debe ser un array con al menos un buque.` }
    }

    const missing = REQUIRED_FIELDS.filter(k => !(k in (parsed as Record<string, unknown>[])[0]))
    if (missing.length > 0) {
      return { instances: null, error: `Bloque ${i + 1}: faltan campos: ${missing.join(', ')}` }
    }

    instances.push({ label, vessels: parsed as VesselWindowInput[] })
  }

  if (instances.length === 0) return { instances: null, error: 'No se encontró ningún array válido.' }

  return { instances, error: null }
}

// ── Single chart component ────────────────────────────────────────────────────

interface InstanceChartProps {
  vessels: VesselWindowInput[]
  instanceIndex: number
  startDate: string
  slotHours: 12 | 24 | 48
}

function InstanceChart({ vessels, instanceIndex, startDate, slotHours }: InstanceChartProps) {
  const [tooltip, setTooltip]   = useState<TooltipState | null>(null)
  const [svgWidth, setSvgWidth] = useState(0)
  const chartWrapRef            = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = chartWrapRef.current
    if (!el) return
    setSvgWidth(el.clientWidth || 900)
    const obs = new ResizeObserver(entries => setSvgWidth(entries[0].contentRect.width || 900))
    obs.observe(el)
    return () => obs.disconnect()
  })

  const renderSvg = () => {
    if (svgWidth === 0) return null

    const slotMs       = slotHours * 3_600_000
    const startDateFmt = startDate.replace(/-/g, '/')
    const toMs         = (slot: number) => slotToDate(slot, startDateFmt, slotHours).getTime()
    const startBaseMs  = new Date(startDate).getTime()

    const allMs     = vessels.flatMap(v => [toMs(v.r_j), toMs(v.d_j)])
    const domainMin = Math.min(...allMs) - slotMs * 2
    const domainMax = Math.max(...allMs) + slotMs * 2
    const chartW    = Math.max(1, svgWidth - LEFT_W - RIGHT_PAD)
    const chartH    = vessels.length * ROW_H
    const svgH      = chartH + AXIS_H
    const timeToX   = (ms: number) => LEFT_W + ((ms - domainMin) / (domainMax - domainMin)) * chartW

    const firstSlot = Math.floor((domainMin - startBaseMs) / slotMs)
    const lastSlot  = Math.ceil((domainMax - startBaseMs) / slotMs)
    const slotBoundaries: { ms: number; slot: number }[] = []
    for (let s = firstSlot; s <= lastSlot; s++) {
      const ms = toMs(s)
      if (ms >= domainMin && ms <= domainMax) slotBoundaries.push({ ms, slot: s })
    }

    const stepMs    = tickStepMs(domainMax - domainMin)
    const firstTick = Math.ceil(domainMin / stepMs) * stepMs
    const dateTicks: number[] = []
    for (let t = firstTick; t <= domainMax; t += stepMs) dateTicks.push(t)

    const clipId    = `vwp-clip-${instanceIndex}`
    const TOOLTIP_W = 230
    const tooltipLeft = tooltip
      ? tooltip.screenX + 14 + TOOLTIP_W > svgWidth ? tooltip.screenX - TOOLTIP_W - 8 : tooltip.screenX + 14
      : 0

    return (
      <>
        <svg width={svgWidth} height={svgH} className={styles.svgRoot} aria-label="Ventanas de arribo">
          <defs>
            <clipPath id={clipId}>
              <rect x={LEFT_W} y={0} width={chartW} height={chartH} />
            </clipPath>
          </defs>

          <rect x={LEFT_W} y={0} width={chartW} height={chartH} fill="rgba(255,255,255,0.012)" />

          <g clipPath={`url(#${clipId})`}>
            {slotBoundaries.map(({ ms, slot }, i) => {
              const x = timeToX(ms)
              return (
                <g key={i}>
                  <line x1={x} y1={0} x2={x} y2={chartH} stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
                  <text x={x + 3} y={13} fontSize="8" fill="rgba(0,196,180,0.3)" fontFamily="var(--font-sans)">
                    s{slot}
                  </text>
                </g>
              )
            })}
          </g>

          {vessels.map((v, idx) => {
            const rowY   = idx * ROW_H
            const isEven = idx % 2 === 0
            const color  = tierColor(v.w_j)
            const relX   = timeToX(toMs(v.r_j))
            const dueX   = timeToX(toMs(v.d_j))
            const procX2 = timeToX(toMs(v.r_j + v.p_j))
            const winY   = rowY + WIN_PAD_V
            const winH   = ROW_H - WIN_PAD_V * 2
            const procY  = rowY + PROC_PAD_V
            const procH  = ROW_H - PROC_PAD_V * 2
            const winW   = Math.max(4, dueX - relX)
            const procW  = Math.max(4, procX2 - relX)
            const badgeW = 38
            const badgeH = 17
            const badgeX = LEFT_W - badgeW - 6
            const badgeY = rowY + ROW_H / 2 - badgeH / 2
            const procClipId = `vwp-p-${instanceIndex}-${idx}`

            return (
              <g key={v.vessel_id}
                onMouseEnter={e => {
                  const wrap = chartWrapRef.current
                  if (!wrap) return
                  const rect = wrap.getBoundingClientRect()
                  setTooltip({ screenX: e.clientX - rect.left, screenY: e.clientY - rect.top, vessel: v })
                }}
                onMouseLeave={() => setTooltip(null)}
                style={{ cursor: 'default' }}
              >
                <rect x={0} y={rowY} width={svgWidth} height={ROW_H}
                  fill={isEven ? 'rgba(255,255,255,0.02)' : 'transparent'} />
                <line x1={LEFT_W} y1={rowY + ROW_H} x2={svgWidth - RIGHT_PAD} y2={rowY + ROW_H}
                  stroke="rgba(255,255,255,0.04)" strokeWidth={1} />

                <text x={8} y={rowY + ROW_H / 2 - 4} fill="rgba(255,255,255,0.75)"
                  fontSize="11" fontWeight="600" fontFamily="var(--font-sans)">
                  {v.vessel_id}
                </text>

                <rect x={badgeX} y={badgeY} width={badgeW} height={badgeH} rx={3}
                  fill={color} fillOpacity={0.18} stroke={color} strokeOpacity={0.5} strokeWidth={0.8} />
                <text x={badgeX + badgeW / 2} y={badgeY + badgeH / 2 + 4}
                  textAnchor="middle" fontSize="9" fontWeight="700" fill={color} fontFamily="var(--font-sans)">
                  {v.w_j.toFixed(1)}d
                </text>

                <rect x={relX} y={winY} width={winW} height={winH} rx={BAR_R}
                  fill={color} fillOpacity={0.12} stroke={color} strokeOpacity={0.35} strokeWidth={1} />

                <rect x={relX} y={procY} width={procW} height={procH} rx={BAR_R}
                  fill={color} fillOpacity={0.65} stroke={color} strokeOpacity={0.9} strokeWidth={1} />

                <clipPath id={procClipId}>
                  <rect x={relX} y={procY} width={procW} height={procH} />
                </clipPath>
                <text x={relX + procW / 2} y={procY + procH / 2 + 4}
                  textAnchor="middle" fontSize="10" fontWeight="700"
                  fill="rgba(0,0,0,0.8)" fontFamily="var(--font-sans)"
                  clipPath={`url(#${procClipId})`}>
                  wj {v.w_j.toFixed(1)}
                </text>

                <line x1={relX} y1={winY - 4} x2={relX} y2={winY + winH + 4}
                  stroke={TEAL} strokeWidth={1.5} strokeDasharray="3 2" strokeOpacity={0.9} />
                <polygon points={`${relX},${winY - 8} ${relX + 5},${winY - 4} ${relX},${winY}`}
                  fill={TEAL} fillOpacity={0.8} />

                <line x1={dueX} y1={winY - 4} x2={dueX} y2={winY + winH + 4}
                  stroke={RED} strokeWidth={1.5} strokeDasharray="3 2" strokeOpacity={0.9} />
                <polygon points={`${dueX},${winY - 8} ${dueX - 5},${winY - 4} ${dueX},${winY}`}
                  fill={RED} fillOpacity={0.8} />
              </g>
            )
          })}

          <line x1={LEFT_W} y1={0} x2={LEFT_W} y2={chartH} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
          <line x1={LEFT_W} y1={chartH} x2={svgWidth - RIGHT_PAD} y2={chartH}
            stroke="rgba(255,255,255,0.15)" strokeWidth={1} />

          {dateTicks.map((ms, i) => {
            const x = timeToX(ms)
            if (x < LEFT_W || x > svgWidth - RIGHT_PAD) return null
            const tickSlot = Math.round((ms - startBaseMs) / slotMs)
            return (
              <g key={i}>
                <line x1={x} y1={chartH} x2={x} y2={chartH + 5} stroke="rgba(255,255,255,0.25)" strokeWidth={1} />
                <text x={x} y={chartH + 18} textAnchor="middle" fontSize="11"
                  fill="rgba(255,255,255,0.5)" fontFamily="var(--font-sans)">
                  {formatDateShort(new Date(ms))}
                </text>
                <text x={x} y={chartH + 33} textAnchor="middle" fontSize="9"
                  fill="rgba(0,196,180,0.5)" fontFamily="var(--font-sans)">
                  s{tickSlot}
                </text>
              </g>
            )
          })}
        </svg>

        {tooltip && (
          <div className={styles.tooltip} style={{ left: tooltipLeft, top: tooltip.screenY - 12 }}>
            <p className={styles.tooltipTitle} style={{ color: tierColor(tooltip.vessel.w_j) }}>
              {tooltip.vessel.vessel_id}
            </p>
            <div className={styles.tooltipGrid}>
              <span className={styles.tooltipKey}>Slot llegada (rj)</span>
              <span className={styles.tooltipVal}>{tooltip.vessel.r_j}</span>
              <span className={styles.tooltipKey}>Slot límite (dj)</span>
              <span className={styles.tooltipVal}>{tooltip.vessel.d_j}</span>
              <span className={styles.tooltipKey}>Ventana</span>
              <span className={styles.tooltipVal}>{tooltip.vessel.d_j - tooltip.vessel.r_j} slots</span>
              <span className={styles.tooltipKey}>Carga (pj)</span>
              <span className={styles.tooltipVal}>{tooltip.vessel.p_j} slots</span>
              <span className={styles.tooltipKey}>Peso (wj)</span>
              <span className={styles.tooltipVal} style={{ color: tierColor(tooltip.vessel.w_j), fontWeight: 700 }}>
                {tooltip.vessel.w_j.toFixed(2)} días ESD
              </span>
              <span className={styles.tooltipKey}>Stock acum.</span>
              <span className={styles.tooltipVal}>
                {tooltip.vessel.stock_acumulado_m3.toLocaleString('es-UY', { maximumFractionDigits: 0 })} m³
              </span>
              <span className={styles.tooltipKey}>Inflow diario</span>
              <span className={styles.tooltipVal}>
                {tooltip.vessel.daily_inflow_m3.toLocaleString('es-UY', { maximumFractionDigits: 0 })} m³/d
              </span>
            </div>
          </div>
        )}
      </>
    )
  }

  return (
    <div ref={chartWrapRef} className={styles.chartWrap}>
      {renderSvg()}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function VesselWindowPreview() {
  const [jsonText, setJsonText]       = useState('')
  const [startDate, setStartDate]     = useState(DEFAULT_DATE)
  const [slotHours, setSlotHours]     = useState<12 | 24 | 48>(12)
  const [instances, setInstances]     = useState<ParsedInstance[] | null>(null)
  const [parseError, setParseError]   = useState<string | null>(null)
  const [activeTab, setActiveTab]     = useState(0)

  const handleRender = () => {
    setParseError(null)
    setInstances(null)
    const { instances: parsed, error } = parseInstances(jsonText)
    if (error) { setParseError(error); return }
    setInstances(parsed)
    setActiveTab(0)
  }

  const handleReset = () => {
    setJsonText('')
    setInstances(null)
    setParseError(null)
    setActiveTab(0)
  }

  const active = instances?.[activeTab] ?? null

  return (
    <div className={styles.root}>
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Ventanas de Arribo</h1>
        <p className={styles.subtitle}>
          Pegá uno o varios arrays JSON separados por <code>---</code> para comparar distribuciones de ventanas.
          Podés agregar una etiqueta con <code># Nombre</code> antes de cada array.
        </p>
      </div>

      <div className={styles.inputsGrid}>
        <div className={styles.card}>
          <p className={styles.cardLabel}>JSON de buques</p>
          <textarea className={styles.textarea} value={jsonText}
            onChange={e => setJsonText(e.target.value)}
            placeholder={
              '[{"vessel_id":"V01","r_j":1,"d_j":6,"p_j":4,"w_j":13.4,...}]\n[{"vessel_id":"V01","r_j":2,"d_j":8,"p_j":4,"w_j":9.1,...}]'
            }
            spellCheck={false} />
          <p className={styles.textareaHint}>
            Un array por línea, o separados con <code>---</code> · Etiqueta opcional: <code># Nombre</code> antes de cada array
          </p>
        </div>

        <div className={styles.card}>
          <p className={styles.cardLabel}>Parámetros</p>
          <div className={styles.paramsGrid}>
            <label className={styles.paramLabel}>
              Fecha inicio (slot 0)
              <input type="date" className={styles.paramInput} value={startDate}
                onChange={e => setStartDate(e.target.value)} />
            </label>
            <label className={styles.paramLabel}>
              Duración del slot
              <select className={styles.paramInput} value={slotHours}
                onChange={e => setSlotHours(Number(e.target.value) as 12 | 24 | 48)}>
                <option value={12}>12 horas</option>
                <option value={24}>24 horas (1 día)</option>
                <option value={48}>48 horas (2 días)</option>
              </select>
            </label>
          </div>
          <div className={styles.colorLegend}>
            <p className={styles.legendTitle}>Color = Peso de prioridad wj</p>
            <div className={styles.legendRows}>
              <span className={styles.dot} style={{ background: TEAL }} />
              <span className={styles.legendTier}>Normal</span>
              <span className={styles.legendRange}>wj &lt; 10 días</span>
              <span className={styles.dot} style={{ background: YELLOW }} />
              <span className={styles.legendTier}>Elevado</span>
              <span className={styles.legendRange}>10 ≤ wj &lt; 15</span>
              <span className={styles.dot} style={{ background: AMBER }} />
              <span className={styles.legendTier}>Alto</span>
              <span className={styles.legendRange}>15 ≤ wj &lt; 25</span>
              <span className={styles.dot} style={{ background: RED }} />
              <span className={styles.legendTier}>Crítico</span>
              <span className={styles.legendRange}>wj ≥ 25 días</span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.actions}>
        <Button variant="primary" onClick={handleRender}>Visualizar ventanas</Button>
        <Button variant="ghost" onClick={handleReset}>
          <RotateCcw size={14} /> Limpiar
        </Button>
      </div>

      {parseError && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={14} /> {parseError}
        </div>
      )}

      {instances && (
        <>
          {/* Tab bar — only shown when there are multiple instances */}
          {instances.length > 1 && (
            <div className={styles.tabBar}>
              {instances.map((inst, i) => (
                <button
                  key={i}
                  className={`${styles.tab} ${i === activeTab ? styles.tabActive : ''}`}
                  onClick={() => setActiveTab(i)}
                >
                  {inst.label}
                  <span className={styles.tabCount}>{inst.vessels.length} buques</span>
                </button>
              ))}
            </div>
          )}

          {active && (
            <>
              <div className={styles.chartSection}>
                <div className={styles.chartHeader}>
                  <p className={styles.chartLabel}>
                    {instances.length > 1 ? active.label + ' · ' : ''}{active.vessels.length} buques · Gantt de ventanas de arribo
                  </p>
                  <div className={styles.markerLegend}>
                    <span className={styles.markerLine} style={{ borderColor: TEAL }} />
                    <span className={styles.markerText}>Slot de llegada (r_j)</span>
                    <span className={styles.markerLine} style={{ borderColor: RED }} />
                    <span className={styles.markerText}>Slot límite (d_j)</span>
                    <span className={styles.barSwatch} />
                    <span className={styles.markerText}>Duración carga (p_j)</span>
                    <span className={styles.windowSwatch} />
                    <span className={styles.markerText}>Ventana disponible</span>
                  </div>
                </div>
                <InstanceChart
                  key={activeTab}
                  vessels={active.vessels}
                  instanceIndex={activeTab}
                  startDate={startDate}
                  slotHours={slotHours}
                />
              </div>

              <div className={styles.card}>
                <p className={styles.cardLabel}>Tabla de buques</p>
                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Buque</th><th>r_j</th><th>d_j</th><th>Ventana</th>
                        <th>p_j</th><th>Peso (wj)</th><th>Stock acum. m³</th><th>Inflow m³/d</th>
                      </tr>
                    </thead>
                    <tbody>
                      {active.vessels.map(v => (
                        <tr key={v.vessel_id}>
                          <td className={styles.tdMono}>{v.vessel_id}</td>
                          <td>{v.r_j}</td>
                          <td>{v.d_j}</td>
                          <td>{v.d_j - v.r_j}</td>
                          <td>{v.p_j}</td>
                          <td>
                            <span className={styles.wjBadge} data-tier={getPriorityTier(v.w_j)}>
                              {v.w_j.toFixed(2)}
                            </span>
                          </td>
                          <td>{v.stock_acumulado_m3.toLocaleString('es-UY', { maximumFractionDigits: 0 })}</td>
                          <td>{v.daily_inflow_m3.toLocaleString('es-UY', { maximumFractionDigits: 0 })}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
