'use client'

import { useRef, useState, useEffect } from 'react'
import type { ScheduleEntry, Vessel } from '@/types/scheduling'
import { slotToDate, formatDateShort, formatDateTimeMed } from '@/lib/utils'
import styles from './GanttChart.module.css'

// ── Layout constants (px) ──────────────────────────────────────
const LEFT_W      = 96   // Y-label column width
const RIGHT_PAD   = 20   // right margin
const ROW_H       = 44   // vessel row height
const HEADER_H    = 28   // monobuoy section header height
const SECTION_GAP = 10   // vertical gap between M1 and M2 blocks
const AXIS_H      = 48   // date axis height at bottom (date + slot number)
const BAR_PAD_V   = 8    // vertical inner padding for bar within row
const BAR_R       = 3    // bar corner radius

// ── SVG color literals (must match tokens.scss values) ────────
const TEAL        = '#00C4B4'
const RED         = '#E24B4A'

// ── Pattern IDs ────────────────────────────────────────────────
const PAT_TEAL    = 'gtt-teal'
const PAT_RED     = 'gtt-red'
const PAT_BLOCKED = 'gtt-blocked'

// ── Interfaces ─────────────────────────────────────────────────

interface GanttChartProps {
  schedule: ScheduleEntry[]
  vessels: Vessel[]
  startDate: string
  slotDurationHours: 12 | 24 | 48
  /** n_machines from TerminalConfig — ensures all monobuoys 1..n are always shown */
  nMachines?: number
  /** blocked_slots from TerminalConfig — keys are monobuoy indices ("1", "2", …) */
  blockedSlots?: Record<string, number[]>
}

interface GanttEntry {
  vessel_id: string
  startMs: number
  endMs: number
  startSlot: number
  endSlot: number
  monobuoy: number
  within_window: boolean
  priority_weight: number
  tardiness_slots: number
  dueMs: number
  releaseMs: number
}

interface TooltipState {
  screenX: number
  screenY: number
  entry: GanttEntry
}

interface MarkerTooltipState {
  screenX: number
  screenY: number
  label: string   // e.g. "Slot de llegada"
  slot: number
  date: Date
  color: string
}

type RowItem =
  | { kind: 'header'; monobuoy: number; y: number }
  | { kind: 'vessel'; entry: GanttEntry; y: number; vesselIdx: number }

// ── Helpers ───────────────────────────────────────────────────

function barFill(entry: GanttEntry): string {
  if (!entry.within_window) return `url(#${PAT_RED})`
  if (entry.monobuoy === 2) return `url(#${PAT_TEAL})`
  return TEAL
}

function barStroke(entry: GanttEntry): string {
  return entry.within_window ? TEAL : RED
}

function tickStepMs(spanMs: number): number {
  const days = spanMs / 86_400_000
  if (days <= 7)  return 86_400_000
  if (days <= 21) return 3 * 86_400_000
  if (days <= 50) return 5 * 86_400_000
  return 7 * 86_400_000
}

// ── Component ──────────────────────────────────────────────────

export default function GanttChart({
  schedule,
  vessels,
  startDate,
  slotDurationHours,
  nMachines,
  blockedSlots = {},
}: GanttChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [svgWidth, setSvgWidth] = useState(0)
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [markerTooltip, setMarkerTooltip] = useState<MarkerTooltipState | null>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    setSvgWidth(el.clientWidth || 800)
    const obs = new ResizeObserver(entries => {
      setSvgWidth(entries[0].contentRect.width || 800)
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  if (!schedule.length) {
    return (
      <div ref={containerRef} className={styles.empty}>
        Sin asignaciones en el resultado.
      </div>
    )
  }

  // ── Time helpers ──────────────────────────────────────────────
  const slotMs = slotDurationHours * 3_600_000
  const toMs = (slot: number) =>
    slotToDate(slot, startDate, slotDurationHours).getTime()

  const vesselDueMap     = new Map(vessels.map(v => [v.vessel_id, v.due_slot]))
  const vesselReleaseMap = new Map(vessels.map(v => [v.vessel_id, v.release_slot]))

  // ── Build GanttEntry list ─────────────────────────────────────
  const entries: GanttEntry[] = schedule.map(e => ({
    vessel_id: e.vessel_id,
    startMs: toMs(e.start_slot),
    endMs: toMs(e.end_slot),
    startSlot: e.start_slot,
    endSlot: e.end_slot,
    monobuoy: e.monobuoy,
    within_window: e.within_window,
    priority_weight: e.priority_weight,
    tardiness_slots: e.tardiness_slots,
    dueMs: toMs(vesselDueMap.get(e.vessel_id) ?? e.end_slot),
    releaseMs: toMs(vesselReleaseMap.get(e.vessel_id) ?? e.start_slot),
  }))

  // ── Group by monobuoy ─────────────────────────────────────────
  // Always show all configured monobuoys (1..nMachines), even if empty.
  const maxFromConfig = nMachines ?? 0
  const maxFromSchedule = entries.length > 0 ? Math.max(...entries.map(e => e.monobuoy)) : 0
  const maxFromBlocked = Object.keys(blockedSlots).length > 0
    ? Math.max(...Object.keys(blockedSlots).map(k => parseInt(k, 10)))
    : 0
  const totalMachines = Math.max(maxFromConfig, maxFromSchedule, maxFromBlocked, 1)
  const monobuoys = Array.from({ length: totalMachines }, (_, i) => i + 1)

  const byMonobuoy = new Map<number, GanttEntry[]>()
  monobuoys.forEach(m => {
    byMonobuoy.set(
      m,
      entries.filter(e => e.monobuoy === m).sort((a, b) => a.startMs - b.startMs),
    )
  })

  // ── Time domain ───────────────────────────────────────────────
  const allBlockedMs = Object.entries(blockedSlots).flatMap(([, slots]) =>
    slots.flatMap(s => [toMs(s), toMs(s + 1)]),
  )
  const allMs = [
    ...entries.flatMap(e => [e.startMs, e.endMs, e.dueMs, e.releaseMs]),
    ...allBlockedMs,
  ]
  const domainMin = Math.min(...allMs) - slotMs
  const domainMax = Math.max(...allMs) + slotMs

  // ── Build row layout ──────────────────────────────────────────
  const rows: RowItem[] = []
  let currentY = 0
  let vesselCounter = 0

  monobuoys.forEach((m, mi) => {
    if (mi > 0) currentY += SECTION_GAP
    rows.push({ kind: 'header', monobuoy: m, y: currentY })
    currentY += HEADER_H
    const mEntries = byMonobuoy.get(m) ?? []
    mEntries.forEach(e => {
      rows.push({ kind: 'vessel', entry: e, y: currentY, vesselIdx: vesselCounter++ })
      currentY += ROW_H
    })
  })

  const chartContentH = currentY
  const svgH = chartContentH + AXIS_H

  // ── Pixel mapper ──────────────────────────────────────────────
  const chartW = Math.max(1, svgWidth - LEFT_W - RIGHT_PAD)
  const timeToX = (ms: number) =>
    LEFT_W + ((ms - domainMin) / (domainMax - domainMin)) * chartW

  // ── Slot boundary grid lines ──────────────────────────────────
  const startBaseMs = new Date(startDate.replace(/\//g, '-')).getTime()
  const firstSlotIdx = Math.floor((domainMin - startBaseMs) / slotMs)
  const lastSlotIdx  = Math.ceil((domainMax - startBaseMs) / slotMs)
  const slotBoundaries: { ms: number; slot: number }[] = []
  for (let s = firstSlotIdx; s <= lastSlotIdx; s++) {
    const ms = toMs(s)
    if (ms >= domainMin && ms <= domainMax) slotBoundaries.push({ ms, slot: s })
  }

  // ── Date axis ticks ───────────────────────────────────────────
  const stepMs = tickStepMs(domainMax - domainMin)
  const firstTick = Math.ceil(domainMin / stepMs) * stepMs
  const dateTicks: number[] = []
  for (let t = firstTick; t <= domainMax; t += stepMs) dateTicks.push(t)

  // ── Blocked slot bands (per monobuoy section) ─────────────────
  interface BlockedBand {
    x: number
    width: number
    y: number
    height: number
    monobuoy: number
  }
  const blockedBands: BlockedBand[] = []

  Object.entries(blockedSlots).forEach(([key, slots]) => {
    const monobuoy = parseInt(key, 10)
    const headerRow = rows.find(r => r.kind === 'header' && r.monobuoy === monobuoy)
    if (!headerRow) return
    const vesselRowsForM = rows.filter(r => r.kind === 'vessel' && r.entry.monobuoy === monobuoy)
    const sectionTop = headerRow.y
    const sectionBottom = vesselRowsForM.length > 0
      ? (vesselRowsForM.at(-1) as { y: number }).y + ROW_H
      : headerRow.y + HEADER_H

    slots.forEach(slot => {
      const x1 = timeToX(toMs(slot))
      const x2 = timeToX(toMs(slot + 1))
      if (x2 <= LEFT_W || x1 >= svgWidth - RIGHT_PAD) return
      blockedBands.push({
        x: Math.max(x1, LEFT_W),
        width: Math.max(0, Math.min(x2, svgWidth - RIGHT_PAD) - Math.max(x1, LEFT_W)),
        y: sectionTop,
        height: sectionBottom - sectionTop,
        monobuoy,
      })
    })
  })

  // ── Tooltip handlers ──────────────────────────────────────────
  const handleBarEnter = (e: React.MouseEvent<SVGRectElement>, entry: GanttEntry) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    setTooltip({ screenX: e.clientX - rect.left, screenY: e.clientY - rect.top, entry })
  }
  const handleBarLeave = () => setTooltip(null)

  const handleMarkerEnter = (
    e: React.MouseEvent<SVGGElement>,
    label: string,
    slot: number,
    ms: number,
    color: string,
  ) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    setMarkerTooltip({
      screenX: e.clientX - rect.left,
      screenY: e.clientY - rect.top,
      label,
      slot,
      date: new Date(ms),
      color,
    })
  }
  const handleMarkerLeave = () => setMarkerTooltip(null)

  // ── Legend counts ─────────────────────────────────────────────
  const onTimeM1 = entries.filter(e => e.within_window && e.monobuoy === 1).length
  const onTimeM2 = entries.filter(e => e.within_window && e.monobuoy === 2).length
  const tardy    = entries.filter(e => !e.within_window).length
  const hasBlocked = Object.values(blockedSlots).some(s => s.length > 0)

  // ── Tooltip position (keep inside container) ──────────────────
  const TOOLTIP_W = 210
  const tooltipLeft = tooltip
    ? tooltip.screenX + 14 + TOOLTIP_W > svgWidth
      ? tooltip.screenX - TOOLTIP_W - 8
      : tooltip.screenX + 14
    : 0

  // ── Clip rect bounds ──────────────────────────────────────────
  const clipId = 'gtt-clip'

  return (
    <div className={styles.root}>
      {/* ── Chart ── */}
      <div ref={containerRef} className={styles.chartWrap}>
        {svgWidth > 0 && (
          <svg
            className={styles.svgRoot}
            width={svgWidth}
            height={svgH}
            aria-label="Diagrama de Gantt — asignación de atraques"
          >
            <defs>
              {/* Teal hatch — M2 on-time */}
              <pattern id={PAT_TEAL} patternUnits="userSpaceOnUse" width="8" height="8">
                <rect width="8" height="8" fill={TEAL} fillOpacity="0.12" />
                <path
                  d="M 0 8 L 8 0 M -1 1 L 1 -1 M 7 9 L 9 7"
                  stroke={TEAL}
                  strokeWidth="1.5"
                  strokeOpacity="0.85"
                />
              </pattern>

              {/* Red hatch — tardy */}
              <pattern id={PAT_RED} patternUnits="userSpaceOnUse" width="8" height="8">
                <rect width="8" height="8" fill={RED} fillOpacity="0.12" />
                <path
                  d="M 0 8 L 8 0 M -1 1 L 1 -1 M 7 9 L 9 7"
                  stroke={RED}
                  strokeWidth="1.5"
                  strokeOpacity="0.85"
                />
              </pattern>

              {/* Gray diagonal hatch — blocked slots */}
              <pattern id={PAT_BLOCKED} patternUnits="userSpaceOnUse" width="6" height="6">
                <rect width="6" height="6" fill="rgba(255,255,255,0.03)" />
                <path
                  d="M 0 6 L 6 0 M -1 1 L 1 -1 M 5 7 L 7 5"
                  stroke="rgba(255,255,255,0.14)"
                  strokeWidth="1"
                />
              </pattern>

              {/* Clip to chart content area */}
              <clipPath id={clipId}>
                <rect x={LEFT_W} y={0} width={chartW} height={chartContentH} />
              </clipPath>
            </defs>

            {/* ── Chart area background ── */}
            <rect
              x={LEFT_W}
              y={0}
              width={chartW}
              height={chartContentH}
              fill="rgba(255,255,255,0.012)"
            />

            {/* ── Slot grid lines + slot number labels ── */}
            <g clipPath={`url(#${clipId})`}>
              {slotBoundaries.map(({ ms, slot }, i) => {
                const x = timeToX(ms)
                return (
                  <g key={i}>
                    <line
                      x1={x} y1={0}
                      x2={x} y2={chartContentH}
                      stroke="rgba(255,255,255,0.055)"
                      strokeWidth={1}
                    />
                    <text
                      x={x + 3}
                      y={10}
                      fill="rgba(0,196,180,0.35)"
                      fontSize="8"
                      fontFamily="var(--font-sans)"
                    >
                      {slot}
                    </text>
                  </g>
                )
              })}
            </g>

            {/* ── Blocked slot bands (clipped) ── */}
            <g clipPath={`url(#${clipId})`}>
              {blockedBands.map((b, i) => (
                <g key={i}>
                  {/* Hatched fill */}
                  <rect
                    x={b.x}
                    y={b.y}
                    width={b.width}
                    height={b.height}
                    fill={`url(#${PAT_BLOCKED})`}
                  />
                  {/* Left/right dotted borders */}
                  <line
                    x1={b.x} y1={b.y}
                    x2={b.x} y2={b.y + b.height}
                    stroke="rgba(255,255,255,0.22)"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                  <line
                    x1={b.x + b.width} y1={b.y}
                    x2={b.x + b.width} y2={b.y + b.height}
                    stroke="rgba(255,255,255,0.22)"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                </g>
              ))}
            </g>

            {/* ── Rows (headers + vessels) ── */}
            {rows.map(row => {
              if (row.kind === 'header') {
                return (
                  <g key={`hdr-${row.monobuoy}`}>
                    {/* Full-width section header bg */}
                    <rect
                      x={0}
                      y={row.y}
                      width={svgWidth}
                      height={HEADER_H}
                      fill="rgba(0,196,180,0.07)"
                    />
                    {/* Top border line */}
                    <line
                      x1={0}
                      y1={row.y}
                      x2={svgWidth}
                      y2={row.y}
                      stroke="rgba(0,196,180,0.3)"
                      strokeWidth={1}
                    />
                    {/* Section label in Y column */}
                    <text
                      x={LEFT_W / 2}
                      y={row.y + HEADER_H / 2 + 4}
                      fill={TEAL}
                      fontSize="11"
                      fontWeight="700"
                      textAnchor="middle"
                      letterSpacing="0.1em"
                      fontFamily="var(--font-sans)"
                    >
                      MONOBOYA {row.monobuoy}
                    </text>
                    {/* Chart-area label on the right side */}
                    <text
                      x={LEFT_W + 10}
                      y={row.y + HEADER_H / 2 + 4}
                      fill="rgba(0,196,180,0.45)"
                      fontSize="10"
                      fontWeight="600"
                      textAnchor="start"
                      letterSpacing="0.06em"
                      fontFamily="var(--font-sans)"
                    >
                      M{row.monobuoy}
                    </text>
                  </g>
                )
              }

              // vessel row
              const e = row.entry
              const barX = timeToX(e.startMs)
              const barW = Math.max(2, timeToX(e.endMs) - barX)
              const barY = row.y + BAR_PAD_V
              const barH = ROW_H - BAR_PAD_V * 2
              const isEven = row.vesselIdx % 2 === 0

              return (
                <g key={`v-${e.vessel_id}`}>
                  {/* Alternating row background */}
                  <rect
                    x={0}
                    y={row.y}
                    width={svgWidth}
                    height={ROW_H}
                    fill={isEven ? 'rgba(255,255,255,0.018)' : 'transparent'}
                  />
                  {/* Row bottom separator */}
                  <line
                    x1={LEFT_W}
                    y1={row.y + ROW_H}
                    x2={svgWidth - RIGHT_PAD}
                    y2={row.y + ROW_H}
                    stroke="rgba(255,255,255,0.045)"
                    strokeWidth={1}
                  />

                  {/* Vessel ID label */}
                  <text
                    x={LEFT_W - 10}
                    y={row.y + ROW_H / 2 + 4}
                    fill="rgba(255,255,255,0.62)"
                    fontSize="11"
                    textAnchor="end"
                    fontFamily="var(--font-sans)"
                  >
                    {e.vessel_id}
                  </text>

                  {/* Gantt bar */}
                  <rect
                    x={barX}
                    y={barY}
                    width={barW}
                    height={barH}
                    fill={barFill(e)}
                    stroke={barStroke(e)}
                    strokeWidth={0.6}
                    strokeOpacity={0.65}
                    rx={BAR_R}
                    className={styles.bar}
                    onMouseEnter={ev => handleBarEnter(ev, e)}
                    onMouseLeave={handleBarLeave}
                  />

                  {/* Release-slot marker — slot de llegada preferida de este buque */}
                  {(() => {
                    const relX = timeToX(e.releaseMs)
                    if (relX < LEFT_W || relX > svgWidth - RIGHT_PAD) return null
                    const releaseSlot = vesselReleaseMap.get(e.vessel_id) ?? e.startSlot
                    return (
                      <g
                        style={{ cursor: 'default' }}
                        onMouseEnter={ev => handleMarkerEnter(ev, 'Slot de llegada', releaseSlot, e.releaseMs, TEAL)}
                        onMouseLeave={handleMarkerLeave}
                      >
                        {/* Wider invisible hit area */}
                        <rect
                          x={relX - 6}
                          y={barY - 3}
                          width={12}
                          height={barH + 9}
                          fill="transparent"
                        />
                        {/* Vertical dashed line, scoped to this row */}
                        <line
                          x1={relX} y1={barY - 3}
                          x2={relX} y2={barY + barH + 3}
                          stroke={TEAL}
                          strokeWidth={1.5}
                          strokeDasharray="3 2"
                          strokeOpacity={0.8}
                        />
                        {/* Upward-pointing triangle cap at bottom */}
                        <polygon
                          points={`${relX - 4},${barY + barH + 6} ${relX + 4},${barY + barH + 6} ${relX},${barY + barH + 1}`}
                          fill={TEAL}
                          fillOpacity={0.75}
                        />
                      </g>
                    )
                  })()}

                  {/* Due-slot marker — slot límite de este buque */}
                  {(() => {
                    const dueX = timeToX(e.dueMs)
                    if (dueX < LEFT_W || dueX > svgWidth - RIGHT_PAD) return null
                    const dueSlot = vesselDueMap.get(e.vessel_id) ?? e.endSlot
                    return (
                      <g
                        style={{ cursor: 'default' }}
                        onMouseEnter={ev => handleMarkerEnter(ev, 'Slot límite', dueSlot, e.dueMs, RED)}
                        onMouseLeave={handleMarkerLeave}
                      >
                        {/* Wider invisible hit area */}
                        <rect
                          x={dueX - 6}
                          y={barY - 9}
                          width={12}
                          height={barH + 12}
                          fill="transparent"
                        />
                        {/* Vertical dashed line, scoped to this row */}
                        <line
                          x1={dueX} y1={barY - 3}
                          x2={dueX} y2={barY + barH + 3}
                          stroke={RED}
                          strokeWidth={1.5}
                          strokeDasharray="3 2"
                          strokeOpacity={0.8}
                        />
                        {/* Downward-pointing triangle cap at top */}
                        <polygon
                          points={`${dueX - 4},${barY - 6} ${dueX + 4},${barY - 6} ${dueX},${barY - 1}`}
                          fill={RED}
                          fillOpacity={0.75}
                        />
                      </g>
                    )
                  })()}
                </g>
              )
            })}

            {/* ── Y-axis divider line ── */}
            <line
              x1={LEFT_W}
              y1={0}
              x2={LEFT_W}
              y2={chartContentH}
              stroke="rgba(255,255,255,0.1)"
              strokeWidth={1}
            />

            {/* ── Date axis ── */}
            <line
              x1={LEFT_W}
              y1={chartContentH}
              x2={svgWidth - RIGHT_PAD}
              y2={chartContentH}
              stroke="rgba(255,255,255,0.15)"
              strokeWidth={1}
            />
            {dateTicks.map((ms, i) => {
              const x = timeToX(ms)
              if (x < LEFT_W || x > svgWidth - RIGHT_PAD) return null
              const tickSlot = Math.round((ms - startBaseMs) / slotMs)
              return (
                <g key={i}>
                  <line
                    x1={x} y1={chartContentH}
                    x2={x} y2={chartContentH + 5}
                    stroke="rgba(255,255,255,0.25)"
                    strokeWidth={1}
                  />
                  <text
                    x={x}
                    y={chartContentH + 17}
                    fill="rgba(255,255,255,0.45)"
                    fontSize="11"
                    textAnchor="middle"
                    fontFamily="var(--font-sans)"
                  >
                    {formatDateShort(new Date(ms))}
                  </text>
                  <text
                    x={x}
                    y={chartContentH + 31}
                    fill="rgba(0,196,180,0.45)"
                    fontSize="9"
                    textAnchor="middle"
                    fontFamily="var(--font-sans)"
                  >
                    s{tickSlot}
                  </text>
                </g>
              )
            })}
          </svg>
        )}

        {/* ── Floating tooltip (bar) ── */}
        {tooltip && (
          <div
            className={styles.tooltip}
            style={{ left: tooltipLeft, top: tooltip.screenY - 12 }}
          >
            <p className={styles.tooltipTitle}>{tooltip.entry.vessel_id}</p>
            <div className={styles.tooltipGrid}>
              <span className={styles.tooltipKey}>Monoboya</span>
              <span className={styles.tooltipVal}>M{tooltip.entry.monobuoy}</span>

              <span className={styles.tooltipKey}>Inicio</span>
              <span className={styles.tooltipVal}>
                {formatDateTimeMed(new Date(tooltip.entry.startMs))}
              </span>

              <span className={styles.tooltipKey}>Fin</span>
              <span className={styles.tooltipVal}>
                {formatDateTimeMed(new Date(tooltip.entry.endMs))}
              </span>

              <span className={styles.tooltipKey}>Slots</span>
              <span className={styles.tooltipVal}>
                {tooltip.entry.startSlot}–{tooltip.entry.endSlot}
              </span>

              <span className={styles.tooltipKey}>Peso (wj)</span>
              <span className={styles.tooltipVal}>
                {tooltip.entry.priority_weight.toFixed(2)} días
              </span>

              <span className={styles.tooltipKey}>Tardanza</span>
              <span className={tooltip.entry.within_window ? styles.tooltipVal : styles.tooltipValWarn}>
                {tooltip.entry.tardiness_slots === 0
                  ? 'En ventana'
                  : `${tooltip.entry.tardiness_slots} slots`}
              </span>
            </div>
          </div>
        )}

        {/* ── Floating tooltip (marker lines) ── */}
        {markerTooltip && (
          <div
            className={styles.tooltip}
            style={{
              left: markerTooltip.screenX + 14 + 170 > svgWidth
                ? markerTooltip.screenX - 178
                : markerTooltip.screenX + 14,
              top: markerTooltip.screenY - 12,
              minWidth: 170,
              borderColor: markerTooltip.color,
            }}
          >
            <p
              className={styles.tooltipTitle}
              style={{ color: markerTooltip.color }}
            >
              {markerTooltip.label}
            </p>
            <div className={styles.tooltipGrid}>
              <span className={styles.tooltipKey}>Slot</span>
              <span className={styles.tooltipVal}>{markerTooltip.slot}</span>
              <span className={styles.tooltipKey}>Fecha</span>
              <span className={styles.tooltipVal}>
                {formatDateTimeMed(markerTooltip.date)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* ── Legend ── */}
      <div className={styles.legend}>
        {onTimeM1 > 0 && (
          <div className={styles.legendItem}>
            <span className={styles.swatchTeal} aria-hidden="true" />
            <span>A tiempo M1 ({onTimeM1})</span>
          </div>
        )}
        {onTimeM2 > 0 && (
          <div className={styles.legendItem}>
            <span className={styles.swatchTealHatch} aria-hidden="true" />
            <span>A tiempo M2 ({onTimeM2})</span>
          </div>
        )}
        {tardy > 0 && (
          <div className={styles.legendItem}>
            <span className={styles.swatchRed} aria-hidden="true" />
            <span>Tardío ({tardy})</span>
          </div>
        )}
        <div className={styles.legendItem}>
          <span className={styles.swatchReleaseLine} aria-hidden="true" />
          <span>Llegada preferida del buque (slot de llegada)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.swatchDueLine} aria-hidden="true" />
          <span>Fecha límite del buque (slot límite)</span>
        </div>
        {hasBlocked && (
          <div className={styles.legendItem}>
            <span className={styles.swatchBlocked} aria-hidden="true" />
            <span>Slot bloqueado</span>
          </div>
        )}
      </div>
    </div>
  )
}
