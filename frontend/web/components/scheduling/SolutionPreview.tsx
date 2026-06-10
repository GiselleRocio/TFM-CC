'use client'

import { useState, useCallback } from 'react'
import * as XLSX from 'xlsx'
import { Upload, Play, AlertTriangle, RotateCcw, Ban } from 'lucide-react'
import type { Vessel, ScheduleEntry, TerminalConfig } from '@/types/scheduling'
import Button from '@/components/ui/Button'
import GanttChart from './GanttChart'
import styles from './SolutionPreview.module.css'

// ── Default config used only for Gantt rendering ──────────────────────────────
// The user can override start_date and slot_duration_hours via inputs.
const DEFAULT_CONFIG: Pick<TerminalConfig, 'start_date' | 'slot_duration_hours' | 'n_machines' | 'blocked_slots'> = {
  start_date: new Date().toISOString().slice(0, 10).replace(/-/g, '/'),
  slot_duration_hours: 12,
  n_machines: 2,
  blocked_slots: {},
}

// ── Variable parser ────────────────────────────────────────────────────────────
// Format: x_{vessel_id}_{monobuoy}_{start_slot}  value >= 0.5 → active
// vessel_id may contain underscores (e.g. V01, CRUDE_01) — we grab last two
// numeric tokens as monobuoy + start_slot, everything before is vessel_id.

interface ParsedVar {
  vessel_id: string
  monobuoy: number
  start_slot: number
}

function parseVariableLine(line: string): ParsedVar | null {
  // Accept formats:
  //   x_V01_2_0: 1.0
  //   x_V01_2_0 = 1.0
  //   x_V01_2_0: 1
  const match = line.match(/^\s*x_(.+?)[\s:=]+([0-9.]+)\s*$/)
  if (!match) return null

  const value = parseFloat(match[2])
  if (value < 0.5) return null   // inactive variable

  // Split key into parts and take the last two as monobuoy + start_slot
  const parts = match[1].split('_')
  if (parts.length < 3) return null

  const start_slot = parseInt(parts[parts.length - 1], 10)
  const monobuoy = parseInt(parts[parts.length - 2], 10)
  const vessel_id = parts.slice(0, parts.length - 2).join('_')

  if (isNaN(monobuoy) || isNaN(start_slot)) return null
  return { vessel_id, monobuoy, start_slot }
}

// ── ArtP_ parser ──────────────────────────────────────────────────────────────
// Format: ArtP_assign_{vessel_id}: 1.0  →  vessel was dropped by the model.

function parseArtPLine(line: string): string | null {
  const match = line.match(/^\s*ArtP_assign_(.+?)[\s:=]+([0-9.]+)\s*$/)
  if (!match) return null
  const value = parseFloat(match[2])
  if (value < 0.5) return null
  return match[1]
}

function buildSchedule(
  parsed: ParsedVar[],
  vesselMap: Map<string, Vessel>,
): { schedule: ScheduleEntry[]; warnings: string[] } {
  const schedule: ScheduleEntry[] = []
  const warnings: string[] = []

  for (const p of parsed) {
    const vessel = vesselMap.get(p.vessel_id)
    if (!vessel) {
      warnings.push(`Buque "${p.vessel_id}" no encontrado en el Excel.`)
      continue
    }

    const end_slot = p.start_slot + vessel.processing_slots
    const tardiness_slots = Math.max(0, end_slot - vessel.due_slot)

    schedule.push({
      vessel_id: p.vessel_id,
      monobuoy: p.monobuoy,
      start_slot: p.start_slot,
      end_slot,
      priority_weight: vessel.priority_weight ?? vessel.volume_m3 / vessel.daily_inflow_m3,
      tardiness_slots,
      within_window: tardiness_slots === 0,
    })
  }

  return { schedule, warnings }
}

// ── Excel parser (reuses same logic as VesselInputStep) ───────────────────────

async function parseExcelFile(file: File): Promise<Vessel[]> {
  const buf = await file.arrayBuffer()
  const wb = XLSX.read(buf)
  const ws = wb.Sheets[wb.SheetNames[0]]
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(ws)

  return rows.map((row): Vessel => {
    const ps = Number(row.processing_slots ?? 0)
    return {
      vessel_id: String(row.vessel_id ?? ''),
      volume_m3: Number(row.volume_m3 ?? 0),
      daily_inflow_m3: Number(row.daily_inflow_m3 ?? 0),
      cargo_m3: row.cargo_m3 !== undefined ? Number(row.cargo_m3) : undefined,
      release_slot: Number(row.release_slot ?? 0),
      due_slot: Number(row.due_slot ?? 0),
      processing_slots: ps,
      priority_weight: row.priority_weight !== undefined ? Number(row.priority_weight) : undefined,
    }
  })
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SolutionPreview() {
  const [vessels, setVessels] = useState<Vessel[]>([])
  const [vesselFileName, setVesselFileName] = useState<string | null>(null)
  const [solutionText, setSolutionText] = useState('')
  const [startDate, setStartDate] = useState(DEFAULT_CONFIG.start_date.replace(/\//g, '-'))
  const [slotHours, setSlotHours] = useState<12 | 24 | 48>(12)

  const [schedule, setSchedule] = useState<ScheduleEntry[] | null>(null)
  const [unassigned, setUnassigned] = useState<string[]>([])
  const [warnings, setWarnings] = useState<string[]>([])
  const [parseError, setParseError] = useState<string | null>(null)

  // ── Excel upload ────────────────────────────────────────────────────────────

  const handleExcelDrop = useCallback(async (file: File) => {
    try {
      const vs = await parseExcelFile(file)
      setVessels(vs)
      setVesselFileName(file.name)
      setSchedule(null)
      setWarnings([])
      setParseError(null)
    } catch {
      setParseError('No se pudo leer el archivo Excel. Verificá el formato.')
    }
  }, [])

  const handleExcelChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleExcelDrop(file)
  }

  const handleDragOver = (e: React.DragEvent) => e.preventDefault()
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleExcelDrop(file)
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  const handleRender = () => {
    setParseError(null)
    setWarnings([])
    setSchedule(null)
    setUnassigned([])

    if (vessels.length === 0) {
      setParseError('Primero cargá el Excel con los buques.')
      return
    }
    if (!solutionText.trim()) {
      setParseError('Pegá las variables de solución antes de renderizar.')
      return
    }

    const lines = solutionText.split('\n')
    const parsed: ParsedVar[] = []
    const dropped: string[] = []

    for (const line of lines) {
      if (!line.trim() || line.trim().startsWith('#') || line.trim().startsWith('//')) continue
      const artP = parseArtPLine(line)
      if (artP) { dropped.push(artP); continue }
      const result = parseVariableLine(line)
      if (result) parsed.push(result)
      // T_ and other prefixes are silently skipped
    }

    if (parsed.length === 0 && dropped.length === 0) {
      setParseError('No se encontraron variables activas. Verificá el formato: x_V01_2_0: 1.0')
      return
    }

    const vesselMap = new Map(vessels.map(v => [v.vessel_id, v]))
    const { schedule: built, warnings: w } = buildSchedule(parsed, vesselMap)

    if (built.length === 0 && dropped.length === 0) {
      setParseError('No se pudo construir ninguna entrada de schedule. Verificá que los vessel_id coincidan con el Excel.')
      return
    }

    setSchedule(built.length > 0 ? built : null)
    setUnassigned(dropped)
    setWarnings(w)
  }

  const handleReset = () => {
    setVessels([])
    setVesselFileName(null)
    setSolutionText('')
    setSchedule(null)
    setUnassigned([])
    setWarnings([])
    setParseError(null)
  }

  // ── Derived config for GanttChart ───────────────────────────────────────────

  const ganttConfig = {
    start_date: startDate.replace(/-/g, '/'),
    slot_duration_hours: slotHours,
    n_machines: schedule
      ? Math.max(...schedule.map(e => e.monobuoy))
      : DEFAULT_CONFIG.n_machines,
    blocked_slots: DEFAULT_CONFIG.blocked_slots,
  }

  return (
    <div className={styles.root}>
      {/* ── Header ── */}
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Preview de Solución</h1>
        <p className={styles.subtitle}>
          Cargá el Excel de buques y pegá las variables QUBO activas para visualizar el schedule sin ejecutar el solver.
        </p>
      </div>

      {/* ── Inputs ── */}
      <div className={styles.inputsGrid}>

        {/* Excel upload */}
        <div className={styles.card}>
          <p className={styles.cardLabel}>1. Excel de buques</p>
          <label
            className={`${styles.dropzone} ${vesselFileName ? styles.dropzoneLoaded : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".xlsx,.xls"
              className={styles.fileInput}
              onChange={handleExcelChange}
            />
            {vesselFileName ? (
              <div className={styles.dropzoneLoaded}>
                <Upload size={16} className={styles.dropzoneIcon} />
                <span className={styles.dropzoneFile}>{vesselFileName}</span>
                <span className={styles.dropzoneHint}>{vessels.length} buques cargados</span>
              </div>
            ) : (
              <div className={styles.dropzoneEmpty}>
                <Upload size={20} className={styles.dropzoneIconDim} />
                <span className={styles.dropzoneText}>Arrastrá o hacé click para subir el .xlsx</span>
                <span className={styles.dropzoneHint}>Columnas requeridas: vessel_id, release_slot, due_slot, processing_slots, volume_m3, daily_inflow_m3</span>
              </div>
            )}
          </label>
        </div>

        {/* Solution variables textarea */}
        <div className={styles.card}>
          <p className={styles.cardLabel}>2. Variables de solución</p>
          <textarea
            className={styles.textarea}
            value={solutionText}
            onChange={e => setSolutionText(e.target.value)}
            placeholder={'x_V01_2_0: 1.0\nx_V02_2_8: 1.0\nx_V03_1_11: 1.0\nx_V04_1_16: 1.0\n...'}
            spellCheck={false}
          />
          <p className={styles.textareaHint}>
            Formato: <code>x_&#123;vessel_id&#125;_&#123;monoboya&#125;_&#123;slot_inicio&#125;: &#123;valor&#125;</code>
            &nbsp;· Solo se procesan variables con valor ≥ 0.5
          </p>
        </div>

        {/* Rendering params */}
        <div className={styles.card}>
          <p className={styles.cardLabel}>3. Parámetros de visualización</p>
          <div className={styles.paramsGrid}>
            <label className={styles.paramLabel}>
              Fecha inicio (slot 0)
              <input
                type="date"
                className={styles.paramInput}
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
              />
            </label>
            <label className={styles.paramLabel}>
              Duración del slot
              <select
                className={styles.paramInput}
                value={slotHours}
                onChange={e => setSlotHours(Number(e.target.value) as 12 | 24 | 48)}
              >
                <option value={12}>12 horas</option>
                <option value={24}>24 horas (1 día)</option>
                <option value={48}>48 horas (2 días)</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      {/* ── Actions ── */}
      <div className={styles.actions}>
        <Button variant="primary" onClick={handleRender}>
          <Play size={14} />
          Renderizar Gantt
        </Button>
        <Button variant="ghost" onClick={handleReset}>
          <RotateCcw size={14} />
          Limpiar
        </Button>
      </div>

      {/* ── Error ── */}
      {parseError && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={14} />
          {parseError}
        </div>
      )}

      {/* ── Warnings ── */}
      {warnings.length > 0 && (
        <div className={styles.warnBanner}>
          <AlertTriangle size={14} />
          <div>
            <p>Advertencias ({warnings.length}):</p>
            <ul className={styles.warnList}>
              {warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* ── Unassigned vessels ── */}
      {unassigned.length > 0 && (
        <div className={styles.unassignedPanel}>
          <div className={styles.unassignedHeader}>
            <Ban size={14} className={styles.unassignedIcon} />
            <span>Buques excluidos del plan ({unassigned.length})</span>
          </div>
          <p className={styles.unassignedDesc}>
            El modelo no pudo programar estos buques sin generar conflictos de capacidad.
          </p>
          <div className={styles.unassignedChips}>
            {unassigned.map(id => (
              <span key={id} className={styles.chip}>{id}</span>
            ))}
          </div>
        </div>
      )}

      {/* ── Gantt ── */}
      {schedule && (
        <div className={styles.ganttSection}>
          <p className={styles.ganttLabel}>
            Diagrama de Gantt — {schedule.length} asignaciones
          </p>
          <GanttChart
            schedule={schedule}
            vessels={vessels}
            startDate={ganttConfig.start_date}
            slotDurationHours={ganttConfig.slot_duration_hours}
            nMachines={ganttConfig.n_machines}
            blockedSlots={ganttConfig.blocked_slots}
          />
        </div>
      )}
    </div>
  )
}
