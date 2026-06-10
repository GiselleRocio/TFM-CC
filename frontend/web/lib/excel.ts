import * as XLSX from 'xlsx'
import type { Vessel, ScheduleEntry } from '@/types/scheduling'

// ── Column order must match parseExcelFile() in VesselInputStep ──────────────
const VESSEL_COLS: (keyof Vessel)[] = [
  'vessel_id',
  'volume_m3',
  'daily_inflow_m3',
  'cargo_m3',
  'release_slot',
  'due_slot',
  'processing_slots',
  'priority_weight',
]

function vesselRows(vessels: Vessel[]): Record<string, unknown>[] {
  return vessels.map(v =>
    Object.fromEntries(VESSEL_COLS.map(col => [col, v[col] ?? ''])),
  )
}

function scheduleRows(schedule: ScheduleEntry[]): Record<string, unknown>[] {
  return schedule.map(e => ({
    vessel_id: e.vessel_id,
    monoboya: e.monobuoy,
    slot_inicio: e.start_slot,
    slot_fin: e.end_slot,
    peso_prioridad: e.priority_weight,
    tardanza_slots: e.tardiness_slots,
    a_tiempo: e.within_window ? 'Sí' : 'No',
  }))
}

function triggerDownload(wb: XLSX.WorkBook, filename: string): void {
  XLSX.writeFile(wb, filename)
}

/** Download only vessel nominations — importable by this same app. */
export function downloadVesselsExcel(vessels: Vessel[], filename = 'buques.xlsx'): void {
  const ws = XLSX.utils.json_to_sheet(vesselRows(vessels), { header: VESSEL_COLS })
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, 'Buques')
  triggerDownload(wb, filename)
}

/**
 * Download a 2-sheet workbook:
 *   Sheet 1 "Buques"    — vessel nominations (importable)
 *   Sheet 2 "Resultados" — schedule assignment
 */
export function downloadResultsExcel(
  vessels: Vessel[],
  schedule: ScheduleEntry[],
  filename = 'resultado.xlsx',
): void {
  const wsVessels = XLSX.utils.json_to_sheet(vesselRows(vessels), { header: VESSEL_COLS })
  const wsResults = XLSX.utils.json_to_sheet(scheduleRows(schedule))

  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, wsVessels, 'Buques')
  XLSX.utils.book_append_sheet(wb, wsResults, 'Resultados')
  triggerDownload(wb, filename)
}
