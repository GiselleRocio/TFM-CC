'use client'

import { useState, useEffect, useRef, useMemo } from 'react'
import { Info } from 'lucide-react'
import Button from '@/components/ui/Button'
import Spinner from '@/components/ui/Spinner'
import { getConfigDefaults } from '@/lib/api'
import type {
  TerminalConfig,
  ConfigDefaults,
  SamplerOption,
} from '@/types/scheduling'
import styles from './TerminalConfigStep.module.css'

// ---- Types -----------------------------------------------------------------

export interface TerminalConfigStepProps {
  /** Committed config from a prior visit to this step — null on first visit */
  initialConfig: TerminalConfig | null
  onBack: () => void
  onNext: (config: TerminalConfig) => void
}

/** Internal mutable form state — dates kept as YYYY-MM-DD for <input type="date"> */
interface FormState {
  start_date: string
  end_date: string
  slot_duration_hours: 12 | 24 | 48
  min_ullage_days: number
  n_tanks: number
  tank_capacity_m3: number
  initial_terminal_stock_m3: number
  daily_inflow_m3: number
  n_machines: number
  /**
   * Assignment of each monobuoy to a pipeline group.
   * Key = monobuoy index (1-based), value = group label ("A", "B", …) or null = independent.
   */
  pipeline_assignment: Record<number, string | null>
  alpha: number
  sampler: SamplerOption
  /** Raw comma-separated text per monobuoy key ("1", "2", …) */
  blocked_slots_text: Record<string, string>
}

// ---- Sampler labels --------------------------------------------------------

const SAMPLER_LABELS: Record<string, string> = {
  leap_hybrid: 'LeapHybridSampler (cloud)',
  simulated_annealing: 'Force Simulated Annealing (offline)',
}

// ---- Date helpers ----------------------------------------------------------

function nextMonthRange(): { start: string; end: string } {
  const now = new Date()
  const first = new Date(now.getFullYear(), now.getMonth() + 1, 1)
  const last = new Date(now.getFullYear(), now.getMonth() + 2, 0)
  return {
    start: first.toISOString().split('T')[0],
    end: last.toISOString().split('T')[0],
  }
}

function addDaysISO(iso: string, days: number): string {
  const d = new Date(iso)
  d.setDate(d.getDate() + days)
  return d.toISOString().split('T')[0]
}

function diffDays(startISO: string, endISO: string): number {
  const ms = new Date(endISO).getTime() - new Date(startISO).getTime()
  return Math.max(0, Math.round(ms / 86_400_000) + 1)
}

/** Convert YYYY-MM-DD → YYYY/MM/DD for API payload */
function isoToApi(date: string): string {
  return date.replace(/-/g, '/')
}

/** Convert YYYY/MM/DD → YYYY-MM-DD for <input type="date"> */
function apiToIso(date: string): string {
  return date.replace(/\//g, '-')
}

// ---- Blocked slots helpers -------------------------------------------------

function blockedToText(slots: number[]): string {
  return slots.join(', ')
}

function textToBlocked(text: string): number[] {
  return text
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !isNaN(n) && n >= 0)
}

// ---- Form state factories --------------------------------------------------

// ---- Pipeline group helpers ------------------------------------------------

/** Convert shared_pipeline_groups array → per-monobuoy assignment map */
function groupsToAssignment(groups: number[][], n_machines: number): Record<number, string | null> {
  const assignment: Record<number, string | null> = {}
  for (let m = 1; m <= n_machines; m++) assignment[m] = null
  const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  groups.forEach((group, idx) => {
    if (group.length >= 2) {
      const label = labels[idx % labels.length]
      for (const m of group) {
        if (m >= 1 && m <= n_machines) assignment[m] = label
      }
    }
  })
  return assignment
}

/** Convert per-monobuoy assignment map → shared_pipeline_groups array */
function assignmentToGroups(assignment: Record<number, string | null>): number[][] {
  const byLabel: Record<string, number[]> = {}
  for (const [mStr, label] of Object.entries(assignment)) {
    if (label !== null) {
      byLabel[label] = [...(byLabel[label] ?? []), Number(mStr)]
    }
  }
  return Object.values(byLabel).filter((g) => g.length >= 2)
}

/** Get distinct group labels currently in use (sorted) */
function usedGroupLabels(assignment: Record<number, string | null>): string[] {
  return [...new Set(Object.values(assignment).filter((v): v is string => v !== null))].sort()
}

/** Next available group label not yet in use */
function nextGroupLabel(used: string[]): string {
  const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  for (const l of labels) if (!used.includes(l)) return l
  return 'A'
}

// ---- Form state factories --------------------------------------------------

function defaultsToForm(defaults: ConfigDefaults): FormState {
  const { start, end } = nextMonthRange()

  const blocked_slots_text: Record<string, string> = {}
  for (let m = 1; m <= defaults.n_machines; m++) {
    const key = String(m)
    blocked_slots_text[key] = blockedToText(defaults.blocked_slots[key] ?? [])
  }

  const pipeline_assignment = groupsToAssignment(
    defaults.shared_pipeline_groups ?? [],
    defaults.n_machines,
  )

  return {
    start_date: start,
    end_date: end,
    slot_duration_hours: defaults.slot_duration_hours,
    min_ullage_days: defaults.min_ullage_days,
    n_tanks: defaults.n_tanks,
    tank_capacity_m3: defaults.tank_capacity_m3,
    initial_terminal_stock_m3: defaults.initial_terminal_stock_m3,
    daily_inflow_m3: defaults.daily_inflow_m3,
    n_machines: defaults.n_machines,
    pipeline_assignment,
    alpha: defaults.alpha,
    sampler: (defaults.sampler_options[0] as SamplerOption) ??
      'simulated_annealing',
    blocked_slots_text,
  }
}

function configToForm(config: TerminalConfig): FormState {
  const blocked_slots_text: Record<string, string> = {}
  for (let m = 1; m <= config.n_machines; m++) {
    const key = String(m)
    blocked_slots_text[key] = blockedToText(config.blocked_slots[key] ?? [])
  }
  const pipeline_assignment = groupsToAssignment(
    config.shared_pipeline_groups ?? [],
    config.n_machines,
  )
  return {
    start_date: apiToIso(config.start_date),
    end_date: apiToIso(config.end_date),
    slot_duration_hours: config.slot_duration_hours,
    min_ullage_days: config.min_ullage_days,
    n_tanks: config.n_tanks ?? 6,
    tank_capacity_m3: config.tank_capacity_m3 ?? 100000,
    initial_terminal_stock_m3: config.initial_terminal_stock_m3 ?? 300000,
    daily_inflow_m3: config.daily_inflow_m3 ?? 20000,
    n_machines: config.n_machines,
    pipeline_assignment,
    alpha: config.alpha,
    sampler: config.sampler,
    blocked_slots_text,
  }
}

function formToConfig(form: FormState): TerminalConfig {
  const horizon_days = diffDays(form.start_date, form.end_date)
  const blocked_slots: Record<string, number[]> = {}
  for (const key of Object.keys(form.blocked_slots_text)) {
    blocked_slots[key] = textToBlocked(form.blocked_slots_text[key])
  }
  return {
    n_machines: form.n_machines,
    start_date: isoToApi(form.start_date),
    end_date: isoToApi(form.end_date),
    horizon_days,
    slot_duration_hours: form.slot_duration_hours,
    min_ullage_days: form.min_ullage_days,
    n_tanks: form.n_tanks,
    tank_capacity_m3: form.tank_capacity_m3,
    initial_terminal_stock_m3: form.initial_terminal_stock_m3,
    daily_inflow_m3: form.daily_inflow_m3,
    shared_pipeline_groups: assignmentToGroups(form.pipeline_assignment),
    alpha: form.alpha,
    sampler: form.sampler,
    blocked_slots,
  }
}

// ---- Sub-component: Stepper ------------------------------------------------

interface StepperProps {
  value: number
  min: number
  max: number
  onChange: (val: number) => void
  ariaLabel: string
}

function Stepper({ value, min, max, onChange, ariaLabel }: StepperProps) {
  return (
    <div className={styles.stepper} role="group" aria-label={ariaLabel}>
      <button
        className={styles.stepperBtn}
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
        aria-label="Reducir"
      >
        −
      </button>
      <input
        type="number"
        className={styles.stepperInput}
        value={value}
        min={min}
        max={max}
        aria-live="polite"
        aria-label={ariaLabel}
        onChange={(e) => {
          const v = parseInt(e.target.value, 10)
          if (!isNaN(v) && v >= min) onChange(Math.min(max, v))
        }}
      />
      <button
        className={styles.stepperBtn}
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
        aria-label="Aumentar"
      >
        +
      </button>
    </div>
  )
}

// ---- Sub-component: PipelineGroupEditor ------------------------------------

const GROUP_COLORS: Record<string, string> = {
  A: 'var(--color-teal)',
  B: '#F59E0B',
  C: '#8B5CF6',
  D: '#EC4899',
  E: '#3B82F6',
  F: '#10B981',
}

function groupColor(label: string): string {
  return GROUP_COLORS[label] ?? 'var(--color-teal)'
}

interface PipelineGroupEditorProps {
  nMachines: number
  assignment: Record<number, string | null>
  onChangeGroup: (machine: number, group: string | null) => void
  onAddGroup: () => void
}

function PipelineGroupEditor({
  nMachines,
  assignment,
  onChangeGroup,
  onAddGroup,
}: PipelineGroupEditorProps) {
  const used = usedGroupLabels(assignment)
  const availableLabels = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'].slice(0, used.length + 1)
  const machines = Array.from({ length: nMachines }, (_, i) => i + 1)

  const summaryGroups = assignmentToGroups(assignment)
  const summaryText =
    summaryGroups.length === 0
      ? 'Todas independientes'
      : summaryGroups
          .map((g, i) => `Grupo ${availableLabels[i] ?? String.fromCharCode(65 + i)}: ${g.map((m) => `M${m}`).join('+') }`)
          .join(' · ')

  return (
    <div className={styles.pipelineEditor}>
      <div className={styles.pipelineChips}>
        {machines.map((m) => {
          const group = assignment[m] ?? null
          return (
            <div
              key={m}
              className={styles.pipelineChip}
              style={group ? { borderColor: groupColor(group), color: groupColor(group) } : {}}
            >
              <span className={styles.pipelineChipLabel}>M{m}</span>
              <select
                className={styles.pipelineSelect}
                value={group ?? ''}
                aria-label={`Pipeline group for M${m}`}
                onChange={(e) => {
                  const val = e.target.value
                  onChangeGroup(m, val === '' ? null : val)
                }}
                style={group ? { color: groupColor(group) } : {}}
              >
                <option value="">Independiente</option>
                {availableLabels.map((lbl) => (
                  <option key={lbl} value={lbl}>
                    Grupo {lbl}
                  </option>
                ))}
              </select>
            </div>
          )
        })}
        <button
          type="button"
          className={styles.addGroupBtn}
          onClick={onAddGroup}
          aria-label="Agregar grupo de oleoducto"
          title="Crear nuevo grupo"
        >
          + Grupo
        </button>
      </div>
      <span className={styles.sublabel}>{summaryText}</span>
    </div>
  )
}

// ---- Sub-component: TooltipLabel -------------------------------------------

interface TooltipLabelProps {
  label: string
  tip: string
}

function TooltipLabel({ label, tip }: TooltipLabelProps) {
  return (
    <span className={styles.label}>
      {label}
      <span className={styles.tooltipWrap}>
        <Info size={13} className={styles.infoIcon} aria-hidden="true" />
        <span role="tooltip" className={styles.tooltip}>
          {tip}
        </span>
      </span>
    </span>
  )
}

// ---- Main component --------------------------------------------------------

export default function TerminalConfigStep({
  initialConfig,
  onBack,
  onNext,
}: TerminalConfigStepProps) {
  const [form, setForm] = useState<FormState | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [alphaRange, setAlphaRange] = useState({ min: 1.0, max: 10.0 })
  const [samplerOptions, setSamplerOptions] = useState<string[]>([
    'simulated_annealing',
    'leap_hybrid',
  ])

  // Use a ref so the effect closure never goes stale
  const initialConfigRef = useRef(initialConfig)

  useEffect(() => {
    const saved = initialConfigRef.current
    if (saved) {
      setForm(configToForm(saved))
      return
    }
    getConfigDefaults()
      .then((defaults: ConfigDefaults) => {
        setAlphaRange({ min: defaults.alpha_min, max: defaults.alpha_max })
        setSamplerOptions(defaults.sampler_options)
        setForm(defaultsToForm(defaults))
      })
      .catch((err: unknown) => {
        const msg =
          err instanceof Error ? err.message : 'Error al cargar configuración.'
        setApiError(msg)
        // Still show form with hardcoded defaults so the user isn't blocked
        setForm(
          defaultsToForm({
            n_machines: 2,
            horizon_days: 31,
            slot_duration_hours: 12,
            min_ullage_days: 4,
            n_tanks: 6,
            tank_capacity_m3: 100000,
            initial_terminal_stock_m3: 300000,
            daily_inflow_m3: 20000,
            alpha: 3.0,
            blocked_slots: { '1': [], '2': [] },
            max_iterations: 10,
            sampler_options: ['simulated_annealing', 'leap_hybrid'],
            shared_pipeline_groups: [[1, 2]],
            alpha_min: 1.0,
            alpha_max: 10.0,
          }),
        )
      })
  }, []) // mount only

  // ---- Derived values -------------------------------------------------------

  const horizonDays = useMemo(() => {
    if (!form) return 0
    return diffDays(form.start_date, form.end_date)
  }, [form?.start_date, form?.end_date]) // eslint-disable-line react-hooks/exhaustive-deps

  const slotsPerDay = form ? 24 / form.slot_duration_hours : 2
  const effectiveT = form ? Math.floor((horizonDays * 24) / form.slot_duration_hours) : 0

  // ---- Update helpers -------------------------------------------------------

  function patch(delta: Partial<FormState>) {
    setForm((prev) => (prev ? { ...prev, ...delta } : prev))
  }

  function patchBlockedSlots(machineKey: string, text: string) {
    setForm((prev) =>
      prev
        ? {
            ...prev,
            blocked_slots_text: {
              ...prev.blocked_slots_text,
              [machineKey]: text,
            },
          }
        : prev,
    )
  }

  function changeNMachines(newN: number) {
    setForm((prev) => {
      if (!prev) return prev
      const text = { ...prev.blocked_slots_text }
      const pa = { ...prev.pipeline_assignment }
      // Add empty entries for new machines
      for (let m = 1; m <= newN; m++) {
        if (!(String(m) in text)) text[String(m)] = ''
        if (!(m in pa)) pa[m] = null
      }
      // Remove entries for removed machines
      for (const key of Object.keys(text)) {
        if (Number(key) > newN) delete text[key]
      }
      for (const key of Object.keys(pa)) {
        if (Number(key) > newN) delete pa[Number(key)]
      }
      return { ...prev, n_machines: newN, blocked_slots_text: text, pipeline_assignment: pa }
    })
  }

  function setPipelineGroup(machine: number, group: string | null) {
    setForm((prev) =>
      prev
        ? {
            ...prev,
            pipeline_assignment: { ...prev.pipeline_assignment, [machine]: group },
          }
        : prev,
    )
  }

  function addPipelineGroup() {
    setForm((prev) => {
      if (!prev) return prev
      // Label for the new group
      const used = usedGroupLabels(prev.pipeline_assignment)
      const label = nextGroupLabel(used)
      // Assign all currently-unassigned machines to the new group? No — just show it as option.
      // The user will assign machines manually. Nothing to change in assignment yet.
      // But we want the new group to appear — we need at least one machine assigned.
      // Find the first unassigned machine and assign it.
      const firstFree = Object.entries(prev.pipeline_assignment).find(
        ([, v]) => v === null,
      )
      if (!firstFree) return prev
      return {
        ...prev,
        pipeline_assignment: {
          ...prev.pipeline_assignment,
          [Number(firstFree[0])]: label,
        },
      }
    })
  }

  // ---- Submit ---------------------------------------------------------------

  function handleNext() {
    if (!form) return
    onNext(formToConfig(form))
  }

  // ---- Render ---------------------------------------------------------------

  if (!form) {
    return (
      <div className={styles.spinnerWrap}>
        <Spinner size="lg" label="Cargando configuración…" />
      </div>
    )
  }

  return (
    <div className={styles.root}>
      <h2 className={styles.title}>Configuración del terminal</h2>

      {apiError !== null && (
        <div className={styles.apiBanner} role="alert">
          {apiError} — usando valores por defecto.
        </div>
      )}

      <div className={styles.form}>

        {/* ── Grupo 1: Planificación ── */}
        <div className={styles.group}>
          <span className={styles.groupLabel}>Planificación</span>
          <div className={styles.row}>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="start_date">Inicio</label>
              <input
                id="start_date"
                className={styles.input}
                type="date"
                value={form.start_date}
                onChange={(e) => {
                  const sd = e.target.value
                  patch({ start_date: sd })
                  if (sd && form.end_date < sd) {
                    patch({ start_date: sd, end_date: addDaysISO(sd, 30) })
                  }
                }}
              />
            </div>

            <div className={styles.field}>
              <label className={styles.label} htmlFor="end_date">Fin</label>
              <input
                id="end_date"
                className={styles.input}
                type="date"
                value={form.end_date}
                min={form.start_date}
                onChange={(e) => patch({ end_date: e.target.value })}
              />
            </div>

            <div className={styles.field}>
              <label className={styles.label} htmlFor="slot_duration">Duración del slot</label>
              <select
                id="slot_duration"
                className={styles.select}
                value={form.slot_duration_hours}
                onChange={(e) =>
                  patch({ slot_duration_hours: Number(e.target.value) as 12 | 24 | 48 })
                }
              >
                <option value={12}>12 h / slot</option>
                <option value={24}>24 h / slot</option>
                <option value={48}>48 h / slot</option>
              </select>
            </div>

            <div className={styles.effectiveT} aria-live="polite">
              <span>Horizonte efectivo</span>
              <span className={styles.effectiveTValue}>T = {effectiveT}</span>
              <span className={styles.sublabel}>{horizonDays}d × {slotsPerDay} slots</span>
            </div>
          </div>
        </div>

        <div className={styles.divider} />

        {/* ── Grupo 2: Buffer y Stock ── */}
        <div className={styles.group}>
          <span className={styles.groupLabel}>Buffer y Stock</span>
          <div className={styles.row}>
            <div className={styles.field}>
              <TooltipLabel
                label="Buffer de ullage"
                tip="Días mínimos de capacidad libre que deben mantenerse en los tanques para recibir crudo entrante."
              />
              <Stepper
                value={form.min_ullage_days}
                min={1}
                max={14}
                onChange={(v) => patch({ min_ullage_days: v })}
                ariaLabel="Buffer mínimo de ullage en días"
              />
              <span className={styles.sublabel}>días</span>
            </div>

            <div className={styles.field}>
              <TooltipLabel
                label="Tanques"
                tip="Número de tanques de almacenamiento en el terminal."
              />
              <Stepper
                value={form.n_tanks}
                min={1}
                max={20}
                onChange={(v) => patch({ n_tanks: v })}
                ariaLabel="Número de tanques"
              />
            </div>

            <div className={styles.field}>
              <TooltipLabel
                label="Capacidad tanque (m³)"
                tip="Volumen físico de un tanque individual en metros cúbicos."
              />
              <input
                className={styles.input}
                type="number"
                step="1000"
                min={10000}
                max={200000}
                value={form.tank_capacity_m3}
                onChange={(e) => {
                  const v = parseFloat(e.target.value)
                  if (!isNaN(v) && v >= 10000)
                    patch({ tank_capacity_m3: v })
                }}
                aria-label="Capacidad de un tanque en m3"
              />
            </div>

            <div className={styles.field}>
              <TooltipLabel
                label="Stock inicial (m³)"
                tip="Volumen total de crudo almacenado en los tanques al inicio del horizonte."
              />
              <input
                className={styles.input}
                type="number"
                step="10000"
                min={0}
                max={1000000}
                value={form.initial_terminal_stock_m3}
                onChange={(e) => {
                  const v = parseFloat(e.target.value)
                  if (!isNaN(v) && v >= 0)
                    patch({ initial_terminal_stock_m3: v })
                }}
                aria-label="Stock inicial del terminal en m3"
              />
            </div>

            <div className={styles.field}>
              <TooltipLabel
                label="Caudal entrante (m³/d)"
                tip="Caudal diario de crudo que ingresa a los tanques desde la estación upstream."
              />
              <input
                className={styles.input}
                type="number"
                step="1000"
                min={0}
                max={100000}
                value={form.daily_inflow_m3}
                onChange={(e) => {
                  const v = parseFloat(e.target.value)
                  if (!isNaN(v) && v >= 0)
                    patch({ daily_inflow_m3: v })
                }}
                aria-label="Caudal entrante diario en m3"
              />
            </div>
          </div>
        </div>

        <div className={styles.divider} />

        {/* ── Grupo 3: Terminal · Solver ── */}
        <div className={styles.group}>
          <span className={styles.groupLabel}>Terminal · Solver</span>
          <div className={styles.row}>
            <div className={styles.field}>
              <span className={styles.label}>Monoboyas</span>
              <Stepper
                value={form.n_machines}
                min={1}
                max={99}
                onChange={changeNMachines}
                ariaLabel="Número de monoboyas"
              />
            </div>

            <div className={`${styles.field} ${styles.fieldWide}`}>
              <TooltipLabel
                label="Oleoductos compartidos"
                tip="Agrupa las monoboyas que comparten un mismo oleoducto submarine. Los buques asignados a monoboyas del mismo grupo no pueden cargar simultáneamente."
              />
              <PipelineGroupEditor
                nMachines={form.n_machines}
                assignment={form.pipeline_assignment}
                onChangeGroup={setPipelineGroup}
                onAddGroup={addPipelineGroup}
              />
            </div>
          </div>

          <div className={styles.row} style={{ marginTop: '16px' }}>
            <div className={styles.field}>
              <TooltipLabel
                label="Penalización α"
                tip="α no es un hiperparámetro arbitrario: es una garantía matemática que impone la jerarquía P₁ > P₂ > P₃ > c_max del modelo QUBO."
              />
              <input
                className={styles.input}
                type="number"
                step="0.1"
                min={alphaRange.min}
                max={alphaRange.max}
                value={form.alpha}
                onChange={(e) => {
                  const v = parseFloat(e.target.value)
                  if (!isNaN(v))
                    patch({ alpha: Math.min(alphaRange.max, Math.max(alphaRange.min, v)) })
                }}
                aria-label={`Penalización alfa (${alphaRange.min}–${alphaRange.max})`}
              />
              <span className={styles.sublabel}>rango {alphaRange.min}–{alphaRange.max}</span>
            </div>

            <div className={styles.field}>
              <label className={styles.label} htmlFor="sampler">Sampler</label>
              <select
                id="sampler"
                className={styles.select}
                value={form.sampler}
                onChange={(e) => patch({ sampler: e.target.value as SamplerOption })}
              >
                {samplerOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {SAMPLER_LABELS[opt] ?? opt}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className={styles.divider} />

        {/* ── Grupo 4: Slots bloqueados ── */}
        <div className={styles.group}>
          <span className={styles.groupLabel}>Slots bloqueados</span>
          <div className={styles.row}>
            {Array.from({ length: form.n_machines }, (_, idx) => {
              const machineKey = String(idx + 1)
              return (
                <div key={machineKey} className={styles.field}>
                  <label className={styles.label} htmlFor={`blocked_${machineKey}`}>
                    M{machineKey}
                  </label>
                  <input
                    id={`blocked_${machineKey}`}
                    className={styles.input}
                    type="text"
                    placeholder="Ej: 10, 11, 30"
                    value={form.blocked_slots_text[machineKey] ?? ''}
                    onChange={(e) => patchBlockedSlots(machineKey, e.target.value)}
                    aria-label={`Slots bloqueados Monoboya ${machineKey}, separados por coma`}
                  />
                </div>
              )
            })}
          </div>
        </div>

      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <Button variant="primary" onClick={handleNext}>
          Siguiente →
        </Button>
      </div>
    </div>
  )
}
