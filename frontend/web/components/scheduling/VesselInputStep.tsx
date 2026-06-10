"use client";

import { useState, useCallback, useRef } from "react";
import * as XLSX from "xlsx";
import { Upload, Trash2, Plus, Shuffle, Download } from "lucide-react";
import { downloadVesselsExcel } from "@/lib/excel";
import Button from "@/components/ui/Button";
import Spinner from "@/components/ui/Spinner";
import { postGenerate } from "@/lib/api";
import { getPriorityTier } from "@/types/scheduling";
import type { Vessel, PriorityTier, TerminalConfig } from "@/types/scheduling";
import styles from "./VesselInputStep.module.css";

// ---- Types ----------------------------------------------------------------

export interface VesselInputStepProps {
  initialVessels: Vessel[];
  config: TerminalConfig;
  onBack: () => void;
  onNext: (vessels: Vessel[]) => void;
}

type InputMode = "generate" | "upload";

// ---- Helpers ---------------------------------------------------------------

async function parseExcelFile(file: File): Promise<Vessel[]> {
  const buf = await file.arrayBuffer();
  const wb = XLSX.read(buf);
  const ws = wb.Sheets[wb.SheetNames[0]];
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(ws);

  return rows.map((row): Vessel => {
    const ps = Number(row.processing_slots ?? 0);
    return {
      vessel_id: String(row.vessel_id ?? ""),
      volume_m3: Number(row.volume_m3 ?? 0),
      daily_inflow_m3: Number(row.daily_inflow_m3 ?? 0),
      cargo_m3: row.cargo_m3 !== undefined ? Number(row.cargo_m3) : undefined,
      release_slot: Number(row.release_slot ?? 0),
      due_slot: Number(row.due_slot ?? 0),
      processing_slots: ps,
      priority_weight:
        row.priority_weight !== undefined
          ? Number(row.priority_weight)
          : undefined,
    };
  });
}

function validateVessels(
  vs: Vessel[],
  procSmall: number,
  procLarge: number,
): string[] {
  const errors: string[] = [];

  if (vs.length === 0) {
    errors.push("Debe ingresar al menos un buque.");
    return errors;
  }

  vs.forEach((v, i) => {
    const label = v.vessel_id.trim()
      ? `Buque "${v.vessel_id}"`
      : `Fila ${i + 1}`;

    if (!v.vessel_id.trim()) errors.push(`${label}: vessel_id requerido.`);
    if (v.volume_m3 <= 0)
      errors.push(`${label}: stock acumulado debe ser > 0 m³.`);
    if (v.daily_inflow_m3 <= 0)
      errors.push(`${label}: inflow diario debe ser > 0 m³/d.`);
    if (v.release_slot < 0)
      errors.push(`${label}: slot de llegada debe ser ≥ 0.`);
    if (v.due_slot <= v.release_slot)
      errors.push(`${label}: slot límite debe ser > slot de llegada.`);
    if (v.processing_slots !== procSmall && v.processing_slots !== procLarge)
      errors.push(
        `${label}: tiempo de procesamiento debe ser ${procSmall} o ${procLarge} slots.`,
      );
  });

  return errors;
}

function makeEmptyVessel(index: number, procLarge: number): Vessel {
  return {
    vessel_id: `V${String(index + 1).padStart(2, "0")}`,
    volume_m3: 0,
    daily_inflow_m3: 0,
    release_slot: 0,
    due_slot: procLarge + 2,
    processing_slots: procLarge,
  };
}

/** Compute priority weight (ESD) for display — volume_m3 / daily_inflow_m3 */
function computeDisplayWeight(v: Vessel): number {
  if (v.priority_weight !== undefined) return v.priority_weight;
  if (v.daily_inflow_m3 > 0) return v.volume_m3 / v.daily_inflow_m3;
  return 0;
}

const TIER_CLASS: Record<PriorityTier, string> = {
  none: styles.tierNone,
  yellow: styles.tierYellow,
  amber: styles.tierAmber,
  red: styles.tierRed,
};

// ---- Component ------------------------------------------------------------

export default function VesselInputStep({
  initialVessels,
  config,
  onBack,
  onNext,
}: VesselInputStepProps) {
  const [mode, setMode] = useState<InputMode>("generate");
  const [vessels, setVessels] = useState<Vessel[]>(initialVessels);
  const [nVessels, setNVessels] = useState(4);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [capacityWarning, setCapacityWarning] = useState<string | null>(null);

  const slotDurationHours = config.slot_duration_hours;
  // Processing times derived from domain: small = 2 days, large = 4 days
  const procSmall = 48 / slotDurationHours;
  const procLarge = 96 / slotDurationHours;
  // Max vessels that fit within the horizon.
  // The backend oversaturation check uses Σ p_j: capacity = parallelism × T.
  // When the pipeline is shared, all monobuoys serialize (1 active at a time),
  // so capacity = T regardless of monobuoy count.
  // When monobuoys are independent each can load in parallel, so multiply by n_machines.
  const T = Math.floor((config.horizon_days * 24) / slotDurationHours);
  // Effective parallel streams: each shared-pipeline group serialises to 1,
  // independent monobuoys each contribute 1. Monobuoys not in any group are independent.
  const parallelism = (() => {
    const groups = config.shared_pipeline_groups ?? [];
    const grouped = new Set(groups.flatMap((g) => g));
    const nSharedGroups = groups.filter((g) => g.length >= 2).length;
    const nIndependent = Array.from(
      { length: config.n_machines },
      (_, i) => i + 1,
    ).filter((m) => !grouped.has(m)).length;
    return nSharedGroups + nIndependent;
  })();
  const maxVessels = parallelism * Math.floor(T / procLarge);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ---- Vessel list mutations -----------------------------------------------

  const updateVessel = useCallback(
    (index: number, patch: Partial<Vessel>) => {
      setVessels((prev) =>
        prev.map((v, i) => {
          if (i !== index) return v;
          const next = { ...v, ...patch };
          return next;
        }),
      );
      // Clear validation on edit
      if (validationErrors.length > 0) setValidationErrors([]);
    },
    [validationErrors.length],
  );

  const addVessel = useCallback(() => {
    setVessels((prev) => [...prev, makeEmptyVessel(prev.length, procLarge)]);
  }, [procLarge]);

  const removeVessel = useCallback((index: number) => {
    setVessels((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // ---- Generate ------------------------------------------------------------

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setApiError(null);
    try {
      const result = await postGenerate({
        n_vessels: nVessels,
        slot_duration_hours: slotDurationHours,
        n_machines: config.n_machines,
      });
      setVessels(result);
    } catch (err) {
      setApiError(
        err instanceof Error ? err.message : "Error al generar buques.",
      );
    } finally {
      setLoading(false);
    }
  }, [nVessels, slotDurationHours]);

  // ---- Excel upload --------------------------------------------------------

  const processFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".xlsx") && !file.name.endsWith(".xls")) {
      setApiError("Solo se aceptan archivos .xlsx");
      return;
    }
    setLoading(true);
    setApiError(null);
    try {
      const parsed = await parseExcelFile(file);
      setVessels(parsed);
    } catch (err) {
      setApiError(
        err instanceof Error ? err.message : "Error al leer el archivo Excel.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) void processFile(file);
    },
    [processFile],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) void processFile(file);
      e.target.value = "";
    },
    [processFile],
  );

  // ---- Proceed to next step ------------------------------------------------

  const handleNext = useCallback(() => {
    const errors = validateVessels(vessels, procSmall, procLarge);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }
    if (vessels.length > maxVessels) {
      setCapacityWarning(
        `Aviso: con ${vessels.length} buques se supera el estimado de ${maxVessels} ` +
          `para buques grandes en un horizonte de ${config.horizon_days} días. ` +
          `Si el dataset es mixto (chicos y grandes) esto es normal — el modelo lo manejará.`,
      );
    } else {
      setCapacityWarning(null);
    }
    onNext(vessels);
  }, [vessels, maxVessels, config.horizon_days, procSmall, procLarge, onNext]);

  // ---- Render --------------------------------------------------------------

  return (
    <div className={styles.root}>
      {/* Header */}
      <div className={styles.pageHeader}>
        <h2 className={styles.title}>Nominación de buques</h2>
        <p className={styles.subtitle}>
          Cargue la lista de buques desde un archivo Excel o genere datos
          sintéticos para pruebas. Puede editar cualquier campo de la tabla
          antes de continuar.
        </p>
      </div>

      {/* Mode toggle */}
      <div>
        <div
          className={styles.modeToggle}
          role="group"
          aria-label="Modo de carga"
        >
          <button
            className={[
              styles.modeBtn,
              mode === "generate" ? styles.modeBtnActive : "",
            ]
              .filter(Boolean)
              .join(" ")}
            onClick={() => setMode("generate")}
            aria-pressed={mode === "generate"}
          >
            <Shuffle size={14} aria-hidden="true" />
            Generar aleatorio
          </button>
          <button
            className={[
              styles.modeBtn,
              mode === "upload" ? styles.modeBtnActive : "",
            ]
              .filter(Boolean)
              .join(" ")}
            onClick={() => setMode("upload")}
            aria-pressed={mode === "upload"}
          >
            <Upload size={14} aria-hidden="true" />
            Cargar Excel
          </button>
        </div>
      </div>

      {/* Generate panel */}
      {mode === "generate" && (
        <div className={styles.generatePanel}>
          <span className={styles.generateLabel}>Número de buques</span>
          <div className={styles.nVesselsInput}>
            <button
              className={styles.stepperBtn}
              onClick={() => setNVessels((n) => Math.max(1, n - 1))}
              disabled={nVessels <= 1}
              aria-label="Reducir"
            >
              −
            </button>
            <input
              type="number"
              className={styles.nVesselsField}
              min={1}
              value={nVessels}
              aria-label="Número de buques"
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (!isNaN(v) && v >= 1) setNVessels(v);
              }}
            />
            <button
              className={styles.stepperBtn}
              onClick={() => setNVessels((n) => n + 1)}
              aria-label="Aumentar"
            >
              +
            </button>
          </div>
          <span className={styles.slotHint}>
            ref. {maxVessels} buques grandes (T={T}, {parallelism} cadena
            {parallelism !== 1 ? "s" : ""})
          </span>
          <Button
            variant="primary"
            size="sm"
            loading={loading}
            onClick={() => void handleGenerate()}
          >
            Generar buques
          </Button>
        </div>
      )}

      {/* Upload zone */}
      {mode === "upload" && (
        <div
          className={[styles.dropZone, dragOver ? styles.dropZoneActive : ""]
            .filter(Boolean)
            .join(" ")}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ")
              fileInputRef.current?.click();
          }}
          aria-label="Zona de carga — arrastra un .xlsx o haz clic para seleccionar"
        >
          {loading ? (
            <Spinner size="md" label="Leyendo archivo…" />
          ) : (
            <>
              <Upload
                className={styles.uploadIcon}
                size={32}
                aria-hidden="true"
              />
              <p className={styles.uploadTitle}>
                Arrastra un archivo .xlsx aquí
              </p>
              <p className={styles.uploadSub}>o haz clic para seleccionar</p>
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx,.xls"
            className={styles.hiddenInput}
            onChange={handleFileChange}
            aria-hidden="true"
            tabIndex={-1}
          />
        </div>
      )}

      {/* API error */}
      {apiError !== null && (
        <div className={styles.apiBanner} role="alert">
          {apiError}
        </div>
      )}

      {/* Capacity advisory (non-blocking) */}
      {capacityWarning !== null && (
        <div className={styles.warningBanner} role="status">
          {capacityWarning}
        </div>
      )}

      {/* Editable table */}
      {vessels.length > 0 && (
        <div className={styles.tableSection}>
          <div className={styles.tableMeta}>
            <span className={styles.vesselCount}>
              {vessels.length} buque{vessels.length !== 1 ? "s" : ""}
            </span>
            <Button variant="ghost" size="sm" onClick={addVessel}>
              <Plus size={12} aria-hidden="true" />
              Añadir fila
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => downloadVesselsExcel(vessels)}
            >
              <Download size={12} aria-hidden="true" />
              Descargar Excel
            </Button>
          </div>

          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead className={styles.thead}>
                <tr>
                  <th className={`${styles.th} ${styles.thAction}`} />
                  <th className={styles.th} scope="col">
                    Buque
                  </th>
                  <th className={styles.th} scope="col">
                    Slot llegada
                  </th>
                  <th className={styles.th} scope="col">
                    Slot límite
                  </th>
                  <th className={styles.th} scope="col">
                    T. proc.
                  </th>
                  <th className={styles.th} scope="col">
                    Stock (m³)
                  </th>
                  <th className={styles.th} scope="col">
                    Cargo (m³)
                  </th>
                  <th className={styles.th} scope="col">
                    Inflow (m³/d)
                  </th>
                  <th className={styles.th} scope="col">
                    Peso (wj)
                  </th>
                </tr>
              </thead>
              <tbody className={styles.tbody}>
                {vessels.map((vessel, i) => {
                  const displayWeight = computeDisplayWeight(vessel);
                  const tier = getPriorityTier(displayWeight);

                  return (
                    <tr key={i} className={styles.tr}>
                      {/* Delete */}
                      <td className={styles.td}>
                        <button
                          className={styles.btnRemove}
                          onClick={() => removeVessel(i)}
                          aria-label={`Eliminar buque ${vessel.vessel_id || i + 1}`}
                        >
                          <Trash2 size={13} aria-hidden="true" />
                        </button>
                      </td>

                      {/* vessel_id */}
                      <td className={styles.td}>
                        <input
                          className={styles.cellInput}
                          type="text"
                          value={vessel.vessel_id}
                          onChange={(e) =>
                            updateVessel(i, { vessel_id: e.target.value })
                          }
                          aria-label="Identificador del buque"
                        />
                      </td>

                      {/* release_slot */}
                      <td className={styles.td}>
                        <input
                          className={styles.cellInput}
                          type="number"
                          step="1"
                          min="0"
                          value={vessel.release_slot}
                          onChange={(e) =>
                            updateVessel(i, {
                              release_slot: parseInt(e.target.value, 10) || 0,
                            })
                          }
                          aria-label="Slot de llegada"
                        />
                      </td>

                      {/* due_slot */}
                      <td className={styles.td}>
                        <input
                          className={styles.cellInput}
                          type="number"
                          step="1"
                          min="1"
                          value={vessel.due_slot}
                          onChange={(e) =>
                            updateVessel(i, {
                              due_slot: parseInt(e.target.value, 10) || 0,
                            })
                          }
                          aria-label="Slot límite"
                        />
                      </td>

                      {/* processing_slots */}
                      <td className={styles.td}>
                        <select
                          className={styles.cellSelect}
                          value={vessel.processing_slots}
                          onChange={(e) =>
                            updateVessel(i, {
                              processing_slots:
                                Number(e.target.value) === procSmall
                                  ? procSmall
                                  : procLarge,
                            })
                          }
                          aria-label="Tiempo de procesamiento en slots"
                        >
                          <option value={procSmall}>Chico (2 días)</option>
                          <option value={procLarge}>Grande (4 días)</option>
                        </select>
                      </td>

                      {/* volume_m3 */}
                      <td className={styles.td}>
                        <input
                          className={styles.cellInput}
                          type="number"
                          step="1"
                          min="0"
                          value={vessel.volume_m3}
                          onChange={(e) =>
                            updateVessel(i, {
                              volume_m3: parseInt(e.target.value, 10) || 0,
                            })
                          }
                          aria-label="Stock acumulado en m³"
                        />
                      </td>

                      {/* cargo_m3 — read-only (computed by API) */}
                      <td className={`${styles.td} ${styles.tdReadOnly}`}>
                        {vessel.cargo_m3 !== undefined
                          ? vessel.cargo_m3.toLocaleString("es-AR")
                          : "—"}
                      </td>

                      {/* daily_inflow_m3 */}
                      <td className={styles.td}>
                        <input
                          className={styles.cellInput}
                          type="number"
                          step="1"
                          min="1"
                          value={vessel.daily_inflow_m3}
                          onChange={(e) =>
                            updateVessel(i, {
                              daily_inflow_m3:
                                parseInt(e.target.value, 10) || 0,
                            })
                          }
                          aria-label="Inflow del cargador en m³/día"
                        />
                      </td>

                      {/* priority_weight — read-only, color-coded */}
                      <td
                        className={[
                          styles.td,
                          styles.tdWeight,
                          TIER_CLASS[tier],
                        ].join(" ")}
                      >
                        {displayWeight > 0 ? displayWeight.toFixed(1) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Validation errors */}
      {validationErrors.length > 0 && (
        <ul className={styles.errorList} role="alert" aria-live="assertive">
          {validationErrors.map((msg, i) => (
            <li key={i} className={styles.errorItem}>
              {msg}
            </li>
          ))}
        </ul>
      )}

      {/* Footer */}
      <div className={styles.footer}>
        <Button variant="ghost" onClick={onBack}>
          ← Volver
        </Button>
        <Button variant="primary" onClick={handleNext} disabled={loading}>
          Siguiente →
        </Button>
      </div>
    </div>
  );
}
