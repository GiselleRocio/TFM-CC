// ============================================================
// TFM — Shared utilities
// ============================================================

/**
 * Converts a slot number to a Date.
 * Source: docs/architecture.md — Slot ↔ Date Conversion
 * Never show raw slot numbers in the UI — always use this function.
 */
export function slotToDate(
  slot: number,
  startDate: string,
  slotHours = 12,
): Date {
  // Normalise separator so both "2026/03/14" and "2026-03-14" work
  const base = new Date(startDate.replace(/\//g, "-"));
  base.setHours(base.getHours() + slot * slotHours);
  return base;
}

/** "14 mar" */
export function formatDateShort(date: Date): string {
  return date.toLocaleDateString("es-AR", { day: "2-digit", month: "short" });
}

/** "14 mar 08:00" */
export function formatDateTimeMed(date: Date): string {
  return date.toLocaleDateString("es-AR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}
