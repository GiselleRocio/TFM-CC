'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { InventoryCurveEntry, TerminalConfig } from '@/types/scheduling'
import styles from './InventoryChart.module.css'

interface InventoryChartProps {
  inventoryCurve: InventoryCurveEntry[]
  config: TerminalConfig
  slotDurationHours: number
}

export default function InventoryChart({
  inventoryCurve,
  config,
  slotDurationHours,
}: InventoryChartProps) {
  if (!inventoryCurve || inventoryCurve.length === 0) {
    return (
      <div className={styles.empty}>
        No hay datos de inventario disponibles.
      </div>
    )
  }

  // Calculate thresholds
  const nTanks = config.n_tanks ?? 6
  const tankCapacity = config.tank_capacity_m3 ?? 100000
  const maxCapacity = nTanks * tankCapacity

  // Use configurable daily inflow rate for threshold
  const dailyInflow = config.daily_inflow_m3 ?? 20000
  const minUllageDays = config.min_ullage_days ?? 4
  const safeThreshold = maxCapacity - (dailyInflow * minUllageDays)

  // Format large numbers with K/M suffix
  const formatVolume = (value: number): string => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(0)}K`
    }
    return value.toString()
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean
    payload?: Array<{ value: number; color: string; name: string }>
    label?: string
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipDate}>{label}</p>
          <p className={styles.tooltipStock}>
            Stock: <strong>{payload[0].value?.toLocaleString()} m³</strong>
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3 className={styles.title}>Curva de Inventario</h3>
        <p className={styles.subtitle}>
          Proyección del nivel de stock en los tanques a lo largo del horizonte de planificación
        </p>
      </div>

      <div className={styles.legendInfo}>
        <div className={styles.legendItem}>
          <span className={styles.legendLineDanger} />
          <span>Capacidad máxima: {maxCapacity.toLocaleString()} m³</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendLineWarning} />
          <span>Umbral seguro: {safeThreshold.toLocaleString()} m³ ({dailyInflow.toLocaleString()} m³/d × {minUllageDays}d)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendLinePrimary} />
          <span>Stock proyectado</span>
        </div>
      </div>

      <div className={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={inventoryCurve}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="date"
              stroke="rgba(255,255,255,0.55)"
              tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.2)' }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="rgba(255,255,255,0.55)"
              tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.2)' }}
              tickFormatter={formatVolume}
              domain={[0, maxCapacity * 1.1]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <ReferenceLine
              y={maxCapacity}
              stroke="#E24B4A"
              strokeWidth={2}
              label={{
                value: 'Cap. máx',
                fill: '#E24B4A',
                fontSize: 10,
                position: 'right',
              }}
            />
            <ReferenceLine
              y={safeThreshold}
              stroke="#F59E0B"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{
                value: 'Umbral',
                fill: '#F59E0B',
                fontSize: 10,
                position: 'right',
              }}
            />
            <Line
              type="monotone"
              dataKey="stock_m3"
              name="Stock (m³)"
              stroke="#00C4B4"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#00C4B4' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.warningNote}>
        <strong>Nota:</strong> El stock debe mantenerse por debajo de la línea punteada
        (umbral de ullage) para evitar desbordamiento de los tanques.
      </div>
    </div>
  )
}