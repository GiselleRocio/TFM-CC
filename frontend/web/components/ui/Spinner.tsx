import styles from './Spinner.module.css'

export type SpinnerSize = 'sm' | 'md' | 'lg'

export interface SpinnerProps {
  size?: SpinnerSize
  /** Accessible label — also rendered visually below the ring when provided */
  label?: string
}

const sizeClass: Record<SpinnerSize, string> = {
  sm: styles.sm,
  md: styles.md,
  lg: styles.lg,
}

export default function Spinner({ size = 'md', label }: SpinnerProps) {
  return (
    <div
      className={styles.root}
      role="status"
      aria-label={label ?? 'Cargando…'}
    >
      <div
        className={[styles.ring, sizeClass[size]].join(' ')}
        aria-hidden="true"
      />
      {label && <span className={styles.label}>{label}</span>}
    </div>
  )
}
