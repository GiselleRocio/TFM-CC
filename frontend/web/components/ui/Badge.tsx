import type { ReactNode } from 'react'
import styles from './Badge.module.css'

export type BadgeVariant = 'success' | 'warning' | 'error' | 'neutral'

export interface BadgeProps {
  variant?: BadgeVariant
  /** Show a coloured dot before the label */
  dot?: boolean
  children: ReactNode
  className?: string
}

const variantClass: Record<BadgeVariant, string> = {
  success: styles.success,
  warning: styles.warning,
  error: styles.error,
  neutral: styles.neutral,
}

export default function Badge({
  variant = 'neutral',
  dot = false,
  children,
  className,
}: BadgeProps) {
  const classes = [styles.badge, variantClass[variant], className]
    .filter(Boolean)
    .join(' ')

  return (
    <span className={classes}>
      {dot && <span className={styles.dot} aria-hidden="true" />}
      {children}
    </span>
  )
}
