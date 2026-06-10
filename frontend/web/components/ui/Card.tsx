import type { ReactNode } from 'react'
import styles from './Card.module.css'

export interface CardProps {
  /** Content rendered in the card header bar */
  header?: ReactNode
  /** Content rendered in the card body */
  children: ReactNode
  /** Use secondary background for the body instead of primary */
  secondaryBody?: boolean
  className?: string
}

export default function Card({
  header,
  children,
  secondaryBody = false,
  className,
}: CardProps) {
  const bodyClass = [
    styles.body,
    secondaryBody ? styles.bodySecondary : undefined,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className={[styles.card, className].filter(Boolean).join(' ')}>
      {header !== undefined && <div className={styles.header}>{header}</div>}
      <div className={bodyClass}>{children}</div>
    </div>
  )
}
