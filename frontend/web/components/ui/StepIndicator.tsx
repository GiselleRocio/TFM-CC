import { Check } from 'lucide-react'
import styles from './StepIndicator.module.css'

export interface StepIndicatorProps {
  /** Labels for each step, in order */
  steps: [string, string, string, string]
  /** 1-based index of the currently active step */
  currentStep: 1 | 2 | 3 | 4
}

export default function StepIndicator({
  steps,
  currentStep,
}: StepIndicatorProps) {
  return (
    <nav aria-label="Progreso del asistente">
      <ol className={styles.root}>
        {steps.map((label, index) => {
          const stepNumber = (index + 1) as 1 | 2 | 3 | 4
          const isCompleted = stepNumber < currentStep
          const isActive = stepNumber === currentStep

          const stepClass = [
            styles.step,
            isCompleted ? styles.stepCompleted : undefined,
            isActive ? styles.stepActive : undefined,
          ]
            .filter(Boolean)
            .join(' ')

          return (
            <li
              key={stepNumber}
              className={stepClass}
              aria-current={isActive ? 'step' : undefined}
            >
              <div className={styles.circle} aria-hidden="true">
                {isCompleted ? (
                  <Check className={styles.checkIcon} strokeWidth={2.5} />
                ) : (
                  stepNumber
                )}
              </div>
              <span className={styles.label}>{label}</span>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
