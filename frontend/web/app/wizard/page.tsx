'use client'

import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import StepIndicator from '@/components/ui/StepIndicator'
import VesselInputStep from '@/components/scheduling/VesselInputStep'
import TerminalConfigStep from '@/components/scheduling/TerminalConfigStep'
import ProcessingStep from '@/components/scheduling/ProcessingStep'
import ResultsStep from '@/components/scheduling/ResultsStep'
import type { Vessel, TerminalConfig, JobDone } from '@/types/scheduling'
import styles from './page.module.css'

const STEP_LABELS: [string, string, string, string] = [
  'Configuración',
  'Buques',
  'Procesamiento',
  'Resultados',
]

type WizardStep = 1 | 2 | 3 | 4

export default function WizardPage() {
  const router = useRouter()

  const [step, setStep] = useState<WizardStep>(1)
  const [vessels, setVessels] = useState<Vessel[]>([])
  const [config, setConfig] = useState<TerminalConfig | null>(null)
  const [jobResult, setJobResult] = useState<JobDone | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)

  const goTo = useCallback(
    (next: WizardStep) => {
      setStep(next)
      router.replace(`/wizard?step=${next}`, { scroll: false })
    },
    [router],
  )

  return (
    <div className={styles.page}>
      <div className={styles.stepBar}>
        <StepIndicator steps={STEP_LABELS} currentStep={step} />
      </div>

      <div className={styles.content}>
        {step === 1 && (
          <TerminalConfigStep
            initialConfig={config}
            onBack={() => router.push('/')}
            onNext={(committed) => {
              setConfig(committed)
              goTo(2)
            }}
          />
        )}

        {step === 2 && config && (
          <VesselInputStep
            initialVessels={vessels}
            config={config}
            onBack={() => goTo(1)}
            onNext={(committed) => {
              setVessels(committed)
              goTo(3)
            }}
          />
        )}

        {step === 3 && config && (
          <ProcessingStep
            vessels={vessels}
            config={config}
            onBack={() => goTo(2)}
            onDone={(result, id) => {
              setJobResult(result)
              setJobId(id)
              goTo(4)
            }}
          />
        )}

        {step === 4 && jobResult && jobId && (
          <ResultsStep
            result={jobResult}
            jobId={jobId}
            vessels={vessels}
            config={config!}
            onNewRun={() => {
              setJobResult(null)
              setJobId(null)
              goTo(1)
            }}
          />
        )}
      </div>
    </div>
  )
}
