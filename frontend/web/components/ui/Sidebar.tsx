'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { LayoutTemplate, History, ScanSearch, Ship } from 'lucide-react'
import styles from './Sidebar.module.css'

const NAV = [
  { label: 'Nuevo Scheduling', href: '/wizard', icon: LayoutTemplate },
  { label: 'Historial', href: '/history', icon: History },
  { label: 'Preview Solución', href: '/preview', icon: ScanSearch },
  { label: 'Ventanas de Arribo', href: '/vessel-windows', icon: Ship },
] as const

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className={styles.sidebar}>
      <nav aria-label="Navegación principal">
        <span className={styles.sectionLabel}>Módulos</span>
        <ul className={styles.list}>
          {NAV.map(({ label, href, icon: Icon }) => {
            const isActive = pathname === href || pathname.startsWith(href + '?')
            return (
              <li key={href}>
                <Link
                  href={href}
                  className={`${styles.item} ${isActive ? styles.itemActive : ''}`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon size={15} className={styles.icon} />
                  <span>{label}</span>
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>
    </aside>
  )
}
