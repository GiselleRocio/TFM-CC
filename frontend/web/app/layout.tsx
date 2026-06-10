import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import { User } from "lucide-react";
import "../styles/globals.css";
import Sidebar from "@/components/ui/Sidebar";
import styles from "./layout.module.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TFM — Scheduling Terminal",
  description:
    "Optimización híbrida cuántico-clásica de programación de buques",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es" className={`${outfit.variable} ${jetbrainsMono.variable}`}>
      <body>
        <div className={styles.shell}>
          <header className={styles.header}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo.svg" alt="TFM" className={styles.logo} />
            <div className={styles.avatar} role="img" aria-label="Usuario">
              <User size={17} strokeWidth={1.5} />
            </div>
          </header>

          <div className={styles.body}>
            <Sidebar />
            <main className={styles.main}>{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
