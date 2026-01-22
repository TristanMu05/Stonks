import "./globals.css";
import type { ReactNode } from "react";
import Link from "next/link";

export const metadata = {
  title: "Stonks Ticker Dashboard",
  description: "Think-or-Swim inspired market ticker board.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100">
        <div className="min-h-screen">
          <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
            <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
              <div className="flex items-center gap-3">
                <span className="h-3 w-3 rounded-full bg-emerald-400 shadow-sm shadow-emerald-500/40" />
                <span className="text-lg font-semibold tracking-wide">Stonks</span>
              </div>
              <nav className="flex items-center gap-4 text-sm text-slate-300">
                <Link className="transition hover:text-white" href="/">
                  Overview
                </Link>
                <Link className="transition hover:text-white" href="/settings">
                  Ticker Settings
                </Link>
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
