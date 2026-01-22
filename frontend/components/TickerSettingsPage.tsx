"use client";

import Link from "next/link";
import { tickerColumns } from "../lib/tickerColumns";
import { useTickerSettings } from "./useTickerSettings";

const TickerSettingsPage = () => {
  const {
    visibleColumns,
    visibleCount,
    updateColumn,
    selectAll,
    clearAll,
    resetDefaults,
  } = useTickerSettings();

  return (
    <div className="space-y-8">
      <section className="rounded-xl border border-slate-800 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 px-6 py-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Ticker Settings</h1>
            <p className="mt-1 text-sm text-slate-400">
              Enable or disable the indicators you want on the tape.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-300">
            <span className="rounded-full border border-slate-700 px-3 py-1">
              {visibleCount} selected
            </span>
            <button
              className="rounded-full border border-slate-700 px-3 py-1 transition hover:border-slate-500 hover:text-white"
              type="button"
              onClick={selectAll}
            >
              Select all
            </button>
            <button
              className="rounded-full border border-slate-700 px-3 py-1 transition hover:border-slate-500 hover:text-white"
              type="button"
              onClick={clearAll}
            >
              Clear all
            </button>
            <button
              className="rounded-full border border-slate-700 px-3 py-1 transition hover:border-slate-500 hover:text-white"
              type="button"
              onClick={resetDefaults}
            >
              Reset defaults
            </button>
            <Link
              className="rounded-full border border-emerald-500/60 bg-emerald-500/10 px-3 py-1 text-emerald-200 transition hover:bg-emerald-500/20"
              href="/"
            >
              Back to board
            </Link>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/40">
        <div className="grid gap-4 p-6 md:grid-cols-2">
          {tickerColumns.map((column) => {
            const enabled = visibleColumns.includes(column.key);
            return (
              <label
                key={column.key}
                className="flex cursor-pointer items-start gap-3 rounded-lg border border-slate-800 bg-slate-950/50 p-4 transition hover:border-slate-600"
              >
                <input
                  className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-900 text-emerald-400 focus:ring-emerald-500"
                  type="checkbox"
                  checked={enabled}
                  onChange={(event) =>
                    updateColumn(column.key, event.target.checked)
                  }
                />
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <span>{column.label}</span>
                    <span className="rounded-full border border-slate-700 px-2 py-0.5 text-[10px] uppercase text-slate-400">
                      {column.group}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-400">
                    {column.description}
                  </p>
                </div>
              </label>
            );
          })}
        </div>
      </section>
    </div>
  );
};

export default TickerSettingsPage;
