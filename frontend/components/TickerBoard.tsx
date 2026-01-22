"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import type { StockTicker } from "../lib/types";
import { mockTickers } from "../lib/mockData";
import { tickerColumns } from "../lib/tickerColumns";
import type { ChartTimeScaleKey } from "../lib/chartData";
import TickerChartPanel from "./TickerChartPanel";
import TickerTape from "./TickerTape";
import TickerTable from "./TickerTable";
import { useDataCollectorQuotes } from "./useFinnhubQuotes";
import { useTickerSettings } from "./useTickerSettings";

const TickerBoard = () => {
  const { visibleColumns, visibleCount, resetDefaults } = useTickerSettings();
  const activeColumns = tickerColumns.filter((column) =>
    visibleColumns.includes(column.key),
  );
  const { tickers: liveTickers, status } = useDataCollectorQuotes(mockTickers);
  const [selectedTicker, setSelectedTicker] = useState<StockTicker | null>(
    mockTickers[0] ?? null,
  );
  const [timeScale, setTimeScale] = useState<ChartTimeScaleKey>("1D:5m");
  const [indicators, setIndicators] = useState({
    ema: true,
    vwap: true,
    macd: true,
  });
  const chartRef = useRef<HTMLDivElement | null>(null);

  const handleSelectTicker = useCallback((ticker: StockTicker) => {
    setSelectedTicker(ticker);
  }, []);

  const handleToggleIndicator = useCallback((key: keyof typeof indicators) => {
    setIndicators((current) => ({ ...current, [key]: !current[key] }));
  }, []);

  const selectedLabel = useMemo(
    () => selectedTicker?.symbol ?? "Select a ticker",
    [selectedTicker],
  );

  useEffect(() => {
    if (selectedTicker && chartRef.current) {
      chartRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedTicker]);

  return (
    <div className="space-y-8">
      <section className="rounded-xl border border-slate-800 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 px-6 py-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Live Stock Tickers</h1>
            <p className="mt-1 text-sm text-slate-400">
              Full quote context with bid/ask depth, liquidity, volatility, and
              fundamentals.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-300">
            <span className="rounded-full border border-slate-700 px-3 py-1">
              {visibleCount} indicators enabled
            </span>
            <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-400">
              Chart: {selectedLabel}
            </span>
            <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-400">
              Quotes: {status}
            </span>
            <button
              className="rounded-full border border-slate-700 px-3 py-1 transition hover:border-slate-500 hover:text-white"
              type="button"
              onClick={resetDefaults}
            >
              Reset defaults
            </button>
            <Link
              className="rounded-full border border-emerald-500/60 bg-emerald-500/10 px-3 py-1 text-emerald-200 transition hover:bg-emerald-500/20"
              href="/settings"
            >
              Manage indicators
            </Link>
          </div>
        </div>
      </section>

      <TickerTape tickers={liveTickers} onSelectTicker={handleSelectTicker} />

      <div ref={chartRef}>
        {selectedTicker ? (
          <TickerChartPanel
            ticker={selectedTicker}
            timeScale={timeScale}
            onTimeScaleChange={setTimeScale}
            indicators={indicators}
            onToggleIndicator={handleToggleIndicator}
            onClose={() => setSelectedTicker(null)}
          />
        ) : (
          <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/40 px-6 py-10 text-center text-sm text-slate-400">
            Click any symbol to open its chart with MACD, VWAP, and EMA overlays.
          </div>
        )}
      </div>

      <TickerTable
        tickers={liveTickers}
        columns={activeColumns}
        onSelectTicker={handleSelectTicker}
      />

      <section className="rounded-xl border border-slate-800 bg-slate-900/40 px-6 py-5 text-sm text-slate-300">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Indicator Sets
            </h3>
            <p className="text-xs text-slate-400">
              Use the settings page to tune which columns are visible per desk or
              strategy.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            {activeColumns.slice(0, 6).map((column) => (
              <span
                key={column.key}
                className="rounded-full border border-slate-700 px-3 py-1 text-slate-300"
              >
                {column.label}
              </span>
            ))}
            {activeColumns.length > 6 && (
              <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-400">
                +{activeColumns.length - 6} more
              </span>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

export default TickerBoard;
