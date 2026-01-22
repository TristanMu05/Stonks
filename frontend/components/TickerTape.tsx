"use client";

import type { StockTicker } from "../lib/types";
import { formatCurrency, formatPercent } from "../lib/format";

type TickerTapeProps = {
  tickers: StockTicker[];
  onSelectTicker?: (ticker: StockTicker) => void;
};

const TickerTape = ({ tickers, onSelectTicker }: TickerTapeProps) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
            Live Ticker Tape
          </h2>
          <p className="text-xs text-slate-400">
            Streaming marks with bid/ask and percent change.
          </p>
        </div>
        <span className="text-xs text-slate-400">
          {tickers.length} symbols
        </span>
      </div>
      <div className="mt-3 flex gap-6 overflow-x-auto pb-2">
        {tickers.map((ticker) => {
          const tone =
            ticker.change >= 0 ? "text-emerald-400" : "text-rose-400";
          return (
            <button
              key={ticker.symbol}
              className="min-w-[190px] rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-left transition hover:border-slate-600"
              type="button"
              onClick={onSelectTicker ? () => onSelectTicker(ticker) : undefined}
            >
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>{ticker.exchange}</span>
                <span>{ticker.symbol}</span>
              </div>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-base font-semibold">
                  {formatCurrency(ticker.last)}
                </span>
                <span className={`text-sm font-semibold ${tone}`}>
                  {formatCurrency(ticker.change)} ({formatPercent(ticker.changePercent)})
                </span>
              </div>
              <div className="mt-1 text-xs text-slate-400">
                Bid {formatCurrency(ticker.bid)} x {formatCurrency(ticker.ask)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default TickerTape;
