"use client";

import type { StockTicker, TickerColumn } from "../lib/types";

type TickerTableProps = {
  tickers: StockTicker[];
  columns: TickerColumn[];
  onSelectTicker?: (ticker: StockTicker) => void;
};

const TickerTable = ({ tickers, columns, onSelectTicker }: TickerTableProps) => {
  if (columns.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-700 bg-slate-900/40 px-6 py-8 text-center text-sm text-slate-400">
        No columns enabled. Head to settings to turn indicators on.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
            Market Indicators
          </h2>
          <p className="text-xs text-slate-400">
            Full quote tape with real-time metrics.
          </p>
        </div>
        <span className="text-xs text-slate-400">
          {columns.length} columns
        </span>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-950/70 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              {columns.map((column) => {
                const alignClass =
                  column.align === "right"
                    ? "text-right"
                    : column.align === "center"
                      ? "text-center"
                      : "text-left";
                return (
                <th
                  key={column.key}
                  className={`border-b border-slate-800 px-3 py-2 ${alignClass} font-medium ${column.widthClass ?? ""}`}
                >
                  <div className="flex flex-col">
                    <span>{column.label}</span>
                    <span className="text-[10px] font-normal normal-case text-slate-500">
                      {column.description}
                    </span>
                  </div>
                </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {tickers.map((ticker) => (
              <tr
                key={ticker.symbol}
                className={`odd:bg-slate-950/30 hover:bg-slate-900/70 ${
                  onSelectTicker ? "cursor-pointer" : ""
                }`}
                role={onSelectTicker ? "button" : undefined}
                tabIndex={onSelectTicker ? 0 : undefined}
                onClick={onSelectTicker ? () => onSelectTicker(ticker) : undefined}
                onKeyDown={
                  onSelectTicker
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onSelectTicker(ticker);
                        }
                      }
                    : undefined
                }
              >
                {columns.map((column) => {
                  const alignClass =
                    column.align === "right"
                      ? "text-right"
                      : column.align === "center"
                        ? "text-center"
                        : "text-left";
                  return (
                    <td
                      key={`${ticker.symbol}-${column.key}`}
                      className={`px-3 py-2 align-top ${alignClass} ${column.widthClass ?? ""}`}
                    >
                      {column.render(ticker)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TickerTable;
