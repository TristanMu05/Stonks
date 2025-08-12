'use client';

import { useMemo } from 'react';
import Sparkline from './Sparkline';
import type { MarketTick } from './LiveTickers';

export default function SymbolList({ rows, historyBySymbol }: { rows: MarketTick[], historyBySymbol: Map<string, MarketTick[]> }) {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900/40">
      <table className="w-full text-sm">
        <thead className="bg-neutral-950/80 text-neutral-400">
          <tr>
            <th className="px-3 py-2 text-left">Symbol</th>
            <th className="px-3 py-2 text-left">Exchange</th>
            <th className="px-3 py-2 text-right">Price</th>
            <th className="px-3 py-2 text-right">Trend</th>
            <th className="px-3 py-2 text-right">Volume</th>
            <th className="px-3 py-2 text-right">Time</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <ListRow key={t.id} tick={t} historyBySymbol={historyBySymbol} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ListRow({ tick, historyBySymbol }: { tick: MarketTick, historyBySymbol?: Map<string, MarketTick[]> }) {
  const prices = (historyBySymbol?.get(tick.symbol) ?? []).slice(-60).map((t) => t.price);
  const min = useMemo(() => (prices.length ? Math.min(...prices) : 0), [prices]);
  const max = useMemo(() => (prices.length ? Math.max(...prices) : 1), [prices]);
  const points = useMemo(() => {
    const n = Math.max(prices.length, 2);
    return prices
      .map((v, i) => {
        const x = (i / (n - 1)) * 100;
        const y = max === min ? 50 : 100 - ((v - min) / (max - min)) * 100;
        return `${x},${y}`;
      })
      .join(' ');
  }, [prices, min, max]);

  const arrowUp = prices.length > 1 && prices[prices.length - 1] >= prices[0];

  return (
    <tr className="border-t border-neutral-800 hover:bg-neutral-900/60">
      <td className="px-3 py-2 font-semibold">{tick.symbol}</td>
      <td className="px-3 py-2 text-neutral-400">{tick.exchange}</td>
      <td className="px-3 py-2 text-right font-mono">{tick.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</td>
      <td className="px-3 py-2">
        <div className="flex items-center justify-end gap-2">
          <div className="h-8 w-24 rounded bg-neutral-950/50">
            {prices.length > 1 && <Sparkline data={prices} />}
          </div>
          <div className={`text-lg ${arrowUp ? 'text-emerald-400' : 'text-rose-400'}`}>
            {arrowUp ? '▲' : '▼'}
          </div>
        </div>
      </td>
      <td className="px-3 py-2 text-right">{tick.volume.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-neutral-400">{new Date(tick.timestamp).toLocaleTimeString()}</td>
    </tr>
  );
}


