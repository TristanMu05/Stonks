'use client';

import { useEffect, useMemo, useState } from 'react';
import Sparkline from './Sparkline';
import type { MarketTick } from './LiveTickers';

type Props = {
  tick: MarketTick;
  history: MarketTick[];
  timeframe?: '1m' | '1d';
  windowSize?: number;
};

function formatTimeAgo(iso: string): string {
  const t = new Date(iso).getTime();
  const diff = Date.now() - t;
  if (diff < 1000) return 'now';
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  return `${hr}h ago`;
}

export default function TickerCard({ tick, history, timeframe = '1m', windowSize = 60 }: Props) {
  const [previousPrice, setPreviousPrice] = useState<number | null>(null);
  const [flash, setFlash] = useState<'up' | 'down' | null>(null);

  useEffect(() => {
    setFlash((prev) => {
      if (previousPrice === null) return null;
      if (tick.price > previousPrice) return 'up';
      if (tick.price < previousPrice) return 'down';
      return null;
    });
    const id = setTimeout(() => setFlash(null), 400);
    setPreviousPrice(tick.price);
    return () => clearTimeout(id);
  }, [tick.price]);

  const priceClass =
    flash === 'up'
      ? 'bg-emerald-500/10 text-emerald-400'
      : flash === 'down'
      ? 'bg-rose-500/10 text-rose-400'
      : 'bg-neutral-800 text-neutral-100';

  const priceDisplay = tick.price.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });

  const series = useMemo(() => history.slice(-windowSize).map((t) => t.price), [history, windowSize]);

  return (
    <a href={`/symbol/${encodeURIComponent(tick.symbol)}`} className="block rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 shadow-sm hover:border-neutral-700 transition-colors">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm uppercase tracking-wide text-neutral-400">{tick.exchange}</div>
          <div className="text-xl font-semibold">{tick.symbol}</div>
        </div>
        <div className={`rounded-lg px-3 py-1 text-lg font-mono ${priceClass}`}>${priceDisplay}</div>
      </div>
      {series.length > 1 && (
        <div className="mt-3 h-20 w-full rounded-md bg-neutral-950/50 p-1">
          <Sparkline data={series} />
        </div>
      )}
      <div className="mt-3 flex items-center justify-between text-sm text-neutral-400">
        <div>Vol: {tick.volume.toLocaleString()}</div>
        <div className="flex items-center gap-2">
          <span>{formatTimeAgo(tick.timestamp)}</span>
          <span className="rounded-md border border-neutral-800 bg-neutral-950 px-2 py-0.5 text-neutral-300">
            Details
          </span>
        </div>
      </div>
    </a>
  );
}


