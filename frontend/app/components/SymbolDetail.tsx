'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Sparkline from './Sparkline';
import type { MarketTick } from './LiveTickers';

const SSE_URL = process.env.NEXT_PUBLIC_SSE_URL || 'http://localhost:8090/events';

function getApiBase(url: string): string {
  try { const u = new URL(url); return `${u.protocol}//${u.host}`; } catch { return 'http://localhost:8090'; }
}

export default function SymbolDetail({ symbol }: { symbol: string }) {
  const [history, setHistory] = useState<MarketTick[]>([]);
  const [connected, setConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const [windowSize, setWindowSize] = useState<number>(300);

  useEffect(() => {
    const base = getApiBase(SSE_URL);
    fetch(`${base}/history?symbol=${encodeURIComponent(symbol)}&limit=1000`)
      .then((r) => r.json())
      .then((arr: MarketTick[]) => setHistory(arr))
      .catch(() => {});
  }, [symbol]);

  useEffect(() => {
    const source = new EventSource(SSE_URL);
    sourceRef.current = source;
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (e) => {
      try {
        const tick = JSON.parse(e.data) as MarketTick;
        if (tick.symbol !== symbol) return;
        setHistory((prev) => [tick, ...prev].slice(0, 1000));
      } catch {}
    };
    return () => source.close();
  }, [symbol]);

  const min = useMemo(() => (history.length ? Math.min(...history.map((t) => t.price)) : 0), [history]);
  const max = useMemo(() => (history.length ? Math.max(...history.map((t) => t.price)) : 1), [history]);
  const pct = (v: number) => (max === min ? 50 : ((v - min) / (max - min)) * 100);

  useEffect(() => {
    // autoscroll to top since newest now at top
    if (scrollerRef.current) scrollerRef.current.scrollTop = 0;
  }, [history.length]);

  return (
    <section className="mt-6 space-y-4">
      <div className="flex items-center gap-2 text-sm text-neutral-400">
        <div className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-rose-500'}`} />
        Live · {symbol}
      </div>

      <div className="relative h-48 w-full overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900/40 p-2">
        {history.length > 1 && (
          <Sparkline data={history.slice(0, windowSize).reverse().map((t) => t.price)} width={100} height={100} />
        )}
      </div>

      <div className="flex items-center gap-2 text-sm">
        <span className="text-neutral-400">Window:</span>
        {[60, 300, 600].map((w) => (
          <button key={w} onClick={() => setWindowSize(w)} className={`rounded-md border px-2 py-0.5 ${windowSize === w ? 'bg-neutral-800 border-neutral-700' : 'border-neutral-800'}`}>
            {w}
          </button>
        ))}
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
        <div className="text-sm text-neutral-400 mb-2">Recent ticks (newest first)</div>
        <div ref={scrollerRef} className="relative h-64 overflow-auto">
          <div className="absolute inset-0">
            {history.map((t) => (
              <div
                key={t.id}
                className="flex items-center gap-4 border-b border-neutral-800 px-2 py-1 text-sm animate-[pop_200ms_ease-out]"
                style={{
                  transformOrigin: 'top left',
                }}
              >
                <div className="w-24 font-mono text-neutral-400">{new Date(t.timestamp).toLocaleTimeString()}</div>
                <div className="w-24 text-right font-mono">{t.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</div>
                <div className="w-24 text-right">{t.volume.toLocaleString()}</div>
                <div className="flex-1">
                  <div className="h-2 rounded bg-neutral-800">
                    <div className="h-2 rounded bg-emerald-500/70" style={{ width: `${pct(t.price)}%` }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}


