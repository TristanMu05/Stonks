'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import TickerCard from './TickerCard';
import SymbolList from './SymbolList';

export type MarketTick = {
  id: string;
  symbol: string;
  price: number;
  volume: number;
  timestamp: string;
  exchange: string;
};

const SSE_URL = process.env.NEXT_PUBLIC_SSE_URL || 'http://localhost:8090/events';

function getApiBase(url: string): string {
  try {
    const u = new URL(url);
    return `${u.protocol}//${u.host}`;
  } catch {
    return 'http://localhost:8090';
  }
}

export default function LiveTickers() {
  const [lastTickBySymbol, setLastTickBySymbol] = useState<Map<string, MarketTick>>(new Map());
  const [historyBySymbol, setHistoryBySymbol] = useState<Map<string, MarketTick[]>>(new Map());
  const [connected, setConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<'cards' | 'list'>('cards');
  const [timeframe, setTimeframe] = useState<'1m' | '1d'>('1m');
  const [windowSize, setWindowSize] = useState<number>(60);
  const sourceRef = useRef<EventSource | null>(null);

  // bootstrap initial history
  useEffect(() => {
    const base = getApiBase(SSE_URL);
    fetch(`${base}/history`)
      .then((r) => r.json() as Promise<MarketTick[]>)
      .then((latest) => {
        const syms = latest.map((t) => t.symbol);
        // set latest
        setLastTickBySymbol(new Map(latest.map((t) => [t.symbol, t])));
        // fetch per-symbol history
        Promise.all(
          syms.map((s) =>
            fetch(`${base}/history?symbol=${encodeURIComponent(s)}&limit=200`).then((r) =>
              r.json() as Promise<MarketTick[]>
            )
          )
        ).then((arrays) => {
          const map = new Map<string, MarketTick[]>();
          arrays.forEach((arr) => {
            if (arr.length > 0) map.set(arr[0].symbol, arr);
          });
          setHistoryBySymbol(map);
        });
      })
      .catch(() => {});
  }, []);

  // live SSE updates
  useEffect(() => {
    const source = new EventSource(SSE_URL, { withCredentials: false });
    sourceRef.current = source;

    source.onopen = () => {
      setConnected(true);
      setError(null);
    };
    source.onerror = () => {
      setConnected(false);
      setError('Disconnected from SSE');
    };
    source.onmessage = (e) => {
      try {
        const tick = JSON.parse(e.data) as MarketTick;
        setLastTickBySymbol((prev) => {
          const next = new Map(prev);
          next.set(tick.symbol, tick);
          return next;
        });
        setHistoryBySymbol((prev) => {
          const next = new Map(prev);
          const arr = next.get(tick.symbol) ?? [];
          const merged = [...arr, tick].slice(-240);
          next.set(tick.symbol, merged);
          return next;
        });
      } catch {}
    };

    return () => {
      source.close();
    };
  }, []);

  const symbols = useMemo(() => Array.from(lastTickBySymbol.keys()).sort(), [lastTickBySymbol]);

  return (
    <section className="mt-6 space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-rose-500'}`} />
          <p className="text-sm text-neutral-400">
            {connected ? 'Live' : error ?? 'Connecting...'} · {SSE_URL}
          </p>
        </div>
        <div className="inline-flex rounded-lg border border-neutral-800 bg-neutral-900 p-1 text-sm">
          <button
            className={`rounded-md px-3 py-1 ${view === 'cards' ? 'bg-neutral-800' : ''}`}
            onClick={() => setView('cards')}
          >
            Cards
          </button>
          <button
            className={`rounded-md px-3 py-1 ${view === 'list' ? 'bg-neutral-800' : ''}`}
            onClick={() => setView('list')}
          >
            List
          </button>
        </div>
        {view === 'cards' && (
          <div className="inline-flex rounded-lg border border-neutral-800 bg-neutral-900 p-1 text-sm">
            <button
              className={`rounded-md px-3 py-1 ${timeframe === '1m' ? 'bg-neutral-800' : ''}`}
              onClick={() => setTimeframe('1m')}
            >
              1m
            </button>
            <button
              className={`rounded-md px-3 py-1 ${timeframe === '1d' ? 'bg-neutral-800' : ''}`}
              onClick={() => setTimeframe('1d')}
            >
              1d
            </button>
          </div>
        )}

        {view === 'cards' && (
          <div className="inline-flex rounded-lg border border-neutral-800 bg-neutral-900 p-1 text-sm">
            {[10, 60, 300].map((w) => (
              <button
                key={w}
                className={`rounded-md px-3 py-1 ${windowSize === w ? 'bg-neutral-800' : ''}`}
                onClick={() => setWindowSize(w)}
              >
                {w}
              </button>
            ))}
          </div>
        )}
      </div>

      {view === 'cards' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {symbols.map((sym) => {
            const tick = lastTickBySymbol.get(sym)!;
            const history = historyBySymbol.get(sym) ?? [];
            return <TickerCard key={sym} tick={tick} history={history} timeframe={timeframe} windowSize={windowSize} />;
          })}
        </div>
      ) : (
        <SymbolList
          rows={symbols.map((sym) => lastTickBySymbol.get(sym)!).filter(Boolean)}
          historyBySymbol={historyBySymbol}
        />
      )}

      {symbols.length === 0 && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-6 text-neutral-400">
          Waiting for ticks...
        </div>
      )}
    </section>
  );
}


