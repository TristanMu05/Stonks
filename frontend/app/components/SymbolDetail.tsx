'use client';

import { useMemo, useRef, useEffect, useState } from 'react';
import ProfessionalChart from './ProfessionalChart';
import type { MarketTick } from './LiveTickers';

interface SymbolDetailProps {
  symbol: string;
  currentTick: MarketTick;
  history: MarketTick[];
  connected: boolean;
}

export default function SymbolDetail({ symbol, currentTick, history, connected }: SymbolDetailProps) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const [windowSize, setWindowSize] = useState<number>(300);

  const { min, max, pct } = useMemo(() => {
    if (!history.length) return { min: 0, max: 1, pct: () => 50 };
    const prices = history.map((t) => t.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    return {
      min: minPrice,
      max: maxPrice,
      pct: (v: number) => (maxPrice === minPrice ? 50 : ((v - minPrice) / (maxPrice - minPrice)) * 100)
    };
  }, [history]);

  useEffect(() => {
    if (scrollerRef.current) scrollerRef.current.scrollTop = 0;
  }, [history.length]);

  const displayHistory = useMemo(() => {
    return history
      .slice(0, windowSize)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [history, windowSize]);

  const recentTicks = useMemo(() => {
    return history
      .slice(0, 200)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [history]);

  return (
    <section className="mt-6 space-y-4">
      <div className="flex items-center gap-2 text-sm text-neutral-400">
        <div className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-rose-500'}`} />
        Live · {symbol}
        <span className="text-neutral-500">•</span>
        <span className="text-emerald-400 font-mono">
          ${currentTick.price.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 4,
          })}
        </span>
        <span className="text-neutral-500">•</span>
        <span className="text-neutral-400">
          {new Date(currentTick.timestamp).toLocaleTimeString()}
        </span>
      </div>

      <div className="relative w-full">
        {displayHistory.length > 1 && (
          <ProfessionalChart 
            data={displayHistory} 
            height={400}
            symbol={symbol}
          />
        )}
        {displayHistory.length <= 1 && (
          <div className="border border-neutral-700 rounded-lg bg-neutral-900/40 p-8 text-center text-neutral-400">
            <div className="animate-pulse">Waiting for more tick data...</div>
            <div className="text-sm mt-2">Currently have {history.length} tick{history.length !== 1 ? 's' : ''}</div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 text-sm">
        <span className="text-neutral-400">Chart Window:</span>
        {[60, 300, 600].map((w) => (
          <button 
            key={w} 
            onClick={() => setWindowSize(w)} 
            className={`rounded-md border px-2 py-0.5 transition-colors ${
              windowSize === w 
                ? 'bg-neutral-800 border-neutral-700 text-white' 
                : 'border-neutral-800 hover:border-neutral-700'
            }`}
          >
            {w}
          </button>
        ))}
        <span className="text-neutral-500 ml-2">• {displayHistory.length} points shown</span>
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm text-neutral-400">Recent ticks (newest first)</div>
          <div className="text-xs text-neutral-500">
            {recentTicks.length} of {history.length} total
          </div>
        </div>
        <div ref={scrollerRef} className="relative h-64 overflow-auto">
          <div className="absolute inset-0">
            {recentTicks.map((t, index) => {
              const prevTick = recentTicks[index + 1];
              const priceChange = prevTick ? t.price - prevTick.price : 0;
              const changeClass = priceChange > 0 ? 'text-emerald-400' : priceChange < 0 ? 'text-rose-400' : '';
              
              return (
                <div
                  key={`${t.id}-${index}`}
                  className="flex items-center gap-4 border-b border-neutral-800 px-2 py-1 text-sm hover:bg-neutral-800/30 transition-colors"
                  style={{
                    transformOrigin: 'top left',
                  }}
                >
                  <div className="w-20 font-mono text-neutral-400 text-xs">
                    {new Date(t.timestamp).toLocaleTimeString('en-US', {
                      hour12: false,
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit'
                    })}
                  </div>
                  <div className={`w-24 text-right font-mono ${changeClass}`}>
                    ${t.price.toLocaleString(undefined, { 
                      minimumFractionDigits: 2, 
                      maximumFractionDigits: 4 
                    })}
                  </div>
                  <div className="w-20 text-right text-neutral-300">
                    {t.volume.toLocaleString()}
                  </div>
                  <div className="flex-1">
                    <div className="h-2 rounded bg-neutral-800">
                      <div 
                        className="h-2 rounded bg-emerald-500/70" 
                        style={{ width: `${pct(t.price)}%` }} 
                      />
                    </div>
                  </div>
                  {priceChange !== 0 && (
                    <div className={`w-16 text-xs text-right ${changeClass}`}>
                      {priceChange > 0 ? '+' : ''}{priceChange.toFixed(3)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}