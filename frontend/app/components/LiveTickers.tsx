'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from '@dnd-kit/sortable';
import SortableTickerCard from './SortableTickerCard';
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

// Load symbol order from localStorage
function loadSymbolOrder(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const saved = localStorage.getItem('symbolOrder');
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
}

// Save symbol order to localStorage
function saveSymbolOrder(order: string[]) {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem('symbolOrder', JSON.stringify(order));
  } catch {
    // Ignore localStorage errors
  }
}

export default function LiveTickers() {
  const [lastTickBySymbol, setLastTickBySymbol] = useState<Map<string, MarketTick>>(new Map());
  const [historyBySymbol, setHistoryBySymbol] = useState<Map<string, MarketTick[]>>(new Map());
  const [connected, setConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<'cards' | 'list'>('cards');
  const [windowSize, setWindowSize] = useState<number>(60);
  const [symbolOrder, setSymbolOrder] = useState<string[]>([]);
  const sourceRef = useRef<EventSource | null>(null);

  // Drag and drop sensors
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Load saved symbol order on mount
  useEffect(() => {
    setSymbolOrder(loadSymbolOrder());
  }, []);

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

  // Handle drag end
  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const oldIndex = symbols.indexOf(active.id as string);
      const newIndex = symbols.indexOf(over.id as string);
      const newOrder = arrayMove(symbols, oldIndex, newIndex);
      setSymbolOrder(newOrder);
      saveSymbolOrder(newOrder);
    }
  }

  // Compute symbols order: use saved order, fall back to alphabetical
  const symbols = useMemo(() => {
    const availableSymbols = Array.from(lastTickBySymbol.keys());
    
    if (symbolOrder.length === 0) {
      // No saved order, use alphabetical
      return availableSymbols.sort();
    }
    
    // Use saved order for existing symbols, append new ones at the end
    const orderedSymbols = symbolOrder.filter(sym => availableSymbols.includes(sym));
    const newSymbols = availableSymbols.filter(sym => !symbolOrder.includes(sym)).sort();
    const finalOrder = [...orderedSymbols, ...newSymbols];
    
    // Update saved order if we have new symbols
    if (newSymbols.length > 0) {
      setSymbolOrder(finalOrder);
      saveSymbolOrder(finalOrder);
    }
    
    return finalOrder;
  }, [lastTickBySymbol, symbolOrder]);

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

        {view === 'cards' && (
          <button
            onClick={() => {
              setSymbolOrder([]);
              saveSymbolOrder([]);
            }}
            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1 text-sm hover:bg-neutral-800 transition-colors"
            title="Reset card order"
          >
            Reset Order
          </button>
        )}
      </div>

      {view === 'cards' ? (
        <div className="space-y-4">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={symbols} strategy={rectSortingStrategy}>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {symbols.map((sym) => {
                  const tick = lastTickBySymbol.get(sym)!;
                  const history = historyBySymbol.get(sym) ?? [];
                                  return (
                  <SortableTickerCard
                    key={sym}
                    id={sym}
                    tick={tick}
                    history={history}
                    windowSize={windowSize}
                  />
                );
                })}
              </div>
            </SortableContext>
          </DndContext>
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


