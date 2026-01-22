"use client";

import { useEffect, useMemo, useState } from "react";
import type { StockTicker } from "../lib/types";
import { buildDataCollectorUrl } from "../lib/dataCollector";

type MarketTick = {
  id: string;
  symbol: string;
  price: number;
  volume: number;
  timestamp: string;
  exchange: string;
};

type QuoteStatus = "idle" | "connecting" | "connected" | "error";

const buildTickerMap = (tickers: StockTicker[]) =>
  tickers.reduce<Record<string, StockTicker>>((acc, ticker) => {
    acc[ticker.symbol] = ticker;
    return acc;
  }, {});

const createEmptyTicker = (tick: MarketTick): StockTicker => ({
  symbol: tick.symbol,
  name: tick.symbol,
  exchange: tick.exchange || "N/A",
  last: tick.price,
  mark: tick.price,
  change: 0,
  changePercent: 0,
  bid: tick.price,
  ask: tick.price,
  bidSize: 0,
  askSize: 0,
  volume: tick.volume,
  avgVolume: 0,
  open: tick.price,
  high: tick.price,
  low: tick.price,
  prevClose: tick.price,
  vwap: tick.price,
  dayRange: [tick.price, tick.price],
  week52Range: [tick.price, tick.price],
  marketCap: 0,
  float: 0,
  beta: 0,
  pe: 0,
  eps: 0,
  dividend: 0,
  dividendYield: 0,
  earningsDate: "",
  iv: 0,
  ivPercentile: 0,
  shortInterest: 0,
  shortFloat: 0,
  sector: "",
  industry: "",
  timestamp: tick.timestamp,
});

const applyMarketTick = (
  current: StockTicker | undefined,
  tick: MarketTick,
) => {
  const base = current ?? createEmptyTicker(tick);
  const prevClose = base.prevClose || base.last || tick.price;
  const change = tick.price - prevClose;
  const changePercent = prevClose ? (change / prevClose) * 100 : 0;
  const nextVolume = base.volume + tick.volume;
  const nextVwap =
    nextVolume > 0
      ? (base.vwap * base.volume + tick.price * tick.volume) / nextVolume
      : base.vwap;
  const nextHigh = Math.max(base.high || tick.price, tick.price);
  const nextLow =
    base.low === 0 ? tick.price : Math.min(base.low, tick.price);
  return {
    ...base,
    symbol: tick.symbol,
    exchange: tick.exchange || base.exchange,
    last: tick.price,
    mark: tick.price,
    bid: tick.price,
    ask: tick.price,
    change,
    changePercent,
    volume: nextVolume,
    vwap: nextVwap,
    high: nextHigh,
    low: nextLow,
    dayRange: [Math.min(base.dayRange[0], nextLow), Math.max(base.dayRange[1], nextHigh)],
    timestamp: tick.timestamp,
  };
};

export const useDataCollectorQuotes = (initial: StockTicker[]) => {
  const [tickers, setTickers] = useState<StockTicker[]>(initial);
  const [status, setStatus] = useState<QuoteStatus>("idle");
  const symbols = useMemo(() => initial.map((ticker) => ticker.symbol), [initial]);

  useEffect(() => {
    setTickers(initial);
  }, [initial]);

  useEffect(() => {
    const controller = new AbortController();
    const fetchHistory = async () => {
      try {
        const url = buildDataCollectorUrl("/history");
        const response = await fetch(url.toString(), {
          signal: controller.signal,
        });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as MarketTick[];
        if (payload.length === 0) {
          return;
        }
        setTickers((current) => {
          const map = buildTickerMap(current);
          payload.forEach((tick) => {
            const existing = map[tick.symbol];
            map[tick.symbol] = applyMarketTick(existing, tick);
          });
          return Object.values(map);
        });
      } catch {
        // Ignore history fetch errors; live stream may still connect
      }
    };

    fetchHistory();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (symbols.length === 0) {
      return;
    }

    setStatus("connecting");
    const url = buildDataCollectorUrl("/events");
    const source = new EventSource(url.toString());

    source.addEventListener("open", () => {
      setStatus("connected");
    });

    source.addEventListener("message", (event) => {
      try {
        const tick = JSON.parse(event.data) as MarketTick;
        if (!tick?.symbol) {
          return;
        }
        setTickers((current) => {
          const map = buildTickerMap(current);
          const existing = map[tick.symbol];
          map[tick.symbol] = applyMarketTick(existing, tick);
          return Object.values(map);
        });
      } catch {
        setStatus("error");
      }
    });

    source.addEventListener("error", () => {
      setStatus("error");
    });

    return () => {
      source.close();
    };
  }, [symbols]);

  return { tickers, status };
};
