import type { ReactNode } from "react";

export type StockTicker = {
  symbol: string;
  name: string;
  exchange: string;
  last: number;
  mark: number;
  change: number;
  changePercent: number;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  volume: number;
  avgVolume: number;
  open: number;
  high: number;
  low: number;
  prevClose: number;
  vwap: number;
  dayRange: [number, number];
  week52Range: [number, number];
  marketCap: number;
  float: number;
  beta: number;
  pe: number;
  eps: number;
  dividend: number;
  dividendYield: number;
  earningsDate: string;
  iv: number;
  ivPercentile: number;
  shortInterest: number;
  shortFloat: number;
  sector: string;
  industry: string;
  timestamp: string;
};

export type TickerColumnKey =
  | "symbol"
  | "last"
  | "mark"
  | "change"
  | "changePercent"
  | "bid"
  | "ask"
  | "bidSize"
  | "askSize"
  | "spread"
  | "spreadPercent"
  | "volume"
  | "avgVolume"
  | "relativeVolume"
  | "open"
  | "high"
  | "low"
  | "prevClose"
  | "vwap"
  | "dayRange"
  | "week52Range"
  | "marketCap"
  | "float"
  | "beta"
  | "pe"
  | "eps"
  | "dividend"
  | "dividendYield"
  | "earningsDate"
  | "iv"
  | "ivPercentile"
  | "shortInterest"
  | "shortFloat"
  | "sector"
  | "industry"
  | "exchange"
  | "timestamp";

export type TickerColumn = {
  key: TickerColumnKey;
  label: string;
  description: string;
  group: string;
  align?: "left" | "right" | "center";
  widthClass?: string;
  render: (ticker: StockTicker) => ReactNode;
};
