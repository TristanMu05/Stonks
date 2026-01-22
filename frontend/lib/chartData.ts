import type { StockTicker } from "./types";

export type ChartTimeScaleKey =
  | "1D:1m"
  | "1D:5m"
  | "5D:15m"
  | "1M:1h"
  | "1Y:1d"
  | "5Y:1w";

export type ChartTimeScale = {
  key: ChartTimeScaleKey;
  label: string;
  points: number;
  intervalMinutes: number;
};

export const chartTimeScales: ChartTimeScale[] = [
  { key: "1D:1m", label: "1D · 1m", points: 390, intervalMinutes: 1 },
  { key: "1D:5m", label: "1D · 5m", points: 78, intervalMinutes: 5 },
  { key: "5D:15m", label: "5D · 15m", points: 130, intervalMinutes: 15 },
  { key: "1M:1h", label: "1M · 1h", points: 150, intervalMinutes: 60 },
  { key: "1Y:1d", label: "1Y · 1d", points: 252, intervalMinutes: 1440 },
  { key: "5Y:1w", label: "5Y · 1w", points: 260, intervalMinutes: 10080 },
];

export type ChartPoint = {
  timestamp: number;
  label: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ema20: number;
  vwap: number;
  macd: number;
  signal: number;
  hist: number;
  // Computed fields for candlestick rendering
  candleBottom: number;  // min(open, close) - base of the body
  candleTop: number;     // max(open, close) - top of the body
  candleHeight: number;  // abs(close - open) - height of the body
  candleRange: number;   // high - low - full candle range for bar positioning
  wickUpper: number;     // high - candleTop (distance above body)
  wickLower: number;     // candleBottom - low (distance below body)
  isGreen: boolean;      // close >= open
};

export type FinnhubCandleResponse = {
  c: number[];
  h: number[];
  l: number[];
  o: number[];
  s: "ok" | "no_data";
  t: number[];
  v: number[];
};

export type DataCollectorCandle = {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type FinnhubQuery = {
  resolution: "5" | "15" | "60" | "D" | "W";
  from: number;
  to: number;
};

export const getFinnhubQuery = (
  timeScale: ChartTimeScaleKey,
): FinnhubQuery => {
  const now = Math.floor(Date.now() / 1000);
  switch (timeScale) {
    case "1D:1m":
      return { resolution: "5", from: now - 60 * 60 * 24, to: now };
    case "1D:5m":
      return { resolution: "5", from: now - 60 * 60 * 24, to: now };
    case "5D:15m":
      return { resolution: "15", from: now - 60 * 60 * 24 * 5, to: now };
    case "1M:1h":
      return { resolution: "60", from: now - 60 * 60 * 24 * 30, to: now };
    case "1Y:1d":
      return { resolution: "D", from: now - 60 * 60 * 24 * 365, to: now };
    case "5Y:1w":
      return { resolution: "W", from: now - 60 * 60 * 24 * 365 * 5, to: now };
    default:
      return { resolution: "5", from: now - 60 * 60 * 24, to: now };
  }
};

const hashSeed = (value: string) =>
  value.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);

const seededRandom = (seed: number) => {
  let x = Math.sin(seed) * 10000;
  return () => {
    x = Math.sin(x) * 10000;
    return x - Math.floor(x);
  };
};

const calculateEma = (values: number[], period: number) => {
  const multiplier = 2 / (period + 1);
  const ema: number[] = [];
  values.forEach((value, index) => {
    if (index === 0) {
      ema.push(value);
    } else {
      ema.push(value * multiplier + ema[index - 1] * (1 - multiplier));
    }
  });
  return ema;
};

export type CandlePointBase = Omit<
  ChartPoint,
  "ema20" | "vwap" | "macd" | "signal" | "hist" | "candleBottom" | "candleTop" | "candleHeight" | "candleRange" | "wickUpper" | "wickLower" | "isGreen"
>;

export const formatChartLabel = (
  timestamp: number,
  timeScale: ChartTimeScaleKey,
) => {
  const scale = chartTimeScales.find((entry) => entry.key === timeScale);
  const date = new Date(timestamp);
  if (scale && scale.intervalMinutes < 60) {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

export const applyIndicators = (points: CandlePointBase[]): ChartPoint[] => {
  if (points.length === 0) {
    return [];
  }

  const closes = points.map((point) => point.close);
  const highs = points.map((point) => point.high);
  const lows = points.map((point) => point.low);
  const volumes = points.map((point) => point.volume);
  const typicalPrices = closes.map(
    (close, index) => (close + highs[index] + lows[index]) / 3,
  );

  const ema20 = calculateEma(closes, 20);
  const ema12 = calculateEma(closes, 12);
  const ema26 = calculateEma(closes, 26);
  const macd = ema12.map((value, index) => value - ema26[index]);
  const signal = calculateEma(macd, 9);

  let cumulativeVolume = 0;
  let cumulativePV = 0;

  return points.map((point, index) => {
    cumulativeVolume += volumes[index];
    cumulativePV += typicalPrices[index] * volumes[index];
    const vwap =
      cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : point.close;
    
    // Compute candlestick rendering fields
    const candleBottom = Math.min(point.open, point.close);
    const candleTop = Math.max(point.open, point.close);
    const candleHeight = Math.max(candleTop - candleBottom, 0.001); // Ensure minimum height
    const candleRange = Math.max(point.high - point.low, 0.001); // Full candle range
    const wickUpper = point.high - candleTop;
    const wickLower = candleBottom - point.low;
    const isGreen = point.close >= point.open;
    
    return {
      ...point,
      ema20: ema20[index],
      vwap,
      macd: macd[index],
      signal: signal[index],
      hist: macd[index] - signal[index],
      candleBottom,
      candleTop,
      candleHeight,
      candleRange,
      wickUpper,
      wickLower,
      isGreen,
    };
  });
};

export const buildMockSeries = (
  ticker: StockTicker,
  timeScale: ChartTimeScaleKey,
): ChartPoint[] => {
  const scale = chartTimeScales.find((entry) => entry.key === timeScale);
  if (!scale) {
    return [];
  }

  const seed = hashSeed(`${ticker.symbol}-${timeScale}`);
  const random = seededRandom(seed);
  const now = Date.now();
  const intervalMs = scale.intervalMinutes * 60 * 1000;
  const points: CandlePointBase[] = [];

  let price = ticker.last;
  const baseVolatility = ticker.last * 0.0025;
  const volumes: number[] = [];

  for (let index = 0; index < scale.points; index += 1) {
    const open = price;
    const drift = (random() - 0.5) * baseVolatility;
    price = Math.max(1, price + drift);
    const volume =
      (ticker.avgVolume / scale.points) * (0.6 + random() * 1.2);
    const wick = Math.abs(drift) * (0.5 + random());
    const high = Math.max(open, price) + wick;
    const low = Math.max(1, Math.min(open, price) - wick);

    volumes.push(volume);

    const timestamp = now - (scale.points - 1 - index) * intervalMs;

    points.push({
      timestamp,
      label: formatChartLabel(timestamp, timeScale),
      open,
      high,
      low,
      close: price,
      volume,
    });
  }

  return applyIndicators(points);
};

export const buildSeriesFromCandles = (
  candle: FinnhubCandleResponse,
  timeScale: ChartTimeScaleKey,
): ChartPoint[] => {
  if (!candle || candle.s !== "ok" || candle.c.length === 0) {
    return [];
  }

  const scale = chartTimeScales.find((entry) => entry.key === timeScale);
  const candles: CandlePointBase[] = candle.c.map((close, index) => ({
    timestamp: candle.t[index] * 1000,
    label: formatChartLabel(candle.t[index] * 1000, timeScale),
    open: candle.o[index],
    high: candle.h[index],
    low: candle.l[index],
    close,
    volume: candle.v[index],
  }));

  const sliced =
    scale && candles.length > scale.points
      ? candles.slice(-scale.points)
      : candles;

  return applyIndicators(sliced);
};

export const buildSeriesFromDataCollectorCandles = (
  candles: DataCollectorCandle[],
  timeScale: ChartTimeScaleKey,
): ChartPoint[] => {
  if (!candles || candles.length === 0) {
    return [];
  }

  const scale = chartTimeScales.find((entry) => entry.key === timeScale);
  const intervalMs = scale ? scale.intervalMinutes * 60 * 1000 : 60000;
  
  // Aggregate candles by time bucket to ensure one candle per interval
  const bucketMap = new Map<number, CandlePointBase>();
  
  for (const candle of candles) {
    const bucketStart = Math.floor(candle.ts / intervalMs) * intervalMs;
    const existing = bucketMap.get(bucketStart);
    
    if (existing) {
      // Merge into existing bucket - keep earliest open, latest close
      bucketMap.set(bucketStart, {
        ...existing,
        high: Math.max(existing.high, candle.high),
        low: Math.min(existing.low, candle.low),
        close: candle.ts > existing.timestamp ? candle.close : existing.close,
        volume: existing.volume + candle.volume,
      });
    } else {
      bucketMap.set(bucketStart, {
        timestamp: bucketStart,
        label: formatChartLabel(bucketStart, timeScale),
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
      });
    }
  }
  
  // Sort by timestamp
  const points = Array.from(bucketMap.values()).sort(
    (a, b) => a.timestamp - b.timestamp,
  );

  const sliced =
    scale && points.length > scale.points
      ? points.slice(-scale.points)
      : points;

  return applyIndicators(sliced);
};
