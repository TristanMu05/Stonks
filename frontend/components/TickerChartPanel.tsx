"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Customized,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  usePlotArea,
} from "recharts";
import type { StockTicker } from "../lib/types";
import {
  applyIndicators,
  buildMockSeries,
  buildSeriesFromDataCollectorCandles,
  chartTimeScales,
  formatChartLabel,
  type ChartTimeScaleKey,
  type CandlePointBase,
  type ChartPoint,
} from "../lib/chartData";
import { formatCurrency, formatNumber, formatPercent } from "../lib/format";
import { buildDataCollectorUrl } from "../lib/dataCollector";

type IndicatorKey = "ema" | "vwap" | "macd";

type IndicatorState = Record<IndicatorKey, boolean>;

type TickerChartPanelProps = {
  ticker: StockTicker;
  timeScale: ChartTimeScaleKey;
  onTimeScaleChange: (timeScale: ChartTimeScaleKey) => void;
  indicators: IndicatorState;
  onToggleIndicator: (key: IndicatorKey) => void;
  onClose: () => void;
};

type MarketTick = {
  symbol: string;
  price: number;
  volume: number;
  timestamp: string;
};

const CandlestickLayer = ({
  data,
  domain,
  xRange,
}: {
  data?: ChartPoint[];
  domain?: readonly [number, number] | readonly ["auto", "auto"];
  xRange?: readonly [number, number];
}) => {
  const plotArea = usePlotArea();
  if (!data?.length || !plotArea) {
    return null;
  }

  const sorted = [...data].sort((a, b) => a.timestamp - b.timestamp);
  const minTimestamp = xRange ? xRange[0] : sorted[0].timestamp;
  const maxTimestamp = xRange ? xRange[1] : sorted[sorted.length - 1].timestamp;
  const timeRange = Math.max(1, maxTimestamp - minTimestamp);

  const minDelta = (() => {
    if (sorted.length < 2) return timeRange;
    let delta = Number.POSITIVE_INFINITY;
    for (let i = 1; i < sorted.length; i += 1) {
      const diff = sorted[i].timestamp - sorted[i - 1].timestamp;
      if (diff > 0 && diff < delta) {
        delta = diff;
      }
    }
    return Number.isFinite(delta) ? delta : timeRange;
  })();

  const bandWidth = Math.max(
    2,
    (minDelta / timeRange) * plotArea.width || plotArea.width / data.length,
  );
  const candleWidth = Math.max(3, bandWidth * 0.6);

  const domainMinMax = (() => {
    if (domain && domain[0] !== "auto" && domain[1] !== "auto") {
      return { min: domain[0], max: domain[1] };
    }
    let min = data[0].low;
    let max = data[0].high;
    data.forEach((point) => {
      if (point.low < min) min = point.low;
      if (point.high > max) max = point.high;
    });
    return { min, max };
  })();

  const toY = (value: number) => {
    const { min, max } = domainMinMax;
    if (max === min) {
      return plotArea.y + plotArea.height / 2;
    }
    const ratio = (max - value) / (max - min);
    return plotArea.y + ratio * plotArea.height;
  };

  return (
    <g>
      {data.map((point, index) => {
        const x =
          plotArea.x +
          ((point.timestamp - minTimestamp) / timeRange) * plotArea.width;
        if (Number.isNaN(x)) {
          return null;
        }
        const centerX = x;
        const bodyTop = toY(Math.max(point.open, point.close));
        const bodyBottom = toY(Math.min(point.open, point.close));
        const wickTop = toY(point.high);
        const wickBottom = toY(point.low);
        const color = point.isGreen ? "#22c55e" : "#ef4444";
        const bodyHeight = Math.max(bodyBottom - bodyTop, 1);

        return (
          <g key={point.timestamp}>
            <line
              x1={centerX}
              x2={centerX}
              y1={wickTop}
              y2={wickBottom}
              stroke={color}
              strokeWidth={1}
            />
            <rect
              x={centerX - candleWidth / 2}
              y={bodyTop}
              width={candleWidth}
              height={bodyHeight}
              fill={color}
              stroke={color}
              strokeWidth={1}
            />
          </g>
        );
      })}
    </g>
  );
};

const TooltipContent = ({
  active,
  payload,
  label,
  timeScale,
}: {
  active?: boolean;
  payload?: Array<{ value?: number; name?: string }>;
  label?: string | number;
  timeScale: ChartTimeScaleKey;
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const candle = payload?.[0]?.payload as ChartPoint | undefined;
  const displayLabel =
    typeof label === "number" ? formatChartLabel(label, timeScale) : label;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950/90 px-3 py-2 text-xs text-slate-200 shadow-lg">
      <div className="text-slate-400">{displayLabel}</div>
      {candle && (
        <>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Open</span>
            <span className="text-slate-200">{formatNumber(candle.open)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">High</span>
            <span className="text-slate-200">{formatNumber(candle.high)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Low</span>
            <span className="text-slate-200">{formatNumber(candle.low)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Close</span>
            <span className="text-slate-200">{formatNumber(candle.close)}</span>
          </div>
        </>
      )}
      {payload.map((entry) => (
        <div key={entry.name} className="flex justify-between gap-4">
          <span className="text-slate-400">{entry.name}</span>
          <span className="text-slate-200">
            {typeof entry.value === "number"
              ? formatNumber(entry.value)
              : "--"}
          </span>
        </div>
      ))}
    </div>
  );
};

const TickerChartPanel = ({
  ticker,
  timeScale,
  onTimeScaleChange,
  indicators,
  onToggleIndicator,
  onClose,
}: TickerChartPanelProps) => {
  const mockSeries = useMemo(
    () => buildMockSeries(ticker, timeScale),
    [ticker, timeScale],
  );
  const [series, setSeries] = useState(mockSeries);
  const [source, setSource] = useState<"mock" | "data_collector">("mock");
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  
  // Zoom state
  const [zoomLeft, setZoomLeft] = useState<number | null>(null);
  const [zoomRight, setZoomRight] = useState<number | null>(null);
  const [refAreaLeft, setRefAreaLeft] = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  
  const scale = useMemo(
    () => chartTimeScales.find((entry) => entry.key === timeScale),
    [timeScale],
  );
  
  // Calculate visible series based on zoom
  const visibleSeries = useMemo(() => {
    if (zoomLeft === null || zoomRight === null) return series;
    const leftIdx = series.findIndex((s) => s.timestamp === zoomLeft);
    const rightIdx = series.findIndex((s) => s.timestamp === zoomRight);
    if (leftIdx === -1 || rightIdx === -1) return series;
    return series.slice(leftIdx, rightIdx + 1);
  }, [series, zoomLeft, zoomRight]);

  const seriesTimestamps = useMemo(() => {
    return [...series]
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((point) => point.timestamp);
  }, [series]);

  const xRange = useMemo(() => {
    if (zoomLeft !== null && zoomRight !== null) {
      return zoomLeft < zoomRight ? [zoomLeft, zoomRight] : [zoomRight, zoomLeft];
    }
    if (!scale || series.length === 0) {
      return null;
    }
    const intervalMs = scale.intervalMinutes * 60 * 1000;
    const end = Math.max(...series.map((point) => point.timestamp));
    const start = end - intervalMs * (scale.points - 1);
    return [start, end] as const;
  }, [scale, series, zoomLeft, zoomRight]);

  useEffect(() => {
    if (!scale || seriesTimestamps.length === 0) {
      return;
    }

    const minTimestamp = seriesTimestamps[0];
    const maxTimestamp = seriesTimestamps[seriesTimestamps.length - 1];
    const intervalMs = scale.intervalMinutes * 60 * 1000;

    const clampRange = (start: number, end: number) => {
      let nextStart = start;
      let nextEnd = end;
      const total = maxTimestamp - minTimestamp;
      const window = Math.max(intervalMs, nextEnd - nextStart);
      if (window >= total) {
        return [minTimestamp, maxTimestamp] as const;
      }
      if (nextStart < minTimestamp) {
        nextStart = minTimestamp;
        nextEnd = minTimestamp + window;
      }
      if (nextEnd > maxTimestamp) {
        nextEnd = maxTimestamp;
        nextStart = maxTimestamp - window;
      }
      return [nextStart, nextEnd] as const;
    };

    const snapToNearest = (value: number) => {
      let best = seriesTimestamps[0];
      let bestDiff = Math.abs(value - best);
      for (const ts of seriesTimestamps) {
        const diff = Math.abs(value - ts);
        if (diff < bestDiff) {
          best = ts;
          bestDiff = diff;
        }
      }
      return best;
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) {
        return;
      }

      const baseRange = xRange;
      if (!baseRange) {
        return;
      }

      const [rangeStart, rangeEnd] = baseRange;
      const rangeWindow = Math.max(intervalMs, rangeEnd - rangeStart);
      const center = rangeStart + rangeWindow / 2;

      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        event.preventDefault();
        const zoomFactor = event.key === "ArrowUp" ? 0.8 : 1.25;
        const nextWindow = Math.max(intervalMs * 5, rangeWindow * zoomFactor);
        const rawEnd =
          event.key === "ArrowUp" ? rangeEnd : center + nextWindow / 2;
        const rawStart =
          event.key === "ArrowUp" ? rawEnd - nextWindow : center - nextWindow / 2;
        const [clampedStart, clampedEnd] = clampRange(rawStart, rawEnd);
        setZoomLeft(snapToNearest(clampedStart));
        setZoomRight(snapToNearest(clampedEnd));
      }

      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const shift = event.key === "ArrowLeft" ? -intervalMs : intervalMs;
        const [clampedStart, clampedEnd] = clampRange(
          rangeStart + shift,
          rangeEnd + shift,
        );
        setZoomLeft(snapToNearest(clampedStart));
        setZoomRight(snapToNearest(clampedEnd));
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [scale, seriesTimestamps, xRange]);
  
  const priceDomain = useMemo(() => {
    const data = visibleSeries.length ? visibleSeries : series;
    if (!data.length) {
      return ["auto", "auto"] as const;
    }
    let min = data[0].low;
    let max = data[0].high;
    data.forEach((point) => {
      if (point.low < min) min = point.low;
      if (point.high > max) max = point.high;
    });
    if (min === max) {
      return [min * 0.995, max * 1.005] as const;
    }
    const padding = (max - min) * 0.05;
    return [min - padding, max + padding] as const;
  }, [visibleSeries, series]);
  
  // Reset zoom when timeScale or ticker changes
  useEffect(() => {
    setZoomLeft(null);
    setZoomRight(null);
  }, [timeScale, ticker.symbol]);

  useEffect(() => {
    setSeries(mockSeries);
    setSource("mock");
    setStatus("idle");
  }, [mockSeries]);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");

    const fetchCandles = async () => {
      try {
        // Map frontend time scale to backend timeframe format
        const timeframeMap: Record<ChartTimeScaleKey, string> = {
          "1D:1m": "1d:1m",
          "1D:5m": "1d:5m",
          "5D:15m": "5d:15m",
          "1M:1h": "1M:1h",
          "1Y:1d": "1Y:1d",
          "5Y:1w": "5Y:1w",
        };
        const timeframe = timeframeMap[timeScale] ?? "1d:1m";
        const url = buildDataCollectorUrl("/candles", {
          symbol: ticker.symbol,
          timeframe,
        });
        const response = await fetch(url.toString(), {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error("Failed to load candles");
        }
        const payload = (await response.json()) as Array<{
          ts: number;
          open: number;
          high: number;
          low: number;
          close: number;
          volume: number;
        }>;
        const nextSeries = buildSeriesFromDataCollectorCandles(
          payload,
          timeScale,
        );
        if (nextSeries.length > 0) {
          setSeries(nextSeries);
          setSource("data_collector");
          setStatus("idle");
          return;
        }
        setStatus("error");
      } catch {
        if (!controller.signal.aborted) {
          setStatus("error");
        }
      }
    };

    fetchCandles();

    return () => controller.abort();
  }, [ticker.symbol, timeScale]);

  useEffect(() => {
    if (!scale) {
      return;
    }
    const url = buildDataCollectorUrl("/events");
    const source = new EventSource(url.toString());

    source.addEventListener("message", (event) => {
      try {
        const tick = JSON.parse(event.data) as MarketTick;
        if (tick.symbol !== ticker.symbol) {
          return;
        }
        const tickTime = new Date(tick.timestamp).getTime();
        if (Number.isNaN(tickTime)) {
          return;
        }
        const intervalMs = scale.intervalMinutes * 60 * 1000;
        const bucketStart = Math.floor(tickTime / intervalMs) * intervalMs;
        setSeries((current) => {
          // Build a map keyed by bucket timestamp to aggregate properly
          const bucketMap = new Map<number, CandlePointBase>();
          
          // Add existing points to the map
          for (const point of current) {
            const pointBucket = Math.floor(point.timestamp / intervalMs) * intervalMs;
            const existing = bucketMap.get(pointBucket);
            if (existing) {
              // Merge into existing bucket
              bucketMap.set(pointBucket, {
                ...existing,
                high: Math.max(existing.high, point.high),
                low: Math.min(existing.low, point.low),
                close: point.timestamp > existing.timestamp ? point.close : existing.close,
                volume: existing.volume + point.volume,
              });
            } else {
              bucketMap.set(pointBucket, {
                timestamp: pointBucket,
                label: formatChartLabel(pointBucket, timeScale),
                open: point.open,
                high: point.high,
                low: point.low,
                close: point.close,
                volume: point.volume,
              });
            }
          }
          
          // Update or create bucket for the new tick
          const existing = bucketMap.get(bucketStart);
          if (existing) {
            bucketMap.set(bucketStart, {
              ...existing,
              high: Math.max(existing.high, tick.price),
              low: Math.min(existing.low, tick.price),
              close: tick.price,
              volume: existing.volume + tick.volume,
            });
          } else {
            bucketMap.set(bucketStart, {
              timestamp: bucketStart,
              label: formatChartLabel(bucketStart, timeScale),
              open: tick.price,
              high: tick.price,
              low: tick.price,
              close: tick.price,
              volume: tick.volume,
            });
          }
          
          // Sort by timestamp and trim to max points
          const sorted = Array.from(bucketMap.values()).sort(
            (a, b) => a.timestamp - b.timestamp,
          );
          const trimmed =
            sorted.length > scale.points
              ? sorted.slice(-scale.points)
              : sorted;
          return applyIndicators(trimmed);
        });
        setSource("data_collector");
      } catch {
        // Ignore parse errors; keep existing series
      }
    });

    return () => {
      source.close();
    };
  }, [scale, ticker.symbol, timeScale]);

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold">
              {ticker.symbol} · {ticker.name}
            </h2>
            <span className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-400">
              {ticker.exchange}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-slate-300">
            <span className="text-lg font-semibold text-white">
              {formatCurrency(ticker.last)}
            </span>
            <span
              className={
                ticker.change >= 0 ? "text-emerald-400" : "text-rose-400"
              }
            >
              {formatCurrency(ticker.change)} ({formatPercent(ticker.changePercent)})
            </span>
            <span className="text-xs text-slate-400">
              VWAP {formatCurrency(ticker.vwap)}
            </span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <select
            className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200"
            value={timeScale}
            onChange={(event) =>
              onTimeScaleChange(event.target.value as ChartTimeScaleKey)
            }
          >
            {chartTimeScales.map((scale) => (
              <option key={scale.key} value={scale.key}>
                {scale.label}
              </option>
            ))}
          </select>
          <span className="text-xs text-slate-400">
            {source === "data_collector" ? "Data collector candles" : "Mock series"}
            {status === "loading" ? " · loading" : ""}
            {status === "error" ? " · fallback" : ""}
          </span>
          <div className="flex flex-wrap gap-2 text-xs">
            {(["ema", "vwap", "macd"] as IndicatorKey[]).map((key) => (
              <button
                key={key}
                className={`rounded-full border px-3 py-1 uppercase tracking-wide ${
                  indicators[key]
                    ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-200"
                    : "border-slate-700 text-slate-400 hover:text-slate-200"
                }`}
                type="button"
                onClick={() => onToggleIndicator(key)}
              >
                {key}
              </button>
            ))}
          </div>
          <button
            className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 transition hover:border-slate-500 hover:text-white"
            type="button"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>

      <div className="mt-6 grid gap-6">
        <div className="h-72 w-full">
          <div className="mb-2 flex items-center justify-end gap-2">
            {(zoomLeft !== null || zoomRight !== null) && (
              <button
                type="button"
                className="rounded border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:border-slate-400 hover:text-white"
                onClick={() => {
                  setZoomLeft(null);
                  setZoomRight(null);
                }}
              >
                Reset Zoom
              </button>
            )}
            <span className="text-xs text-slate-500">
              Click and drag to zoom
            </span>
          </div>
          <ResponsiveContainer>
            <ComposedChart
              data={visibleSeries}
              barGap={0}
              barCategoryGap="5%"
              onMouseDown={(e) => {
                const label = typeof e?.activeLabel === "number"
                  ? e.activeLabel
                  : Number(e?.activeLabel);
                if (Number.isFinite(label)) {
                  setRefAreaLeft(label);
                  setIsSelecting(true);
                }
              }}
              onMouseMove={(e) => {
                const label = typeof e?.activeLabel === "number"
                  ? e.activeLabel
                  : Number(e?.activeLabel);
                if (isSelecting && Number.isFinite(label)) {
                  setRefAreaRight(label);
                }
              }}
              onMouseUp={() => {
                if (
                  refAreaLeft !== null &&
                  refAreaRight !== null &&
                  refAreaLeft !== refAreaRight
                ) {
                  // Find indices to ensure left < right
                  const leftIdx = visibleSeries.findIndex(
                    (s) => s.timestamp === refAreaLeft,
                  );
                  const rightIdx = visibleSeries.findIndex(
                    (s) => s.timestamp === refAreaRight,
                  );
                  if (leftIdx !== -1 && rightIdx !== -1) {
                    const [left, right] =
                      leftIdx < rightIdx
                        ? [refAreaLeft, refAreaRight]
                        : [refAreaRight, refAreaLeft];
                    setZoomLeft(left);
                    setZoomRight(right);
                  }
                }
                setRefAreaLeft(null);
                setRefAreaRight(null);
                setIsSelecting(false);
              }}
              onMouseLeave={() => {
                setRefAreaLeft(null);
                setRefAreaRight(null);
                setIsSelecting(false);
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                type="number"
                tick={{ fontSize: 10 }}
                stroke="#64748b"
                domain={xRange ?? ["auto", "auto"]}
                tickFormatter={(value) =>
                  typeof value === "number"
                    ? formatChartLabel(value, timeScale)
                    : value
                }
              />
              <YAxis
                yAxisId="price"
                tick={{ fontSize: 10 }}
                stroke="#64748b"
                domain={priceDomain}
                allowDataOverflow={false}
                tickFormatter={(value) => typeof value === "number" ? value.toFixed(2) : value}
              />
              <YAxis
                yAxisId="volume"
                orientation="right"
                hide
                domain={[0, "dataMax * 5"]}
              />
              <Tooltip content={<TooltipContent timeScale={timeScale} />} />

              <Bar
                yAxisId="volume"
                dataKey="volume"
                fill="#1f2937"
                opacity={0.4}
                name="Volume"
                isAnimationActive={false}
              />
              
              <Customized
                component={
                  <CandlestickLayer
                    data={visibleSeries}
                    domain={priceDomain}
                    xRange={xRange ?? undefined}
                  />
                }
              />
              
              {indicators.ema && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="ema20"
                  stroke="#a78bfa"
                  dot={false}
                  strokeWidth={1.5}
                  name="EMA 20"
                  isAnimationActive={false}
                />
              )}
              {indicators.vwap && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="vwap"
                  stroke="#fbbf24"
                  dot={false}
                  strokeWidth={1.5}
                  name="VWAP"
                  isAnimationActive={false}
                />
              )}
              {/* Zoom selection area */}
              {refAreaLeft !== null && refAreaRight !== null && (
                <ReferenceArea
                  yAxisId="price"
                  x1={refAreaLeft}
                  x2={refAreaRight}
                  strokeOpacity={0.3}
                  fill="#38bdf8"
                  fillOpacity={0.2}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {indicators.macd && (
          <div className="h-40 w-full">
            <ResponsiveContainer>
              <ComposedChart data={visibleSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="timestamp"
                  type="number"
                  tick={{ fontSize: 10 }}
                  stroke="#64748b"
                  domain={xRange ?? ["auto", "auto"]}
                  tickFormatter={(value) =>
                    typeof value === "number"
                      ? formatChartLabel(value, timeScale)
                      : value
                  }
                />
                <YAxis tick={{ fontSize: 10 }} stroke="#64748b" />
                <Tooltip content={<TooltipContent timeScale={timeScale} />} />
                <Bar dataKey="hist" fill="#334155" name="Histogram" />
                <Line
                  type="monotone"
                  dataKey="macd"
                  stroke="#22c55e"
                  dot={false}
                  strokeWidth={1.5}
                  name="MACD"
                />
                <Line
                  type="monotone"
                  dataKey="signal"
                  stroke="#f97316"
                  dot={false}
                  strokeWidth={1.5}
                  name="Signal"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </section>
  );
};

export default TickerChartPanel;
