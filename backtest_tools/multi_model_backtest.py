"""
Multi-Model Backtesting System

Uses different strategies based on stock characteristics:
1. MOMENTUM STRATEGY - For trending tech (QQQ, CRWD, MSFT, NVDA, PANW)
2. MEAN REVERSION STRATEGY - For range-bound stocks (WMT, XOM, C, BAC)
3. CROSS-ASSET SIGNALS - Use component stocks to confirm ETF trades

Key Insight: Classification is based on EMPIRICAL backtest results, not just statistics.
"""

import csv
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo
from collections import defaultdict

ET = ZoneInfo("America/New_York")

# Classification based on backtest performance
MOMENTUM_STOCKS = {"QQQ", "CRWD", "MSFT", "NVDA", "PANW"}  # Profitable with momentum
MEAN_REVERSION_STOCKS = {"WMT", "XOM", "BAC", "C", "SOFI"}  # Need different approach
SKIP_STOCKS = {"AAPL", "GOOG", "META", "SPY", "BA", "TSLA", "CAT"}  # Neither strategy works well

# Correlation-based relationships
CROSS_ASSET_MAP = {
    "QQQ": {
        "components": ["NVDA", "MSFT", "META", "GOOG", "AAPL"],
        "weights": [0.25, 0.20, 0.15, 0.15, 0.10],  # Rough importance
        "correlation_threshold": 0.7,
    },
    "SPY": {
        "components": ["MSFT", "NVDA", "AAPL", "META", "GOOG"],
        "weights": [0.15, 0.10, 0.15, 0.10, 0.10],
        "correlation_threshold": 0.6,
    },
}


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_ratio(self) -> float:
        rng = self.high - self.low
        return abs(self.close - self.open) / rng if rng > 0 else 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class Trade:
    symbol: str
    strategy: str  # "momentum" or "mean_reversion"
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop: float
    target: float
    r_multiple: float
    cross_asset_confirmed: bool = False


@dataclass
class MomentumConfig:
    """Config for momentum/breakout strategy"""
    min_impulse_bars: int = 3
    min_impulse_atr: float = 1.5
    stop_atr_multiple: float = 2.0
    target_r: float = 3.0
    confirmation_bars: int = 2
    cooldown_bars: int = 15


@dataclass
class MeanReversionConfig:
    """Config for mean reversion strategy - TIGHTER PARAMETERS"""
    lookback_period: int = 50      # Longer lookback for more stable mean
    entry_z_score: float = 2.5     # More extreme entry (was 2.0)
    exit_z_score: float = 0.0      # Exit at mean (was 0.5)
    stop_z_score: float = 3.5      # Tighter stop relative to target
    max_hold_bars: int = 30        # Shorter max hold
    cooldown_bars: int = 20        # Longer cooldown to reduce trades
    min_std_pct: float = 0.01      # Minimum std as % of price (avoid low vol)


def load_csv(path: Path) -> list[Bar]:
    bars = []
    with path.open() as f:
        for row in csv.DictReader(f):
            ts = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=timezone.utc)
            bars.append(Bar(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0),
            ))
    return sorted(bars, key=lambda b: b.timestamp)


def resample(bars: list[Bar], minutes: int) -> list[Bar]:
    if not bars:
        return []
    grouped: dict[datetime, list[Bar]] = {}
    for bar in bars:
        ts = bar.timestamp.replace(second=0, microsecond=0)
        bucket = ts.replace(minute=(ts.minute // minutes) * minutes)
        grouped.setdefault(bucket, []).append(bar)

    return [Bar(
        timestamp=ts,
        open=group[0].open,
        high=max(b.high for b in group),
        low=min(b.low for b in group),
        close=group[-1].close,
        volume=sum(b.volume for b in group),
    ) for ts, group in sorted(grouped.items())]


def compute_atr(bars: list[Bar], period: int = 14) -> list[float]:
    if len(bars) < 2:
        return [0.0] * len(bars)

    atrs = [bars[0].high - bars[0].low]
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        if i < period:
            atrs.append((atrs[-1] * i + tr) / (i + 1))
        else:
            atrs.append((atrs[-1] * (period - 1) + tr) / period)
    return atrs


def compute_moving_stats(bars: list[Bar], period: int = 20) -> tuple[list[float], list[float]]:
    """Compute moving average and standard deviation"""
    mas = []
    stds = []

    for i in range(len(bars)):
        if i < period:
            closes = [b.close for b in bars[:i+1]]
        else:
            closes = [b.close for b in bars[i-period+1:i+1]]

        mas.append(np.mean(closes))
        stds.append(np.std(closes) if len(closes) > 1 else 0)

    return mas, stds


def is_killzone(bar: Bar) -> bool:
    et = bar.timestamp.astimezone(ET)
    h, m = et.hour, et.minute
    # Morning: 9:30-11:00, Afternoon: 14:30-16:00
    return (9 <= h < 11 and (h > 9 or m >= 30)) or (14 <= h < 16 and (h > 14 or m >= 30))


def get_cross_asset_signal(
    symbol: str,
    current_time: datetime,
    all_bars: dict[str, list[Bar]],
) -> tuple[str, float]:
    """
    Check cross-asset confirmation for a symbol.
    Returns (signal_direction, confidence)
    """
    if symbol not in CROSS_ASSET_MAP:
        return "neutral", 0.0

    config = CROSS_ASSET_MAP[symbol]
    components = config["components"]
    weights = config["weights"]

    bullish_score = 0.0
    bearish_score = 0.0
    total_weight = 0.0

    for comp, weight in zip(components, weights):
        if comp not in all_bars:
            continue

        comp_bars = all_bars[comp]
        # Find bars up to current_time
        relevant_bars = [b for b in comp_bars if b.timestamp <= current_time]
        if len(relevant_bars) < 20:
            continue

        # Simple trend check: 5-bar vs 20-bar MA
        closes = [b.close for b in relevant_bars[-20:]]
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes)

        if ma5 > ma20 * 1.005:  # 0.5% above
            bullish_score += weight
        elif ma5 < ma20 * 0.995:
            bearish_score += weight

        total_weight += weight

    if total_weight == 0:
        return "neutral", 0.0

    bullish_pct = bullish_score / total_weight
    bearish_pct = bearish_score / total_weight

    if bullish_pct > 0.5:
        return "bullish", bullish_pct
    elif bearish_pct > 0.5:
        return "bearish", bearish_pct
    return "neutral", 0.5


# ============================================================
# MOMENTUM STRATEGY (for trending stocks)
# ============================================================

def run_momentum_strategy(
    symbol: str,
    bars: list[Bar],
    all_bars: dict[str, list[Bar]],
    cfg: MomentumConfig,
) -> list[Trade]:
    """
    Momentum/Breakout strategy for trending stocks.
    Same logic as backtest_final.py but with cross-asset confirmation.
    """
    bars_5m = resample(bars, 5)
    if len(bars_5m) < 100:
        return []

    atr = compute_atr(bars_5m)
    trades = []
    last_exit_idx = -100

    tick_size = 0.01
    slippage = tick_size

    i = 20
    while i < len(bars_5m) - cfg.confirmation_bars - 2:
        bar = bars_5m[i]

        if i <= last_exit_idx + cfg.cooldown_bars:
            i += 1
            continue

        if not is_killzone(bar):
            i += 1
            continue

        current_atr = atr[i] if i < len(atr) else atr[-1]
        if current_atr <= 0:
            i += 1
            continue

        # Look for impulse move
        lookback = bars_5m[i - cfg.min_impulse_bars:i + 1]
        bullish_count = sum(1 for b in lookback if b.is_bullish)
        bearish_count = sum(1 for b in lookback if b.is_bearish)

        move_size = bar.close - bars_5m[i - cfg.min_impulse_bars].open
        atr_multiple = abs(move_size) / current_atr

        direction = None
        if bullish_count >= cfg.min_impulse_bars - 1 and move_size > 0 and atr_multiple >= cfg.min_impulse_atr:
            direction = "long"
        elif bearish_count >= cfg.min_impulse_bars - 1 and move_size < 0 and atr_multiple >= cfg.min_impulse_atr:
            direction = "short"

        if direction is None:
            i += 1
            continue

        # Check cross-asset confirmation for ETFs
        cross_signal, cross_confidence = get_cross_asset_signal(symbol, bar.timestamp, all_bars)
        cross_confirmed = False
        if symbol in CROSS_ASSET_MAP:
            if direction == "long" and cross_signal == "bullish":
                cross_confirmed = True
            elif direction == "short" and cross_signal == "bearish":
                cross_confirmed = True
            # Skip if cross-asset disagrees strongly
            if (direction == "long" and cross_signal == "bearish" and cross_confidence > 0.6) or \
               (direction == "short" and cross_signal == "bullish" and cross_confidence > 0.6):
                i += 1
                continue

        # Wait for confirmation bars
        confirmed = True
        for j in range(1, cfg.confirmation_bars + 1):
            cb = bars_5m[i + j]
            if direction == "long" and not cb.is_bullish:
                confirmed = False
            if direction == "short" and not cb.is_bearish:
                confirmed = False
            if cb.body_ratio < 0.3:
                confirmed = False

        if not confirmed:
            i += 1
            continue

        # Entry
        entry_idx = i + cfg.confirmation_bars + 1
        entry_bar = bars_5m[entry_idx]
        entry_price = entry_bar.open + (slippage if direction == "long" else -slippage)

        # Stop and target
        stop_buffer = cfg.stop_atr_multiple * current_atr
        if direction == "long":
            # Find recent low for stop
            recent_low = min(b.low for b in bars_5m[i-5:i+1])
            stop = recent_low - stop_buffer
            stop_dist = entry_price - stop
            target = entry_price + cfg.target_r * stop_dist
        else:
            recent_high = max(b.high for b in bars_5m[i-5:i+1])
            stop = recent_high + stop_buffer
            stop_dist = stop - entry_price
            target = entry_price - cfg.target_r * stop_dist

        # Simulate trade
        exit_price = None
        exit_time = None

        for k in range(entry_idx + 1, len(bars_5m)):
            b = bars_5m[k]

            if direction == "long":
                if b.low <= stop:
                    exit_price = stop - slippage
                    exit_time = b.timestamp
                    break
                if b.high >= target:
                    exit_price = target - slippage
                    exit_time = b.timestamp
                    break
            else:
                if b.high >= stop:
                    exit_price = stop + slippage
                    exit_time = b.timestamp
                    break
                if b.low <= target:
                    exit_price = target + slippage
                    exit_time = b.timestamp
                    break

        if exit_price is None:
            exit_price = bars_5m[-1].close
            exit_time = bars_5m[-1].timestamp

        # Calculate R
        if direction == "long":
            r_mult = (exit_price - entry_price) / stop_dist
        else:
            r_mult = (entry_price - exit_price) / stop_dist

        # Apply costs
        cost_r = (2 * slippage + slippage) / stop_dist
        r_mult -= cost_r

        trades.append(Trade(
            symbol=symbol,
            strategy="momentum",
            direction=direction,
            entry_time=entry_bar.timestamp,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            stop=stop,
            target=target,
            r_multiple=r_mult,
            cross_asset_confirmed=cross_confirmed,
        ))

        # Find exit index
        for exit_idx, b in enumerate(bars_5m):
            if b.timestamp >= exit_time:
                last_exit_idx = exit_idx
                break

        i = last_exit_idx + 1

    return trades


# ============================================================
# MEAN REVERSION STRATEGY (for range-bound stocks)
# ============================================================

def run_mean_reversion_strategy(
    symbol: str,
    bars: list[Bar],
    cfg: MeanReversionConfig,
) -> list[Trade]:
    """
    Mean reversion strategy for range-bound stocks.
    IMPROVED: Tighter entries, better R:R, confirmation required.
    """
    bars_1h = resample(bars, 60)  # Hourly for mean reversion (was 15m)
    if len(bars_1h) < cfg.lookback_period + 20:
        return []

    ma, std = compute_moving_stats(bars_1h, cfg.lookback_period)
    trades = []
    last_exit_idx = -100

    tick_size = 0.01
    slippage = tick_size * 2  # Higher slippage for hourly

    i = cfg.lookback_period + 5
    while i < len(bars_1h) - 3:
        bar = bars_1h[i]

        if i <= last_exit_idx + cfg.cooldown_bars:
            i += 1
            continue

        if not is_killzone(bar):
            i += 1
            continue

        current_ma = ma[i]
        current_std = std[i]

        # Skip low volatility environments
        if current_std <= 0 or current_std / current_ma < cfg.min_std_pct:
            i += 1
            continue

        # Calculate z-score
        z_score = (bar.close - current_ma) / current_std

        direction = None
        # Oversold - look for long (more extreme threshold)
        if z_score <= -cfg.entry_z_score:
            # CONFIRMATION: Need bullish candle after extreme
            if i + 1 < len(bars_1h) and bars_1h[i + 1].is_bullish:
                direction = "long"
        # Overbought - look for short
        elif z_score >= cfg.entry_z_score:
            if i + 1 < len(bars_1h) and bars_1h[i + 1].is_bearish:
                direction = "short"

        if direction is None:
            i += 1
            continue

        # Entry after confirmation bar
        entry_idx = i + 2
        if entry_idx >= len(bars_1h):
            i += 1
            continue

        entry_bar = bars_1h[entry_idx]
        entry_price = entry_bar.open + (slippage if direction == "long" else -slippage)

        # IMPROVED R:R: Target = mean, Stop = 1.5x distance to mean
        if direction == "long":
            target = current_ma  # Target the mean
            target_dist = target - entry_price
            stop_dist = target_dist * 1.5  # Stop 1.5x away for 1.5:1 R:R
            stop = entry_price - stop_dist
        else:
            target = current_ma
            target_dist = entry_price - target
            stop_dist = target_dist * 1.5
            stop = entry_price + stop_dist

        if stop_dist <= 0 or target_dist <= 0:
            i += 1
            continue

        # Simulate trade
        exit_price = None
        exit_time = None

        for k in range(entry_idx + 1, min(len(bars_1h), entry_idx + cfg.max_hold_bars)):
            b = bars_1h[k]

            if direction == "long":
                if b.low <= stop:
                    exit_price = stop - slippage
                    exit_time = b.timestamp
                    break
                if b.high >= target:
                    exit_price = target - slippage
                    exit_time = b.timestamp
                    break
            else:
                if b.high >= stop:
                    exit_price = stop + slippage
                    exit_time = b.timestamp
                    break
                if b.low <= target:
                    exit_price = target + slippage
                    exit_time = b.timestamp
                    break

        # Max hold time exit at current price
        if exit_price is None:
            exit_idx = min(entry_idx + cfg.max_hold_bars, len(bars_1h) - 1)
            exit_price = bars_1h[exit_idx].close
            exit_time = bars_1h[exit_idx].timestamp

        # Calculate R (using actual stop distance)
        if direction == "long":
            r_mult = (exit_price - entry_price) / stop_dist
        else:
            r_mult = (entry_price - exit_price) / stop_dist

        # Apply costs
        cost_r = (2 * slippage + slippage) / stop_dist
        r_mult -= cost_r

        trades.append(Trade(
            symbol=symbol,
            strategy="mean_reversion",
            direction=direction,
            entry_time=entry_bar.timestamp,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            stop=stop,
            target=target,
            r_multiple=r_mult,
        ))

        for exit_idx, b in enumerate(bars_1h):
            if b.timestamp >= exit_time:
                last_exit_idx = exit_idx
                break

        i = last_exit_idx + 1

    return trades


# ============================================================
# MAIN
# ============================================================

def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"count": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0}
    wins = sum(1 for t in trades if t.r_multiple > 0)
    total_r = sum(t.r_multiple for t in trades)
    return {
        "count": len(trades),
        "win_rate": wins / len(trades),
        "avg_r": total_r / len(trades),
        "total_r": total_r,
    }


def equity_curve(trades: list[Trade], start: float = 10000.0, risk: float = 0.01) -> tuple[float, float]:
    if not trades:
        return start, 0.0

    equity = start
    peak = start
    max_dd = 0.0

    for t in sorted(trades, key=lambda x: x.entry_time):
        pnl = start * risk * t.r_multiple
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return equity, max_dd


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "polygon"

    if not data_dir.exists():
        print("No data directory")
        return

    # Load all data
    all_bars: dict[str, list[Bar]] = {}
    for f in data_dir.glob("*.csv"):
        sym = f.stem.split("_")[0].upper()
        bars = load_csv(f)
        if sym in all_bars:
            all_bars[sym].extend(bars)
        else:
            all_bars[sym] = bars

    for sym in all_bars:
        all_bars[sym].sort(key=lambda b: b.timestamp)

    print(f"Loaded {len(all_bars)} symbols")

    momentum_cfg = MomentumConfig()
    mr_cfg = MeanReversionConfig()

    all_trades: list[Trade] = []
    results_by_strategy: dict[str, list[Trade]] = defaultdict(list)
    results_by_symbol: dict[str, dict] = {}

    # Run appropriate strategy for each symbol
    for sym, bars in sorted(all_bars.items()):
        if sym in SKIP_STOCKS:
            print(f"Skipping {sym} (no profitable strategy)")
            continue

        if sym in MOMENTUM_STOCKS:
            print(f"Running MOMENTUM strategy on {sym}...")
            trades = run_momentum_strategy(sym, bars, all_bars, momentum_cfg)
            results_by_strategy["momentum"].extend(trades)
        elif sym in MEAN_REVERSION_STOCKS:
            print(f"Running MEAN REVERSION strategy on {sym}...")
            trades = run_mean_reversion_strategy(sym, bars, mr_cfg)
            results_by_strategy["mean_reversion"].extend(trades)
        else:
            # Try momentum by default
            print(f"Running MOMENTUM strategy on {sym} (default)...")
            trades = run_momentum_strategy(sym, bars, all_bars, momentum_cfg)
            results_by_strategy["momentum"].extend(trades)

        all_trades.extend(trades)
        results_by_symbol[sym] = summarize(trades)

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL BACKTEST RESULTS")
    print("=" * 70)

    overall = summarize(all_trades)
    equity, max_dd = equity_curve(all_trades)

    print(f"\nOVERALL: {overall['count']} trades, {overall['win_rate']:.1%} win, {overall['avg_r']:.2f} avg R")
    print(f"Total R: {overall['total_r']:.2f}")
    print(f"Equity: ${equity:,.2f} (from $10k)")
    print(f"Max DD: {max_dd:.1%}")

    print("\n## By Strategy")
    for strategy, trades in results_by_strategy.items():
        s = summarize(trades)
        print(f"{strategy.upper()}: {s['count']} trades, {s['win_rate']:.1%} win, {s['avg_r']:.2f} avg R, {s['total_r']:.1f} total R")

    print("\n## By Symbol")
    for sym, stats in sorted(results_by_symbol.items(), key=lambda x: -x[1].get('total_r', 0)):
        if stats['count'] > 0:
            print(f"  {sym:6}: {stats['count']:3} trades, {stats['win_rate']:.1%} win, {stats['avg_r']:+.2f} avg R")

    # Cross-asset analysis
    cross_confirmed = [t for t in all_trades if t.cross_asset_confirmed]
    cross_not = [t for t in all_trades if not t.cross_asset_confirmed]

    print("\n## Cross-Asset Confirmation Impact")
    if cross_confirmed:
        cc_stats = summarize(cross_confirmed)
        print(f"With confirmation:    {cc_stats['count']} trades, {cc_stats['win_rate']:.1%} win, {cc_stats['avg_r']:.2f} avg R")
    if cross_not:
        nc_stats = summarize(cross_not)
        print(f"Without confirmation: {nc_stats['count']} trades, {nc_stats['win_rate']:.1%} win, {nc_stats['avg_r']:.2f} avg R")

    # Save results
    output = ["# Multi-Model Backtest Results\n"]
    output.append("## Strategy Assignment\n")
    output.append(f"- MOMENTUM: {', '.join(sorted(MOMENTUM_STOCKS))}")
    output.append(f"- MEAN REVERSION: {', '.join(sorted(MEAN_REVERSION_STOCKS))}")
    output.append(f"- SKIPPED: {', '.join(sorted(SKIP_STOCKS))}\n")

    output.append("## Overall Results\n")
    output.append(f"- Total trades: {overall['count']}")
    output.append(f"- Win rate: {overall['win_rate']:.1%}")
    output.append(f"- Avg R: {overall['avg_r']:.2f}")
    output.append(f"- Total R: {overall['total_r']:.2f}")
    output.append(f"- Equity: ${equity:,.2f}")
    output.append(f"- Max Drawdown: {max_dd:.1%}\n")

    output.append("## By Strategy\n")
    for strategy, trades in results_by_strategy.items():
        s = summarize(trades)
        output.append(f"### {strategy.title()}")
        output.append(f"- Trades: {s['count']}, Win: {s['win_rate']:.1%}, Avg R: {s['avg_r']:.2f}, Total R: {s['total_r']:.1f}\n")

    output.append("## By Symbol\n")
    for sym, stats in sorted(results_by_symbol.items()):
        strategy = "momentum" if sym in MOMENTUM_STOCKS else "mean_reversion"
        output.append(f"- **{sym}** ({strategy}): {stats['count']} trades, {stats['win_rate']:.1%} win, {stats['avg_r']:.2f} avg R")

    results_path = root / "backtest_docs" / "multi_model_results.md"
    results_path.write_text("\n".join(output))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
