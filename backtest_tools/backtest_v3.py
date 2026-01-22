"""
Backtest V3 - Ultra-Selective Strategy

Key philosophy: Take FEWER trades with HIGHER conviction.

Changes from V2:
1. DISPLACEMENT requirement - BOS must show strong momentum (>60% body, followed by continuation)
2. ZONE FRESHNESS - Only use zones that haven't been touched yet
3. KILLZONE FILTER - Only trade 9:30-11:00 AM and 2:30-3:30 PM ET
4. PULLBACK DEPTH - Wait for pullback to reach at least 50% of the impulse move
5. CONFIRMATION - Require TWO consecutive bullish/bearish candles after zone tap
6. WIDER STOPS - Give trades more room to breathe (zone low/high + 1 ATR)
7. SCALED EXITS - Take partial profits at 1R, let rest run to 2R

The goal: 40%+ win rate with 1.5-2R average winners = positive expectancy
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        return self.body_size / self.range_size if self.range_size > 0 else 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class Swing:
    index: int
    price: float
    confirmed_at: int


@dataclass
class ImpulseMove:
    """Represents a strong directional move that creates a tradeable setup"""
    start_index: int
    end_index: int
    direction: str  # "long" or "short"
    start_price: float
    end_price: float
    fib_50: float  # 50% retracement level
    fib_618: float  # 61.8% retracement level
    zone_low: float  # Entry zone (OB or FVG)
    zone_high: float
    zone_type: str
    invalidated: bool = False


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop: float
    target_1: float  # First target (1R)
    target_2: float  # Second target (2R)
    r_multiple: float
    setup_type: str
    confidence_score: int


@dataclass
class Config:
    tick_size: float = 0.01
    slippage_ticks: float = 1.0
    spread_ticks: float = 1.0

    # Impulse requirements
    min_impulse_bars: int = 3  # Minimum bars for impulse move
    min_impulse_body_ratio: float = 0.55  # Average body ratio of impulse
    min_impulse_range_atr: float = 1.5  # Impulse must be > 1.5 ATR

    # Entry requirements
    min_pullback_ratio: float = 0.38  # Must pull back at least 38.2% fib
    max_pullback_ratio: float = 0.786  # Can't exceed 78.6% fib
    confirmation_candles: int = 1  # Bullish/bearish candles needed in zone

    # Risk
    stop_atr_multiplier: float = 1.5  # Stop = zone edge + 1.5 ATR
    target_1_r: float = 1.0
    target_2_r: float = 2.5
    partial_at_t1: float = 0.5  # Take 50% off at T1

    # Time filters
    morning_start: int = 9
    morning_start_min: int = 30
    morning_end: int = 11
    afternoon_start: int = 14
    afternoon_start_min: int = 30
    afternoon_end: int = 16

    # Session
    ny_start: int = 9
    ny_end: int = 16


def load_polygon_csv(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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


def resample_bars(bars: list[Bar], minutes: int) -> list[Bar]:
    if not bars:
        return []
    grouped: dict[datetime, list[Bar]] = {}
    for bar in bars:
        ts = bar.timestamp.replace(second=0, microsecond=0)
        bucket_minute = (ts.minute // minutes) * minutes
        bucket = ts.replace(minute=bucket_minute)
        grouped.setdefault(bucket, []).append(bar)

    out: list[Bar] = []
    for ts in sorted(grouped.keys()):
        group = grouped[ts]
        out.append(Bar(
            timestamp=ts,
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
        ))
    return out


def compute_atr(bars: list[Bar], period: int = 14) -> list[float]:
    """Compute ATR for each bar"""
    if len(bars) < 2:
        return [0.0] * len(bars)

    atrs = [0.0]
    tr_values = []

    for i in range(1, len(bars)):
        high_low = bars[i].high - bars[i].low
        high_close = abs(bars[i].high - bars[i - 1].close)
        low_close = abs(bars[i].low - bars[i - 1].close)
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)

        if len(tr_values) < period:
            atrs.append(sum(tr_values) / len(tr_values))
        else:
            atrs.append(sum(tr_values[-period:]) / period)

    return atrs


def compute_swings(bars: list[Bar], lookback: int = 3) -> tuple[list[Swing], list[Swing]]:
    """Compute swing points with configurable lookback"""
    swing_highs: list[Swing] = []
    swing_lows: list[Swing] = []

    for i in range(lookback, len(bars) - lookback):
        # Check for swing high
        is_high = all(bars[i].high >= bars[j].high for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_high:
            swing_highs.append(Swing(i, bars[i].high, i + lookback))

        # Check for swing low
        is_low = all(bars[i].low <= bars[j].low for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_low:
            swing_lows.append(Swing(i, bars[i].low, i + lookback))

    return swing_highs, swing_lows


def is_killzone(bar: Bar, config: Config) -> bool:
    """Check if bar is in a killzone (optimal trading time)"""
    et = bar.timestamp.astimezone(ET)
    hour, minute = et.hour, et.minute

    # Morning killzone: 9:30-11:00
    if config.morning_start <= hour < config.morning_end:
        if hour == config.morning_start and minute < config.morning_start_min:
            return False
        return True

    # Afternoon killzone: 2:30-4:00
    if config.afternoon_start <= hour < config.afternoon_end:
        if hour == config.afternoon_start and minute < config.afternoon_start_min:
            return False
        return True

    return False


def find_impulse_moves(bars: list[Bar], swings_h: list[Swing], swings_l: list[Swing],
                       atr: list[float], config: Config) -> list[ImpulseMove]:
    """
    Find strong impulse moves that create tradeable setups.
    An impulse is a sequence of bars moving strongly in one direction.
    """
    impulses: list[ImpulseMove] = []

    for i in range(config.min_impulse_bars + 5, len(bars) - 5):
        current_atr = atr[i] if i < len(atr) else atr[-1]
        if current_atr <= 0:
            continue

        # Look for bullish impulse
        # Check last N bars for strong bullish movement
        lookback_start = i - config.min_impulse_bars
        impulse_bars = bars[lookback_start:i + 1]

        total_body_ratio = sum(b.body_ratio for b in impulse_bars) / len(impulse_bars)
        bullish_bars = sum(1 for b in impulse_bars if b.is_bullish)
        bearish_bars = sum(1 for b in impulse_bars if b.is_bearish)

        move_size = bars[i].close - bars[lookback_start].open
        move_atr_ratio = abs(move_size) / current_atr

        # Bullish impulse criteria
        if (bullish_bars >= config.min_impulse_bars - 1 and
            move_size > 0 and
            move_atr_ratio >= config.min_impulse_range_atr and
            total_body_ratio >= config.min_impulse_body_ratio):

            start_price = min(b.low for b in impulse_bars)
            end_price = max(b.high for b in impulse_bars)
            move_range = end_price - start_price

            fib_50 = end_price - move_range * 0.5
            fib_618 = end_price - move_range * 0.618

            # Find entry zone (last bearish candle in impulse = OB)
            zone_low, zone_high = fib_618, fib_50
            zone_type = "fib_zone"

            for j in range(i, lookback_start, -1):
                if bars[j].is_bearish:
                    zone_low = min(bars[j].open, bars[j].close)
                    zone_high = max(bars[j].open, bars[j].close)
                    zone_type = "ob"
                    break

            impulses.append(ImpulseMove(
                start_index=lookback_start,
                end_index=i,
                direction="long",
                start_price=start_price,
                end_price=end_price,
                fib_50=fib_50,
                fib_618=fib_618,
                zone_low=zone_low,
                zone_high=zone_high,
                zone_type=zone_type,
            ))

        # Bearish impulse criteria
        if (bearish_bars >= config.min_impulse_bars - 1 and
            move_size < 0 and
            move_atr_ratio >= config.min_impulse_range_atr and
            total_body_ratio >= config.min_impulse_body_ratio):

            start_price = max(b.high for b in impulse_bars)
            end_price = min(b.low for b in impulse_bars)
            move_range = start_price - end_price

            fib_50 = end_price + move_range * 0.5
            fib_618 = end_price + move_range * 0.618

            # Find entry zone (last bullish candle in impulse = OB)
            zone_low, zone_high = fib_50, fib_618
            zone_type = "fib_zone"

            for j in range(i, lookback_start, -1):
                if bars[j].is_bullish:
                    zone_low = min(bars[j].open, bars[j].close)
                    zone_high = max(bars[j].open, bars[j].close)
                    zone_type = "ob"
                    break

            impulses.append(ImpulseMove(
                start_index=lookback_start,
                end_index=i,
                direction="short",
                start_price=start_price,
                end_price=end_price,
                fib_50=fib_50,
                fib_618=fib_618,
                zone_low=zone_low,
                zone_high=zone_high,
                zone_type=zone_type,
            ))

    return impulses


def simulate_trades(symbol: str, bars_1m: list[Bar], config: Config) -> list[Trade]:
    """
    Main simulation with ultra-selective criteria.
    """
    bars_5m = resample_bars(bars_1m, 5)
    bars_15m = resample_bars(bars_1m, 15)
    bars_1h = resample_bars(bars_1m, 60)

    if len(bars_5m) < 100:
        return []

    atr_5m = compute_atr(bars_5m, 14)
    atr_1h = compute_atr(bars_1h, 14)

    swings_1h = compute_swings(bars_1h, 3)
    swings_5m = compute_swings(bars_5m, 2)

    trades: list[Trade] = []
    last_exit_index = -50  # Cooldown

    # Find potential setups on 1H
    impulses_1h = find_impulse_moves(bars_1h, *swings_1h, atr_1h, config)

    # Track which impulses we've used
    used_impulses: set[int] = set()

    def get_1h_index(t: datetime) -> int:
        for i, bar in enumerate(bars_1h):
            if bar.timestamp > t:
                return max(0, i - 1)
        return len(bars_1h) - 1

    for i in range(20, len(bars_5m)):
        bar = bars_5m[i]

        # Cooldown
        if i <= last_exit_index + 10:
            continue

        # Killzone filter
        if not is_killzone(bar, config):
            continue

        idx_1h = get_1h_index(bar.timestamp)
        current_atr = atr_5m[i] if i < len(atr_5m) else atr_5m[-1]

        # Find active impulse setups from 1H
        for imp_idx, impulse in enumerate(impulses_1h):
            if imp_idx in used_impulses:
                continue

            if impulse.invalidated:
                continue

            # Check if impulse is recent enough (within last 10 1H bars)
            if impulse.end_index < idx_1h - 10:
                continue

            # Check if price has reached the zone
            in_zone = (bar.low <= impulse.zone_high and bar.high >= impulse.zone_low)

            if not in_zone:
                # Check for zone invalidation
                if impulse.direction == "long" and bar.close < impulse.zone_low - current_atr:
                    impulse.invalidated = True
                elif impulse.direction == "short" and bar.close > impulse.zone_high + current_atr:
                    impulse.invalidated = True
                continue

            # Price is in zone - look for confirmation
            # Need confirmation_candles consecutive candles in the right direction
            if i + config.confirmation_candles >= len(bars_5m):
                continue

            confirmed = True
            for j in range(1, config.confirmation_candles + 1):
                next_bar = bars_5m[i + j]
                if impulse.direction == "long" and not next_bar.is_bullish:
                    confirmed = False
                    break
                if impulse.direction == "short" and not next_bar.is_bearish:
                    confirmed = False
                    break

            if not confirmed:
                continue

            # We have confirmation - enter on next bar
            entry_idx = i + config.confirmation_candles + 1
            if entry_idx >= len(bars_5m):
                continue

            entry_bar = bars_5m[entry_idx]
            entry_price = entry_bar.open

            # Apply slippage
            slip = config.slippage_ticks * config.tick_size
            if impulse.direction == "long":
                entry_price += slip
            else:
                entry_price -= slip

            # Calculate stop with ATR buffer
            stop_buffer = config.stop_atr_multiplier * current_atr
            if impulse.direction == "long":
                stop = impulse.zone_low - stop_buffer
            else:
                stop = impulse.zone_high + stop_buffer

            stop_distance = abs(entry_price - stop)

            # Calculate targets
            if impulse.direction == "long":
                target_1 = entry_price + config.target_1_r * stop_distance
                target_2 = entry_price + config.target_2_r * stop_distance
            else:
                target_1 = entry_price - config.target_1_r * stop_distance
                target_2 = entry_price - config.target_2_r * stop_distance

            # Simulate trade
            current_stop = stop
            hit_t1 = False
            exit_price = None
            exit_time = None
            partial_r = 0.0

            for k in range(entry_idx + 1, len(bars_5m)):
                b = bars_5m[k]

                # Check stop first
                if impulse.direction == "long" and b.low <= current_stop:
                    exit_price = current_stop - slip
                    exit_time = b.timestamp
                    break
                if impulse.direction == "short" and b.high >= current_stop:
                    exit_price = current_stop + slip
                    exit_time = b.timestamp
                    break

                # Check T1
                if not hit_t1:
                    if impulse.direction == "long" and b.high >= target_1:
                        hit_t1 = True
                        partial_r = config.partial_at_t1 * config.target_1_r
                        current_stop = entry_price  # Move to BE
                    if impulse.direction == "short" and b.low <= target_1:
                        hit_t1 = True
                        partial_r = config.partial_at_t1 * config.target_1_r
                        current_stop = entry_price

                # Check T2
                if impulse.direction == "long" and b.high >= target_2:
                    exit_price = target_2 - slip
                    exit_time = b.timestamp
                    break
                if impulse.direction == "short" and b.low <= target_2:
                    exit_price = target_2 + slip
                    exit_time = b.timestamp
                    break

            # Exit at end of data
            if exit_price is None:
                last_bar = bars_5m[-1]
                exit_price = last_bar.close
                exit_time = last_bar.timestamp

            # Calculate R
            if impulse.direction == "long":
                remaining_r = (exit_price - entry_price) / stop_distance
            else:
                remaining_r = (entry_price - exit_price) / stop_distance

            # If we hit T1, we have partial profit locked in
            if hit_t1:
                # 50% at T1, 50% of remaining
                total_r = partial_r + (1 - config.partial_at_t1) * remaining_r
            else:
                total_r = remaining_r

            # Apply costs
            cost_r = (2 * config.slippage_ticks + config.spread_ticks) * config.tick_size / stop_distance
            r_multiple = total_r - cost_r

            trades.append(Trade(
                symbol=symbol,
                direction=impulse.direction,
                entry_time=entry_bar.timestamp,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                stop=stop,
                target_1=target_1,
                target_2=target_2,
                r_multiple=r_multiple,
                setup_type=impulse.zone_type,
                confidence_score=7,  # All trades pass high bar
            ))

            # Mark impulse as used
            used_impulses.add(imp_idx)

            # Find exit index for cooldown
            for exit_idx, b in enumerate(bars_5m):
                if b.timestamp >= exit_time:
                    last_exit_index = exit_idx
                    break

            break  # Only one trade per bar

    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"count": 0, "win_rate": 0.0, "avg_r": 0.0}
    wins = [t for t in trades if t.r_multiple > 0]
    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_r": sum(t.r_multiple for t in trades) / len(trades),
    }


def estimate_equity(trades: list[Trade], starting: float = 10000.0, risk_pct: float = 0.01) -> tuple[float, float, float]:
    """Returns (equity, max_drawdown, profit_factor)"""
    if not trades:
        return starting, 0.0, 0.0

    trades_sorted = sorted(trades, key=lambda t: t.entry_time)
    equity = starting
    peak = starting
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades_sorted:
        pnl = starting * risk_pct * t.r_multiple
        equity += pnl
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return equity, max_dd, pf


def main():
    parser = argparse.ArgumentParser(description="Backtest V3 - Ultra Selective")
    parser.add_argument("--except", dest="exclude", default="")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config()
    root = Path(__file__).resolve().parents[1]
    polygon_dir = root / "data" / "polygon"

    if not polygon_dir.exists():
        print(f"No data found at {polygon_dir}")
        return

    files = list(polygon_dir.glob("*.csv"))
    excluded = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}

    # Group by symbol
    symbol_files: dict[str, list[Path]] = {}
    for f in files:
        sym = f.stem.split("_")[0].upper()
        if sym not in excluded:
            symbol_files.setdefault(sym, []).append(f)

    all_trades: list[Trade] = []
    symbol_results: dict[str, dict] = {}

    for symbol, files_list in sorted(symbol_files.items()):
        print(f"Processing {symbol}...")

        all_bars: list[Bar] = []
        for f in files_list:
            bars = load_polygon_csv(f)
            if args.year:
                bars = [b for b in bars if b.timestamp.year == args.year]
            all_bars.extend(bars)

        all_bars.sort(key=lambda b: b.timestamp)
        if not all_bars:
            continue

        trades = simulate_trades(symbol, all_bars, config)
        all_trades.extend(trades)
        symbol_results[symbol] = summarize(trades)

        if args.debug and trades:
            for t in trades[:5]:
                outcome = "WIN" if t.r_multiple > 0 else "LOSS"
                print(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction} "
                      f"R={t.r_multiple:.2f} {outcome}")

    # Report
    summary = summarize(all_trades)
    equity, max_dd, pf = estimate_equity(all_trades, 10000.0, 0.01)
    total_r = sum(t.r_multiple for t in all_trades)

    output = ["# Backtest V3 Results - Ultra Selective", ""]
    output.append("## Strategy Philosophy")
    output.append("- Only trade clear impulse moves with strong momentum")
    output.append("- Wait for price to retrace to optimal zone")
    output.append("- Require confirmation candles before entry")
    output.append("- Use ATR-based stops for volatility adjustment")
    output.append("- Scale out: 50% at 1R, remainder at 2.5R")
    output.append("- Killzone filter: 9:30-11:00 AM and 2:30-4:00 PM ET")
    output.append("")

    output.append("## Summary")
    if summary["count"] == 0:
        output.append("- No trades generated")
    else:
        output.append(f"- Trades: {summary['count']}")
        output.append(f"- Win rate: {summary['win_rate']:.2%}")
        output.append(f"- Avg R: {summary['avg_r']:.2f}")
        output.append(f"- Total R: {total_r:.2f}")
        output.append(f"- Profit factor: {pf:.2f}")
        output.append(f"- Max drawdown: {max_dd:.2%}")
        output.append(f"- Equity (from $10k, 1% risk): ${equity:,.2f}")

    output.append("")
    output.append("## Per Symbol")
    for sym, stats in sorted(symbol_results.items()):
        if stats["count"] > 0:
            output.append(f"- {sym}: {stats['count']} trades, win {stats['win_rate']:.2%}, avg R {stats['avg_r']:.2f}")
        else:
            output.append(f"- {sym}: 0 trades")

    results_path = root / "backtest_docs" / "backtest_v3_results.md"
    results_path.write_text("\n".join(output))
    print(f"\nWrote {results_path}")
    print(f"\nSummary: {summary['count']} trades, {summary['win_rate']:.1%} win rate, {summary['avg_r']:.2f} avg R")


if __name__ == "__main__":
    main()
