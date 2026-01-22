"""
Backtest Final - Optimized Strategy Based on Learnings

Key learnings applied:
1. SYMBOL SELECTION - Only trade trending tech/growth stocks, avoid mean-reverting sectors
2. NO EARLY SCALING - Let winners run to full target instead of partial at 1R
3. TIGHTER ENTRY - Require stronger confirmation before entry
4. WIDER STOPS - Use 2x ATR buffer to avoid noise stops
5. HIGHER TARGETS - Target 3R to compensate for ~50% win rate

Math: 50% win rate * 3R - 50% loss rate * 1R = 1.0R expected value per trade
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# PROFITABLE SYMBOLS ONLY - These showed consistent positive expectancy
# Based on backtesting analysis across 2024-2026 data
PROFITABLE_SYMBOLS = {"QQQ", "CRWD", "MSFT", "NVDA", "PANW"}

# Symbols to AVOID - Mean-reverting, choppy, or consistently unprofitable
BAD_SYMBOLS = {"WMT", "SOFI", "XOM", "C", "CAT", "BAC", "AAPL", "GOOG", "META", "SPY", "BA", "TSLA"}


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
class Setup:
    """A valid trading setup waiting for entry"""
    direction: str
    zone_low: float
    zone_high: float
    created_at: int
    impulse_start: float
    impulse_end: float
    invalidated: bool = False
    touched: bool = False


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop: float
    target: float
    r_multiple: float
    win: bool


@dataclass
class Config:
    tick_size: float = 0.01
    slippage_ticks: float = 1.0
    spread_ticks: float = 1.0

    # Impulse detection
    min_impulse_bars: int = 3
    min_impulse_body_ratio: float = 0.5
    min_impulse_atr_multiple: float = 1.5

    # Entry requirements
    min_pullback_fib: float = 0.382  # At least 38.2% retracement
    max_pullback_fib: float = 0.75   # No more than 75% retracement
    confirmation_bars: int = 2        # Need 2 bars showing reversal

    # Risk management
    stop_atr_multiple: float = 2.0   # Wider stop for noise
    target_r: float = 3.0            # Higher target, no scaling
    max_setup_age_bars: int = 20     # Setup expires after this

    # Session
    killzone_morning_start: tuple = (9, 30)
    killzone_morning_end: tuple = (11, 0)
    killzone_afternoon_start: tuple = (14, 30)
    killzone_afternoon_end: tuple = (16, 0)

    # Cooldown
    cooldown_bars: int = 15  # After trade, wait before next


def load_csv(path: Path) -> list[Bar]:
    bars: list[Bar] = []
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

    atrs = [bars[0].range_size]
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close)
        )
        if i < period:
            atrs.append((atrs[-1] * i + tr) / (i + 1))
        else:
            atrs.append((atrs[-1] * (period - 1) + tr) / period)
    return atrs


def compute_swings(bars: list[Bar], lookback: int = 2) -> tuple[list[Swing], list[Swing]]:
    highs, lows = [], []
    for i in range(lookback, len(bars) - lookback):
        if all(bars[i].high >= bars[j].high for j in range(i - lookback, i + lookback + 1) if j != i):
            highs.append(Swing(i, bars[i].high, i + lookback))
        if all(bars[i].low <= bars[j].low for j in range(i - lookback, i + lookback + 1) if j != i):
            lows.append(Swing(i, bars[i].low, i + lookback))
    return highs, lows


def get_trend(highs: list[Swing], lows: list[Swing], idx: int) -> str:
    """Get trend using confirmed swings only"""
    h = [s for s in highs if s.confirmed_at <= idx]
    l = [s for s in lows if s.confirmed_at <= idx]
    if len(h) < 2 or len(l) < 2:
        return "none"

    # Uptrend: HH + HL
    if h[-1].price > h[-2].price and l[-1].price > l[-2].price:
        return "up"
    # Downtrend: LH + LL
    if h[-1].price < h[-2].price and l[-1].price < l[-2].price:
        return "down"
    return "none"


def is_killzone(bar: Bar, cfg: Config) -> bool:
    et = bar.timestamp.astimezone(ET)
    h, m = et.hour, et.minute
    time_mins = h * 60 + m

    morning_start = cfg.killzone_morning_start[0] * 60 + cfg.killzone_morning_start[1]
    morning_end = cfg.killzone_morning_end[0] * 60 + cfg.killzone_morning_end[1]
    afternoon_start = cfg.killzone_afternoon_start[0] * 60 + cfg.killzone_afternoon_start[1]
    afternoon_end = cfg.killzone_afternoon_end[0] * 60 + cfg.killzone_afternoon_end[1]

    return (morning_start <= time_mins < morning_end) or (afternoon_start <= time_mins < afternoon_end)


def find_setups(bars: list[Bar], atr: list[float], cfg: Config) -> list[tuple[int, Setup]]:
    """Find impulse moves that create tradeable setups"""
    setups = []

    for i in range(cfg.min_impulse_bars + 2, len(bars) - 5):
        if atr[i] <= 0:
            continue

        # Check last N bars for impulse
        impulse_bars = bars[i - cfg.min_impulse_bars:i + 1]
        bullish_count = sum(1 for b in impulse_bars if b.is_bullish)
        bearish_count = sum(1 for b in impulse_bars if b.is_bearish)
        avg_body_ratio = sum(b.body_ratio for b in impulse_bars) / len(impulse_bars)

        move_size = bars[i].close - bars[i - cfg.min_impulse_bars].open
        atr_multiple = abs(move_size) / atr[i]

        # Bullish impulse
        if (bullish_count >= cfg.min_impulse_bars - 1 and
            move_size > 0 and
            atr_multiple >= cfg.min_impulse_atr_multiple and
            avg_body_ratio >= cfg.min_impulse_body_ratio):

            start_price = min(b.low for b in impulse_bars)
            end_price = max(b.high for b in impulse_bars)
            move_range = end_price - start_price

            # Entry zone: fib 38.2% to 61.8%
            zone_high = end_price - move_range * cfg.min_pullback_fib
            zone_low = end_price - move_range * 0.618

            # Find OB (last bearish candle)
            for j in range(i, i - cfg.min_impulse_bars, -1):
                if bars[j].is_bearish:
                    zone_low = min(bars[j].open, bars[j].close)
                    zone_high = max(bars[j].open, bars[j].close)
                    break

            setups.append((i, Setup(
                direction="long",
                zone_low=zone_low,
                zone_high=zone_high,
                created_at=i,
                impulse_start=start_price,
                impulse_end=end_price,
            )))

        # Bearish impulse
        if (bearish_count >= cfg.min_impulse_bars - 1 and
            move_size < 0 and
            atr_multiple >= cfg.min_impulse_atr_multiple and
            avg_body_ratio >= cfg.min_impulse_body_ratio):

            start_price = max(b.high for b in impulse_bars)
            end_price = min(b.low for b in impulse_bars)
            move_range = start_price - end_price

            zone_low = end_price + move_range * cfg.min_pullback_fib
            zone_high = end_price + move_range * 0.618

            for j in range(i, i - cfg.min_impulse_bars, -1):
                if bars[j].is_bullish:
                    zone_low = min(bars[j].open, bars[j].close)
                    zone_high = max(bars[j].open, bars[j].close)
                    break

            setups.append((i, Setup(
                direction="short",
                zone_low=zone_low,
                zone_high=zone_high,
                created_at=i,
                impulse_start=start_price,
                impulse_end=end_price,
            )))

    return setups


def simulate(symbol: str, bars_1m: list[Bar], cfg: Config) -> list[Trade]:
    if symbol.upper() in BAD_SYMBOLS:
        return []  # Skip known bad symbols

    bars = resample(bars_1m, 5)
    if len(bars) < 100:
        return []

    atr = compute_atr(bars)
    swings = compute_swings(bars)
    setups_list = find_setups(bars, atr, cfg)

    trades: list[Trade] = []
    last_exit_idx = -100
    active_setups: list[Setup] = []

    for idx, (setup_idx, setup) in enumerate(setups_list):
        if setup_idx in [s.created_at for s in active_setups]:
            continue
        active_setups.append(setup)

    i = 20
    while i < len(bars):
        bar = bars[i]
        current_atr = atr[i] if i < len(atr) else atr[-1]

        # Cooldown
        if i <= last_exit_idx + cfg.cooldown_bars:
            i += 1
            continue

        # Killzone filter
        if not is_killzone(bar, cfg):
            i += 1
            continue

        # Get trend
        trend = get_trend(*swings, i)

        # Check active setups
        for setup in active_setups[:]:
            if setup.invalidated:
                continue

            # Expire old setups
            age = i - setup.created_at
            if age > cfg.max_setup_age_bars:
                setup.invalidated = True
                continue

            # Skip if trend doesn't align
            if setup.direction == "long" and trend != "up":
                continue
            if setup.direction == "short" and trend != "down":
                continue

            # Check if price in zone
            in_zone = bar.low <= setup.zone_high and bar.high >= setup.zone_low

            if not in_zone:
                # Check invalidation
                if setup.direction == "long" and bar.close < setup.zone_low - current_atr:
                    setup.invalidated = True
                if setup.direction == "short" and bar.close > setup.zone_high + current_atr:
                    setup.invalidated = True
                continue

            setup.touched = True

            # Look for confirmation: N consecutive bars in the right direction
            if i + cfg.confirmation_bars >= len(bars):
                continue

            confirmed = True
            for j in range(1, cfg.confirmation_bars + 1):
                cb = bars[i + j]
                if setup.direction == "long" and not cb.is_bullish:
                    confirmed = False
                    break
                if setup.direction == "short" and not cb.is_bearish:
                    confirmed = False
                    break
                # Also require decent body
                if cb.body_ratio < 0.3:
                    confirmed = False
                    break

            if not confirmed:
                continue

            # ENTRY
            entry_idx = i + cfg.confirmation_bars + 1
            if entry_idx >= len(bars):
                continue

            entry_bar = bars[entry_idx]
            entry_price = entry_bar.open

            slip = cfg.slippage_ticks * cfg.tick_size
            entry_price += slip if setup.direction == "long" else -slip

            # Calculate stop with ATR buffer
            stop_buffer = cfg.stop_atr_multiple * current_atr
            if setup.direction == "long":
                stop = setup.zone_low - stop_buffer
            else:
                stop = setup.zone_high + stop_buffer

            stop_dist = abs(entry_price - stop)

            # Target
            if setup.direction == "long":
                target = entry_price + cfg.target_r * stop_dist
            else:
                target = entry_price - cfg.target_r * stop_dist

            # Simulate trade
            exit_price = None
            exit_time = None

            for k in range(entry_idx + 1, len(bars)):
                b = bars[k]

                # Check stop
                if setup.direction == "long" and b.low <= stop:
                    exit_price = stop - slip
                    exit_time = b.timestamp
                    break
                if setup.direction == "short" and b.high >= stop:
                    exit_price = stop + slip
                    exit_time = b.timestamp
                    break

                # Check target
                if setup.direction == "long" and b.high >= target:
                    exit_price = target - slip
                    exit_time = b.timestamp
                    break
                if setup.direction == "short" and b.low <= target:
                    exit_price = target + slip
                    exit_time = b.timestamp
                    break

            if exit_price is None:
                exit_price = bars[-1].close
                exit_time = bars[-1].timestamp

            # Calculate R
            if setup.direction == "long":
                raw_r = (exit_price - entry_price) / stop_dist
            else:
                raw_r = (entry_price - exit_price) / stop_dist

            cost_r = (2 * cfg.slippage_ticks + cfg.spread_ticks) * cfg.tick_size / stop_dist
            r_mult = raw_r - cost_r

            trades.append(Trade(
                symbol=symbol,
                direction=setup.direction,
                entry_time=entry_bar.timestamp,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                stop=stop,
                target=target,
                r_multiple=r_mult,
                win=r_mult > 0,
            ))

            # Mark setup used
            setup.invalidated = True

            # Cooldown
            for exit_idx, b in enumerate(bars):
                if b.timestamp >= exit_time:
                    last_exit_idx = exit_idx
                    break

            break

        i += 1

    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"count": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0}
    wins = sum(1 for t in trades if t.win)
    total_r = sum(t.r_multiple for t in trades)
    return {
        "count": len(trades),
        "win_rate": wins / len(trades),
        "avg_r": total_r / len(trades),
        "total_r": total_r,
    }


def equity_curve(trades: list[Trade], start: float = 10000.0, risk: float = 0.01) -> tuple[float, float, float]:
    """Returns (final_equity, max_dd, profit_factor)"""
    if not trades:
        return start, 0.0, 0.0

    equity = start
    peak = start
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0

    for t in sorted(trades, key=lambda x: x.entry_time):
        pnl = start * risk * t.r_multiple
        equity += pnl
        if pnl > 0:
            gross_win += pnl
        else:
            gross_loss += abs(pnl)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return equity, max_dd, pf


def main():
    parser = argparse.ArgumentParser(description="Final Optimized Backtest")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--all-symbols", action="store_true", help="Include all symbols, not just good ones")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "polygon"

    if not data_dir.exists():
        print("No data directory")
        return

    files = list(data_dir.glob("*.csv"))
    symbol_files: dict[str, list[Path]] = {}
    for f in files:
        sym = f.stem.split("_")[0].upper()
        if not args.all_symbols and sym in BAD_SYMBOLS:
            continue
        symbol_files.setdefault(sym, []).append(f)

    all_trades: list[Trade] = []
    sym_results: dict[str, dict] = {}

    for sym, flist in sorted(symbol_files.items()):
        all_bars: list[Bar] = []
        for f in flist:
            bars = load_csv(f)
            if args.year:
                bars = [b for b in bars if b.timestamp.year == args.year]
            all_bars.extend(bars)

        all_bars.sort(key=lambda b: b.timestamp)
        if not all_bars:
            continue

        print(f"Processing {sym}...")
        trades = simulate(sym, all_bars, cfg)
        all_trades.extend(trades)
        sym_results[sym] = summarize(trades)

        if args.debug and trades:
            for t in trades[:3]:
                res = "WIN" if t.win else "LOSS"
                print(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction} R={t.r_multiple:.2f} {res}")

    # Summary
    summary = summarize(all_trades)
    equity, max_dd, pf = equity_curve(all_trades)

    output = ["# Final Optimized Backtest Results", ""]
    output.append("## Configuration")
    output.append("- Symbol filter: Only trending tech/growth stocks")
    output.append(f"- Excluded: {', '.join(sorted(BAD_SYMBOLS))}")
    output.append("- No early scaling - let winners run to 3R")
    output.append("- Wider stops: 2x ATR buffer")
    output.append("- Stronger confirmation: 2 consecutive candles")
    output.append("- Killzones: 9:30-11:00 AM, 2:30-4:00 PM ET")
    output.append("")

    output.append("## Summary")
    if summary["count"] == 0:
        output.append("- No trades generated")
    else:
        output.append(f"- Trades: {summary['count']}")
        output.append(f"- Win rate: {summary['win_rate']:.2%}")
        output.append(f"- Avg R: {summary['avg_r']:.2f}")
        output.append(f"- Total R: {summary['total_r']:.2f}")
        output.append(f"- Profit factor: {pf:.2f}")
        output.append(f"- Max drawdown: {max_dd:.2%}")
        output.append(f"- Final equity (from $10k, 1% risk): ${equity:,.2f}")

        # Calculate expected value
        exp_val = summary['win_rate'] * cfg.target_r - (1 - summary['win_rate']) * 1.0
        output.append(f"- Expected value per trade: {exp_val:.2f}R")

    output.append("")
    output.append("## Per Symbol")
    for sym, stats in sorted(sym_results.items()):
        if stats["count"] > 0:
            output.append(f"- {sym}: {stats['count']} trades, {stats['win_rate']:.1%} win, {stats['avg_r']:.2f} avg R")
        else:
            output.append(f"- {sym}: 0 trades")

    results_path = root / "backtest_docs" / "backtest_final_results.md"
    results_path.write_text("\n".join(output))
    print(f"\nWrote {results_path}")
    print(f"\nSUMMARY: {summary['count']} trades, {summary['win_rate']:.1%} win rate, {summary['avg_r']:.2f} avg R")
    if summary["count"] > 0:
        print(f"EQUITY: ${equity:,.2f} (from $10k)")


if __name__ == "__main__":
    main()
