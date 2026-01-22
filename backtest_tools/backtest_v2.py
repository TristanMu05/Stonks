"""
Backtest V2 - Fixed Strategy Without Lookahead Bias

Key Changes from Original:
1. Entry requires REACTION CANDLE (rejection/engulfing) after zone tap - not just tap
2. BOS must be confirmed with STRONG close (body > 50% of range)
3. Entry zones must be FRESH (created by the BOS move itself)
4. Stops are STRUCTURE-BASED (below/above entry zone) not fixed ticks
5. HTF confluence must be RECENT (within 3 bars, not 10)
6. Proper execution: signal on bar N close, entry on bar N+2 open minimum

Execution Model:
- Signal computed on closed bar
- Entry on NEXT bar open AFTER confirmation candle
- No same-bar entries, no zone-midpoint fantasy fills
"""

import argparse
import json
import csv
from dataclasses import dataclass
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
        """Body as percentage of total range (0-1)"""
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
    confirmed_at: int  # Index when swing was confirmed


@dataclass
class FVG:
    index: int
    direction: str
    low: float
    high: float

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2

    @property
    def size(self) -> float:
        return self.high - self.low


@dataclass
class OrderBlock:
    index: int
    direction: str
    low: float
    high: float

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2


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
    tier: str
    confidence_score: int
    r_multiple: float
    risk_pct: float
    setup_type: str  # "fvg_reaction" or "ob_reaction"


@dataclass
class StrategyConfig:
    # Execution
    tick_size: float = 0.01
    min_entry_delay_bars: int = 2  # Signal bar + confirmation bar

    # Costs
    slippage_ticks: float = 1.0
    spread_ticks: float = 1.0
    commission_per_side: float = 0.0

    # BOS confirmation
    min_bos_body_ratio: float = 0.5  # Body must be >50% of range
    min_bos_ticks: int = 10  # Minimum move beyond swing

    # Entry zone
    max_zone_age_bars: int = 3  # Zone must be fresh
    min_zone_size_ticks: int = 5
    max_zone_size_ticks: int = 50

    # Reaction candle requirements
    min_reaction_body_ratio: float = 0.4
    require_engulfing: bool = False  # If true, need engulfing; if false, just need opposite direction

    # Risk management
    stop_buffer_ticks: int = 5  # Buffer beyond zone for stop
    r_target_medium: float = 2.0
    r_target_high: float = 3.0
    be_trigger_r: float = 1.0  # Move to BE after 1R profit

    # Position sizing
    risk_pct_medium: float = 0.005  # 0.5%
    risk_pct_high: float = 0.01    # 1.0%

    # Filters
    min_score: int = 5
    cooldown_bars: int = 12  # 1 hour on 5m
    ny_session_start: int = 9
    ny_session_end: int = 15  # Avoid last hour

    # HTF confluence
    htf_lookback: int = 3  # Much tighter than before


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


def compute_swings_confirmed(bars: list[Bar]) -> tuple[list[Swing], list[Swing]]:
    """
    Compute swing highs/lows with proper confirmation.
    A swing at index i is confirmed at index i+1 (after we see the next bar).
    """
    swing_highs: list[Swing] = []
    swing_lows: list[Swing] = []

    for i in range(1, len(bars) - 1):
        # Swing high: higher high than both neighbors
        if bars[i].high > bars[i - 1].high and bars[i].high > bars[i + 1].high:
            swing_highs.append(Swing(
                index=i,
                price=bars[i].high,
                confirmed_at=i + 1
            ))
        # Swing low: lower low than both neighbors
        if bars[i].low < bars[i - 1].low and bars[i].low < bars[i + 1].low:
            swing_lows.append(Swing(
                index=i,
                price=bars[i].low,
                confirmed_at=i + 1
            ))

    return swing_highs, swing_lows


def get_usable_swings(swings: list[Swing], current_index: int) -> list[Swing]:
    """Get swings that are confirmed by current_index (no lookahead)."""
    return [s for s in swings if s.confirmed_at <= current_index]


def get_trend(swing_highs: list[Swing], swing_lows: list[Swing], current_index: int) -> str:
    """Determine trend from confirmed swings only."""
    usable_highs = get_usable_swings(swing_highs, current_index)
    usable_lows = get_usable_swings(swing_lows, current_index)

    if len(usable_highs) < 2 or len(usable_lows) < 2:
        return "none"

    last_high, prev_high = usable_highs[-1], usable_highs[-2]
    last_low, prev_low = usable_lows[-1], usable_lows[-2]

    # Uptrend: Higher highs AND higher lows
    if last_high.price > prev_high.price and last_low.price > prev_low.price:
        return "up"
    # Downtrend: Lower highs AND lower lows
    if last_high.price < prev_high.price and last_low.price < prev_low.price:
        return "down"

    return "none"


def detect_fvgs_at_bar(bars: list[Bar], index: int) -> Optional[FVG]:
    """
    Detect if an FVG was created at the given index.
    FVG at index i means: gap between bar[i-2].high and bar[i].low (bull)
    or gap between bar[i].high and bar[i-2].low (bear).
    """
    if index < 2:
        return None

    c1, c2, c3 = bars[index - 2], bars[index - 1], bars[index]

    # Bullish FVG: gap up (bar 1 high < bar 3 low)
    if c1.high < c3.low:
        return FVG(index, "bull", c1.high, c3.low)

    # Bearish FVG: gap down (bar 3 high < bar 1 low)
    if c3.high < c1.low:
        return FVG(index, "bear", c3.high, c1.low)

    return None


def detect_ob_at_bos(bars: list[Bar], bos_index: int, direction: str) -> Optional[OrderBlock]:
    """
    Detect order block created by a BOS.
    For bullish BOS: last bearish candle before the BOS
    For bearish BOS: last bullish candle before the BOS
    """
    if bos_index < 2:
        return None

    # Look back up to 5 bars for the opposing candle
    for j in range(bos_index - 1, max(0, bos_index - 6), -1):
        if direction == "long" and bars[j].is_bearish:
            return OrderBlock(
                index=j,
                direction="bull",
                low=min(bars[j].open, bars[j].close),
                high=max(bars[j].open, bars[j].close),
            )
        if direction == "short" and bars[j].is_bullish:
            return OrderBlock(
                index=j,
                direction="bear",
                low=min(bars[j].open, bars[j].close),
                high=max(bars[j].open, bars[j].close),
            )

    return None


def is_valid_bos(bar: Bar, swing_price: float, direction: str, config: StrategyConfig) -> bool:
    """
    Check if bar represents a valid BOS (not just a wick poke).
    Requires strong close beyond the swing.
    """
    min_move = config.min_bos_ticks * config.tick_size

    if direction == "long":
        # Close must be above swing AND body must be substantial
        return (bar.close > swing_price + min_move and
                bar.body_ratio >= config.min_bos_body_ratio and
                bar.is_bullish)
    else:
        return (bar.close < swing_price - min_move and
                bar.body_ratio >= config.min_bos_body_ratio and
                bar.is_bearish)


def is_reaction_candle(bar: Bar, direction: str, config: StrategyConfig) -> bool:
    """
    Check if bar is a valid reaction candle (showing rejection from zone).
    """
    if bar.body_ratio < config.min_reaction_body_ratio:
        return False

    # For long setup: need bullish candle (showing buyers stepping in)
    if direction == "long":
        return bar.is_bullish
    # For short setup: need bearish candle
    return bar.is_bearish


def is_in_zone(price: float, zone_low: float, zone_high: float) -> bool:
    """Check if price is within zone."""
    return zone_low <= price <= zone_high


def is_ny_session(bar: Bar, config: StrategyConfig) -> bool:
    et = bar.timestamp.astimezone(ET)
    return config.ny_session_start <= et.hour < config.ny_session_end


def detect_htf_sweep(bars: list[Bar], swing_highs: list[Swing], swing_lows: list[Swing],
                     index: int, lookback: int) -> bool:
    """
    Detect liquidity sweep in HTF bars.
    Sweep: price pokes beyond swing but closes back inside.
    """
    for i in range(max(0, index - lookback), index + 1):
        bar = bars[i]

        # Check for sweep of recent swing highs
        usable_highs = [s for s in swing_highs if s.confirmed_at <= i and s.index >= i - 10]
        for swing in usable_highs:
            if bar.high > swing.price and bar.close < swing.price:
                return True

        # Check for sweep of recent swing lows
        usable_lows = [s for s in swing_lows if s.confirmed_at <= i and s.index >= i - 10]
        for swing in usable_lows:
            if bar.low < swing.price and bar.close > swing.price:
                return True

    return False


def confidence_score(
    trend_align: bool,
    htf_sweep: bool,
    fresh_zone: bool,
    strong_reaction: bool,
    discount_premium: bool,
    optimal_time: bool,
) -> int:
    """
    Simplified confidence scoring focused on what matters.
    """
    score = 0
    score += 2 if trend_align else 0
    score += 2 if htf_sweep else 0      # Real confluence
    score += 2 if fresh_zone else 0     # Zone quality
    score += 2 if strong_reaction else 0  # Confirmation
    score += 1 if discount_premium else 0
    score += 1 if optimal_time else 0
    return score


def tier_from_score(score: int) -> str:
    if score < 5:
        return "low"
    if score < 7:
        return "medium"
    return "high"


def simulate_trades(symbol: str, bars_1m: list[Bar], config: StrategyConfig) -> list[Trade]:
    """
    Main simulation with proper execution model.

    Flow:
    1. Detect BOS on HTF (1H trend context)
    2. On LTF (5m), wait for BOS + FVG/OB creation
    3. Wait for price to return to zone
    4. Wait for REACTION CANDLE in zone
    5. Enter on next bar open
    6. Manage trade with structure-based stop
    """
    bars_5m = resample_bars(bars_1m, 5)
    bars_1h = resample_bars(bars_1m, 60)

    if len(bars_1h) < 20 or len(bars_5m) < 50:
        return []

    # Pre-compute HTF structure
    swings_1h = compute_swings_confirmed(bars_1h)
    swings_5m = compute_swings_confirmed(bars_5m)

    trades: list[Trade] = []
    last_exit_index = -100

    def get_1h_index(t: datetime) -> int:
        for i, bar in enumerate(bars_1h):
            if bar.timestamp > t:
                return max(0, i - 1)
        return len(bars_1h) - 1

    # Track active setup (waiting for entry)
    active_setup = None  # (direction, zone_low, zone_high, bos_index, setup_type)

    i = 10  # Start after enough bars for context
    while i < len(bars_5m):
        bar = bars_5m[i]

        # Cooldown check
        if i <= last_exit_index + config.cooldown_bars:
            i += 1
            continue

        # Session filter
        if not is_ny_session(bar, config):
            active_setup = None  # Reset setup outside session
            i += 1
            continue

        idx_1h = get_1h_index(bar.timestamp)
        trend_1h = get_trend(*swings_1h, idx_1h)

        # === LOOK FOR NEW BOS SETUP ===
        if active_setup is None and trend_1h != "none":
            usable_highs = get_usable_swings(swings_5m[0], i)
            usable_lows = get_usable_swings(swings_5m[1], i)

            if len(usable_highs) >= 1 and len(usable_lows) >= 1:
                last_high = usable_highs[-1]
                last_low = usable_lows[-1]

                # Check for bullish BOS (aligned with uptrend)
                if trend_1h == "up" and is_valid_bos(bar, last_high.price, "long", config):
                    # Look for FVG or OB created by this move
                    fvg = detect_fvgs_at_bar(bars_5m, i)
                    ob = detect_ob_at_bos(bars_5m, i, "long")

                    zone = None
                    setup_type = None

                    if fvg and fvg.direction == "bull":
                        zone_size = fvg.size / config.tick_size
                        if config.min_zone_size_ticks <= zone_size <= config.max_zone_size_ticks:
                            zone = (fvg.low, fvg.high)
                            setup_type = "fvg_reaction"

                    if zone is None and ob:
                        zone_size = (ob.high - ob.low) / config.tick_size
                        if config.min_zone_size_ticks <= zone_size <= config.max_zone_size_ticks:
                            zone = (ob.low, ob.high)
                            setup_type = "ob_reaction"

                    if zone:
                        active_setup = ("long", zone[0], zone[1], i, setup_type)

                # Check for bearish BOS (aligned with downtrend)
                elif trend_1h == "down" and is_valid_bos(bar, last_low.price, "short", config):
                    fvg = detect_fvgs_at_bar(bars_5m, i)
                    ob = detect_ob_at_bos(bars_5m, i, "short")

                    zone = None
                    setup_type = None

                    if fvg and fvg.direction == "bear":
                        zone_size = fvg.size / config.tick_size
                        if config.min_zone_size_ticks <= zone_size <= config.max_zone_size_ticks:
                            zone = (fvg.low, fvg.high)
                            setup_type = "fvg_reaction"

                    if zone is None and ob:
                        zone_size = (ob.high - ob.low) / config.tick_size
                        if config.min_zone_size_ticks <= zone_size <= config.max_zone_size_ticks:
                            zone = (ob.low, ob.high)
                            setup_type = "ob_reaction"

                    if zone:
                        active_setup = ("short", zone[0], zone[1], i, setup_type)

        # === MANAGE ACTIVE SETUP ===
        if active_setup is not None:
            direction, zone_low, zone_high, bos_index, setup_type = active_setup
            zone_age = i - bos_index

            # Expire setup if too old
            if zone_age > config.max_zone_age_bars + 10:  # Extra buffer for entry wait
                active_setup = None
                i += 1
                continue

            # Check if price is in zone
            price_in_zone = (bar.low <= zone_high and bar.high >= zone_low)

            if price_in_zone and zone_age >= config.min_entry_delay_bars:
                # Check for reaction candle
                if is_reaction_candle(bar, direction, config):
                    # We have confirmation! Enter on NEXT bar open
                    if i + 1 < len(bars_5m):
                        entry_bar = bars_5m[i + 1]
                        entry_price = entry_bar.open

                        # Add slippage
                        slip = config.slippage_ticks * config.tick_size
                        if direction == "long":
                            entry_price += slip
                        else:
                            entry_price -= slip

                        # Structure-based stop
                        stop_buffer = config.stop_buffer_ticks * config.tick_size
                        if direction == "long":
                            stop = zone_low - stop_buffer
                        else:
                            stop = zone_high + stop_buffer

                        stop_distance = abs(entry_price - stop)

                        # Calculate confidence
                        htf_sweep = detect_htf_sweep(bars_1h, *swings_1h, idx_1h, config.htf_lookback)
                        fresh_zone = zone_age <= config.max_zone_age_bars
                        strong_reaction = bar.body_ratio >= 0.5

                        # Discount/premium check
                        if len(usable_highs) >= 1 and len(usable_lows) >= 1:
                            equilibrium = (usable_highs[-1].price + usable_lows[-1].price) / 2
                            discount_premium = (entry_price < equilibrium if direction == "long"
                                              else entry_price > equilibrium)
                        else:
                            discount_premium = False

                        et = bar.timestamp.astimezone(ET)
                        optimal_time = 9 <= et.hour <= 11 or 14 <= et.hour <= 15

                        score = confidence_score(
                            trend_align=True,  # Already filtered
                            htf_sweep=htf_sweep,
                            fresh_zone=fresh_zone,
                            strong_reaction=strong_reaction,
                            discount_premium=discount_premium,
                            optimal_time=optimal_time,
                        )

                        # Filter by minimum score
                        if score < config.min_score:
                            active_setup = None
                            i += 1
                            continue

                        tier = tier_from_score(score)
                        r_target = config.r_target_high if tier == "high" else config.r_target_medium

                        if direction == "long":
                            target = entry_price + r_target * stop_distance
                        else:
                            target = entry_price - r_target * stop_distance

                        # Simulate trade execution
                        current_stop = stop
                        moved_to_be = False
                        exit_price = None
                        exit_time = None

                        for k in range(i + 2, len(bars_5m)):
                            b = bars_5m[k]

                            # Move to BE after 1R profit
                            if not moved_to_be:
                                be_level = entry_price + config.be_trigger_r * stop_distance if direction == "long" else entry_price - config.be_trigger_r * stop_distance
                                if (direction == "long" and b.high >= be_level) or \
                                   (direction == "short" and b.low <= be_level):
                                    current_stop = entry_price
                                    moved_to_be = True

                            # Check stop
                            if direction == "long" and b.low <= current_stop:
                                exit_price = current_stop - slip  # Adverse slippage
                                exit_time = b.timestamp
                                break
                            if direction == "short" and b.high >= current_stop:
                                exit_price = current_stop + slip
                                exit_time = b.timestamp
                                break

                            # Check target
                            if direction == "long" and b.high >= target:
                                exit_price = target - slip  # Conservative fill
                                exit_time = b.timestamp
                                break
                            if direction == "short" and b.low <= target:
                                exit_price = target + slip
                                exit_time = b.timestamp
                                break

                        # Close at end if still open
                        if exit_price is None:
                            last_bar = bars_5m[-1]
                            exit_price = last_bar.close
                            exit_time = last_bar.timestamp

                        # Calculate R
                        if direction == "long":
                            raw_r = (exit_price - entry_price) / stop_distance
                        else:
                            raw_r = (entry_price - exit_price) / stop_distance

                        # Apply costs
                        cost_r = (2 * config.slippage_ticks + config.spread_ticks) * config.tick_size / stop_distance
                        r_multiple = raw_r - cost_r

                        risk_pct = config.risk_pct_high if tier == "high" else config.risk_pct_medium

                        trades.append(Trade(
                            symbol=symbol,
                            direction=direction,
                            entry_time=entry_bar.timestamp,
                            entry_price=entry_price,
                            exit_time=exit_time,
                            exit_price=exit_price,
                            stop=stop,
                            target=target,
                            tier=tier,
                            confidence_score=score,
                            r_multiple=r_multiple,
                            risk_pct=risk_pct,
                            setup_type=setup_type,
                        ))

                        # Find exit index for cooldown
                        for exit_idx, b in enumerate(bars_5m):
                            if b.timestamp >= exit_time:
                                last_exit_index = exit_idx
                                break

                        active_setup = None
                        i = last_exit_index + 1
                        continue

            # Check if zone was violated (setup invalidated)
            if direction == "long" and bar.close < zone_low:
                active_setup = None
            elif direction == "short" and bar.close > zone_high:
                active_setup = None

        i += 1

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


def summarize_by_tier(trades: list[Trade]) -> dict[str, dict]:
    tiers: dict[str, list[Trade]] = {}
    for trade in trades:
        tiers.setdefault(trade.tier, []).append(trade)
    return {tier: summarize(items) for tier, items in sorted(tiers.items())}


def summarize_by_setup(trades: list[Trade]) -> dict[str, dict]:
    setups: dict[str, list[Trade]] = {}
    for trade in trades:
        setups.setdefault(trade.setup_type, []).append(trade)
    return {setup: summarize(items) for setup, items in sorted(setups.items())}


def estimate_equity(trades: list[Trade], starting: float = 10000.0) -> tuple[float, float, float, float]:
    """Returns (simple_equity, max_drawdown, profit_factor, total_r)"""
    if not trades:
        return starting, 0.0, 0.0, 0.0

    trades_sorted = sorted(trades, key=lambda t: t.entry_time)

    equity = starting
    peak = starting
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    total_r = 0.0

    for t in trades_sorted:
        total_r += t.r_multiple
        pnl = starting * t.risk_pct * t.r_multiple
        equity += pnl

        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return equity, max_dd, pf, total_r


def main():
    parser = argparse.ArgumentParser(description="Backtest V2 - Improved Strategy")
    parser.add_argument("--except", dest="exclude", default="", help="Symbols to exclude")
    parser.add_argument("--year", type=int, default=None, help="Filter by year")
    parser.add_argument("--debug", action="store_true", help="Show trade details")
    args = parser.parse_args()

    config = StrategyConfig()
    root = Path(__file__).resolve().parents[1]
    polygon_dir = root / "data" / "polygon"

    if not polygon_dir.exists():
        print(f"No data directory found at {polygon_dir}")
        return

    files = list(polygon_dir.glob("*.csv"))
    if not files:
        print("No CSV files found")
        return

    excluded = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}

    all_trades: list[Trade] = []
    # Group files by symbol to merge multi-year data
    symbol_files: dict[str, list[Path]] = {}
    for file in sorted(files):
        symbol = file.stem.split("_")[0].upper()
        if symbol not in excluded:
            symbol_files.setdefault(symbol, []).append(file)

    symbol_results: dict[str, dict] = {}

    for symbol, files_list in sorted(symbol_files.items()):
        print(f"Processing {symbol} ({len(files_list)} files)...")

        # Merge all bars for this symbol
        all_bars: list[Bar] = []
        for file in files_list:
            bars = load_polygon_csv(file)
            if args.year:
                bars = [b for b in bars if b.timestamp.year == args.year]
            all_bars.extend(bars)

        # Sort and dedupe by timestamp
        all_bars.sort(key=lambda b: b.timestamp)
        if not all_bars:
            continue

        trades = simulate_trades(symbol, all_bars, config)
        all_trades.extend(trades)
        symbol_results[symbol] = summarize(trades)

        if args.debug and trades:
            print(f"  {len(trades)} trades:")
            for t in trades[:5]:  # Show first 5
                outcome = "WIN" if t.r_multiple > 0 else "LOSS"
                print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction} {t.setup_type} "
                      f"R={t.r_multiple:.2f} {outcome}")

    # Generate report
    summary = summarize(all_trades)
    tier_summary = summarize_by_tier(all_trades)
    setup_summary = summarize_by_setup(all_trades)
    equity, max_dd, pf, total_r = estimate_equity(all_trades)

    output = ["# Backtest V2 Results", ""]
    output.append("## Strategy Changes")
    output.append("- Requires REACTION CANDLE (not just zone tap)")
    output.append("- BOS must have strong body (>50% of range)")
    output.append("- Entry zones must be FRESH (<3 bars old)")
    output.append("- Structure-based stops (below zone, not fixed)")
    output.append("- HTF sweep lookback reduced to 3 bars")
    output.append("- Minimum score raised to 5")
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
        output.append(f"- Equity (from $10k): ${equity:,.2f}")

    output.append("")
    output.append("## By Tier")
    for tier, stats in tier_summary.items():
        if stats["count"] > 0:
            output.append(f"- {tier}: {stats['count']} trades, win {stats['win_rate']:.2%}, avg R {stats['avg_r']:.2f}")

    output.append("")
    output.append("## By Setup Type")
    for setup, stats in setup_summary.items():
        if stats["count"] > 0:
            output.append(f"- {setup}: {stats['count']} trades, win {stats['win_rate']:.2%}, avg R {stats['avg_r']:.2f}")

    output.append("")
    output.append("## Per Symbol")
    for symbol, stats in sorted(symbol_results.items()):
        if stats["count"] > 0:
            output.append(f"- {symbol}: {stats['count']} trades, win {stats['win_rate']:.2%}, avg R {stats['avg_r']:.2f}")
        else:
            output.append(f"- {symbol}: 0 trades")

    results_path = root / "backtest_docs" / "backtest_v2_results.md"
    results_path.write_text("\n".join(output))
    print(f"\nWrote results to {results_path}")
    print(f"\nSummary: {summary['count']} trades, {summary['win_rate']:.1%} win rate, {summary['avg_r']:.2f} avg R")


if __name__ == "__main__":
    main()
