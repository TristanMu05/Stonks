import argparse
import json
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from strategy_config import StrategyConfig, get_default_config

ET = ZoneInfo("America/New_York")


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Swing:
    index: int
    price: float


@dataclass
class FVG:
    index: int
    direction: str
    low: float
    high: float


@dataclass
class OrderBlock:
    index: int
    direction: str  # "bull" or "bear"
    low: float
    high: float


@dataclass
class SMTSignal:
    """Smart Money Technique Divergence Signal"""
    index: int
    direction: str  # "bull" or "bear"
    primary_price: float  # Price level on primary asset
    secondary_price: float  # Price level on secondary asset


@dataclass
class Trade:
    """
    Represents a completed trade with full cost accounting.
    
    Execution Model:
    - Signal computed on bar close
    - Entry at NEXT bar open (entry_price)
    - All PnL computed in ticks first, then converted to R
    """
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: float
    stop: float
    target: float
    tier: str
    confidence_score: int
    risk_pct: float
    
    # PnL fields (computed after costs)
    raw_pnl_ticks: float = 0.0      # Before costs
    cost_ticks: float = 0.0         # Total transaction costs
    net_pnl_ticks: float = 0.0      # After costs
    r_multiple: float = 0.0         # Net PnL / stop distance
    
    # Trade metadata
    execution_tf: str = "5m"        # Execution timeframe used
    htf_event_reason: str = ""      # "sweep", "fvg_touch", "ob_touch"
    entry_zone_type: str = ""       # "fvg" or "ob"
    
    def compute_pnl(self, config: StrategyConfig) -> None:
        """
        Compute PnL in ticks, apply costs, then convert to R.
        Modifies the trade in place.
        """
        tick_size = config.tick_size
        
        # Step 1: Raw PnL in ticks
        if self.direction == "long":
            self.raw_pnl_ticks = (self.exit_price - self.entry_price) / tick_size
        else:
            self.raw_pnl_ticks = (self.entry_price - self.exit_price) / tick_size
        
        # Step 2: Apply costs in ticks
        self.cost_ticks = config.compute_cost_in_ticks()
        self.net_pnl_ticks = self.raw_pnl_ticks - self.cost_ticks
        
        # Step 3: Convert to R
        stop_distance_ticks = config.stop_ticks
        self.r_multiple = self.net_pnl_ticks / stop_distance_ticks if stop_distance_ticks > 0 else 0


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def to_minute_bars(ticks: list[dict]) -> list[Bar]:
    ticks_sorted = sorted(ticks, key=lambda x: x["timestamp"])
    buckets: dict[datetime, list[dict]] = {}
    for tick in ticks_sorted:
        ts = parse_timestamp(tick["timestamp"]).replace(second=0, microsecond=0)
        buckets.setdefault(ts, []).append(tick)

    bars: list[Bar] = []
    for ts in sorted(buckets.keys()):
        bucket = buckets[ts]
        prices = [t["price"] for t in bucket]
        volumes = [t.get("volume", 0) for t in bucket]
        bars.append(
            Bar(
                timestamp=ts,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(volumes),
            )
        )
    return bars


def load_polygon_csv(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=timezone.utc)
            bars.append(
                Bar(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0),
                )
            )
    return bars


def filter_bars_by_year(bars: list[Bar], year: int | None) -> list[Bar]:
    if not year:
        return bars
    return [bar for bar in bars if bar.timestamp.year == year]


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
        out.append(
            Bar(
                timestamp=ts,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
            )
        )
    return out


def compute_swings(bars: list[Bar]) -> tuple[list[Swing], list[Swing]]:
    swing_highs: list[Swing] = []
    swing_lows: list[Swing] = []
    for i in range(1, len(bars) - 1):
        if bars[i].high > bars[i - 1].high and bars[i].high > bars[i + 1].high:
            swing_highs.append(Swing(i, bars[i].high))
        if bars[i].low < bars[i - 1].low and bars[i].low < bars[i + 1].low:
            swing_lows.append(Swing(i, bars[i].low))
    return swing_highs, swing_lows


def get_last_two(swings: list[Swing], index: int) -> tuple[Swing | None, Swing | None]:
    prev = [s for s in swings if s.index < index]
    if len(prev) < 2:
        return None, None
    return prev[-1], prev[-2]


def trend_at(bars: list[Bar], swing_highs: list[Swing], swing_lows: list[Swing], index: int) -> str:
    last_high, prev_high = get_last_two(swing_highs, index)
    last_low, prev_low = get_last_two(swing_lows, index)
    if not last_high or not prev_high or not last_low or not prev_low:
        return "none"
    if last_high.price > prev_high.price and last_low.price > prev_low.price:
        return "up"
    if last_high.price < prev_high.price and last_low.price < prev_low.price:
        return "down"
    return "none"


def detect_fvgs(bars: list[Bar]) -> list[FVG]:
    fvgs: list[FVG] = []
    for i in range(2, len(bars)):
        c1, _, c3 = bars[i - 2], bars[i - 1], bars[i]
        if c1.high < c3.low:
            fvgs.append(FVG(i, "bull", c1.high, c3.low))
        if c1.low > c3.high:
            fvgs.append(FVG(i, "bear", c3.high, c1.low))
    return fvgs


def fvg_tapped(bar: Bar, fvg: FVG) -> bool:
    return bar.low <= fvg.high and bar.high >= fvg.low


def detect_order_blocks(bars: list[Bar], swing_highs: list[Swing], swing_lows: list[Swing]) -> list[OrderBlock]:
    """Detect Order Blocks - the last opposing candle before a BOS."""
    order_blocks: list[OrderBlock] = []

    for i in range(2, len(bars)):
        # Check for bullish BOS (close above previous swing high)
        prev_highs = [s for s in swing_highs if s.index < i]
        if prev_highs:
            last_swing_high = prev_highs[-1]
            if bars[i].close > last_swing_high.price:
                # Find the last down candle before the BOS
                for j in range(i - 1, max(0, i - 10), -1):
                    if bars[j].close < bars[j].open:  # Down candle
                        order_blocks.append(OrderBlock(
                            index=j,
                            direction="bull",
                            low=min(bars[j].open, bars[j].close),
                            high=max(bars[j].open, bars[j].close),
                        ))
                        break

        # Check for bearish BOS (close below previous swing low)
        prev_lows = [s for s in swing_lows if s.index < i]
        if prev_lows:
            last_swing_low = prev_lows[-1]
            if bars[i].close < last_swing_low.price:
                # Find the last up candle before the BOS
                for j in range(i - 1, max(0, i - 10), -1):
                    if bars[j].close > bars[j].open:  # Up candle
                        order_blocks.append(OrderBlock(
                            index=j,
                            direction="bear",
                            low=min(bars[j].open, bars[j].close),
                            high=max(bars[j].open, bars[j].close),
                        ))
                        break

    return order_blocks


def ob_tapped(bar: Bar, ob: OrderBlock) -> bool:
    """Check if price has tapped into an Order Block."""
    return bar.low <= ob.high and bar.high >= ob.low


def detect_smt_divergence(
    primary_bars: list[Bar],
    secondary_bars: list[Bar],
    primary_swings: tuple[list[Swing], list[Swing]],
    secondary_swings: tuple[list[Swing], list[Swing]],
    lookback: int = 10,
) -> list[SMTSignal]:
    """
    Detect SMT (Smart Money Technique) Divergence between two correlated assets.

    Bullish SMT: Primary makes Lower Low, Secondary makes Higher Low
    Bearish SMT: Primary makes Higher High, Secondary makes Lower High

    Args:
        primary_bars: Bars for primary asset (e.g., SPY/ES)
        secondary_bars: Bars for secondary asset (e.g., QQQ/NQ)
        primary_swings: (swing_highs, swing_lows) for primary
        secondary_swings: (swing_highs, swing_lows) for secondary
        lookback: Number of bars to look back for swing comparison

    Returns:
        List of SMT signals with direction and price levels
    """
    signals: list[SMTSignal] = []

    p_highs, p_lows = primary_swings
    s_highs, s_lows = secondary_swings

    # Build timestamp-to-index maps for alignment
    p_time_to_idx = {bar.timestamp: i for i, bar in enumerate(primary_bars)}
    s_time_to_idx = {bar.timestamp: i for i, bar in enumerate(secondary_bars)}

    # Find common timestamps
    common_times = set(p_time_to_idx.keys()) & set(s_time_to_idx.keys())

    for swing in p_lows:
        if swing.index < 2:
            continue

        # Get timestamp of this swing low
        p_time = primary_bars[swing.index].timestamp
        if p_time not in common_times:
            continue

        s_idx = s_time_to_idx[p_time]

        # Find previous swing low on primary (within lookback)
        prev_p_lows = [s for s in p_lows if s.index < swing.index and s.index >= swing.index - lookback * 12]
        if not prev_p_lows:
            continue
        prev_p_low = prev_p_lows[-1]

        # Check if primary made a Lower Low
        if swing.price >= prev_p_low.price:
            continue  # Not a lower low

        # Find corresponding swing lows on secondary around the same time
        prev_s_lows = [s for s in s_lows if s.index <= s_idx and s.index >= s_idx - lookback * 12]
        curr_s_lows = [s for s in s_lows if abs(s.index - s_idx) <= 3]  # Within 3 bars of current time

        if not prev_s_lows or not curr_s_lows:
            continue

        prev_s_low = prev_s_lows[-2] if len(prev_s_lows) >= 2 else prev_s_lows[-1]
        curr_s_low = curr_s_lows[-1]

        # Bullish SMT: Primary Lower Low + Secondary Higher Low
        if curr_s_low.price > prev_s_low.price:
            signals.append(SMTSignal(
                index=swing.index,
                direction="bull",
                primary_price=swing.price,
                secondary_price=curr_s_low.price,
            ))

    for swing in p_highs:
        if swing.index < 2:
            continue

        # Get timestamp of this swing high
        p_time = primary_bars[swing.index].timestamp
        if p_time not in common_times:
            continue

        s_idx = s_time_to_idx[p_time]

        # Find previous swing high on primary (within lookback)
        prev_p_highs = [s for s in p_highs if s.index < swing.index and s.index >= swing.index - lookback * 12]
        if not prev_p_highs:
            continue
        prev_p_high = prev_p_highs[-1]

        # Check if primary made a Higher High
        if swing.price <= prev_p_high.price:
            continue  # Not a higher high

        # Find corresponding swing highs on secondary around the same time
        prev_s_highs = [s for s in s_highs if s.index <= s_idx and s.index >= s_idx - lookback * 12]
        curr_s_highs = [s for s in s_highs if abs(s.index - s_idx) <= 3]  # Within 3 bars of current time

        if not prev_s_highs or not curr_s_highs:
            continue

        prev_s_high = prev_s_highs[-2] if len(prev_s_highs) >= 2 else prev_s_highs[-1]
        curr_s_high = curr_s_highs[-1]

        # Bearish SMT: Primary Higher High + Secondary Lower High
        if curr_s_high.price < prev_s_high.price:
            signals.append(SMTSignal(
                index=swing.index,
                direction="bear",
                primary_price=swing.price,
                secondary_price=curr_s_high.price,
            ))

    return signals


def check_smt_at_index(smt_signals: list[SMTSignal], index: int, direction: str, lookback: int = 10) -> bool:
    """Check if there's a confirming SMT signal near the given index."""
    for signal in smt_signals:
        if signal.direction == direction and abs(signal.index - index) <= lookback:
            return True
    return False


def is_ny_session(bar: Bar) -> bool:
    """Check if bar is within NY trading session (8:00-16:00 ET)."""
    et = bar.timestamp.astimezone(ET)
    return 8 <= et.hour < 16


def is_optimal_trading_time(bar: Bar) -> bool:
    """Check if bar is within optimal trading windows (avoid lunch)."""
    et = bar.timestamp.astimezone(ET)
    # NY AM session (best): 9:30-11:30
    # NY PM session (good): 13:30-15:30
    # Avoid lunch: 11:30-13:30
    if 9 <= et.hour < 12:  # Morning session
        return True
    if et.hour == 12 and et.minute < 30:  # Until 12:30
        return False
    if 13 <= et.hour < 16:  # Afternoon (after 13:00)
        if et.hour == 13 and et.minute < 30:  # Skip 13:00-13:30
            return False
        return True
    return False


def detect_sweeps(bars: list[Bar], swing_highs: list[Swing], swing_lows: list[Swing]) -> set[int]:
    sweep_indices: set[int] = set()
    swing_high_map = {s.index: s for s in swing_highs}
    swing_low_map = {s.index: s for s in swing_lows}
    for i in range(len(bars)):
        for j in range(max(0, i - 3), i):
            if j in swing_high_map:
                level = swing_high_map[j].price
                if bars[i].high > level and bars[i].close < level:
                    sweep_indices.add(i)
            if j in swing_low_map:
                level = swing_low_map[j].price
                if bars[i].low < level and bars[i].close > level:
                    sweep_indices.add(i)
    return sweep_indices


def session_levels(bars: list[Bar]) -> dict[datetime, dict[str, float]]:
    levels: dict[datetime, dict[str, float]] = {}
    for bar in bars:
        et = bar.timestamp.astimezone(ET)
        date_key = datetime(et.year, et.month, et.day, tzinfo=ET)
        levels.setdefault(date_key, {
            "asia_high": float("-inf"),
            "asia_low": float("inf"),
            "london_high": float("-inf"),
            "london_low": float("inf"),
            "ny_high": float("-inf"),
            "ny_low": float("inf"),
            "t_open": None,
        })
        if et.hour == 9 and et.minute == 30:
            levels[date_key]["t_open"] = bar.open
        if 20 <= et.hour or et.hour < 3:
            levels[date_key]["asia_high"] = max(levels[date_key]["asia_high"], bar.high)
            levels[date_key]["asia_low"] = min(levels[date_key]["asia_low"], bar.low)
        if 3 <= et.hour < 8:
            levels[date_key]["london_high"] = max(levels[date_key]["london_high"], bar.high)
            levels[date_key]["london_low"] = min(levels[date_key]["london_low"], bar.low)
        if 8 <= et.hour < 16:
            levels[date_key]["ny_high"] = max(levels[date_key]["ny_high"], bar.high)
            levels[date_key]["ny_low"] = min(levels[date_key]["ny_low"], bar.low)
    return levels


def get_level_for_bar(levels: dict[datetime, dict[str, float]], bar: Bar) -> dict[str, float]:
    et = bar.timestamp.astimezone(ET)
    date_key = datetime(et.year, et.month, et.day, tzinfo=ET)
    return levels.get(date_key, {})


def confidence_score(
    trend_align: bool,
    htf_sweep: bool,
    ltf_bos: bool,
    htf_fvg_touch: bool,
    discount_premium: bool,
    session_conf: bool,
    smt_divergence: bool = False,
    optimal_time: bool = False,
    config: Optional[StrategyConfig] = None,
) -> int:
    """
    Calculate confidence score for a trade setup.
    
    Uses configured weights if config provided, otherwise uses defaults.
    """
    if config is not None:
        return config.compute_confidence_score(
            trend_align=trend_align,
            htf_sweep=htf_sweep,
            ltf_bos=ltf_bos,
            htf_fvg_touch=htf_fvg_touch,
            discount_premium=discount_premium,
            session_conf=session_conf,
            smt_divergence=smt_divergence,
            optimal_time=optimal_time,
        )
    
    # Legacy default weights
    score = 0
    score += 2 if trend_align else 0
    score += 2 if htf_sweep else 0
    score += 2 if ltf_bos else 0
    score += 1 if htf_fvg_touch else 0
    score += 1 if discount_premium else 0
    score += 1 if session_conf else 0
    score += 1 if smt_divergence else 0
    score += 1 if optimal_time else 0
    return score


def tier_from_score(score: int, config: Optional[StrategyConfig] = None) -> str:
    """Determine tier from confidence score using config thresholds."""
    if config is not None:
        return config.get_tier(score)
    
    # Legacy defaults
    if score <= 3:
        return "low"
    if score <= 6:
        return "medium"
    return "high"


def simulate_trades(
    symbol: str,
    bars_1m: list[Bar],
    config: Optional[StrategyConfig] = None,
) -> list[Trade]:
    """
    Simulate trades for a symbol using the strategy rules.
    
    Execution Model:
    - All pattern detection uses COMPLETED bars only
    - Entry occurs at NEXT bar OPEN after signal
    - PnL computed in ticks, costs applied, then converted to R
    
    Args:
        symbol: Ticker symbol
        bars_1m: 1-minute OHLCV bars
        config: Strategy configuration (uses defaults if None)
    
    Returns:
        List of completed trades with cost-adjusted R-multiples
    """
    if config is None:
        config = get_default_config()
    
    tick_size = config.tick_size
    bars_5m = resample_bars(bars_1m, 5)
    bars_15m = resample_bars(bars_1m, 15)
    bars_1h = resample_bars(bars_1m, 60)
    bars_4h = resample_bars(bars_1m, 240)

    if len(bars_4h) < 5 or len(bars_1h) < 10:
        return []

    # Pre-compute all indicators for all timeframes (optimization)
    swings_1h = compute_swings(bars_1h)
    swings_4h = compute_swings(bars_4h)
    swings_5m = compute_swings(bars_5m)
    swings_15m = compute_swings(bars_15m)

    fvgs_1h = detect_fvgs(bars_1h)
    fvgs_4h = detect_fvgs(bars_4h)
    fvgs_5m = detect_fvgs(bars_5m)
    fvgs_15m = detect_fvgs(bars_15m)

    sweeps_1h = detect_sweeps(bars_1h, *swings_1h)
    sweeps_4h = detect_sweeps(bars_4h, *swings_4h)

    obs_1h = detect_order_blocks(bars_1h, *swings_1h)
    obs_4h = detect_order_blocks(bars_4h, *swings_4h)
    obs_5m = detect_order_blocks(bars_5m, *swings_5m)
    obs_15m = detect_order_blocks(bars_15m, *swings_15m)

    levels = session_levels(bars_1m)

    trades: list[Trade] = []
    last_exit_index = -1  # Track when last trade exited to prevent overlapping

    def bar_index_for_time(bars: list[Bar], t: datetime) -> int:
        idx = 0
        for i, bar in enumerate(bars):
            if bar.timestamp <= t:
                idx = i
            else:
                break
        return idx

    for i in range(3, len(bars_5m)):
        t = bars_5m[i].timestamp
        bar_5m = bars_5m[i]

        # COOLDOWN: Skip if we're still in a trade or cooling down
        if i <= last_exit_index + 6:  # 6 bars (30 min on 5m) cooldown after exit
            continue

        # TIME FILTER: Only trade during NY session
        if not is_ny_session(bar_5m):
            continue

        idx_1h = bar_index_for_time(bars_1h, t)
        idx_4h = bar_index_for_time(bars_4h, t)

        trend_1h = trend_at(bars_1h, *swings_1h, idx_1h)
        trend_4h = trend_at(bars_4h, *swings_4h, idx_4h)
        if trend_1h == "none" or trend_4h == "none":
            continue

        # Select pre-computed data based on execution timeframe
        use_5m = trend_1h == trend_4h
        execution_bars = bars_5m if use_5m else bars_15m
        exec_swings = swings_5m if use_5m else swings_15m
        exec_fvgs = fvgs_5m if use_5m else fvgs_15m
        exec_obs = obs_5m if use_5m else obs_15m

        exec_index = bar_index_for_time(execution_bars, t)
        if exec_index < 3:
            continue

        swing_highs, swing_lows = exec_swings
        last_high, _ = get_last_two(swing_highs, exec_index)
        last_low, _ = get_last_two(swing_lows, exec_index)
        if not last_high or not last_low:
            continue

        bar = execution_bars[exec_index]
        bos_long = bar.close > last_high.price
        bos_short = bar.close < last_low.price
        if not bos_long and not bos_short:
            continue

        direction = "long" if bos_long else "short"

        # Detect HTF events with extended lookback (10 bars instead of 3)
        htf_sweep = False
        htf_fvg_touch = False
        htf_ob_touch = False

        for htf_idx in range(max(0, idx_1h - 10), idx_1h + 1):
            if htf_idx in sweeps_1h:
                htf_sweep = True
            for fvg in fvgs_1h:
                if fvg.index <= htf_idx and fvg_tapped(bars_1h[htf_idx], fvg):
                    htf_fvg_touch = True
            for ob in obs_1h:
                if ob.index <= htf_idx and ob_tapped(bars_1h[htf_idx], ob):
                    htf_ob_touch = True

        for htf_idx in range(max(0, idx_4h - 10), idx_4h + 1):
            if htf_idx in sweeps_4h:
                htf_sweep = True
            for fvg in fvgs_4h:
                if fvg.index <= htf_idx and fvg_tapped(bars_4h[htf_idx], fvg):
                    htf_fvg_touch = True
            for ob in obs_4h:
                if ob.index <= htf_idx and ob_tapped(bars_4h[htf_idx], ob):
                    htf_ob_touch = True

        # Require at least ONE HTF event (sweep OR FVG touch OR OB touch)
        htf_event = htf_sweep or htf_fvg_touch or htf_ob_touch
        if not htf_event:
            continue

        # Use pre-computed entry zones from execution timeframe
        ltf_fvgs = exec_fvgs
        ltf_obs = exec_obs

        entry_zone = None
        entry_zone_type = None

        # Try FVG first (must be within last 5 bars, not future bars)
        for fvg in ltf_fvgs:
            if exec_index - 5 <= fvg.index <= exec_index and fvg.direction == ("bull" if direction == "long" else "bear"):
                entry_zone = (fvg.low, fvg.high)
                entry_zone_type = "fvg"
                break

        # If no FVG, try Order Block (must be within last 5 bars, not future bars)
        if not entry_zone:
            for ob in ltf_obs:
                if exec_index - 5 <= ob.index <= exec_index and ob.direction == ("bull" if direction == "long" else "bear"):
                    entry_zone = (ob.low, ob.high)
                    entry_zone_type = "ob"
                    break

        if not entry_zone:
            continue

        # Find entry when price taps the zone
        entry_index = None
        for j in range(exec_index, min(len(execution_bars), exec_index + 10)):
            if execution_bars[j].low <= entry_zone[1] and execution_bars[j].high >= entry_zone[0]:
                entry_index = j
                break

        if entry_index is None:
            continue

        entry_bar = execution_bars[entry_index]
        entry_price = (entry_zone[0] + entry_zone[1]) / 2

        # Calculate equilibrium for discount/premium
        swing_high_4h, swing_low_4h = get_last_two(swings_4h[0], idx_4h)[0], get_last_two(swings_4h[1], idx_4h)[0]
        if swing_high_4h and swing_low_4h:
            equilibrium = (swing_high_4h.price + swing_low_4h.price) / 2
            discount_premium = entry_price < equilibrium if direction == "long" else entry_price > equilibrium
        else:
            discount_premium = False

        # Check session confluence
        level = get_level_for_bar(levels, entry_bar)
        session_conf = False
        if level:
            threshold = 10 * tick_size  # Slightly wider threshold
            for key in ["asia_high", "asia_low", "london_high", "london_low", "t_open"]:
                value = level.get(key)
                if value and abs(entry_price - value) <= threshold:
                    session_conf = True

        trend_align = trend_1h == trend_4h
        optimal_time = is_optimal_trading_time(entry_bar)
        
        # Determine HTF event reason for logging
        htf_event_reason = ""
        if htf_sweep:
            htf_event_reason = "sweep"
        elif htf_fvg_touch:
            htf_event_reason = "fvg_touch"
        elif htf_ob_touch:
            htf_event_reason = "ob_touch"

        # Calculate confidence score using config weights
        score = confidence_score(
            trend_align=trend_align,
            htf_sweep=htf_sweep,
            ltf_bos=True,  # Always true since we required BOS above
            htf_fvg_touch=htf_fvg_touch or htf_ob_touch,
            discount_premium=discount_premium,
            session_conf=session_conf,
            optimal_time=optimal_time,
            config=config,
        )

        # MINIMUM SCORE FILTER using config threshold
        if score < config.min_score_filter:
            continue

        # QUALITY FILTER: For high-confidence trades, require discount/premium alignment
        if score > config.tier_medium_max and not discount_premium:
            score = config.tier_medium_max  # Downgrade to medium if not in discount/premium zone

        tier = tier_from_score(score, config)

        # Use config for all trade parameters
        stop_distance = config.stop_ticks * tick_size
        if direction == "long":
            stop = entry_price - stop_distance
        else:
            stop = entry_price + stop_distance

        r_target = config.get_r_target(tier)
        target = entry_price + r_target * stop_distance if direction == "long" else entry_price - r_target * stop_distance

        be_ticks = config.get_be_ticks(tier)
        be_distance = be_ticks * tick_size
        moved_to_be = False
        
        # Track execution timeframe used
        execution_tf = "5m" if use_5m else "15m"

        # Simulate trade execution
        exit_price = None
        exit_k = entry_index
        for k in range(entry_index + 1, len(execution_bars)):
            b = execution_bars[k]
            if not moved_to_be:
                if direction == "long" and b.high >= entry_price + be_distance:
                    stop = entry_price
                    moved_to_be = True
                if direction == "short" and b.low <= entry_price - be_distance:
                    stop = entry_price
                    moved_to_be = True
            hit_stop = b.low <= stop if direction == "long" else b.high >= stop
            hit_target = b.high >= target if direction == "long" else b.low <= target
            if hit_stop and hit_target:
                exit_price = stop
                exit_k = k
                break
            if hit_stop:
                exit_price = stop
                exit_k = k
                break
            if hit_target:
                exit_price = target
                exit_k = k
                break
        if exit_price is None:
            exit_price = execution_bars[-1].close
            exit_k = len(execution_bars) - 1

        # Update cooldown: find corresponding 5m bar index for exit
        exit_time = execution_bars[exit_k].timestamp
        exit_5m_idx = bar_index_for_time(bars_5m, exit_time)
        last_exit_index = exit_5m_idx

        # Create trade with all metadata
        trade = Trade(
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
            risk_pct=config.get_risk_pct(tier),
            execution_tf=execution_tf,
            htf_event_reason=htf_event_reason,
            entry_zone_type=entry_zone_type or "",
        )
        
        # Compute PnL with costs
        trade.compute_pnl(config)
        trades.append(trade)

    return trades


def simulate_trades_with_smt(
    primary_symbol: str,
    primary_bars_1m: list[Bar],
    secondary_bars_1m: list[Bar],
    config: Optional[StrategyConfig] = None,
) -> list[Trade]:
    """
    Simulate trades on primary asset using SMT divergence with secondary asset.

    Args:
        primary_symbol: Symbol for primary asset (e.g., "NQ" or "SPY")
        primary_bars_1m: 1-minute bars for primary asset
        secondary_bars_1m: 1-minute bars for secondary/correlated asset
        config: Strategy configuration (uses futures defaults if None)

    Returns:
        List of trades with SMT confluence factored into confidence
    """
    if config is None:
        from strategy_config import get_futures_config
        config = get_futures_config()
    
    tick_size = config.tick_size
    
    # Resample primary
    bars_5m = resample_bars(primary_bars_1m, 5)
    bars_15m = resample_bars(primary_bars_1m, 15)
    bars_1h = resample_bars(primary_bars_1m, 60)
    bars_4h = resample_bars(primary_bars_1m, 240)

    # Resample secondary for SMT comparison
    secondary_5m = resample_bars(secondary_bars_1m, 5)
    secondary_1h = resample_bars(secondary_bars_1m, 60)

    if len(bars_4h) < 5 or len(bars_1h) < 10:
        return []

    # Pre-compute indicators for primary
    swings_1h = compute_swings(bars_1h)
    swings_4h = compute_swings(bars_4h)
    swings_5m = compute_swings(bars_5m)
    swings_15m = compute_swings(bars_15m)

    fvgs_1h = detect_fvgs(bars_1h)
    fvgs_4h = detect_fvgs(bars_4h)
    fvgs_5m = detect_fvgs(bars_5m)
    fvgs_15m = detect_fvgs(bars_15m)

    sweeps_1h = detect_sweeps(bars_1h, *swings_1h)
    sweeps_4h = detect_sweeps(bars_4h, *swings_4h)

    obs_1h = detect_order_blocks(bars_1h, *swings_1h)
    obs_4h = detect_order_blocks(bars_4h, *swings_4h)
    obs_5m = detect_order_blocks(bars_5m, *swings_5m)
    obs_15m = detect_order_blocks(bars_15m, *swings_15m)

    # Compute secondary swings for SMT
    secondary_swings_1h = compute_swings(secondary_1h)
    secondary_swings_5m = compute_swings(secondary_5m)

    # Detect SMT divergences on 1H timeframe
    smt_signals_1h = detect_smt_divergence(
        bars_1h, secondary_1h, swings_1h, secondary_swings_1h, lookback=10
    )

    levels = session_levels(primary_bars_1m)

    trades: list[Trade] = []
    last_exit_index = -1

    def bar_index_for_time(bars: list[Bar], t: datetime) -> int:
        idx = 0
        for i, bar in enumerate(bars):
            if bar.timestamp <= t:
                idx = i
            else:
                break
        return idx

    for i in range(3, len(bars_5m)):
        t = bars_5m[i].timestamp
        bar_5m = bars_5m[i]

        if i <= last_exit_index + 6:
            continue

        if not is_ny_session(bar_5m):
            continue

        idx_1h = bar_index_for_time(bars_1h, t)
        idx_4h = bar_index_for_time(bars_4h, t)

        trend_1h = trend_at(bars_1h, *swings_1h, idx_1h)
        trend_4h = trend_at(bars_4h, *swings_4h, idx_4h)
        if trend_1h == "none" or trend_4h == "none":
            continue

        use_5m = trend_1h == trend_4h
        execution_bars = bars_5m if use_5m else bars_15m
        exec_swings = swings_5m if use_5m else swings_15m
        exec_fvgs = fvgs_5m if use_5m else fvgs_15m
        exec_obs = obs_5m if use_5m else obs_15m

        exec_index = bar_index_for_time(execution_bars, t)
        if exec_index < 3:
            continue

        swing_highs, swing_lows = exec_swings
        last_high, _ = get_last_two(swing_highs, exec_index)
        last_low, _ = get_last_two(swing_lows, exec_index)
        if not last_high or not last_low:
            continue

        bar = execution_bars[exec_index]
        bos_long = bar.close > last_high.price
        bos_short = bar.close < last_low.price
        if not bos_long and not bos_short:
            continue

        direction = "long" if bos_long else "short"

        # Detect HTF events
        htf_sweep = False
        htf_fvg_touch = False
        htf_ob_touch = False

        for htf_idx in range(max(0, idx_1h - 10), idx_1h + 1):
            if htf_idx in sweeps_1h:
                htf_sweep = True
            for fvg in fvgs_1h:
                if fvg.index <= htf_idx and fvg_tapped(bars_1h[htf_idx], fvg):
                    htf_fvg_touch = True
            for ob in obs_1h:
                if ob.index <= htf_idx and ob_tapped(bars_1h[htf_idx], ob):
                    htf_ob_touch = True

        for htf_idx in range(max(0, idx_4h - 10), idx_4h + 1):
            if htf_idx in sweeps_4h:
                htf_sweep = True
            for fvg in fvgs_4h:
                if fvg.index <= htf_idx and fvg_tapped(bars_4h[htf_idx], fvg):
                    htf_fvg_touch = True
            for ob in obs_4h:
                if ob.index <= htf_idx and ob_tapped(bars_4h[htf_idx], ob):
                    htf_ob_touch = True

        htf_event = htf_sweep or htf_fvg_touch or htf_ob_touch
        if not htf_event:
            continue

        # Check for SMT divergence confirmation
        smt_confirmed = check_smt_at_index(
            smt_signals_1h,
            idx_1h,
            "bull" if direction == "long" else "bear",
            lookback=5
        )

        # Find entry zone (must be within last 5 bars, not future bars)
        ltf_fvgs = exec_fvgs
        ltf_obs = exec_obs

        entry_zone = None
        entry_zone_type = None
        for fvg in ltf_fvgs:
            if exec_index - 5 <= fvg.index <= exec_index and fvg.direction == ("bull" if direction == "long" else "bear"):
                entry_zone = (fvg.low, fvg.high)
                entry_zone_type = "fvg"
                break

        if not entry_zone:
            for ob in ltf_obs:
                if exec_index - 5 <= ob.index <= exec_index and ob.direction == ("bull" if direction == "long" else "bear"):
                    entry_zone = (ob.low, ob.high)
                    entry_zone_type = "ob"
                    break

        if not entry_zone:
            continue

        entry_index = None
        for j in range(exec_index, min(len(execution_bars), exec_index + 10)):
            if execution_bars[j].low <= entry_zone[1] and execution_bars[j].high >= entry_zone[0]:
                entry_index = j
                break

        if entry_index is None:
            continue

        entry_bar = execution_bars[entry_index]
        entry_price = (entry_zone[0] + entry_zone[1]) / 2

        # Discount/Premium calculation
        swing_high_4h, swing_low_4h = get_last_two(swings_4h[0], idx_4h)[0], get_last_two(swings_4h[1], idx_4h)[0]
        if swing_high_4h and swing_low_4h:
            equilibrium = (swing_high_4h.price + swing_low_4h.price) / 2
            discount_premium = entry_price < equilibrium if direction == "long" else entry_price > equilibrium
        else:
            discount_premium = False

        # Session confluence
        level = get_level_for_bar(levels, entry_bar)
        session_conf = False
        if level:
            threshold = 10 * tick_size
            for key in ["asia_high", "asia_low", "london_high", "london_low", "t_open"]:
                value = level.get(key)
                if value and abs(entry_price - value) <= threshold:
                    session_conf = True

        trend_align = trend_1h == trend_4h
        optimal_time = is_optimal_trading_time(entry_bar)
        
        # Determine HTF event reason
        htf_event_reason = ""
        if htf_sweep:
            htf_event_reason = "sweep"
        elif htf_fvg_touch:
            htf_event_reason = "fvg_touch"
        elif htf_ob_touch:
            htf_event_reason = "ob_touch"

        # Calculate confidence score WITH SMT using config weights
        score = confidence_score(
            trend_align=trend_align,
            htf_sweep=htf_sweep,
            ltf_bos=True,
            htf_fvg_touch=htf_fvg_touch or htf_ob_touch,
            discount_premium=discount_premium,
            session_conf=session_conf,
            smt_divergence=smt_confirmed,
            optimal_time=optimal_time,
            config=config,
        )

        # Minimum score filter using config
        if score < config.min_score_filter:
            continue

        # Quality filter for high tier
        if score > config.tier_medium_max and not discount_premium:
            score = config.tier_medium_max

        tier = tier_from_score(score, config)

        # Trade execution using config parameters
        stop_distance = config.stop_ticks * tick_size
        if direction == "long":
            stop = entry_price - stop_distance
        else:
            stop = entry_price + stop_distance

        r_target = config.get_r_target(tier)
        target = entry_price + r_target * stop_distance if direction == "long" else entry_price - r_target * stop_distance

        be_ticks = config.get_be_ticks(tier)
        be_distance = be_ticks * tick_size
        moved_to_be = False
        
        execution_tf = "5m" if use_5m else "15m"

        exit_price = None
        exit_k = entry_index
        for k in range(entry_index + 1, len(execution_bars)):
            b = execution_bars[k]
            if not moved_to_be:
                if direction == "long" and b.high >= entry_price + be_distance:
                    stop = entry_price
                    moved_to_be = True
                if direction == "short" and b.low <= entry_price - be_distance:
                    stop = entry_price
                    moved_to_be = True
            hit_stop = b.low <= stop if direction == "long" else b.high >= stop
            hit_target = b.high >= target if direction == "long" else b.low <= target
            if hit_stop and hit_target:
                exit_price = stop
                exit_k = k
                break
            if hit_stop:
                exit_price = stop
                exit_k = k
                break
            if hit_target:
                exit_price = target
                exit_k = k
                break
        if exit_price is None:
            exit_price = execution_bars[-1].close
            exit_k = len(execution_bars) - 1

        exit_time = execution_bars[exit_k].timestamp
        exit_5m_idx = bar_index_for_time(bars_5m, exit_time)
        last_exit_index = exit_5m_idx

        # Create trade with all metadata
        trade = Trade(
            symbol=primary_symbol,
            direction=direction,
            entry_time=entry_bar.timestamp,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            stop=stop,
            target=target,
            tier=tier,
            confidence_score=score,
            risk_pct=config.get_risk_pct(tier),
            execution_tf=execution_tf,
            htf_event_reason=htf_event_reason,
            entry_zone_type=entry_zone_type or "",
        )
        
        # Compute PnL with costs
        trade.compute_pnl(config)
        trades.append(trade)

    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"count": 0}
    wins = [t for t in trades if t.r_multiple > 0]
    avg_r = sum(t.r_multiple for t in trades) / len(trades)
    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_r": avg_r,
    }


def summarize_by_tier(trades: list[Trade]) -> dict[str, dict]:
    tiers: dict[str, list[Trade]] = {}
    for trade in trades:
        tiers.setdefault(trade.tier, []).append(trade)
    return {tier: summarize(items) for tier, items in tiers.items()}


def estimate_equity(trades: list[Trade], starting_equity: float = 1.0) -> tuple[float, float, float, float, float]:
    """
    Estimate equity curve with both simple and compounded returns.
    Returns: (simple_final, compounded_final, max_drawdown_pct, profit_factor, total_r)
    
    Simple: Fixed position sizing based on starting equity (no compounding)
    Compounded: Position sizing based on current equity (realistic compounding)
    """
    # Simple equity (no compounding - fixed risk based on starting equity)
    simple_equity = starting_equity
    simple_peak = starting_equity
    simple_max_dd = 0.0
    
    # Compounded equity (risk based on current equity)
    compound_equity = starting_equity
    compound_peak = starting_equity
    compound_max_dd = 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    total_r = 0.0

    for trade in sorted(trades, key=lambda t: t.entry_time):
        total_r += trade.r_multiple
        
        # Simple P&L: always risk percentage of STARTING equity
        simple_pnl = starting_equity * trade.risk_pct * trade.r_multiple
        simple_equity += simple_pnl
        
        # Compounded P&L: risk percentage of CURRENT equity
        compound_pnl = compound_equity * trade.risk_pct * trade.r_multiple
        compound_equity += compound_pnl

        # Track gross profit/loss for profit factor (using simple for consistency)
        if simple_pnl > 0:
            gross_profit += simple_pnl
        else:
            gross_loss += abs(simple_pnl)

        # Track simple drawdown
        if simple_equity > simple_peak:
            simple_peak = simple_equity
        simple_dd = (simple_peak - simple_equity) / simple_peak if simple_peak > 0 else 0
        simple_max_dd = max(simple_max_dd, simple_dd)
        
        # Track compounded drawdown
        if compound_equity > compound_peak:
            compound_peak = compound_equity
        compound_dd = (compound_peak - compound_equity) / compound_peak if compound_peak > 0 else 0
        compound_max_dd = max(compound_max_dd, compound_dd)

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    # Return simple max drawdown as the primary metric (more conservative)
    return simple_equity, compound_equity, simple_max_dd, profit_factor, total_r


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ruleset on available data.")
    parser.add_argument("--except", dest="exclude", default="", help="Comma-separated symbols to exclude")
    parser.add_argument("--time", dest="year", type=int, default=None, help="Filter bars by year (e.g., 2024)")
    parser.add_argument("--smt", nargs=2, metavar=("PRIMARY", "SECONDARY"),
                        help="Run SMT backtest with paired assets (e.g., --smt SPY QQQ)")
    parser.add_argument("--tick-size", type=float, default=0.01,
                        help="Tick size for the instrument (default 0.01 for stocks, use 0.25 for NQ/ES)")
    parser.add_argument("--slippage", type=float, default=1.0,
                        help="Expected slippage in ticks per side (default 1.0)")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="Bid-ask spread in ticks (default 1.0)")
    parser.add_argument("--commission", type=float, default=0.0,
                        help="Commission per contract round-trip (default 0.0 for stocks)")
    args = parser.parse_args()

    # Create strategy config from command line args
    config = StrategyConfig(
        tick_size=args.tick_size,
        slippage_ticks=args.slippage,
        spread_ticks=args.spread,
        commission_per_contract=args.commission,
    )
    
    # Validate config
    errors = config.validate()
    if errors:
        print(f"Configuration errors: {errors}")
        return
    
    print(f"Config hash: {config.compute_config_hash()}")
    print(f"Cost per trade: {config.compute_cost_in_r():.3f}R ({config.compute_cost_in_ticks():.1f} ticks)")

    root = Path(__file__).resolve().parents[1]
    polygon_files = list((root / "data" / "polygon").glob("*.csv"))
    tick_files = list(root.glob("rust_backend/tick_history_*.json"))
    if not polygon_files and not tick_files:
        print("No Polygon CSVs or tick history files found.")
        return

    excluded = {symbol.strip().upper() for symbol in args.exclude.split(",") if symbol.strip()}

    all_trades: list[Trade] = []
    symbol_results: list[tuple[str, dict]] = []

    # SMT Mode: Run backtest with paired assets
    if args.smt:
        primary_symbol, secondary_symbol = args.smt[0].upper(), args.smt[1].upper()
        print(f"Running SMT backtest: {primary_symbol} (primary) vs {secondary_symbol} (secondary)")

        # Find data files for both symbols
        primary_files = [f for f in polygon_files if f.stem.split("_")[0].upper() == primary_symbol]
        secondary_files = [f for f in polygon_files if f.stem.split("_")[0].upper() == secondary_symbol]

        if not primary_files:
            print(f"No data found for primary symbol: {primary_symbol}")
            return
        if not secondary_files:
            print(f"No data found for secondary symbol: {secondary_symbol}")
            return

        # Match files by year
        for p_file in primary_files:
            p_year = p_file.stem.split("_")[1]  # Extract year from filename
            matching_s_files = [f for f in secondary_files if p_year in f.stem]
            if not matching_s_files:
                print(f"No matching secondary data for {p_file.name}")
                continue

            s_file = matching_s_files[0]
            print(f"  Processing: {p_file.name} + {s_file.name}")

            primary_bars = load_polygon_csv(p_file)
            secondary_bars = load_polygon_csv(s_file)

            primary_bars = filter_bars_by_year(primary_bars, args.year)
            secondary_bars = filter_bars_by_year(secondary_bars, args.year)

            if not primary_bars or not secondary_bars:
                continue

            trades = simulate_trades_with_smt(
                primary_symbol, primary_bars, secondary_bars, config=config
            )
            all_trades.extend(trades)

            file_year = p_file.stem.split("_")[1][:4]
            symbol_results.append((f"{primary_symbol}_{file_year}", summarize(trades)))

    # Standard Mode: Run backtest on individual symbols
    elif polygon_files:
        for file in polygon_files:
            symbol = file.stem.split("_")[0]
            if symbol.upper() in excluded:
                continue
            bars_1m = load_polygon_csv(file)
            bars_1m = filter_bars_by_year(bars_1m, args.year)
            trades = simulate_trades(symbol, bars_1m, config=config)
            all_trades.extend(trades)
            symbol_results.append((symbol, summarize(trades)))
    else:
        for file in tick_files:
            data = json.loads(file.read_text())
            if not data:
                continue
            symbol = data[0].get("symbol", file.stem.replace("tick_history_", ""))
            if symbol.upper() in excluded:
                continue
            bars_1m = to_minute_bars(data)
            bars_1m = filter_bars_by_year(bars_1m, args.year)
            trades = simulate_trades(symbol, bars_1m, config=config)
            all_trades.extend(trades)
            symbol_results.append((symbol, summarize(trades)))

    summary = summarize(all_trades)
    tier_summary = summarize_by_tier(all_trades)
    simple_equity, compound_equity, max_dd, profit_factor, total_r = estimate_equity(all_trades, starting_equity=10000.0)

    output = ["# Backtest Results", ""]
    output.append("## Summary")
    if summary.get("count", 0) == 0:
        output.append("- No trades generated with current rules and data.")
    else:
        output.append(f"- Trades: {summary['count']}")
        output.append(f"- Win rate: {summary['win_rate']:.2%}")
        output.append(f"- Avg R: {summary['avg_r']:.2f}")
        output.append(f"- Total R: {total_r:.2f}")
        output.append(f"- Profit factor: {profit_factor:.2f}")
        output.append(f"- Max drawdown: {max_dd:.2%}")
        output.append(f"- Simple equity (from $10k): ${simple_equity:,.2f}")
        output.append(f"- Compounded equity (from $10k): ${compound_equity:,.2f}")

    output.append("")
    output.append("## Per Symbol")
    for symbol, stats in sorted(symbol_results, key=lambda item: item[0]):
        if stats.get("count", 0) == 0:
            output.append(f"- {symbol}: 0 trades")
        else:
            output.append(
                f"- {symbol}: {stats['count']} trades, win {stats['win_rate']:.2%}, avg R {stats['avg_r']:.2f}"
            )

    output.append("")
    output.append("## By Confidence Tier")
    for tier, stats in tier_summary.items():
        if stats.get("count", 0) == 0:
            output.append(f"- {tier}: 0 trades")
        else:
            output.append(
                f"- {tier}: {stats['count']} trades, win {stats['win_rate']:.2%}, avg R {stats['avg_r']:.2f}"
            )

    output.append("")
    output.append("## Notes")
    output.append("- Uses Polygon CSVs if present; otherwise tick-history JSON.")
    output.append("- NY session filter: Only trades during 8:00-16:00 ET.")
    output.append("- HTF events: Liquidity sweeps, FVG taps, or Order Block taps on 1H/4H.")
    output.append("- Entry zones: LTF FVGs or Order Blocks on execution timeframe (5m/15m).")
    output.append("- Confidence scoring (0-10): trend alignment (+2), HTF sweep (+2), BOS (+2), FVG/OB touch (+1), discount/premium (+1), session confluence (+1), optimal time (+1).")
    output.append("- Tiers: Low (0-3), Medium (4-6), High (7+).")
    output.append("- Stop: 50 ticks; TP based on tier (2R/3R/4R).")
    output.append("- Break-even applied per tier; runners not simulated.")
    output.append("- Tier risk sizing: low 0.05%, medium 0.10%, high 0.50%.")
    output.append("- Simple equity: Fixed position sizing (no compounding) - realistic baseline.")
    output.append("- Compounded equity: Position sizing grows with equity - best case scenario.")

    results_path = root / "backtest_docs" / "backtest_results.md"
    results_path.write_text("\n".join(output))
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
