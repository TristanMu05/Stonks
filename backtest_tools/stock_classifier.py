"""
Stock Classification and Clustering System

Groups stocks by:
1. Sector/Industry
2. Statistical characteristics (volatility, trend strength, mean reversion)
3. Correlation clusters
4. Component relationships (individual stocks -> ETFs)

Uses these classifications to:
- Select appropriate trading strategy
- Generate cross-asset signals
- Filter trades based on regime
"""

import csv
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StockProfile:
    """Statistical profile of a stock's behavior"""
    symbol: str

    # Classification
    sector: str
    category: str  # "trending", "mean_reverting", "mixed"

    # Volatility metrics
    avg_daily_range_pct: float  # Average (high-low)/close
    volatility_20d: float       # 20-day realized volatility
    atr_pct: float              # ATR as % of price

    # Trend metrics
    trend_strength: float       # ADX-like metric (0-100)
    trend_consistency: float    # % of days price closes in trend direction
    avg_swing_size: float       # Average swing high to swing low

    # Mean reversion metrics
    mean_reversion_score: float  # How quickly price returns to mean
    range_bound_score: float     # % of time in defined range

    # Correlation data
    correlations: dict = field(default_factory=dict)  # symbol -> correlation

    # Strategy recommendation
    recommended_strategy: str = ""
    confidence: float = 0.0


# Manual sector classification for our symbols
SECTOR_MAP = {
    # Tech - High Growth
    "NVDA": ("Technology", "Semiconductors/AI"),
    "CRWD": ("Technology", "Cybersecurity"),
    "PANW": ("Technology", "Cybersecurity"),

    # Tech - Large Cap Mature
    "MSFT": ("Technology", "Software/Cloud"),
    "AAPL": ("Technology", "Consumer Electronics"),
    "GOOG": ("Technology", "Internet/Advertising"),
    "META": ("Technology", "Internet/Social"),

    # Tech ETF
    "QQQ": ("ETF", "Tech-Heavy Index"),

    # Broad Market
    "SPY": ("ETF", "Broad Market Index"),

    # High Volatility Growth
    "TSLA": ("Consumer Discretionary", "EV/High Volatility"),
    "SOFI": ("Financials", "Fintech/High Volatility"),

    # Traditional Financials
    "BAC": ("Financials", "Banking"),
    "C": ("Financials", "Banking"),

    # Industrials
    "BA": ("Industrials", "Aerospace"),
    "CAT": ("Industrials", "Machinery"),

    # Consumer/Defensive
    "WMT": ("Consumer Staples", "Retail"),

    # Energy
    "XOM": ("Energy", "Oil & Gas"),
}

# ETF component relationships
ETF_COMPONENTS = {
    "QQQ": ["AAPL", "MSFT", "NVDA", "META", "GOOG", "TSLA", "CRWD", "PANW"],
    "SPY": ["AAPL", "MSFT", "NVDA", "META", "GOOG", "TSLA", "BA", "CAT", "WMT", "XOM", "BAC", "C"],
}


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


def resample_daily(bars: list[Bar]) -> list[Bar]:
    """Resample to daily bars"""
    if not bars:
        return []

    grouped: dict[str, list[Bar]] = {}
    for bar in bars:
        date_key = bar.timestamp.strftime("%Y-%m-%d")
        grouped.setdefault(date_key, []).append(bar)

    daily = []
    for date_key in sorted(grouped.keys()):
        group = grouped[date_key]
        daily.append(Bar(
            timestamp=group[0].timestamp.replace(hour=0, minute=0, second=0),
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
        ))
    return daily


def compute_returns(bars: list[Bar]) -> np.ndarray:
    """Compute daily returns"""
    closes = np.array([b.close for b in bars])
    returns = np.diff(closes) / closes[:-1]
    return returns


def compute_volatility(returns: np.ndarray, window: int = 20) -> float:
    """Annualized volatility"""
    if len(returns) < window:
        return np.std(returns) * np.sqrt(252)
    return np.std(returns[-window:]) * np.sqrt(252)


def compute_avg_daily_range(bars: list[Bar]) -> float:
    """Average daily range as % of close"""
    ranges = [(b.high - b.low) / b.close for b in bars if b.close > 0]
    return np.mean(ranges) if ranges else 0


def compute_trend_strength(bars: list[Bar], period: int = 14) -> float:
    """
    ADX-like trend strength metric (0-100).
    Higher = stronger trend, Lower = range-bound.
    """
    if len(bars) < period + 1:
        return 50.0

    # Compute directional movement
    plus_dm = []
    minus_dm = []
    tr = []

    for i in range(1, len(bars)):
        high_diff = bars[i].high - bars[i-1].high
        low_diff = bars[i-1].low - bars[i].low

        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

        tr_val = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        tr.append(tr_val)

    if not tr or sum(tr[-period:]) == 0:
        return 50.0

    # Smoothed averages
    atr = np.mean(tr[-period:])
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0

    # ADX calculation
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 50.0

    dx = 100 * abs(plus_di - minus_di) / di_sum
    return min(100, dx)


def compute_trend_consistency(bars: list[Bar], lookback: int = 50) -> float:
    """
    What % of days does price close in the direction of the overall trend?
    Higher = more consistent trending behavior.
    """
    if len(bars) < lookback:
        return 0.5

    recent_bars = bars[-lookback:]
    overall_direction = 1 if recent_bars[-1].close > recent_bars[0].close else -1

    consistent_days = 0
    for i in range(1, len(recent_bars)):
        daily_direction = 1 if recent_bars[i].close > recent_bars[i-1].close else -1
        if daily_direction == overall_direction:
            consistent_days += 1

    return consistent_days / (len(recent_bars) - 1)


def compute_mean_reversion_score(returns: np.ndarray) -> float:
    """
    Hurst exponent approximation.
    < 0.5 = mean reverting
    = 0.5 = random walk
    > 0.5 = trending

    Returns a score where higher = more mean reverting.
    """
    if len(returns) < 20:
        return 0.5

    # Simple autocorrelation-based estimate
    # Negative autocorrelation suggests mean reversion
    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

    # Convert to 0-1 scale where higher = more mean reverting
    # autocorr of -1 = very mean reverting, +1 = very trending
    mean_reversion_score = (1 - autocorr) / 2
    return mean_reversion_score


def compute_range_bound_score(bars: list[Bar], lookback: int = 50) -> float:
    """
    What % of time is price within a defined range?
    Higher = more range-bound behavior.
    """
    if len(bars) < lookback:
        return 0.5

    recent_bars = bars[-lookback:]
    closes = [b.close for b in recent_bars]

    # Define range as 20th to 80th percentile
    p20 = np.percentile(closes, 20)
    p80 = np.percentile(closes, 80)

    in_range = sum(1 for c in closes if p20 <= c <= p80)
    return in_range / len(closes)


def compute_correlations(symbol_bars: dict[str, list[Bar]]) -> dict[str, dict[str, float]]:
    """Compute pairwise correlations between all symbols"""

    # Get common date range
    all_dates = None
    for sym, bars in symbol_bars.items():
        dates = set(b.timestamp.strftime("%Y-%m-%d") for b in bars)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates &= dates

    if not all_dates or len(all_dates) < 20:
        return {}

    # Build return series for common dates
    returns_by_symbol = {}
    for sym, bars in symbol_bars.items():
        daily = resample_daily(bars)
        date_to_close = {b.timestamp.strftime("%Y-%m-%d"): b.close for b in daily}

        sorted_dates = sorted(all_dates)
        closes = [date_to_close.get(d, np.nan) for d in sorted_dates]
        closes = np.array(closes)

        # Compute returns
        returns = np.diff(closes) / closes[:-1]
        returns_by_symbol[sym] = returns

    # Compute correlations
    correlations = {}
    symbols = list(returns_by_symbol.keys())

    for i, sym1 in enumerate(symbols):
        correlations[sym1] = {}
        for sym2 in symbols:
            if sym1 == sym2:
                correlations[sym1][sym2] = 1.0
            else:
                r1 = returns_by_symbol[sym1]
                r2 = returns_by_symbol[sym2]
                # Handle NaN
                mask = ~(np.isnan(r1) | np.isnan(r2))
                if mask.sum() > 10:
                    correlations[sym1][sym2] = np.corrcoef(r1[mask], r2[mask])[0, 1]
                else:
                    correlations[sym1][sym2] = 0.0

    return correlations


def classify_stock(symbol: str, bars: list[Bar], correlations: dict) -> StockProfile:
    """Create a complete profile for a stock"""

    daily = resample_daily(bars)
    returns = compute_returns(daily)

    sector, subsector = SECTOR_MAP.get(symbol, ("Unknown", "Unknown"))

    # Compute metrics
    volatility = compute_volatility(returns)
    avg_range = compute_avg_daily_range(daily)
    trend_strength = compute_trend_strength(daily)
    trend_consistency = compute_trend_consistency(daily)
    mean_reversion = compute_mean_reversion_score(returns)
    range_bound = compute_range_bound_score(daily)

    # Determine category
    if trend_strength > 30 and trend_consistency > 0.55 and mean_reversion < 0.55:
        category = "trending"
    elif mean_reversion > 0.55 or range_bound > 0.7:
        category = "mean_reverting"
    else:
        category = "mixed"

    # Strategy recommendation
    if category == "trending":
        strategy = "momentum_breakout"
        confidence = min(1.0, (trend_strength / 50) * trend_consistency)
    elif category == "mean_reverting":
        strategy = "mean_reversion"
        confidence = mean_reversion * range_bound
    else:
        strategy = "adaptive"
        confidence = 0.5

    return StockProfile(
        symbol=symbol,
        sector=sector,
        category=category,
        avg_daily_range_pct=avg_range,
        volatility_20d=volatility,
        atr_pct=avg_range,
        trend_strength=trend_strength,
        trend_consistency=trend_consistency,
        avg_swing_size=avg_range * 5,  # Rough estimate
        mean_reversion_score=mean_reversion,
        range_bound_score=range_bound,
        correlations=correlations.get(symbol, {}),
        recommended_strategy=strategy,
        confidence=confidence,
    )


def identify_clusters(profiles: list[StockProfile]) -> dict[str, list[str]]:
    """Group stocks into trading clusters"""

    clusters = {
        "high_momentum_tech": [],      # NVDA, CRWD, PANW - our winners
        "large_cap_tech": [],          # MSFT, AAPL, GOOG, META
        "tech_etf": [],                # QQQ
        "broad_market": [],            # SPY
        "high_volatility": [],         # TSLA, SOFI
        "financials": [],              # BAC, C
        "industrials": [],             # BA, CAT
        "defensive": [],               # WMT
        "energy": [],                  # XOM
    }

    for p in profiles:
        sym = p.symbol

        # ETFs
        if sym == "QQQ":
            clusters["tech_etf"].append(sym)
        elif sym == "SPY":
            clusters["broad_market"].append(sym)

        # High momentum tech (our winners)
        elif sym in ["NVDA", "CRWD", "PANW"] or (
            p.sector == "Technology" and
            p.category == "trending" and
            p.trend_strength > 25
        ):
            clusters["high_momentum_tech"].append(sym)

        # Large cap tech
        elif p.sector == "Technology" and sym in ["MSFT", "AAPL", "GOOG", "META"]:
            clusters["large_cap_tech"].append(sym)

        # High volatility
        elif p.volatility_20d > 0.5 or sym in ["TSLA", "SOFI"]:
            clusters["high_volatility"].append(sym)

        # Financials
        elif p.sector == "Financials":
            clusters["financials"].append(sym)

        # Industrials
        elif p.sector == "Industrials":
            clusters["industrials"].append(sym)

        # Defensive
        elif p.sector == "Consumer Staples":
            clusters["defensive"].append(sym)

        # Energy
        elif p.sector == "Energy":
            clusters["energy"].append(sym)

    return {k: v for k, v in clusters.items() if v}


def get_cross_asset_signals(
    profiles: list[StockProfile],
    symbol_bars: dict[str, list[Bar]],
    target_etf: str = "QQQ"
) -> dict:
    """
    Generate signals for ETF based on component stock behavior.

    Idea: If multiple QQQ components are showing strength, QQQ likely to follow.
    """

    if target_etf not in ETF_COMPONENTS:
        return {}

    components = ETF_COMPONENTS[target_etf]
    available_components = [c for c in components if c in symbol_bars]

    if len(available_components) < 3:
        return {}

    signals = {
        "bullish_components": [],
        "bearish_components": [],
        "neutral_components": [],
        "aggregate_signal": "neutral",
        "signal_strength": 0.0,
    }

    for comp in available_components:
        if comp not in symbol_bars:
            continue

        bars = symbol_bars[comp]
        daily = resample_daily(bars)

        if len(daily) < 20:
            continue

        # Simple trend check: is 5-day MA above 20-day MA?
        closes = [b.close for b in daily[-20:]]
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes)

        if ma5 > ma20 * 1.01:  # 1% above
            signals["bullish_components"].append(comp)
        elif ma5 < ma20 * 0.99:  # 1% below
            signals["bearish_components"].append(comp)
        else:
            signals["neutral_components"].append(comp)

    # Aggregate signal
    total = len(available_components)
    bullish_pct = len(signals["bullish_components"]) / total if total > 0 else 0
    bearish_pct = len(signals["bearish_components"]) / total if total > 0 else 0

    if bullish_pct > 0.6:
        signals["aggregate_signal"] = "bullish"
        signals["signal_strength"] = bullish_pct
    elif bearish_pct > 0.6:
        signals["aggregate_signal"] = "bearish"
        signals["signal_strength"] = bearish_pct
    else:
        signals["aggregate_signal"] = "neutral"
        signals["signal_strength"] = 0.5

    return signals


def analyze_all_stocks(data_dir: Path) -> tuple[list[StockProfile], dict, dict]:
    """
    Analyze all available stocks and return profiles, correlations, and clusters.
    """

    # Load all data
    symbol_bars = {}
    files = list(data_dir.glob("*.csv"))

    for f in files:
        sym = f.stem.split("_")[0].upper()
        bars = load_csv(f)
        if sym in symbol_bars:
            symbol_bars[sym].extend(bars)
        else:
            symbol_bars[sym] = bars

    # Sort bars by timestamp
    for sym in symbol_bars:
        symbol_bars[sym].sort(key=lambda b: b.timestamp)

    print(f"Loaded data for {len(symbol_bars)} symbols")

    # Compute correlations
    print("Computing correlations...")
    correlations = compute_correlations(symbol_bars)

    # Create profiles
    print("Creating stock profiles...")
    profiles = []
    for sym, bars in symbol_bars.items():
        profile = classify_stock(sym, bars, correlations)
        profiles.append(profile)

    # Identify clusters
    print("Identifying clusters...")
    clusters = identify_clusters(profiles)

    return profiles, correlations, clusters


def print_analysis(profiles: list[StockProfile], correlations: dict, clusters: dict):
    """Print analysis results"""

    print("\n" + "=" * 70)
    print("STOCK CLASSIFICATION ANALYSIS")
    print("=" * 70)

    # Print by category
    print("\n## By Category")
    trending = [p for p in profiles if p.category == "trending"]
    mean_rev = [p for p in profiles if p.category == "mean_reverting"]
    mixed = [p for p in profiles if p.category == "mixed"]

    print(f"\nTRENDING ({len(trending)}):")
    for p in sorted(trending, key=lambda x: -x.trend_strength):
        print(f"  {p.symbol:6} | Trend: {p.trend_strength:5.1f} | Consistency: {p.trend_consistency:.1%} | Strategy: {p.recommended_strategy}")

    print(f"\nMEAN REVERTING ({len(mean_rev)}):")
    for p in sorted(mean_rev, key=lambda x: -x.mean_reversion_score):
        print(f"  {p.symbol:6} | MR Score: {p.mean_reversion_score:.2f} | Range Bound: {p.range_bound_score:.1%} | Strategy: {p.recommended_strategy}")

    print(f"\nMIXED ({len(mixed)}):")
    for p in mixed:
        print(f"  {p.symbol:6} | Trend: {p.trend_strength:5.1f} | MR: {p.mean_reversion_score:.2f} | Strategy: {p.recommended_strategy}")

    # Print clusters
    print("\n## Trading Clusters")
    for cluster_name, symbols in clusters.items():
        if symbols:
            print(f"\n{cluster_name.upper().replace('_', ' ')}:")
            print(f"  {', '.join(symbols)}")

    # Print correlation highlights
    print("\n## High Correlations (>0.7)")
    printed = set()
    for sym1, corrs in correlations.items():
        for sym2, corr in corrs.items():
            if sym1 != sym2 and corr > 0.7:
                pair = tuple(sorted([sym1, sym2]))
                if pair not in printed:
                    print(f"  {sym1} <-> {sym2}: {corr:.2f}")
                    printed.add(pair)

    # Strategy recommendations
    print("\n## Strategy Recommendations")
    print("\nMOMENTUM BREAKOUT (use our winning strategy):")
    momentum_stocks = [p for p in profiles if p.recommended_strategy == "momentum_breakout"]
    for p in momentum_stocks:
        print(f"  {p.symbol:6} - Confidence: {p.confidence:.1%}")

    print("\nMEAN REVERSION (need different strategy):")
    mr_stocks = [p for p in profiles if p.recommended_strategy == "mean_reversion"]
    for p in mr_stocks:
        print(f"  {p.symbol:6} - Confidence: {p.confidence:.1%}")


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "polygon"

    if not data_dir.exists():
        print(f"No data directory at {data_dir}")
        return

    profiles, correlations, clusters = analyze_all_stocks(data_dir)
    print_analysis(profiles, correlations, clusters)

    # Save results
    output = ["# Stock Classification Results\n"]

    output.append("## Category Summary\n")
    for p in sorted(profiles, key=lambda x: (x.category, -x.trend_strength)):
        output.append(f"- **{p.symbol}**: {p.category} | {p.sector} | Strategy: {p.recommended_strategy}")

    output.append("\n## Clusters\n")
    for cluster, symbols in clusters.items():
        if symbols:
            output.append(f"### {cluster.replace('_', ' ').title()}")
            output.append(f"Symbols: {', '.join(symbols)}\n")

    output.append("\n## Correlations\n")
    output.append("| Symbol 1 | Symbol 2 | Correlation |")
    output.append("|----------|----------|-------------|")
    printed = set()
    for sym1, corrs in sorted(correlations.items()):
        for sym2, corr in sorted(corrs.items(), key=lambda x: -x[1]):
            if sym1 != sym2 and corr > 0.5:
                pair = tuple(sorted([sym1, sym2]))
                if pair not in printed:
                    output.append(f"| {sym1} | {sym2} | {corr:.2f} |")
                    printed.add(pair)

    results_path = root / "backtest_docs" / "stock_classification.md"
    results_path.write_text("\n".join(output))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
