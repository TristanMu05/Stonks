"""
Strategy Configuration Module

Defines all tunable parameters for the trading strategy, including:
- Instrument configuration (tick size, multiplier)
- Cost model (commissions, slippage, spread)
- Confidence scoring weights
- Tier thresholds and risk management
- Pattern detection parameters
- Session filters

All parameters have defined bounds for optimization and validation rules
to prevent invalid configurations.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration with all tunable parameters.
    
    Execution Model:
    - All signals computed on COMPLETED bars only (no lookahead)
    - Entries occur at NEXT bar OPEN after signal
    - PnL computed in ticks first, then converted to R
    """
    
    # === INSTRUMENT CONFIG ===
    tick_size: float = 0.01           # 0.01 for stocks, 0.25 for NQ/ES
    contract_multiplier: float = 1.0  # 1 for stocks, 20 for NQ, 50 for ES
    timezone: str = "America/New_York"
    
    # === COST MODEL ===
    commission_per_contract: float = 0.0   # e.g., 2.25 for futures round-trip
    slippage_ticks: float = 1.0            # Expected slippage per side
    spread_ticks: float = 1.0              # Bid-ask spread in ticks
    
    # === CONFIDENCE WEIGHTS (bounds: 0-4) ===
    weight_trend_align: int = 2
    weight_htf_sweep: int = 2
    weight_ltf_bos: int = 2
    weight_htf_fvg_touch: int = 1
    weight_discount_premium: int = 1
    weight_session_conf: int = 1
    weight_smt_divergence: int = 1
    weight_optimal_time: int = 1
    
    # === TIER THRESHOLDS ===
    tier_low_max: int = 3              # bounds: 1-5, scores <= this are "low"
    tier_medium_max: int = 6           # bounds: 4-8, scores <= this are "medium"
    min_score_filter: int = 4          # bounds: 2-6, minimum score to take trade
    
    # === RISK MANAGEMENT ===
    stop_ticks: int = 50               # bounds: 20-100
    r_target_low: float = 2.0          # bounds: 1.0-3.0
    r_target_medium: float = 3.0       # bounds: 2.0-4.0
    r_target_high: float = 4.0         # bounds: 3.0-6.0
    be_ticks_low: int = 50             # bounds: 30-80, move to BE after this profit
    be_ticks_medium: int = 75          # bounds: 50-100
    be_ticks_high: int = 100           # bounds: 75-150
    risk_pct_low: float = 0.0005       # bounds: 0.0001-0.001 (0.05%)
    risk_pct_medium: float = 0.001     # bounds: 0.0005-0.002 (0.1%)
    risk_pct_high: float = 0.005       # bounds: 0.001-0.01 (0.5%, CAPPED at 1%)
    
    # === PATTERN DETECTION ===
    htf_lookback: int = 10             # bounds: 5-20, bars to look back for HTF events
    entry_zone_lookback: int = 5       # bounds: 3-10, bars to find entry zone
    entry_wait_window: int = 10        # bounds: 5-20, bars to wait for zone tap
    sweep_lookback: int = 3            # bounds: 2-5, bars to detect sweep
    cooldown_bars: int = 6             # bounds: 3-12, bars between trades
    
    # === SESSION FILTERS ===
    ny_session_start: int = 8          # bounds: 7-9 (ET)
    ny_session_end: int = 16           # bounds: 15-17 (ET)
    optimal_am_start: int = 9          # Start of optimal morning window
    optimal_am_end: int = 12           # End of optimal morning window
    lunch_start: int = 12              # Lunch avoid start
    lunch_end: int = 13                # Lunch avoid end
    optimal_pm_start: int = 13         # Start of optimal afternoon window
    optimal_pm_end: int = 16           # End of optimal afternoon window
    
    # === SESSION DEFINITIONS (for session level tracking) ===
    asia_start_hour: int = 20          # 8 PM ET
    asia_end_hour: int = 3             # 3 AM ET
    london_start_hour: int = 3         # 3 AM ET
    london_end_hour: int = 8           # 8 AM ET
    
    # === VERSIONING ===
    config_version: int = 1            # Increment when params change
    ruleset_version: int = 1           # Increment when logic changes
    feature_schema_version: int = 1    # Increment when features change
    
    # === METADATA (set at export time) ===
    trained_on: str = ""               # Date range used for training
    train_symbols: list[str] = field(default_factory=list)
    holdout_symbols: list[str] = field(default_factory=list)
    median_daily_r: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    exported_at: str = ""
    python_commit: str = ""
    rust_commit: str = ""
    
    def __post_init__(self):
        """Set git commit if not provided."""
        if not self.python_commit:
            self.python_commit = get_git_commit()
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.
        Returns empty list if valid.
        """
        errors: list[str] = []
        
        # Tier thresholds must be ordered
        if self.tier_low_max >= self.tier_medium_max:
            errors.append("tier_low_max must be < tier_medium_max")
        
        # Min score filter should allow some trades
        if self.min_score_filter > self.tier_medium_max:
            errors.append("min_score_filter too high - only high tier trades possible")
        
        # Risk caps to prevent blowups
        if self.risk_pct_high > 0.01:
            errors.append("risk_pct_high capped at 1% to prevent blowups")
        if self.risk_pct_medium > self.risk_pct_high:
            errors.append("risk_pct_medium should not exceed risk_pct_high")
        if self.risk_pct_low > self.risk_pct_medium:
            errors.append("risk_pct_low should not exceed risk_pct_medium")
        
        # Stop must be reasonable
        if self.stop_ticks < 20:
            errors.append("stop_ticks too tight (<20) - will get stopped on noise")
        if self.stop_ticks > 200:
            errors.append("stop_ticks too wide (>200) - poor risk/reward")
        
        # R targets must be ordered
        if not (self.r_target_low <= self.r_target_medium <= self.r_target_high):
            errors.append("R targets must be ordered: low <= medium <= high")
        
        # BE thresholds must be ordered
        if not (self.be_ticks_low <= self.be_ticks_medium <= self.be_ticks_high):
            errors.append("BE ticks must be ordered: low <= medium <= high")
        
        # Session times must be valid
        if self.ny_session_start >= self.ny_session_end:
            errors.append("ny_session_start must be < ny_session_end")
        
        # Weights should be non-negative
        weight_fields = [
            "weight_trend_align", "weight_htf_sweep", "weight_ltf_bos",
            "weight_htf_fvg_touch", "weight_discount_premium", "weight_session_conf",
            "weight_smt_divergence", "weight_optimal_time"
        ]
        for field_name in weight_fields:
            value = getattr(self, field_name)
            if value < 0:
                errors.append(f"{field_name} cannot be negative")
            if value > 5:
                errors.append(f"{field_name} unusually high (>5)")
        
        # Tick size must be positive
        if self.tick_size <= 0:
            errors.append("tick_size must be positive")
        
        # Slippage/spread should be non-negative
        if self.slippage_ticks < 0:
            errors.append("slippage_ticks cannot be negative")
        if self.spread_ticks < 0:
            errors.append("spread_ticks cannot be negative")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def compute_config_hash(self) -> str:
        """
        Compute SHA256 hash of all tunable parameters (excluding metadata).
        Used for traceability between Python training and Rust execution.
        """
        # Include only parameters that affect trading logic
        params = {
            "tick_size": self.tick_size,
            "contract_multiplier": self.contract_multiplier,
            "timezone": self.timezone,
            "commission_per_contract": self.commission_per_contract,
            "slippage_ticks": self.slippage_ticks,
            "spread_ticks": self.spread_ticks,
            "weight_trend_align": self.weight_trend_align,
            "weight_htf_sweep": self.weight_htf_sweep,
            "weight_ltf_bos": self.weight_ltf_bos,
            "weight_htf_fvg_touch": self.weight_htf_fvg_touch,
            "weight_discount_premium": self.weight_discount_premium,
            "weight_session_conf": self.weight_session_conf,
            "weight_smt_divergence": self.weight_smt_divergence,
            "weight_optimal_time": self.weight_optimal_time,
            "tier_low_max": self.tier_low_max,
            "tier_medium_max": self.tier_medium_max,
            "min_score_filter": self.min_score_filter,
            "stop_ticks": self.stop_ticks,
            "r_target_low": self.r_target_low,
            "r_target_medium": self.r_target_medium,
            "r_target_high": self.r_target_high,
            "be_ticks_low": self.be_ticks_low,
            "be_ticks_medium": self.be_ticks_medium,
            "be_ticks_high": self.be_ticks_high,
            "risk_pct_low": self.risk_pct_low,
            "risk_pct_medium": self.risk_pct_medium,
            "risk_pct_high": self.risk_pct_high,
            "htf_lookback": self.htf_lookback,
            "entry_zone_lookback": self.entry_zone_lookback,
            "entry_wait_window": self.entry_wait_window,
            "sweep_lookback": self.sweep_lookback,
            "cooldown_bars": self.cooldown_bars,
            "ny_session_start": self.ny_session_start,
            "ny_session_end": self.ny_session_end,
            "asia_start_hour": self.asia_start_hour,
            "asia_end_hour": self.asia_end_hour,
            "london_start_hour": self.london_start_hour,
            "london_end_hour": self.london_end_hour,
            "ruleset_version": self.ruleset_version,
            "feature_schema_version": self.feature_schema_version,
        }
        
        # Stable JSON serialization
        json_str = json.dumps(params, sort_keys=True)
        return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()[:16]}"
    
    def get_r_target(self, tier: str) -> float:
        """Get R target for given tier."""
        if tier == "low":
            return self.r_target_low
        elif tier == "medium":
            return self.r_target_medium
        else:
            return self.r_target_high
    
    def get_be_ticks(self, tier: str) -> int:
        """Get break-even trigger in ticks for given tier."""
        if tier == "low":
            return self.be_ticks_low
        elif tier == "medium":
            return self.be_ticks_medium
        else:
            return self.be_ticks_high
    
    def get_risk_pct(self, tier: str) -> float:
        """Get risk percentage for given tier."""
        if tier == "low":
            return self.risk_pct_low
        elif tier == "medium":
            return self.risk_pct_medium
        else:
            return self.risk_pct_high
    
    def get_tier(self, score: int) -> str:
        """Determine tier from confidence score."""
        if score <= self.tier_low_max:
            return "low"
        elif score <= self.tier_medium_max:
            return "medium"
        else:
            return "high"
    
    def compute_confidence_score(
        self,
        trend_align: bool,
        htf_sweep: bool,
        ltf_bos: bool,
        htf_fvg_touch: bool,
        discount_premium: bool,
        session_conf: bool,
        smt_divergence: bool = False,
        optimal_time: bool = False,
    ) -> int:
        """
        Calculate confidence score using configured weights.
        
        Returns score from 0 to max possible (sum of all weights).
        """
        score = 0
        score += self.weight_trend_align if trend_align else 0
        score += self.weight_htf_sweep if htf_sweep else 0
        score += self.weight_ltf_bos if ltf_bos else 0
        score += self.weight_htf_fvg_touch if htf_fvg_touch else 0
        score += self.weight_discount_premium if discount_premium else 0
        score += self.weight_session_conf if session_conf else 0
        score += self.weight_smt_divergence if smt_divergence else 0
        score += self.weight_optimal_time if optimal_time else 0
        return score
    
    def max_possible_score(self) -> int:
        """Return maximum possible confidence score."""
        return (
            self.weight_trend_align +
            self.weight_htf_sweep +
            self.weight_ltf_bos +
            self.weight_htf_fvg_touch +
            self.weight_discount_premium +
            self.weight_session_conf +
            self.weight_smt_divergence +
            self.weight_optimal_time
        )
    
    def compute_cost_in_ticks(self) -> float:
        """
        Compute total round-trip cost in ticks.
        Includes: entry slippage + exit slippage + spread + commissions
        """
        # Slippage on both entry and exit (adverse direction)
        slippage_cost = 2 * self.slippage_ticks
        
        # Spread (paid once, effectively)
        spread_cost = self.spread_ticks
        
        # Commission converted to ticks
        if self.contract_multiplier > 0 and self.tick_size > 0:
            commission_ticks = (
                2 * self.commission_per_contract / 
                (self.tick_size * self.contract_multiplier)
            )
        else:
            commission_ticks = 0
        
        return slippage_cost + spread_cost + commission_ticks
    
    def compute_cost_in_r(self) -> float:
        """Compute total round-trip cost as fraction of 1R (stop distance)."""
        cost_ticks = self.compute_cost_in_ticks()
        return cost_ticks / self.stop_ticks if self.stop_ticks > 0 else 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        """Create from dictionary."""
        return cls(**data)
    
    def copy(self, **changes) -> StrategyConfig:
        """Create a copy with optional changes."""
        data = self.to_dict()
        data.update(changes)
        return StrategyConfig.from_dict(data)


# === PARAMETER BOUNDS FOR OPTIMIZATION ===

PARAM_BOUNDS: dict[str, tuple[float | int, float | int]] = {
    # Weights
    "weight_trend_align": (0, 4),
    "weight_htf_sweep": (0, 4),
    "weight_ltf_bos": (0, 4),
    "weight_htf_fvg_touch": (0, 3),
    "weight_discount_premium": (0, 3),
    "weight_session_conf": (0, 3),
    "weight_smt_divergence": (0, 3),
    "weight_optimal_time": (0, 3),
    
    # Tiers
    "tier_low_max": (1, 5),
    "tier_medium_max": (4, 8),
    "min_score_filter": (2, 6),
    
    # Risk
    "stop_ticks": (20, 100),
    "r_target_low": (1.0, 3.0),
    "r_target_medium": (2.0, 4.0),
    "r_target_high": (3.0, 6.0),
    "be_ticks_low": (30, 80),
    "be_ticks_medium": (50, 100),
    "be_ticks_high": (75, 150),
    "risk_pct_low": (0.0001, 0.001),
    "risk_pct_medium": (0.0005, 0.002),
    "risk_pct_high": (0.001, 0.01),
    
    # Pattern detection
    "htf_lookback": (5, 20),
    "entry_zone_lookback": (3, 10),
    "entry_wait_window": (5, 20),
    "sweep_lookback": (2, 5),
    "cooldown_bars": (3, 12),
    
    # Session
    "ny_session_start": (7, 9),
    "ny_session_end": (15, 17),
}


def get_default_config() -> StrategyConfig:
    """Get default configuration with standard parameters."""
    return StrategyConfig()


def get_futures_config() -> StrategyConfig:
    """Get configuration preset for NQ/ES futures."""
    return StrategyConfig(
        tick_size=0.25,
        contract_multiplier=20.0,  # NQ
        commission_per_contract=2.25,
        slippage_ticks=1.0,
        spread_ticks=1.0,
    )


def get_stocks_config() -> StrategyConfig:
    """Get configuration preset for equities."""
    return StrategyConfig(
        tick_size=0.01,
        contract_multiplier=1.0,
        commission_per_contract=0.0,  # Most brokers are commission-free
        slippage_ticks=1.0,
        spread_ticks=1.0,
    )
