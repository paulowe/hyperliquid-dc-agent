"""Configuration for DC Trend-Adaptive strategy.

Extends DCAdaptiveConfig with trend direction filter parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from strategies.dc_adaptive.config import DCAdaptiveConfig


@dataclass
class DCTrendAdaptiveConfig(DCAdaptiveConfig):
    """Validated config for DC Trend-Adaptive strategy.

    Extends DCAdaptiveConfig with Guard 4: TrendDirectionFilter parameters.
    """

    # Trend direction filter (Guard 4)
    trend_lookback_seconds: float = 900.0
    trend_min_events: int = 5
    trend_min_consistency: float = 0.6
    trend_bias_mode: str = "tmv_weighted"
    trend_strict_threshold: float = 0.8
    counter_trend_action: str = "block"
    counter_trend_size_fraction: float = 0.5
    counter_trend_sl_pct: Optional[float] = None
    close_on_trend_flip: bool = True

    # Nuclear options
    long_only: bool = False
    short_only: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DCTrendAdaptiveConfig:
        """Create config from dict, converting list thresholds to tuples."""
        # Start with parent config
        parent = DCAdaptiveConfig.from_dict(d)
        cfg = cls()

        # Copy all parent fields
        for f in DCAdaptiveConfig.__dataclass_fields__:
            setattr(cfg, f, getattr(parent, f))

        # Trend direction filter
        if "trend_lookback_seconds" in d:
            cfg.trend_lookback_seconds = float(d["trend_lookback_seconds"])
        if "trend_min_events" in d:
            cfg.trend_min_events = int(d["trend_min_events"])
        if "trend_min_consistency" in d:
            cfg.trend_min_consistency = float(d["trend_min_consistency"])
        if "trend_bias_mode" in d:
            cfg.trend_bias_mode = str(d["trend_bias_mode"])
        if "trend_strict_threshold" in d:
            cfg.trend_strict_threshold = float(d["trend_strict_threshold"])
        if "counter_trend_action" in d:
            cfg.counter_trend_action = str(d["counter_trend_action"])
        if "counter_trend_size_fraction" in d:
            cfg.counter_trend_size_fraction = float(d["counter_trend_size_fraction"])
        if "counter_trend_sl_pct" in d:
            val = d["counter_trend_sl_pct"]
            cfg.counter_trend_sl_pct = float(val) if val is not None else None
        if "close_on_trend_flip" in d:
            cfg.close_on_trend_flip = bool(d["close_on_trend_flip"])
        if "long_only" in d:
            cfg.long_only = bool(d["long_only"])
        if "short_only" in d:
            cfg.short_only = bool(d["short_only"])

        return cfg
