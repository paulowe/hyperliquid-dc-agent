"""Configuration for DC Adaptive strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class DCAdaptiveConfig:
    """Validated config for DC Adaptive strategy.

    Extends the DC Overshoot config with adaptive guard parameters.
    """

    # Core DC parameters
    symbol: str = "BTC"
    dc_thresholds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.001, 0.001)]
    )
    position_size_usd: float = 50.0
    max_position_size_usd: float = 200.0
    initial_stop_loss_pct: float = 0.003
    initial_take_profit_pct: float = 0.002
    trail_pct: float = 0.5
    min_profit_to_trail_pct: float = 0.001
    cooldown_seconds: float = 10.0
    max_open_positions: int = 1
    log_events: bool = True

    # Regime detector
    sensor_threshold: Tuple[float, float] = (0.004, 0.004)
    lookback_seconds: float = 600.0
    choppy_rate_threshold: float = 4.0
    trending_consistency_threshold: float = 0.6

    # Overshoot tracker
    os_window_size: int = 20
    os_min_samples: int = 5
    tp_fraction: float = 0.8
    min_tp_pct: float = 0.003
    default_tp_pct: float = 0.005

    # Loss streak guard
    max_consecutive_losses: int = 3
    base_cooldown_seconds: float = 300.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DCAdaptiveConfig:
        """Create config from dict, converting list thresholds to tuples."""
        cfg = cls()

        # Core parameters
        if "symbol" in d:
            cfg.symbol = d["symbol"]
        if "dc_thresholds" in d:
            cfg.dc_thresholds = [
                tuple(t) if isinstance(t, list) else t for t in d["dc_thresholds"]
            ]
        if "position_size_usd" in d:
            cfg.position_size_usd = float(d["position_size_usd"])
        if "max_position_size_usd" in d:
            cfg.max_position_size_usd = float(d["max_position_size_usd"])
        if "initial_stop_loss_pct" in d:
            cfg.initial_stop_loss_pct = float(d["initial_stop_loss_pct"])
        if "initial_take_profit_pct" in d:
            cfg.initial_take_profit_pct = float(d["initial_take_profit_pct"])
        if "trail_pct" in d:
            cfg.trail_pct = float(d["trail_pct"])
        if "min_profit_to_trail_pct" in d:
            cfg.min_profit_to_trail_pct = float(d["min_profit_to_trail_pct"])
        if "cooldown_seconds" in d:
            cfg.cooldown_seconds = float(d["cooldown_seconds"])
        if "max_open_positions" in d:
            cfg.max_open_positions = int(d["max_open_positions"])
        if "log_events" in d:
            cfg.log_events = bool(d["log_events"])

        # Sensor threshold
        if "sensor_threshold" in d:
            st = d["sensor_threshold"]
            cfg.sensor_threshold = tuple(st) if isinstance(st, list) else st

        # Regime detector
        if "lookback_seconds" in d:
            cfg.lookback_seconds = float(d["lookback_seconds"])
        if "choppy_rate_threshold" in d:
            cfg.choppy_rate_threshold = float(d["choppy_rate_threshold"])
        if "trending_consistency_threshold" in d:
            cfg.trending_consistency_threshold = float(
                d["trending_consistency_threshold"]
            )

        # Overshoot tracker
        if "os_window_size" in d:
            cfg.os_window_size = int(d["os_window_size"])
        if "os_min_samples" in d:
            cfg.os_min_samples = int(d["os_min_samples"])
        if "tp_fraction" in d:
            cfg.tp_fraction = float(d["tp_fraction"])
        if "min_tp_pct" in d:
            cfg.min_tp_pct = float(d["min_tp_pct"])
        if "default_tp_pct" in d:
            cfg.default_tp_pct = float(d["default_tp_pct"])

        # Loss streak guard
        if "max_consecutive_losses" in d:
            cfg.max_consecutive_losses = int(d["max_consecutive_losses"])
        if "base_cooldown_seconds" in d:
            cfg.base_cooldown_seconds = float(d["base_cooldown_seconds"])

        return cfg
