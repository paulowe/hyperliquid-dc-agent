"""Configuration for the Multi-Scale DC Overshoot strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


def _parse_threshold_pair(t: Any) -> Tuple[float, float]:
    """Parse a threshold value into a (down, up) pair."""
    if isinstance(t, (list, tuple)) and len(t) == 2:
        return (float(t[0]), float(t[1]))
    elif isinstance(t, (int, float)):
        return (float(t), float(t))
    else:
        raise ValueError(f"Invalid threshold: {t}. Expected (down, up) pair or single float.")


def _validate_threshold_pair(pair: Tuple[float, float], label: str) -> None:
    """Validate that both threshold values are positive."""
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{label} values must be positive, got {pair}")


@dataclass
class MultiScaleConfig:
    """Configuration for the Multi-Scale DC Overshoot strategy.

    Combines low-threshold DC sensors (intelligence) with a high-threshold
    DC trade trigger, filtered by momentum consensus from the sensors.
    """

    symbol: str = "BTC"

    # Sensor thresholds: provide directional intelligence, never trigger trades
    sensor_thresholds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.002, 0.002), (0.004, 0.004), (0.008, 0.008)]
    )

    # Trade threshold: the high threshold that actually triggers entries
    trade_threshold: Tuple[float, float] = (0.015, 0.015)

    # Momentum scorer parameters
    momentum_alpha: float = 0.3
    min_momentum_score: float = 0.3

    # Position sizing
    position_size_usd: float = 50.0
    max_position_size_usd: float = 200.0

    # Risk management
    initial_stop_loss_pct: float = 0.015
    initial_take_profit_pct: float = 0.005
    trail_pct: float = 0.3
    min_profit_to_trail_pct: float = 0.002

    # Cooldown between entries
    cooldown_seconds: float = 10.0

    # Maximum concurrent positions
    max_open_positions: int = 1

    # Logging
    log_events: bool = True
    log_sensor_events: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MultiScaleConfig:
        """Create config from a flat dict (e.g., from YAML or CLI args).

        Validates all parameters and raises ValueError for invalid inputs.
        """
        # Parse sensor thresholds
        raw_sensors = d.get(
            "sensor_thresholds",
            [(0.002, 0.002), (0.004, 0.004), (0.008, 0.008)],
        )
        sensor_thresholds = [_parse_threshold_pair(t) for t in raw_sensors]
        for pair in sensor_thresholds:
            _validate_threshold_pair(pair, "Sensor threshold")

        # Parse trade threshold
        raw_trade = d.get("trade_threshold", (0.015, 0.015))
        trade_threshold = _parse_threshold_pair(raw_trade)
        _validate_threshold_pair(trade_threshold, "Trade threshold")

        # Parse numeric fields
        sl_pct = float(d.get("initial_stop_loss_pct", 0.015))
        tp_pct = float(d.get("initial_take_profit_pct", 0.005))
        trail = float(d.get("trail_pct", 0.3))
        pos_size = float(d.get("position_size_usd", 50.0))
        max_pos = float(d.get("max_position_size_usd", 200.0))
        cooldown = float(d.get("cooldown_seconds", 10.0))
        min_profit = float(d.get("min_profit_to_trail_pct", 0.002))
        alpha = float(d.get("momentum_alpha", 0.3))
        min_score = float(d.get("min_momentum_score", 0.3))

        # Validate ranges
        if sl_pct < 0:
            raise ValueError(f"initial_stop_loss_pct must be non-negative, got {sl_pct}")
        if tp_pct < 0:
            raise ValueError(f"initial_take_profit_pct must be non-negative, got {tp_pct}")
        if trail <= 0 or trail > 1.0:
            raise ValueError(f"trail_pct must be in (0, 1], got {trail}")
        if pos_size <= 0:
            raise ValueError(f"position_size_usd must be positive, got {pos_size}")
        if max_pos <= 0:
            raise ValueError(f"max_position_size_usd must be positive, got {max_pos}")
        if cooldown < 0:
            raise ValueError(f"cooldown_seconds must be non-negative, got {cooldown}")

        return cls(
            symbol=d.get("symbol", "BTC"),
            sensor_thresholds=sensor_thresholds,
            trade_threshold=trade_threshold,
            momentum_alpha=alpha,
            min_momentum_score=min_score,
            position_size_usd=pos_size,
            max_position_size_usd=max_pos,
            initial_stop_loss_pct=sl_pct,
            initial_take_profit_pct=tp_pct,
            trail_pct=trail,
            min_profit_to_trail_pct=min_profit,
            cooldown_seconds=cooldown,
            max_open_positions=int(d.get("max_open_positions", 1)),
            log_events=bool(d.get("log_events", True)),
            log_sensor_events=bool(d.get("log_sensor_events", False)),
        )

    def all_thresholds(self) -> List[Tuple[float, float]]:
        """Return combined list of sensor + trade thresholds for LiveDCDetector."""
        return list(self.sensor_thresholds) + [self.trade_threshold]

    def sensor_threshold_keys(self) -> set:
        """Return threshold keys for sensor-only thresholds (for event routing)."""
        return {f"{d}:{u}" for d, u in self.sensor_thresholds}

    def trade_threshold_key(self) -> str:
        """Return the threshold key for the trade threshold."""
        d, u = self.trade_threshold
        return f"{d}:{u}"
