"""Configuration for the DC Overshoot strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class DCOvershootConfig:
    """Configuration for the DC Overshoot trading strategy.

    The DC Overshoot strategy exploits the empirical observation that after
    a Directional Change confirmation (PDCC event), price tends to overshoot
    in the same direction by approximately the same threshold amount.

    No ML model is needed — the threshold itself is the prediction.

    Risk management uses a "greedy trailing" approach:
    - When in a loss: SL and TP remain at initial levels
    - When in profit: SL ratchets toward profit (locks in gains),
      TP pushes further in the profit direction
    """

    symbol: str = "BTC"

    # DC detector thresholds: list of (down, up) pairs
    dc_thresholds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.001, 0.001)]
    )

    # Position sizing
    position_size_usd: float = 50.0
    max_position_size_usd: float = 200.0

    # Risk management (percentages as decimals: 0.003 = 0.3%)
    initial_stop_loss_pct: float = 0.003  # 0.3% initial SL from entry
    initial_take_profit_pct: float = 0.002  # 0.2% initial TP (≈ 2x threshold)
    trail_pct: float = 0.5  # Lock in 50% of profit as new SL floor
    min_profit_to_trail_pct: float = 0.001  # Don't trail until 0.1% profit (prevents noise ratcheting)

    # Cooldown between entry signals
    cooldown_seconds: float = 10.0

    # Maximum concurrent positions (1 = one at a time)
    max_open_positions: int = 1

    # Logging
    log_events: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DCOvershootConfig:
        """Create config from a flat dict (e.g., from YAML).

        Validates all parameters and raises ValueError for invalid inputs.
        """
        # Parse and validate thresholds
        raw_thresholds = d.get("dc_thresholds", [(0.001, 0.001)])
        thresholds = []
        for t in raw_thresholds:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                thresholds.append((float(t[0]), float(t[1])))
            elif isinstance(t, (int, float)):
                thresholds.append((float(t), float(t)))
            else:
                raise ValueError(f"Invalid threshold: {t}. Expected (down, up) pair or single float.")

        # Validate threshold values are positive
        for down, up in thresholds:
            if down <= 0 or up <= 0:
                raise ValueError(
                    f"DC threshold values must be positive, got ({down}, {up})"
                )

        # Parse numeric fields
        sl_pct = float(d.get("initial_stop_loss_pct", 0.003))
        tp_pct = float(d.get("initial_take_profit_pct", 0.002))
        trail = float(d.get("trail_pct", 0.5))
        pos_size = float(d.get("position_size_usd", 50.0))
        max_pos = float(d.get("max_position_size_usd", 200.0))
        cooldown = float(d.get("cooldown_seconds", 10.0))

        min_profit = float(d.get("min_profit_to_trail_pct", 0.001))

        # Validate ranges
        if sl_pct < 0:
            raise ValueError(
                f"initial_stop_loss_pct must be non-negative, got {sl_pct}"
            )
        if tp_pct < 0:
            raise ValueError(
                f"initial_take_profit_pct must be non-negative, got {tp_pct}"
            )
        if trail <= 0 or trail > 1.0:
            raise ValueError(
                f"trail_pct must be in (0, 1], got {trail}"
            )
        if pos_size <= 0:
            raise ValueError(
                f"position_size_usd must be positive, got {pos_size}"
            )
        if max_pos <= 0:
            raise ValueError(
                f"max_position_size_usd must be positive, got {max_pos}"
            )
        if cooldown < 0:
            raise ValueError(
                f"cooldown_seconds must be non-negative, got {cooldown}"
            )

        return cls(
            symbol=d.get("symbol", "BTC"),
            dc_thresholds=thresholds,
            position_size_usd=pos_size,
            max_position_size_usd=max_pos,
            initial_stop_loss_pct=sl_pct,
            initial_take_profit_pct=tp_pct,
            trail_pct=trail,
            min_profit_to_trail_pct=min_profit,
            cooldown_seconds=cooldown,
            max_open_positions=int(d.get("max_open_positions", 1)),
            log_events=bool(d.get("log_events", True)),
        )
