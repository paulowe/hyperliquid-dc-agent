"""Configuration for Archon strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ArchonConfig:
    """Validated config for Archon strategy.

    Combines DC detection parameters with Claude intelligence settings.
    """

    # --- DC Detection ---
    symbol: str = "HYPE"
    dc_threshold: Tuple[float, float] = (0.02, 0.02)
    sensor_threshold: Tuple[float, float] = (0.004, 0.004)

    # --- Position Management ---
    position_size_usd: float = 10.0
    leverage: int = 5
    initial_stop_loss_pct: float = 0.015
    default_tp_pct: float = 0.008
    trail_pct: float = 0.35
    min_profit_to_trail_pct: float = 0.002
    backstop_sl_pct: float = 0.04
    backstop_tp_pct: float = 0.04

    # --- Claude Intelligence ---
    # Model to use for trade decisions
    model: str = "claude-haiku-4-5-20251001"
    # Whether to use Claude AI (False = heuristic-only mode)
    use_ai: bool = True
    # Minimum confidence from Claude to act on a decision
    min_confidence: float = 0.6
    # Max API calls per hour (cost control)
    max_calls_per_hour: int = 30
    # System prompt for trade decisions
    system_prompt: str = ""

    # --- Context Window ---
    # Number of recent ticks to include in context
    context_ticks: int = 60
    # Number of recent DC events to include
    context_dc_events: int = 10
    # Number of recent trades to include
    context_trades: int = 10

    # --- Regime Detection (sensor-based) ---
    lookback_seconds: float = 600.0
    choppy_rate_threshold: float = 4.0
    trending_consistency_threshold: float = 0.6

    # --- Safety ---
    cooldown_seconds: float = 30.0
    max_consecutive_losses: int = 4
    direction_filter: str = "long"  # "long", "short", or "both"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArchonConfig:
        """Create config from dict."""
        cfg = cls()
        for key, val in d.items():
            if hasattr(cfg, key):
                field_type = type(getattr(cfg, key))
                if field_type == tuple and isinstance(val, list):
                    setattr(cfg, key, tuple(val))
                else:
                    setattr(cfg, key, field_type(val))
        # Validate direction_filter
        if cfg.direction_filter not in ("long", "short", "both"):
            raise ValueError(
                f"direction_filter must be 'long', 'short', or 'both', "
                f"got '{cfg.direction_filter}'"
            )
        return cfg
