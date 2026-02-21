"""Configuration for the observer agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Default exploration thresholds (from backtesting/sweep.py)
DEFAULT_THRESHOLDS = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015]

# SL/TP defaults paired with each threshold (SL ~= threshold, TP ~= 0.5*threshold)
# These are reasonable starting points informed by backtest sweep results
DEFAULT_SL_MULTIPLIER = 1.0  # SL = threshold * multiplier
DEFAULT_TP_MULTIPLIER = 0.5  # TP = threshold * multiplier


@dataclass
class SessionConfig:
    """Configuration for a single observe-only session."""

    symbol: str
    threshold: float
    sl_pct: float
    tp_pct: float
    trail_pct: float = 0.5
    min_profit_to_trail_pct: float = 0.001
    position_size_usd: float = 100.0
    leverage: int = 10

    def to_bridge_args(self) -> list[str]:
        """Convert to live_bridge.py CLI argument list."""
        return [
            "--symbol", self.symbol,
            "--threshold", str(self.threshold),
            "--sl-pct", str(self.sl_pct),
            "--tp-pct", str(self.tp_pct),
            "--trail-pct", str(self.trail_pct),
            "--min-profit-to-trail-pct", str(self.min_profit_to_trail_pct),
            "--position-size", str(self.position_size_usd),
            "--leverage", str(self.leverage),
            "--observe-only",
        ]


@dataclass
class ObserverConfig:
    """Configuration for the observer agent."""

    # Sessions to run concurrently
    sessions: list[SessionConfig]

    # Where to write JSON reports
    report_dir: Path

    # State file
    state_file: Path

    # Observation period per round (minutes)
    observation_duration_minutes: int = 30

    # Max concurrent sessions
    max_concurrent: int = 8

    # Claude model for reasoning
    claude_model: str = "claude-sonnet-4-20250514"

    @classmethod
    def default_exploration(
        cls,
        symbol: str,
        report_dir: Path,
        state_file: Path,
        thresholds: list[float] | None = None,
        position_size_usd: float = 100.0,
        leverage: int = 10,
        observation_duration_minutes: int = 30,
    ) -> ObserverConfig:
        """Generate a default exploration config testing multiple thresholds.

        Each threshold gets a paired SL/TP based on backtest-informed defaults:
        SL = threshold (stop at 1x the threshold move)
        TP = 0.5 * threshold (take profit at half-threshold overshoot)
        """
        thresholds = thresholds or DEFAULT_THRESHOLDS
        sessions = []
        for thresh in thresholds:
            sessions.append(SessionConfig(
                symbol=symbol,
                threshold=thresh,
                sl_pct=round(thresh * DEFAULT_SL_MULTIPLIER, 6),
                tp_pct=round(thresh * DEFAULT_TP_MULTIPLIER, 6),
                position_size_usd=position_size_usd,
                leverage=leverage,
            ))
        return cls(
            sessions=sessions,
            report_dir=report_dir,
            state_file=state_file,
            observation_duration_minutes=observation_duration_minutes,
        )
