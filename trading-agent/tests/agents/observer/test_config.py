"""Tests for ObserverConfig and SessionConfig."""

import pytest
from pathlib import Path

from agents.observer.config import ObserverConfig, SessionConfig


class TestSessionConfig:
    """SessionConfig dataclass validation."""

    def test_creates_with_required_fields(self):
        cfg = SessionConfig(symbol="SOL", threshold=0.015, sl_pct=0.015, tp_pct=0.005)
        assert cfg.symbol == "SOL"
        assert cfg.threshold == 0.015
        assert cfg.sl_pct == 0.015
        assert cfg.tp_pct == 0.005
        # Defaults
        assert cfg.trail_pct == 0.5
        assert cfg.min_profit_to_trail_pct == 0.001
        assert cfg.position_size_usd == 100.0
        assert cfg.leverage == 10

    def test_override_defaults(self):
        cfg = SessionConfig(
            symbol="BTC", threshold=0.004, sl_pct=0.008, tp_pct=0.02,
            trail_pct=0.7, leverage=3,
        )
        assert cfg.trail_pct == 0.7
        assert cfg.leverage == 3

    def test_to_bridge_args(self):
        """to_bridge_args() should produce correct CLI arg list."""
        cfg = SessionConfig(symbol="SOL", threshold=0.015, sl_pct=0.015, tp_pct=0.005)
        args = cfg.to_bridge_args()
        assert "--symbol" in args
        assert "SOL" in args
        assert "--threshold" in args
        assert "0.015" in args
        assert "--sl-pct" in args
        assert "--tp-pct" in args
        assert "--observe-only" in args


class TestObserverConfig:
    """ObserverConfig dataclass and factory methods."""

    def test_creates_with_sessions(self, tmp_path):
        sessions = [
            SessionConfig(symbol="SOL", threshold=0.004, sl_pct=0.005, tp_pct=0.01),
            SessionConfig(symbol="SOL", threshold=0.01, sl_pct=0.01, tp_pct=0.03),
        ]
        cfg = ObserverConfig(
            sessions=sessions,
            report_dir=tmp_path / "reports",
            state_file=tmp_path / "state.json",
        )
        assert len(cfg.sessions) == 2
        assert cfg.observation_duration_minutes == 30
        assert cfg.max_concurrent == 8

    def test_default_exploration_generates_sessions(self, tmp_path):
        """default_exploration() should generate sessions across threshold range."""
        cfg = ObserverConfig.default_exploration(
            symbol="SOL",
            report_dir=tmp_path / "reports",
            state_file=tmp_path / "state.json",
        )
        assert len(cfg.sessions) >= 4
        # All sessions should be for the requested symbol
        for s in cfg.sessions:
            assert s.symbol == "SOL"
        # Thresholds should span a range
        thresholds = [s.threshold for s in cfg.sessions]
        assert min(thresholds) < max(thresholds)

    def test_default_exploration_uses_known_thresholds(self, tmp_path):
        """Should use thresholds from the backtest sweep defaults."""
        cfg = ObserverConfig.default_exploration(
            symbol="BTC",
            report_dir=tmp_path / "reports",
            state_file=tmp_path / "state.json",
        )
        thresholds = {s.threshold for s in cfg.sessions}
        # At least some of these known-good thresholds should appear
        known = {0.002, 0.004, 0.006, 0.008, 0.01, 0.015}
        assert thresholds & known, f"Expected overlap with {known}, got {thresholds}"
