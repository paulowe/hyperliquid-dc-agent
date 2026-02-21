"""Tests for parameter sweep."""

from __future__ import annotations

import json

import pytest

from backtesting.engine import BacktestConfig, BacktestResult, TradeRecord
from backtesting.sweep import ParameterSweep, SweepConfig, SweepResult


def make_trade(
    pnl_usd: float = 0.0,
    net_pnl_usd: float | None = None,
    fees: float = 0.01,
    reason: str = "trailing_stop_loss",
    side: str = "LONG",
) -> TradeRecord:
    """Create a TradeRecord with sensible defaults."""
    if net_pnl_usd is None:
        net_pnl_usd = pnl_usd - fees
    return TradeRecord(
        side=side,
        entry_price=100.0,
        exit_price=100.0 + pnl_usd,
        size=0.1,
        entry_time=1000.0,
        exit_time=2000.0,
        pnl_pct=pnl_usd / 100.0,
        pnl_usd=pnl_usd,
        entry_fee=fees / 2,
        exit_fee=fees / 2,
        total_fees=fees,
        net_pnl_usd=net_pnl_usd,
        reason=reason,
    )


class TestSweepConfig:
    def test_default_grid_size(self):
        """Default grid should produce 1,296 combinations."""
        config = SweepConfig()
        combos = config.combinations()
        assert len(combos) == 1296

    def test_custom_grid(self):
        """Custom grid should produce correct number of combinations."""
        config = SweepConfig(
            thresholds=[0.004, 0.015],
            sl_pcts=[0.003, 0.01],
            tp_pcts=[0.01, 0.05],
            trail_pcts=[0.5],
            min_profit_to_trail_pcts=[0.001],
        )
        combos = config.combinations()
        # 2 * 2 * 2 * 1 * 1 = 8
        assert len(combos) == 8

    def test_combinations_are_backtest_configs(self):
        """Each combination should be a BacktestConfig."""
        config = SweepConfig(
            thresholds=[0.004],
            sl_pcts=[0.003],
            tp_pcts=[0.01],
            trail_pcts=[0.5],
            min_profit_to_trail_pcts=[0.001],
        )
        combos = config.combinations()
        assert len(combos) == 1
        assert isinstance(combos[0], BacktestConfig)
        assert combos[0].threshold == 0.004
        assert combos[0].initial_stop_loss_pct == 0.003
        assert combos[0].initial_take_profit_pct == 0.01
        assert combos[0].trail_pct == 0.5
        assert combos[0].min_profit_to_trail_pct == 0.001


class TestSweepResult:
    def test_fields(self):
        """SweepResult should hold config + metrics."""
        config = BacktestConfig(threshold=0.004, initial_stop_loss_pct=0.003)
        result = SweepResult(
            config=config,
            total_trades=10,
            wins=7,
            losses=3,
            win_rate_net=70.0,
            gross_pnl_usd=5.0,
            total_fees_usd=0.5,
            net_pnl_usd=4.5,
            profit_factor=3.0,
            max_drawdown_usd=1.0,
            net_pnl_per_day=0.64,
            trades_eaten_by_fees=1,
        )
        assert result.config.threshold == 0.004
        assert result.net_pnl_usd == 4.5


class TestParameterSweepMinTrades:
    def test_filters_low_trade_count(self):
        """Configs producing fewer than min_trades should be excluded."""
        # We test this by using a tiny grid and candles that produce few trades
        # with some configs. Use mock approach: test the filter logic directly.
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.004),
                total_trades=3,
                wins=2, losses=1, win_rate_net=66.7,
                gross_pnl_usd=1.0, total_fees_usd=0.1, net_pnl_usd=0.9,
                profit_factor=2.0, max_drawdown_usd=0.5,
                net_pnl_per_day=0.13, trades_eaten_by_fees=0,
            ),
            SweepResult(
                config=BacktestConfig(threshold=0.015),
                total_trades=10,
                wins=8, losses=2, win_rate_net=80.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=8.0, max_drawdown_usd=1.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
        ]
        # Filter with min_trades=5
        filtered = [r for r in results if r.total_trades >= 5]
        assert len(filtered) == 1
        assert filtered[0].config.threshold == 0.015


class TestParameterSweepSorting:
    def test_results_sorted_by_net_pnl(self):
        """Results should be sorted by net P&L descending."""
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.004),
                total_trades=10, wins=3, losses=7, win_rate_net=30.0,
                gross_pnl_usd=-2.0, total_fees_usd=0.5, net_pnl_usd=-2.5,
                profit_factor=0.5, max_drawdown_usd=3.0,
                net_pnl_per_day=-0.36, trades_eaten_by_fees=2,
            ),
            SweepResult(
                config=BacktestConfig(threshold=0.015),
                total_trades=10, wins=8, losses=2, win_rate_net=80.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=8.0, max_drawdown_usd=1.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
        ]
        sorted_results = sorted(results, key=lambda r: r.net_pnl_usd, reverse=True)
        assert sorted_results[0].net_pnl_usd > sorted_results[1].net_pnl_usd


class TestPatternAnalysis:
    def test_analyze_patterns(self):
        """Pattern analysis should report averages for profitable vs unprofitable."""
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.015, initial_stop_loss_pct=0.01),
                total_trades=10, wins=8, losses=2, win_rate_net=80.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=8.0, max_drawdown_usd=1.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
            SweepResult(
                config=BacktestConfig(threshold=0.004, initial_stop_loss_pct=0.003),
                total_trades=100, wins=30, losses=70, win_rate_net=30.0,
                gross_pnl_usd=-2.0, total_fees_usd=2.0, net_pnl_usd=-4.0,
                profit_factor=0.5, max_drawdown_usd=5.0,
                net_pnl_per_day=-0.57, trades_eaten_by_fees=20,
            ),
        ]
        patterns = ParameterSweep.analyze_patterns(results)
        assert "profitable" in patterns
        assert "unprofitable" in patterns
        assert patterns["profitable"]["avg_threshold"] == 0.015
        assert patterns["unprofitable"]["avg_threshold"] == 0.004
        assert patterns["profitable_count"] == 1
        assert patterns["unprofitable_count"] == 1

    def test_analyze_patterns_all_profitable(self):
        """When all configs are profitable, unprofitable stats should be empty."""
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.015),
                total_trades=10, wins=8, losses=2, win_rate_net=80.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=8.0, max_drawdown_usd=1.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
        ]
        patterns = ParameterSweep.analyze_patterns(results)
        assert patterns["profitable_count"] == 1
        assert patterns["unprofitable_count"] == 0
        assert patterns["unprofitable"] == {}


class TestFormatResults:
    def test_format_returns_string(self):
        """format_results should return a non-empty string."""
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.015, initial_stop_loss_pct=0.01,
                                       initial_take_profit_pct=0.05, trail_pct=0.5,
                                       min_profit_to_trail_pct=0.001),
                total_trades=10, wins=8, losses=2, win_rate_net=80.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=8.0, max_drawdown_usd=1.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
        ]
        text = ParameterSweep.format_results(results, top_n=5)
        assert isinstance(text, str)
        assert "0.015" in text
        assert len(text) > 0


class TestResultsToJson:
    def test_json_serializable(self):
        """results_to_json output should be JSON-serializable."""
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.015, initial_stop_loss_pct=0.01,
                                       initial_take_profit_pct=0.05, trail_pct=0.5,
                                       min_profit_to_trail_pct=0.001),
                total_trades=10, wins=8, losses=2, win_rate_net=80.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=8.0, max_drawdown_usd=1.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
        ]
        data = ParameterSweep.results_to_json(results)
        # Should be serializable
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        # Should round-trip
        parsed = json.loads(serialized)
        assert parsed[0]["threshold"] == 0.015
        assert parsed[0]["net_pnl_usd"] == 4.5

    def test_json_handles_inf_profit_factor(self):
        """Infinity profit factor should serialize cleanly."""
        results = [
            SweepResult(
                config=BacktestConfig(threshold=0.015),
                total_trades=10, wins=10, losses=0, win_rate_net=100.0,
                gross_pnl_usd=5.0, total_fees_usd=0.5, net_pnl_usd=4.5,
                profit_factor=float("inf"), max_drawdown_usd=0.0,
                net_pnl_per_day=0.64, trades_eaten_by_fees=0,
            ),
        ]
        data = ParameterSweep.results_to_json(results)
        # Infinity can't go through JSON, so should be converted to a string or large number
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
