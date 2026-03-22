"""Tests for DC Trend-Adaptive backtest comparison."""

import math

import pytest

from interfaces.strategy import MarketData, SignalType
from strategies.dc_adaptive.backtest_compare import run_strategy_on_candles
from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy
from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from strategies.dc_trend_adaptive.backtest_compare import directional_breakdown
from strategies.dc_trend_adaptive.dc_trend_adaptive_strategy import DCTrendAdaptiveStrategy


def make_candles(start_price=100.0, n=500, trend="up"):
    """Generate synthetic candles for testing.

    For 'up' trend: gentle uptrend with small oscillations.
    For 'zigzag': alternating up/down legs.
    """
    candles = []
    price = start_price
    t = 1_000_000  # ms

    if trend == "up":
        # Uptrend with noise
        for i in range(n):
            drift = 0.001 * price  # 0.1% drift up per candle
            noise = (0.002 * price) * (1 if i % 3 else -1)  # oscillation
            price += drift + noise
            candles.append({"c": str(price), "t": str(t)})
            t += 60_000
    elif trend == "zigzag":
        # Up/down legs to generate trades in both directions
        leg_len = 20
        for i in range(n):
            leg = (i // leg_len) % 2
            if leg == 0:
                price += 0.003 * price
            else:
                price -= 0.003 * price
            candles.append({"c": str(price), "t": str(t)})
            t += 60_000

    return candles


def make_base_config(symbol="TEST"):
    return {
        "symbol": symbol,
        "dc_thresholds": [(0.01, 0.01)],
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.02,
        "initial_take_profit_pct": 0.005,
        "trail_pct": 0.3,
        "min_profit_to_trail_pct": 0.003,
        "cooldown_seconds": 10.0,
        "max_open_positions": 1,
        "log_events": False,
    }


def make_adaptive_config(symbol="TEST"):
    cfg = make_base_config(symbol)
    cfg.update({
        "sensor_threshold": (0.004, 0.004),
        "lookback_seconds": 600,
        "choppy_rate_threshold": 4.0,
        "trending_consistency_threshold": 0.6,
        "os_window_size": 20,
        "os_min_samples": 5,
        "tp_fraction": 0.4,
        "min_tp_pct": 0.003,
        "default_tp_pct": 0.005,
        "max_consecutive_losses": 3,
        "base_cooldown_seconds": 300,
    })
    return cfg


def make_trend_config(symbol="TEST"):
    cfg = make_adaptive_config(symbol)
    cfg.update({
        "trend_lookback_seconds": 900,
        "trend_min_events": 5,
        "trend_min_consistency": 0.6,
        "trend_bias_mode": "tmv_weighted",
        "counter_trend_action": "block",
        "counter_trend_size_fraction": 0.5,
        "close_on_trend_flip": True,
    })
    return cfg


class TestThreeWayComparison:
    """All three strategies produce results on synthetic data."""

    def test_three_way_comparison_runs(self):
        """All three strategies run without errors on zigzag candles."""
        candles = make_candles(trend="zigzag", n=500)

        baseline = DCOvershootStrategy(make_base_config())
        baseline_trades = run_strategy_on_candles(baseline, candles, make_base_config())

        adaptive = DCAdaptiveStrategy(make_adaptive_config())
        adaptive_trades = run_strategy_on_candles(adaptive, candles, make_adaptive_config())

        trend = DCTrendAdaptiveStrategy(make_trend_config())
        trend_trades = run_strategy_on_candles(trend, candles, make_trend_config())

        # All should produce some trades on zigzag data
        assert isinstance(baseline_trades, list)
        assert isinstance(adaptive_trades, list)
        assert isinstance(trend_trades, list)


class TestDirectionalBreakdown:
    """directional_breakdown() works correctly."""

    def test_directional_breakdown_in_output(self):
        """Output includes per-direction metrics."""
        candles = make_candles(trend="zigzag", n=500)
        config = make_base_config()
        strategy = DCOvershootStrategy(config)
        trades = run_strategy_on_candles(strategy, candles, config)

        breakdown = directional_breakdown(trades)
        assert "LONG" in breakdown
        assert "SHORT" in breakdown
        assert "count" in breakdown["LONG"]
        assert "net_pnl" in breakdown["LONG"]
        assert "wins" in breakdown["LONG"]
        assert "losses" in breakdown["LONG"]

    def test_empty_trades(self):
        """Empty trade list produces zero counts."""
        breakdown = directional_breakdown([])
        assert breakdown["LONG"]["count"] == 0
        assert breakdown["SHORT"]["count"] == 0


class TestTrendAdaptiveOnUptrend:
    """On uptrend data, trend-adaptive should block more shorts."""

    def test_trend_adaptive_fewer_counter_trend_trades(self):
        """On uptrend candles, trend-adaptive should skip more shorts than baseline."""
        candles = make_candles(trend="up", n=1000)

        baseline_config = make_base_config()
        baseline = DCOvershootStrategy(baseline_config)
        baseline_trades = run_strategy_on_candles(baseline, candles, baseline_config)

        trend_config = make_trend_config()
        trend = DCTrendAdaptiveStrategy(trend_config)
        trend_trades = run_strategy_on_candles(trend, candles, trend_config)

        baseline_shorts = sum(1 for t in baseline_trades if t.side == "SHORT")
        trend_shorts = sum(1 for t in trend_trades if t.side == "SHORT")

        # Trend-adaptive should have fewer shorts (or equal) on an uptrend
        assert trend_shorts <= baseline_shorts
