"""Tests for DCForecastStrategy."""

import math
import pytest
from strategies.dc_forecast.dc_forecast_strategy import DCForecastStrategy
from strategies.dc_forecast.config import DCForecastConfig
from interfaces.strategy import MarketData, Position


class TestDCForecastStrategy:
    """Test the DC Forecast strategy integration."""

    def _make_config(self, **overrides) -> dict:
        defaults = {
            "symbol": "BTC",
            "dc_thresholds": [(0.01, 0.01)],
            "model_uri": "",
            "scaler_uri": "",
            "log_dc_events": False,
        }
        defaults.update(overrides)
        return defaults

    def _make_market_data(self, price: float, ts: float, asset: str = "BTC") -> MarketData:
        return MarketData(
            asset=asset,
            price=price,
            volume_24h=0.0,
            timestamp=ts,
        )

    def test_initialization(self):
        config = self._make_config()
        strategy = DCForecastStrategy(config)
        assert strategy.name == "dc_forecast"
        assert strategy.is_active is True

    def test_generate_signals_returns_empty_phase1(self):
        """Phase 1: No signals generated, only logging."""
        strategy = DCForecastStrategy(self._make_config())
        md = self._make_market_data(100.0, 1000.0)
        signals = strategy.generate_signals(md, [], 10000.0)
        assert signals == []

    def test_ignores_non_matching_asset(self):
        """Strategy should ignore price updates for other assets."""
        strategy = DCForecastStrategy(self._make_config(symbol="BTC"))
        md = self._make_market_data(100.0, 1000.0, asset="ETH")
        signals = strategy.generate_signals(md, [], 10000.0)
        assert signals == []

    def test_tick_counter_increments(self):
        strategy = DCForecastStrategy(self._make_config())
        for i in range(5):
            md = self._make_market_data(100.0, 1000.0 + i)
            strategy.generate_signals(md, [], 10000.0)
        assert strategy._tick_count == 5

    def test_dc_events_detected(self):
        """Verify DC events are counted when prices move enough."""
        strategy = DCForecastStrategy(self._make_config(dc_thresholds=[(0.01, 0.01)]))

        # Feed stable prices then a 2% drop
        for i in range(5):
            md = self._make_market_data(100.0, 1000.0 + i)
            strategy.generate_signals(md, [], 10000.0)

        # Sharp drop - should trigger PDCC_Down
        md = self._make_market_data(97.5, 1005.0)
        strategy.generate_signals(md, [], 10000.0)

        assert strategy._total_events > 0
        assert strategy._pdcc_down_count > 0

    def test_get_status(self):
        strategy = DCForecastStrategy(self._make_config())
        md = self._make_market_data(100.0, 1000.0)
        strategy.generate_signals(md, [], 10000.0)
        status = strategy.get_status()

        assert status["name"] == "dc_forecast"
        assert status["symbol"] == "BTC"
        assert status["tick_count"] == 1
        assert "regimes" in status
        assert "0.01:0.01" in status["regimes"]

    def test_start_stop_lifecycle(self):
        strategy = DCForecastStrategy(self._make_config())
        strategy.start()
        assert strategy.is_active is True
        strategy.stop()
        assert strategy.is_active is False

    def test_full_regime_cycle(self):
        """Feed prices that create a down->up regime cycle."""
        strategy = DCForecastStrategy(self._make_config(dc_thresholds=[(0.01, 0.01)]))

        # Stable at 100
        for i in range(3):
            strategy.generate_signals(self._make_market_data(100.0, 1000.0 + i), [], 10000.0)

        # Drop to 98 (2% -> PDCC_Down)
        strategy.generate_signals(self._make_market_data(98.0, 1003.0), [], 10000.0)
        assert strategy._pdcc_down_count >= 1

        # Hold at 98
        for i in range(3):
            strategy.generate_signals(self._make_market_data(98.0, 1004.0 + i), [], 10000.0)

        # Rise to 100 (>1% from 98 -> PDCC2_UP)
        strategy.generate_signals(self._make_market_data(100.0, 1007.0), [], 10000.0)
        assert strategy._pdcc_up_count >= 1

    def test_config_from_dict(self):
        """Verify DCForecastConfig.from_dict works."""
        d = {
            "symbol": "ETH",
            "dc_thresholds": [[0.005, 0.005]],
            "window_size": 30,
            "signal_threshold_pct": 0.2,
        }
        cfg = DCForecastConfig.from_dict(d)
        assert cfg.symbol == "ETH"
        assert cfg.dc_thresholds == [(0.005, 0.005)]
        assert cfg.window_size == 30
        assert cfg.signal_threshold_pct == 0.2


class TestDCForecastIntegration:
    """End-to-end integration test with synthetic price data."""

    def test_oscillating_prices_produce_events(self):
        """Oscillating prices should produce multiple DC events."""
        config = {
            "symbol": "BTC",
            "dc_thresholds": [(0.005, 0.005)],  # 0.5% threshold
            "log_dc_events": False,
        }
        strategy = DCForecastStrategy(config)

        # Simulate oscillating price: 100 -> 99 -> 101 -> 99.5 -> 100.5
        prices = []
        for i in range(200):
            prices.append(100.0 + 2.0 * math.sin(i * 0.15))

        for i, price in enumerate(prices):
            md = MarketData(
                asset="BTC", price=price, volume_24h=0.0, timestamp=1000.0 + i
            )
            strategy.generate_signals(md, [], 10000.0)

        # Should have detected multiple regime changes
        status = strategy.get_status()
        assert status["total_dc_events"] > 0
        assert status["tick_count"] == 200
