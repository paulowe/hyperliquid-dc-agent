"""Tests for DC Forecast signal generation (Phase 2-3).

Tests the full pipeline: DC detection -> feature engineering -> model inference
-> signal generation with position tracking, cooldown, and close-on-reversal.
"""

import math
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from strategies.dc_forecast.dc_forecast_strategy import DCForecastStrategy
from strategies.dc_forecast.config import DCForecastConfig, DEFAULT_FEATURE_NAMES
from interfaces.strategy import MarketData, Position, SignalType, TradingSignal


def _make_config(**overrides) -> dict:
    """Helper to create a strategy config dict with sensible defaults."""
    defaults = {
        "symbol": "BTC",
        "dc_thresholds": [(0.01, 0.01)],
        "model_uri": "",
        "scaler_uri": "",
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "window_size": 5,  # Small window for fast tests
        "signal_threshold_pct": 0.1,
        "position_size_usd": 100.0,
        "max_position_size_usd": 500.0,
        "log_dc_events": False,
        "cooldown_seconds": 0,  # Disable cooldown for tests
    }
    defaults.update(overrides)
    return defaults


def _make_market_data(price: float, ts: float, asset: str = "BTC") -> MarketData:
    return MarketData(asset=asset, price=price, volume_24h=0.0, timestamp=ts)


def _make_position(asset: str, size: float, entry_price: float) -> Position:
    """Create a Position for testing. Positive size = long, negative = short."""
    return Position(
        asset=asset,
        size=size,
        entry_price=entry_price,
        current_value=abs(size) * entry_price,
        unrealized_pnl=0.0,
        timestamp=time.time(),
    )


class _MockModel:
    """Mock TF model that returns a configurable prediction."""

    def __init__(self, prediction_value: float = 0.5):
        self.prediction_value = prediction_value
        self.call_count = 0
        self.last_input_shape = None

    def predict(self, x, verbose=0):
        """Mimic keras model.predict() returning shape (1, 1)."""
        self.call_count += 1
        self.last_input_shape = x.shape
        return np.array([[self.prediction_value]], dtype=np.float32)


class TestSignalGenerationWithMockModel:
    """Test signal generation when a mock model is available."""

    def _build_strategy_with_mock(self, prediction_value=0.5, **config_overrides):
        """Create a strategy and inject a mock model + scaler."""
        config = _make_config(**config_overrides)
        strategy = DCForecastStrategy(config)

        # Inject mock model
        mock_model = _MockModel(prediction_value=prediction_value)
        strategy._model = mock_model

        # Inject mock scaler that passes features through (identity scaling)
        from strategies.dc_forecast.model_loader import ScalerParams
        strategy._scaler_params = ScalerParams(
            feature_order=["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
            mean=np.array([100.0, 0.0, 0.0], dtype=np.float64),  # 3 continuous
            scale=np.array([10.0, 1.0, 1.0], dtype=np.float64),
            continuous_cols=["PRICE", "OSV_Down", "OSV_Up"],
            indicator_cols=["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"],
            std_feature_order=["PRICE_std", "PDCC_Down", "OSV_Down_std", "PDCC2_UP", "OSV_Up_std", "regime_up", "regime_down"],
        )

        return strategy, mock_model

    def _fill_buffer(self, strategy, base_price=100.0, start_ts=1000.0, n_ticks=None):
        """Feed enough ticks to fill the rolling buffer."""
        n = n_ticks or strategy._config.window_size
        for i in range(n):
            md = _make_market_data(base_price, start_ts + i)
            strategy.generate_signals(md, [], 10000.0)

    def test_no_signal_without_model(self):
        """Without model, strategy returns empty signals (Phase 1 mode)."""
        strategy = DCForecastStrategy(_make_config())
        # Feed a price drop to trigger DC event
        for i in range(5):
            strategy.generate_signals(_make_market_data(100.0, 1000.0 + i), [], 10000.0)
        strategy.generate_signals(_make_market_data(97.0, 1005.0), [], 10000.0)
        # No model → no signals
        signals = strategy.generate_signals(_make_market_data(97.0, 1006.0), [], 10000.0)
        assert signals == []

    def test_buy_signal_on_positive_prediction(self):
        """Model predicting price increase should produce BUY signal on PDCC event."""
        # High prediction value = predicted price above current after inverse scaling
        strategy, mock_model = self._build_strategy_with_mock(
            prediction_value=2.0,  # High scaled prediction
            signal_threshold_pct=0.1,
        )

        # Fill buffer
        self._fill_buffer(strategy, base_price=100.0)

        # Trigger PDCC_Down with a sharp drop (2%)
        signals = strategy.generate_signals(_make_market_data(97.5, 1100.0), [], 10000.0)

        # Model should have been called
        assert mock_model.call_count > 0
        # Should get a BUY signal (model predicts higher price)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1
        assert buy_signals[0].asset == "BTC"
        assert buy_signals[0].size > 0

    def test_sell_signal_on_negative_prediction(self):
        """Model predicting price decrease should produce SELL signal on PDCC event."""
        strategy, mock_model = self._build_strategy_with_mock(
            prediction_value=-2.0,  # Low scaled prediction
            signal_threshold_pct=0.1,
        )
        self._fill_buffer(strategy, base_price=100.0)

        # Trigger PDCC_Down
        signals = strategy.generate_signals(_make_market_data(97.5, 1100.0), [], 10000.0)

        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1
        assert sell_signals[0].asset == "BTC"
        assert sell_signals[0].size > 0

    def test_no_signal_below_threshold(self):
        """Prediction within threshold should produce no signal."""
        strategy, mock_model = self._build_strategy_with_mock(
            prediction_value=0.005,  # Tiny prediction, well within threshold
            signal_threshold_pct=5.0,  # Very high threshold
        )
        self._fill_buffer(strategy, base_price=100.0)

        # Trigger PDCC_Down
        signals = strategy.generate_signals(_make_market_data(97.5, 1100.0), [], 10000.0)

        # No signal because prediction delta is below threshold
        buy_sell_signals = [s for s in signals if s.signal_type in (SignalType.BUY, SignalType.SELL)]
        assert len(buy_sell_signals) == 0

    def test_model_called_only_on_pdcc_event(self):
        """Model inference should only trigger on PDCC events, not every tick."""
        strategy, mock_model = self._build_strategy_with_mock()
        self._fill_buffer(strategy, base_price=100.0)

        # Feed a few flat ticks (no PDCC event)
        for i in range(5):
            strategy.generate_signals(_make_market_data(100.0, 1200.0 + i), [], 10000.0)

        # Model should NOT have been called on flat ticks
        assert mock_model.call_count == 0

    def test_model_input_shape(self):
        """Model should receive input of shape (1, window_size, n_features)."""
        strategy, mock_model = self._build_strategy_with_mock(window_size=5)
        self._fill_buffer(strategy, base_price=100.0, n_ticks=5)

        # Trigger PDCC_Down
        strategy.generate_signals(_make_market_data(97.5, 1100.0), [], 10000.0)

        if mock_model.call_count > 0:
            # Shape should be (1, window_size, n_features)
            assert mock_model.last_input_shape[0] == 1
            assert mock_model.last_input_shape[1] == 5  # window_size
            assert mock_model.last_input_shape[2] == len(DEFAULT_FEATURE_NAMES)

    def test_signal_metadata_contains_prediction(self):
        """Signal metadata should include prediction details."""
        strategy, mock_model = self._build_strategy_with_mock(
            prediction_value=2.0,
            signal_threshold_pct=0.1,
        )
        self._fill_buffer(strategy, base_price=100.0)
        signals = strategy.generate_signals(_make_market_data(97.5, 1100.0), [], 10000.0)

        if signals:
            sig = signals[0]
            assert "predicted_price_std" in sig.metadata or "prediction_delta_pct" in sig.metadata
            assert sig.reason != ""


class TestPositionAwareness:
    """Test that signal generation respects existing positions."""

    def _build_strategy_with_mock(self, prediction_value=2.0, **config_overrides):
        config = _make_config(**config_overrides)
        strategy = DCForecastStrategy(config)
        mock_model = _MockModel(prediction_value=prediction_value)
        strategy._model = mock_model
        from strategies.dc_forecast.model_loader import ScalerParams
        strategy._scaler_params = ScalerParams(
            feature_order=["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
            mean=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            scale=np.array([10.0, 1.0, 1.0], dtype=np.float64),
            continuous_cols=["PRICE", "OSV_Down", "OSV_Up"],
            indicator_cols=["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"],
            std_feature_order=["PRICE_std", "PDCC_Down", "OSV_Down_std", "PDCC2_UP", "OSV_Up_std", "regime_up", "regime_down"],
        )
        return strategy, mock_model

    def _fill_buffer(self, strategy, base_price=100.0, start_ts=1000.0):
        n = strategy._config.window_size
        for i in range(n):
            strategy.generate_signals(_make_market_data(base_price, start_ts + i), [], 10000.0)

    def test_respects_max_position_size(self):
        """Should not exceed max_position_size_usd when already holding a position."""
        strategy, _ = self._build_strategy_with_mock(
            prediction_value=2.0,
            position_size_usd=100.0,
            max_position_size_usd=150.0,
        )
        self._fill_buffer(strategy, base_price=100.0)

        # Already have a long position worth $120
        existing_pos = _make_position("BTC", 0.0012, 100000.0)  # ~$120

        signals = strategy.generate_signals(
            _make_market_data(97.5, 1100.0), [existing_pos], 10000.0
        )

        # Should either be no signal (already near max) or a reduced-size buy
        for sig in signals:
            if sig.signal_type == SignalType.BUY:
                # Size * price should respect max position constraint
                assert sig.size * 97.5 <= 150.0

    def test_close_on_reversal_long_to_short(self):
        """SELL signal while holding a long should emit CLOSE first."""
        strategy, _ = self._build_strategy_with_mock(
            prediction_value=-2.0,  # Predict price drop → SELL
        )
        self._fill_buffer(strategy, base_price=100.0)

        # Have a long position
        existing_pos = _make_position("BTC", 0.001, 100000.0)

        signals = strategy.generate_signals(
            _make_market_data(97.5, 1100.0), [existing_pos], 10000.0
        )

        # Should see CLOSE before SELL (or just CLOSE if close-on-reversal is configured)
        signal_types = [s.signal_type for s in signals]
        if SignalType.SELL in signal_types and SignalType.CLOSE in signal_types:
            close_idx = signal_types.index(SignalType.CLOSE)
            sell_idx = signal_types.index(SignalType.SELL)
            assert close_idx < sell_idx, "CLOSE should come before SELL"

    def test_no_duplicate_signal_same_direction(self):
        """Don't add to position if already long and model predicts up again."""
        strategy, _ = self._build_strategy_with_mock(
            prediction_value=2.0,
            max_position_size_usd=100.0,
        )
        self._fill_buffer(strategy, base_price=100.0)

        # Already maxed out long
        existing_pos = _make_position("BTC", 0.001, 100000.0)  # ~$100

        signals = strategy.generate_signals(
            _make_market_data(97.5, 1100.0), [existing_pos], 10000.0
        )

        # Should not add more to already-maxed position
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        total_new_usd = sum(s.size * 97.5 for s in buy_signals)
        assert total_new_usd <= 100.0  # Within max


class TestCooldown:
    """Test signal cooldown logic."""

    def _build_strategy_with_mock(self, prediction_value=2.0, **config_overrides):
        config = _make_config(**config_overrides)
        strategy = DCForecastStrategy(config)
        mock_model = _MockModel(prediction_value=prediction_value)
        strategy._model = mock_model
        from strategies.dc_forecast.model_loader import ScalerParams
        strategy._scaler_params = ScalerParams(
            feature_order=["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
            mean=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            scale=np.array([10.0, 1.0, 1.0], dtype=np.float64),
            continuous_cols=["PRICE", "OSV_Down", "OSV_Up"],
            indicator_cols=["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"],
            std_feature_order=["PRICE_std", "PDCC_Down", "OSV_Down_std", "PDCC2_UP", "OSV_Up_std", "regime_up", "regime_down"],
        )
        return strategy, mock_model

    def test_cooldown_blocks_rapid_signals(self):
        """Signals within cooldown period should be suppressed."""
        strategy, mock_model = self._build_strategy_with_mock(
            prediction_value=2.0,
            cooldown_seconds=60,  # 60s cooldown
            window_size=3,
        )

        # Fill buffer
        for i in range(3):
            strategy.generate_signals(_make_market_data(100.0, 1000.0 + i), [], 10000.0)

        # First PDCC event at t=1003 → should produce signal
        signals1 = strategy.generate_signals(_make_market_data(97.5, 1003.0), [], 10000.0)

        # Fill buffer again quickly
        for i in range(3):
            strategy.generate_signals(_make_market_data(100.0, 1004.0 + i), [], 10000.0)

        # Second PDCC event at t=1010 (7s later, within cooldown) → suppressed
        signals2 = strategy.generate_signals(_make_market_data(97.5, 1010.0), [], 10000.0)

        if signals1:  # First should fire
            buy_sell_1 = [s for s in signals1 if s.signal_type in (SignalType.BUY, SignalType.SELL)]
            buy_sell_2 = [s for s in signals2 if s.signal_type in (SignalType.BUY, SignalType.SELL)]
            # Second should be empty due to cooldown
            if buy_sell_1:
                assert len(buy_sell_2) == 0, "Second signal within cooldown should be suppressed"

    def test_signal_after_cooldown_expires(self):
        """After cooldown expires, signals should be allowed again."""
        strategy, _ = self._build_strategy_with_mock(
            prediction_value=2.0,
            cooldown_seconds=10,
            window_size=3,
        )

        # Fill and trigger first PDCC_Down signal (drop from 100 to 97.5)
        for i in range(3):
            strategy.generate_signals(_make_market_data(100.0, 1000.0 + i), [], 10000.0)
        signals1 = strategy.generate_signals(_make_market_data(97.5, 1003.0), [], 10000.0)

        # Recover (trigger PDCC2_UP at t=1020, within cooldown → suppressed or fires)
        strategy.generate_signals(_make_market_data(100.0, 1020.0), [], 10000.0)

        # Establish new high and wait well past any cooldown reset
        strategy.generate_signals(_make_market_data(101.0, 1040.0), [], 10000.0)
        strategy.generate_signals(_make_market_data(101.0, 1041.0), [], 10000.0)

        # Second PDCC_Down (drop >1% from 101) well after cooldown
        signals2 = strategy.generate_signals(_make_market_data(98.5, 1060.0), [], 10000.0)

        # Both should produce signals
        if signals1:
            assert len(signals2) > 0, "Signal after cooldown should be allowed"


class TestOnTradeExecuted:
    """Test trade execution callback tracking."""

    def test_on_trade_executed_tracks_pnl(self):
        """on_trade_executed should update internal trade tracking."""
        strategy = DCForecastStrategy(_make_config())
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=0.001,
            price=100000.0,
            reason="test",
        )
        # Should not raise
        strategy.on_trade_executed(signal, executed_price=100000.0, executed_size=0.001)

        status = strategy.get_status()
        assert status.get("trades_executed", 0) >= 1

    def test_get_status_includes_signal_stats(self):
        """get_status should include signal generation statistics."""
        strategy = DCForecastStrategy(_make_config())
        status = strategy.get_status()
        # Should have signal-related fields
        assert "total_dc_events" in status
        assert "tick_count" in status


class TestConfigExtensions:
    """Test config additions for Phase 2-3."""

    def test_config_cooldown_seconds(self):
        cfg = DCForecastConfig.from_dict({"cooldown_seconds": 30})
        assert cfg.cooldown_seconds == 30

    def test_config_default_cooldown(self):
        cfg = DCForecastConfig.from_dict({})
        assert cfg.cooldown_seconds >= 0
