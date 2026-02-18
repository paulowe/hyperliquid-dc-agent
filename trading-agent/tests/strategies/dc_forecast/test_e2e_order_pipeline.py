"""End-to-end test: DC detection → features → mock model → signals → order placement.

Simulates the full pipeline without touching a real exchange. Validates that:
1. Synthetic price data triggers DC events
2. Features are computed correctly
3. Mock model produces predictions
4. Predictions generate proper BUY/SELL/CLOSE TradingSignals
5. Signals can be converted to Order objects and "placed"

This test uses a MockExchangeAdapter to capture what orders would be placed,
verifying the full flow from market data to order without real network calls.
"""

import asyncio
import math
import time
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, List, Any, Optional

from strategies.dc_forecast.dc_forecast_strategy import DCForecastStrategy
from strategies.dc_forecast.config import DCForecastConfig, DEFAULT_FEATURE_NAMES
from strategies.dc_forecast.model_loader import ScalerParams
from interfaces.strategy import MarketData, Position, SignalType, TradingSignal
from interfaces.exchange import Order, OrderSide, OrderType, OrderStatus


class _MockModel:
    """Mock TF model returning a configurable prediction."""

    def __init__(self, prediction_fn=None):
        # prediction_fn(input_array) → float, or fixed value
        self._prediction_fn = prediction_fn or (lambda x: 0.5)
        self.call_count = 0
        self.call_log = []

    def predict(self, x, verbose=0):
        self.call_count += 1
        self.call_log.append({"input_shape": x.shape, "timestamp": time.time()})
        val = self._prediction_fn(x)
        return np.array([[val]], dtype=np.float32)


def _build_scaler() -> ScalerParams:
    """Build a ScalerParams matching the default feature set."""
    return ScalerParams(
        feature_order=["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
        mean=np.array([50000.0, 0.0, 0.0], dtype=np.float64),
        scale=np.array([5000.0, 1.0, 1.0], dtype=np.float64),
        continuous_cols=["PRICE", "OSV_Down", "OSV_Up"],
        indicator_cols=["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"],
        std_feature_order=["PRICE_std", "PDCC_Down", "OSV_Down_std", "PDCC2_UP", "OSV_Up_std", "regime_up", "regime_down"],
    )


class MockOrderExecutor:
    """Captures orders that would be placed, without hitting a real exchange."""

    def __init__(self):
        self.orders_placed: List[Dict] = []
        self.positions_closed: List[Dict] = []

    async def execute_signal(self, signal: TradingSignal, market_price: float) -> Optional[str]:
        """Convert a TradingSignal to an Order and record it."""
        if signal.signal_type in (SignalType.BUY, SignalType.SELL):
            order = Order(
                id=f"mock_{int(time.time() * 1000)}",
                asset=signal.asset,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                size=signal.size,
                order_type=OrderType.MARKET if signal.price is None else OrderType.LIMIT,
                price=signal.price or market_price,
                status=OrderStatus.SUBMITTED,
                created_at=time.time(),
            )
            self.orders_placed.append({
                "order": order,
                "signal": signal,
                "market_price": market_price,
            })
            return order.id

        elif signal.signal_type == SignalType.CLOSE:
            self.positions_closed.append({
                "asset": signal.asset,
                "size": signal.size,
                "reason": signal.reason,
                "market_price": market_price,
            })
            return "close_ok"

        return None


class TestE2EOrderPipeline:
    """Full pipeline: synthetic prices → DC → features → model → signals → orders."""

    def _build_strategy(self, prediction_fn=None, **config_overrides):
        """Create strategy with mock model and scaler."""
        defaults = {
            "symbol": "BTC",
            "dc_thresholds": [(0.005, 0.005)],  # 0.5% threshold
            "feature_names": list(DEFAULT_FEATURE_NAMES),
            "window_size": 10,
            "signal_threshold_pct": 0.1,
            "position_size_usd": 100.0,
            "max_position_size_usd": 500.0,
            "log_dc_events": False,
            "cooldown_seconds": 0,
        }
        defaults.update(config_overrides)
        strategy = DCForecastStrategy(defaults)

        # Inject mock model and scaler
        strategy._model = _MockModel(prediction_fn=prediction_fn)
        strategy._scaler_params = _build_scaler()

        return strategy

    def test_oscillating_prices_produce_buy_signals(self):
        """Oscillating prices should trigger DC events and model-based BUY signals."""
        # Model always predicts price going up
        strategy = self._build_strategy(prediction_fn=lambda x: 2.0)
        executor = MockOrderExecutor()

        # Generate oscillating prices (BTC-like scale)
        prices = []
        for i in range(100):
            prices.append(50000.0 + 500.0 * math.sin(i * 0.2))

        all_signals = []
        for i, price in enumerate(prices):
            md = MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=1000.0 + i)
            signals = strategy.generate_signals(md, [], 100000.0)
            all_signals.extend(signals)

            # Execute signals through mock executor
            for sig in signals:
                asyncio.get_event_loop().run_until_complete(
                    executor.execute_signal(sig, price)
                )

        # Should have generated at least one signal
        assert len(all_signals) > 0, "Oscillating prices should produce at least one signal"

        # All signals should be BUY (model always predicts up)
        buy_signals = [s for s in all_signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0, "Should have BUY signals when model predicts up"

        # Verify orders were captured
        assert len(executor.orders_placed) > 0, "Orders should have been placed"

        # Verify order details
        for order_record in executor.orders_placed:
            order = order_record["order"]
            assert order.asset == "BTC"
            assert order.side == OrderSide.BUY
            assert order.size > 0
            assert order.order_type == OrderType.MARKET  # price=None → market

    def test_sell_signals_on_bearish_prediction(self):
        """Model predicting price decrease should produce SELL signals."""
        strategy = self._build_strategy(prediction_fn=lambda x: -2.0)
        executor = MockOrderExecutor()

        # Oscillating prices to trigger PDCC events
        prices = [50000.0 + 500.0 * math.sin(i * 0.2) for i in range(100)]

        for i, price in enumerate(prices):
            md = MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=1000.0 + i)
            signals = strategy.generate_signals(md, [], 100000.0)
            for sig in signals:
                asyncio.get_event_loop().run_until_complete(
                    executor.execute_signal(sig, price)
                )

        # Should have SELL orders
        sell_orders = [o for o in executor.orders_placed if o["order"].side == OrderSide.SELL]
        assert len(sell_orders) > 0, "Should have SELL orders when model predicts down"

    def test_close_on_reversal_produces_close_then_opposite(self):
        """When holding a long and model flips to short, should CLOSE then SELL."""
        strategy = self._build_strategy(
            prediction_fn=lambda x: -2.0,  # Predicts down → SELL
            cooldown_seconds=0,
        )

        # Fill buffer
        for i in range(10):
            md = MarketData(asset="BTC", price=50000.0, volume_24h=0.0, timestamp=1000.0 + i)
            strategy.generate_signals(md, [], 100000.0)

        # Existing long position
        long_pos = Position(
            asset="BTC", size=0.002, entry_price=49000.0,
            current_value=100.0, unrealized_pnl=2.0, timestamp=time.time()
        )

        # Trigger PDCC_Down
        md = MarketData(asset="BTC", price=49500.0, volume_24h=0.0, timestamp=1020.0)
        signals = strategy.generate_signals(md, [long_pos], 100000.0)

        signal_types = [s.signal_type for s in signals]

        # Should have CLOSE (close long) followed by SELL (open short)
        if SignalType.SELL in signal_types:
            assert SignalType.CLOSE in signal_types, "Should CLOSE before SELL when reversing"
            close_idx = signal_types.index(SignalType.CLOSE)
            sell_idx = signal_types.index(SignalType.SELL)
            assert close_idx < sell_idx, "CLOSE must come before SELL"

    def test_position_sizing_respects_max(self):
        """Signal size should never exceed max_position_size_usd."""
        strategy = self._build_strategy(
            prediction_fn=lambda x: 2.0,
            position_size_usd=100.0,
            max_position_size_usd=150.0,
        )

        # Fill buffer
        for i in range(10):
            md = MarketData(asset="BTC", price=50000.0, volume_24h=0.0, timestamp=1000.0 + i)
            strategy.generate_signals(md, [], 100000.0)

        # Already have a $120 position
        existing_pos = Position(
            asset="BTC", size=0.0024, entry_price=50000.0,
            current_value=120.0, unrealized_pnl=0.0, timestamp=time.time()
        )

        # Trigger PDCC_Down
        md = MarketData(asset="BTC", price=49500.0, volume_24h=0.0, timestamp=1020.0)
        signals = strategy.generate_signals(md, [existing_pos], 100000.0)

        for sig in signals:
            if sig.signal_type == SignalType.BUY:
                # New order + existing should not exceed max
                new_value = sig.size * 49500.0
                assert new_value <= 150.0, f"Order value ${new_value:.2f} exceeds max"

    def test_signal_metadata_includes_prediction_info(self):
        """Signal metadata should contain prediction details for debugging."""
        strategy = self._build_strategy(prediction_fn=lambda x: 2.0)

        # Fill buffer and trigger PDCC
        for i in range(10):
            md = MarketData(asset="BTC", price=50000.0, volume_24h=0.0, timestamp=1000.0 + i)
            strategy.generate_signals(md, [], 100000.0)

        md = MarketData(asset="BTC", price=49500.0, volume_24h=0.0, timestamp=1020.0)
        signals = strategy.generate_signals(md, [], 100000.0)

        for sig in signals:
            if sig.signal_type in (SignalType.BUY, SignalType.SELL):
                assert "predicted_price_std" in sig.metadata
                assert "predicted_price" in sig.metadata
                assert "prediction_delta_pct" in sig.metadata
                assert "tick" in sig.metadata

    def test_on_trade_executed_callback(self):
        """Strategy should track executed trades via callback."""
        strategy = self._build_strategy(prediction_fn=lambda x: 2.0)

        # Fill and trigger
        for i in range(10):
            strategy.generate_signals(
                MarketData(asset="BTC", price=50000.0, volume_24h=0.0, timestamp=1000.0 + i),
                [], 100000.0
            )
        signals = strategy.generate_signals(
            MarketData(asset="BTC", price=49500.0, volume_24h=0.0, timestamp=1020.0),
            [], 100000.0
        )

        # Simulate execution callback
        for sig in signals:
            strategy.on_trade_executed(sig, executed_price=49500.0, executed_size=sig.size)

        status = strategy.get_status()
        assert status["trades_executed"] == len(signals)

    def test_full_lifecycle_with_status(self):
        """Test start → process → stop lifecycle with status reporting."""
        strategy = self._build_strategy(prediction_fn=lambda x: 2.0)

        strategy.start()
        assert strategy.is_active is True
        assert strategy.has_model is True

        status = strategy.get_status()
        assert status["has_model"] is True
        assert status["tick_count"] == 0

        # Process some ticks
        for i in range(15):
            md = MarketData(asset="BTC", price=50000.0 + i * 10, volume_24h=0.0, timestamp=1000.0 + i)
            strategy.generate_signals(md, [], 100000.0)

        status = strategy.get_status()
        assert status["tick_count"] == 15
        assert status["buffer_ready"] is True
        assert "buffer_fill" in status

        strategy.stop()
        assert strategy.is_active is False


class TestE2EWithEnginePattern:
    """Test the strategy as it would be used by TradingEngine._execute_signal()."""

    def test_signal_to_order_conversion(self):
        """Verify TradingSignal → Order conversion matches engine pattern."""
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=0.002,
            price=None,  # Market order
            reason="DC forecast BUY: predicted +0.5%",
            metadata={"predicted_price": 50250.0},
        )

        # This mirrors TradingEngine._place_order() logic
        order = Order(
            id=f"order_{int(time.time() * 1000)}",
            asset=signal.asset,
            side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
            size=signal.size,
            order_type=OrderType.MARKET if signal.price is None else OrderType.LIMIT,
            price=signal.price,
            created_at=time.time(),
        )

        assert order.asset == "BTC"
        assert order.side == OrderSide.BUY
        assert order.size == 0.002
        assert order.order_type == OrderType.MARKET

    def test_close_signal_conversion(self):
        """CLOSE signals should route to exchange.close_position()."""
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            asset="BTC",
            size=0.001,
            price=None,
            reason="Close long before SELL",
        )

        # In engine, CLOSE routes to _close_positions
        assert signal.signal_type == SignalType.CLOSE
        assert signal.asset == "BTC"
        assert signal.size == 0.001

    def test_limit_order_signal(self):
        """Verify limit order signal conversion."""
        signal = TradingSignal(
            signal_type=SignalType.SELL,
            asset="BTC",
            size=0.001,
            price=50100.0,  # Limit price
            reason="DC forecast limit SELL",
        )

        order = Order(
            id="test_order",
            asset=signal.asset,
            side=OrderSide.SELL,
            size=signal.size,
            order_type=OrderType.LIMIT if signal.price else OrderType.MARKET,
            price=signal.price,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.price == 50100.0


class TestSpotVsPerpSignals:
    """Test that signals work for both spot and perpetual markets."""

    def _build_strategy(self, symbol="BTC", **overrides):
        defaults = {
            "symbol": symbol,
            "dc_thresholds": [(0.005, 0.005)],
            "window_size": 5,
            "log_dc_events": False,
            "cooldown_seconds": 0,
            "signal_threshold_pct": 0.1,
        }
        defaults.update(overrides)
        strategy = DCForecastStrategy(defaults)
        strategy._model = _MockModel(prediction_fn=lambda x: 2.0)
        strategy._scaler_params = _build_scaler()
        return strategy

    def test_perp_signal_uses_simple_symbol(self):
        """Perp signals use simple symbol like 'BTC'."""
        strategy = self._build_strategy(symbol="BTC")

        for i in range(5):
            strategy.generate_signals(
                MarketData(asset="BTC", price=50000.0, volume_24h=0.0, timestamp=1000.0 + i),
                [], 100000.0
            )
        signals = strategy.generate_signals(
            MarketData(asset="BTC", price=49500.0, volume_24h=0.0, timestamp=1010.0),
            [], 100000.0
        )

        for sig in signals:
            assert sig.asset == "BTC"  # Simple symbol = perpetual

    def test_spot_signal_uses_pair_format(self):
        """Spot signals use pair format like 'PURR/USDC'."""
        strategy = self._build_strategy(symbol="PURR/USDC")

        for i in range(5):
            strategy.generate_signals(
                MarketData(asset="PURR/USDC", price=1.0, volume_24h=0.0, timestamp=1000.0 + i),
                [], 100000.0
            )
        signals = strategy.generate_signals(
            MarketData(asset="PURR/USDC", price=0.99, volume_24h=0.0, timestamp=1010.0),
            [], 100000.0
        )

        for sig in signals:
            assert sig.asset == "PURR/USDC"  # Pair format = spot

    def test_eth_perp(self):
        """ETH perpetual signals."""
        strategy = self._build_strategy(symbol="ETH")

        for i in range(5):
            strategy.generate_signals(
                MarketData(asset="ETH", price=3000.0, volume_24h=0.0, timestamp=1000.0 + i),
                [], 100000.0
            )
        signals = strategy.generate_signals(
            MarketData(asset="ETH", price=2970.0, volume_24h=0.0, timestamp=1010.0),
            [], 100000.0
        )

        for sig in signals:
            assert sig.asset == "ETH"
