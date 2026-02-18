"""Tests for TradingEngine signal execution, focusing on CLOSE signal handling."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from interfaces.strategy import TradingSignal, SignalType, MarketData, Position
from interfaces.exchange import Order, OrderSide, OrderType, OrderStatus


class MockExchange:
    """Minimal mock exchange for testing engine signal execution."""

    def __init__(self):
        self.orders_placed = []
        self.positions_closed = []
        self.orders_cancelled = 0
        self.is_connected = True

    async def place_order(self, order: Order) -> str:
        self.orders_placed.append(order)
        return f"mock_oid_{len(self.orders_placed)}"

    async def close_position(self, asset: str, size=None) -> bool:
        self.positions_closed.append({"asset": asset, "size": size})
        return True

    async def cancel_all_orders(self) -> int:
        self.orders_cancelled += 1
        return 3  # Pretend 3 orders were cancelled

    async def get_positions(self):
        return []


class MockStrategy:
    """Minimal mock strategy for testing engine callbacks."""

    def __init__(self):
        self.trade_callbacks = []
        self.error_callbacks = []
        self.is_active = True

    def on_trade_executed(self, signal, executed_price, executed_size):
        self.trade_callbacks.append({
            "signal": signal,
            "executed_price": executed_price,
            "executed_size": executed_size,
        })

    def on_error(self, error, context):
        self.error_callbacks.append({"error": error, "context": context})


def _build_engine():
    """Create a TradingEngine with mock dependencies."""
    from core.engine import TradingEngine

    config = {"log_level": "WARNING"}
    engine = TradingEngine(config)
    engine.exchange = MockExchange()
    engine.strategy = MockStrategy()
    engine.running = True
    return engine


class TestExecuteSignalRouting:
    """Test that _execute_signal routes to the correct handler."""

    @pytest.mark.asyncio
    async def test_buy_signal_routes_to_place_order(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=0.001,
            price=50000.0,
            reason="test buy",
        )
        await engine._execute_signal(signal)

        assert len(engine.exchange.orders_placed) == 1
        order = engine.exchange.orders_placed[0]
        assert order.side == OrderSide.BUY
        assert order.asset == "BTC"

    @pytest.mark.asyncio
    async def test_sell_signal_routes_to_place_order(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.SELL,
            asset="BTC",
            size=0.001,
            price=None,  # Market order
            reason="test sell",
        )
        await engine._execute_signal(signal)

        assert len(engine.exchange.orders_placed) == 1
        order = engine.exchange.orders_placed[0]
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_close_signal_routes_to_close_positions(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            asset="BTC",
            size=0.002,
            reason="Close before reversal",
        )
        await engine._execute_signal(signal)

        # Should call close_position, not place_order
        assert len(engine.exchange.orders_placed) == 0
        assert len(engine.exchange.positions_closed) == 1
        assert engine.exchange.positions_closed[0]["asset"] == "BTC"
        assert engine.exchange.positions_closed[0]["size"] == 0.002


class TestClosePositionsHandler:
    """Test _close_positions specifically."""

    @pytest.mark.asyncio
    async def test_close_position_for_asset(self):
        """CLOSE signal should call exchange.close_position(asset, size)."""
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            asset="ETH",
            size=0.5,
            reason="Reversal detected",
        )
        await engine._close_positions(signal)

        assert len(engine.exchange.positions_closed) == 1
        closed = engine.exchange.positions_closed[0]
        assert closed["asset"] == "ETH"
        assert closed["size"] == 0.5

    @pytest.mark.asyncio
    async def test_cancel_all_action(self):
        """cancel_all metadata should cancel all orders."""
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            asset="BTC",
            size=0,
            reason="Grid rebalance",
            metadata={"action": "cancel_all"},
        )
        await engine._close_positions(signal)

        # Should cancel orders, not close position
        assert engine.exchange.orders_cancelled == 1
        assert len(engine.exchange.positions_closed) == 0

    @pytest.mark.asyncio
    async def test_close_triggers_trade_executed_callback(self):
        """Successful close should notify strategy via on_trade_executed."""
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            asset="BTC",
            size=0.001,
            reason="test close callback",
        )
        await engine._close_positions(signal)

        assert len(engine.strategy.trade_callbacks) == 1
        callback = engine.strategy.trade_callbacks[0]
        assert callback["signal"] == signal
        assert callback["executed_size"] == 0.001

    @pytest.mark.asyncio
    async def test_close_increments_trade_counter(self):
        """Successful close should increment executed_trades."""
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            asset="BTC",
            size=0.001,
            reason="test counter",
        )
        assert engine.executed_trades == 0
        await engine._close_positions(signal)
        assert engine.executed_trades == 1


class TestPlaceOrderHandler:
    """Test _place_order specifically."""

    @pytest.mark.asyncio
    async def test_limit_order_creation(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=0.001,
            price=50000.0,
            reason="limit buy",
        )
        await engine._place_order(signal)

        order = engine.exchange.orders_placed[0]
        assert order.order_type == OrderType.LIMIT
        assert order.price == 50000.0
        assert order.size == 0.001

    @pytest.mark.asyncio
    async def test_market_order_creation(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.SELL,
            asset="BTC",
            size=0.002,
            price=None,  # Market order
            reason="market sell",
        )
        await engine._place_order(signal)

        order = engine.exchange.orders_placed[0]
        assert order.order_type == OrderType.MARKET
        assert order.price is None
        assert order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_order_tracked_as_pending(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=0.001,
            price=50000.0,
        )
        await engine._place_order(signal)

        assert len(engine.pending_orders) == 1
        order = list(engine.pending_orders.values())[0]
        assert order.status == OrderStatus.SUBMITTED
        assert order.exchange_order_id is not None

    @pytest.mark.asyncio
    async def test_strategy_notified_of_execution(self):
        engine = _build_engine()
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=0.001,
            price=50000.0,
        )
        await engine._place_order(signal)

        assert len(engine.strategy.trade_callbacks) == 1
        assert engine.executed_trades == 1
