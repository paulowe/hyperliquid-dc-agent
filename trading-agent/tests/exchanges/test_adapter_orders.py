"""Tests for HyperliquidAdapter order placement and position management.

Uses mocks to verify the adapter calls the SDK correctly without
requiring a live connection.
"""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from interfaces.exchange import Order, OrderSide, OrderType, OrderStatus
from interfaces.strategy import Position


class MockInfo:
    """Mock Hyperliquid SDK Info object."""

    def __init__(self, mids=None, user_state=None):
        self._mids = mids or {"BTC": "50000.0", "ETH": "3000.0"}
        self._user_state = user_state or {
            "assetPositions": [],
            "crossMarginSummary": {"accountValue": "10000"},
        }

    def all_mids(self):
        return self._mids

    def user_state(self, address):
        return self._user_state

    def open_orders(self, address):
        return []

    def meta(self):
        return {"universe": [{"name": "BTC", "szDecimals": 5, "priceDecimals": 0}]}


class MockExchangeSDK:
    """Mock Hyperliquid SDK Exchange object."""

    def __init__(self):
        self.wallet = MagicMock()
        self.wallet.address = "0xTestAddress"
        self.order_calls = []
        self.cancel_calls = []

    def order(self, **kwargs):
        """Capture order calls and return a success response."""
        self.order_calls.append(kwargs)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [{"resting": {"oid": 12345}}]
                }
            },
        }

    def cancel(self, **kwargs):
        self.cancel_calls.append(kwargs)
        return {"status": "ok", "response": {"data": {"statuses": ["success"]}}}


def _build_adapter(mids=None, positions=None):
    """Create a HyperliquidAdapter with mocked SDK components."""
    from exchanges.hyperliquid.adapter import HyperliquidAdapter

    adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
    adapter.is_connected = True

    # Inject mock SDK objects
    user_state = {
        "assetPositions": positions or [],
        "crossMarginSummary": {"accountValue": "10000"},
    }
    adapter.info = MockInfo(mids=mids, user_state=user_state)
    adapter.exchange = MockExchangeSDK()

    # Build precision cache from mock meta()
    adapter._build_precision_cache()

    return adapter


class TestPlaceOrder:
    """Test the place_order method."""

    @pytest.mark.asyncio
    async def test_market_buy_order(self):
        adapter = _build_adapter()
        order = Order(
            id="test_1",
            asset="BTC",
            side=OrderSide.BUY,
            size=0.001,
            order_type=OrderType.MARKET,
            price=None,
        )
        oid = await adapter.place_order(order)

        assert oid == "12345"
        call = adapter.exchange.order_calls[0]
        assert call["name"] == "BTC"
        assert call["is_buy"] is True
        assert call["sz"] == 0.001
        # Market order uses IOC
        assert call["reduce_only"] is False

    @pytest.mark.asyncio
    async def test_limit_sell_order(self):
        adapter = _build_adapter()
        order = Order(
            id="test_2",
            asset="BTC",
            side=OrderSide.SELL,
            size=0.002,
            order_type=OrderType.LIMIT,
            price=51000.0,
        )
        oid = await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        assert call["name"] == "BTC"
        assert call["is_buy"] is False
        assert call["limit_px"] == 51000.0  # Rounded for BTC
        assert call["reduce_only"] is False

    @pytest.mark.asyncio
    async def test_btc_price_rounded_to_integer(self):
        """BTC prices should be rounded to whole dollars."""
        adapter = _build_adapter()
        order = Order(
            id="test_3",
            asset="BTC",
            side=OrderSide.BUY,
            size=0.001,
            order_type=OrderType.LIMIT,
            price=50123.456,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        assert call["limit_px"] == 50123.0  # int() truncation

    @pytest.mark.asyncio
    async def test_minimum_size_enforced(self):
        """Size should be at least 1 unit at the asset's szDecimals (10^-5 for BTC)."""
        adapter = _build_adapter()
        order = Order(
            id="test_4",
            asset="BTC",
            side=OrderSide.BUY,
            size=0.000001,  # Below BTC min (szDecimals=5 → 0.00001)
            order_type=OrderType.MARKET,
            price=None,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # BTC szDecimals=5 → min size = 10^-5 = 0.00001
        assert call["sz"] >= 0.00001


class TestClosePosition:
    """Test the close_position method."""

    @pytest.mark.asyncio
    async def test_close_long_position(self):
        """Closing a long should place a SELL (reduce-only) order."""
        # Position: long 0.001 BTC
        positions = [{
            "position": {
                "coin": "BTC",
                "szi": "0.001",
                "entryPx": "50000",
            }
        }]
        adapter = _build_adapter(positions=positions)

        success = await adapter.close_position("BTC")

        assert success is True
        call = adapter.exchange.order_calls[0]
        assert call["name"] == "BTC"
        assert call["is_buy"] is False  # Sell to close long
        assert call["sz"] == 0.001
        assert call["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_close_short_position(self):
        """Closing a short should place a BUY (reduce-only) order."""
        positions = [{
            "position": {
                "coin": "BTC",
                "szi": "-0.002",
                "entryPx": "50000",
            }
        }]
        adapter = _build_adapter(positions=positions)

        success = await adapter.close_position("BTC")

        call = adapter.exchange.order_calls[0]
        assert call["is_buy"] is True  # Buy to close short
        assert call["sz"] == 0.002
        assert call["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_partial_close(self):
        """Can close a partial amount of a position."""
        positions = [{
            "position": {
                "coin": "BTC",
                "szi": "0.01",
                "entryPx": "50000",
            }
        }]
        adapter = _build_adapter(positions=positions)

        success = await adapter.close_position("BTC", size=0.005)

        call = adapter.exchange.order_calls[0]
        assert call["sz"] == 0.005  # Only close half

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self):
        """Closing a position that doesn't exist returns False."""
        adapter = _build_adapter(positions=[])

        success = await adapter.close_position("BTC")

        assert success is False
        assert len(adapter.exchange.order_calls) == 0

    @pytest.mark.asyncio
    async def test_close_uses_slippage_price(self):
        """Close order should use ±1% slippage for market-like fill."""
        positions = [{
            "position": {
                "coin": "BTC",
                "szi": "0.001",
                "entryPx": "50000",
            }
        }]
        adapter = _build_adapter(mids={"BTC": "50000.0"})
        adapter.info._user_state["assetPositions"] = positions

        await adapter.close_position("BTC")

        call = adapter.exchange.order_calls[0]
        # Selling to close long → price = 50000 * 0.99 = 49500
        assert call["limit_px"] == 49500.0  # int(49500.0) = 49500


class TestGetPositions:
    """Test position retrieval."""

    @pytest.mark.asyncio
    async def test_returns_positions(self):
        positions = [{
            "position": {
                "coin": "BTC",
                "szi": "0.005",
                "entryPx": "48000",
            }
        }]
        adapter = _build_adapter(positions=positions)

        result = await adapter.get_positions()

        assert len(result) == 1
        pos = result[0]
        assert pos.asset == "BTC"
        assert pos.size == 0.005
        assert pos.entry_price == 48000.0

    @pytest.mark.asyncio
    async def test_empty_positions(self):
        adapter = _build_adapter(positions=[])

        result = await adapter.get_positions()

        assert result == []

    @pytest.mark.asyncio
    async def test_zero_size_filtered_out(self):
        """Positions with size 0 should be excluded."""
        positions = [{
            "position": {
                "coin": "BTC",
                "szi": "0",
                "entryPx": "50000",
            }
        }]
        adapter = _build_adapter(positions=positions)

        result = await adapter.get_positions()

        assert result == []


class TestGetMarketPrice:
    """Test market price retrieval."""

    @pytest.mark.asyncio
    async def test_returns_btc_price(self):
        adapter = _build_adapter(mids={"BTC": "67500.5"})
        price = await adapter.get_market_price("BTC")
        assert price == 67500.5

    @pytest.mark.asyncio
    async def test_unknown_asset_raises(self):
        adapter = _build_adapter(mids={"BTC": "50000"})
        with pytest.raises(RuntimeError):
            await adapter.get_market_price("UNKNOWN_COIN")
