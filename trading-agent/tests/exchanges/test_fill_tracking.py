"""Tests for order fill tracking in HyperliquidAdapter.

Validates:
- get_order_status queries SDK and returns proper status
- IOC fill detection from place_order response ("filled" vs "resting")
- get_user_fills retrieves fill history
- place_order returns fill info for IOC orders
"""

import pytest
from unittest.mock import MagicMock

from interfaces.exchange import Order, OrderSide, OrderType, OrderStatus


# ---- Mock objects ----

class MockInfoWithFills:
    """Mock Info that supports order queries and fills."""

    def __init__(self):
        self._fills = []
        self._order_status = None

    def meta(self):
        return {"universe": [{"name": "BTC", "szDecimals": 5}]}

    def all_mids(self):
        return {"BTC": "67580.0"}

    def user_state(self, address):
        return {
            "assetPositions": [],
            "crossMarginSummary": {"accountValue": "10000"},
        }

    def open_orders(self, address):
        return []

    def query_order_by_oid(self, user, oid):
        return self._order_status

    def user_fills_by_time(self, address, start_time, end_time=None):
        return self._fills


class MockExchangeSDK:
    def __init__(self, response=None):
        self.wallet = MagicMock()
        self.wallet.address = "0xTest"
        self.order_calls = []
        self._response = response or {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 100}}]}},
        }

    def order(self, **kwargs):
        self.order_calls.append(kwargs)
        return self._response


def _build_adapter(response=None, fills=None, order_status=None):
    from exchanges.hyperliquid.adapter import HyperliquidAdapter

    adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
    adapter.is_connected = True
    adapter.info = MockInfoWithFills()
    if fills:
        adapter.info._fills = fills
    if order_status:
        adapter.info._order_status = order_status
    adapter.exchange = MockExchangeSDK(response=response)
    adapter._build_precision_cache()
    return adapter


# ---- get_order_status ----

class TestGetOrderStatus:
    """Test real order status queries."""

    @pytest.mark.asyncio
    async def test_resting_order_returns_submitted(self):
        """A resting order should return SUBMITTED status."""
        adapter = _build_adapter(order_status={
            "order": {
                "coin": "BTC",
                "side": "B",
                "limitPx": "67000",
                "sz": "0.001",
                "oid": 100,
                "orderType": "Limit",
            },
            "status": "open",
        })
        order = await adapter.get_order_status("100")
        assert order.status == OrderStatus.SUBMITTED
        assert order.asset == "BTC"
        assert order.exchange_order_id == "100"

    @pytest.mark.asyncio
    async def test_filled_order_returns_filled(self):
        """A fully filled order should return FILLED status."""
        adapter = _build_adapter(order_status={
            "order": {
                "coin": "BTC",
                "side": "A",
                "limitPx": "67000",
                "sz": "0.001",
                "oid": 101,
                "orderType": "Limit",
            },
            "status": "filled",
        })
        order = await adapter.get_order_status("101")
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancelled_order(self):
        adapter = _build_adapter(order_status={
            "order": {
                "coin": "BTC",
                "side": "B",
                "limitPx": "67000",
                "sz": "0.001",
                "oid": 102,
                "orderType": "Limit",
            },
            "status": "canceled",
        })
        order = await adapter.get_order_status("102")
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_unknown_order_returns_none_status(self):
        """If SDK returns None, return PENDING as fallback."""
        adapter = _build_adapter(order_status=None)
        order = await adapter.get_order_status("999")
        assert order.status == OrderStatus.PENDING


# ---- IOC fill detection ----

class TestIOCFillDetection:
    """Test that place_order detects immediate fills from IOC orders."""

    @pytest.mark.asyncio
    async def test_ioc_filled_returns_oid(self):
        """IOC order that fills immediately should still return the oid."""
        filled_response = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [{"filled": {"oid": 200, "totalSz": "0.001", "avgPx": "67580.0"}}]
                }
            },
        }
        adapter = _build_adapter(response=filled_response)
        order = Order(
            id="t1", asset="BTC", side=OrderSide.BUY,
            size=0.001, order_type=OrderType.MARKET, price=None,
        )
        oid = await adapter.place_order(order)
        assert oid == "200"

    @pytest.mark.asyncio
    async def test_resting_returns_oid(self):
        """GTC order that rests returns the resting oid."""
        resting_response = {
            "status": "ok",
            "response": {
                "data": {"statuses": [{"resting": {"oid": 300}}]}
            },
        }
        adapter = _build_adapter(response=resting_response)
        order = Order(
            id="t2", asset="BTC", side=OrderSide.BUY,
            size=0.001, order_type=OrderType.LIMIT, price=67000.0,
        )
        oid = await adapter.place_order(order)
        assert oid == "300"


# ---- get_user_fills ----

class TestGetUserFills:
    """Test fill history retrieval."""

    @pytest.mark.asyncio
    async def test_returns_fills(self):
        fills = [
            {
                "coin": "BTC", "side": "B", "px": "67500.0",
                "sz": "0.001", "oid": 100, "time": 1700000000000,
                "closedPnl": "0.0", "crossed": True,
            },
            {
                "coin": "BTC", "side": "A", "px": "67600.0",
                "sz": "0.001", "oid": 101, "time": 1700000060000,
                "closedPnl": "0.1", "crossed": True,
            },
        ]
        adapter = _build_adapter(fills=fills)
        result = await adapter.get_user_fills(start_time=1700000000)
        assert len(result) == 2
        assert result[0]["coin"] == "BTC"
        assert result[1]["oid"] == 101

    @pytest.mark.asyncio
    async def test_empty_fills(self):
        adapter = _build_adapter(fills=[])
        result = await adapter.get_user_fills(start_time=1700000000)
        assert result == []

    @pytest.mark.asyncio
    async def test_fills_not_connected(self):
        adapter = _build_adapter()
        adapter.is_connected = False
        result = await adapter.get_user_fills(start_time=0)
        assert result == []
