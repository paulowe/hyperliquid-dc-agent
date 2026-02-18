"""Tests for leverage management in HyperliquidAdapter.

Validates:
- set_leverage calls SDK's update_leverage with correct args
- Only applies to perps (skips spot)
- Handles SDK errors gracefully
"""

import pytest
from unittest.mock import MagicMock

from interfaces.exchange import Order, OrderSide, OrderType


class MockInfoBasic:
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


class MockExchangeSDK:
    def __init__(self):
        self.wallet = MagicMock()
        self.wallet.address = "0xTest"
        self.leverage_calls = []
        self.order_calls = []

    def update_leverage(self, **kwargs):
        self.leverage_calls.append(kwargs)
        return {"status": "ok"}

    def order(self, **kwargs):
        self.order_calls.append(kwargs)
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}},
        }


def _build_adapter():
    from exchanges.hyperliquid.adapter import HyperliquidAdapter

    adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
    adapter.is_connected = True
    adapter.info = MockInfoBasic()
    adapter.exchange = MockExchangeSDK()
    adapter._build_precision_cache()
    return adapter


class TestSetLeverage:
    """Test set_leverage method."""

    @pytest.mark.asyncio
    async def test_sets_cross_leverage(self):
        adapter = _build_adapter()
        success = await adapter.set_leverage("BTC", 5, is_cross=True)

        assert success is True
        call = adapter.exchange.leverage_calls[0]
        assert call["leverage"] == 5
        assert call["name"] == "BTC"
        assert call["is_cross"] is True

    @pytest.mark.asyncio
    async def test_sets_isolated_leverage(self):
        adapter = _build_adapter()
        success = await adapter.set_leverage("BTC", 10, is_cross=False)

        assert success is True
        call = adapter.exchange.leverage_calls[0]
        assert call["is_cross"] is False
        assert call["leverage"] == 10

    @pytest.mark.asyncio
    async def test_skips_spot_assets(self):
        """Setting leverage on a spot pair should be a no-op and return True."""
        adapter = _build_adapter()
        success = await adapter.set_leverage("PURR/USDC", 5, is_cross=True)

        assert success is True
        # No SDK call should have been made
        assert len(adapter.exchange.leverage_calls) == 0

    @pytest.mark.asyncio
    async def test_not_connected_returns_false(self):
        adapter = _build_adapter()
        adapter.is_connected = False
        success = await adapter.set_leverage("BTC", 5, is_cross=True)

        assert success is False

    @pytest.mark.asyncio
    async def test_sdk_error_returns_false(self):
        adapter = _build_adapter()
        # Make SDK raise an exception
        adapter.exchange.update_leverage = MagicMock(side_effect=Exception("API error"))
        success = await adapter.set_leverage("BTC", 5, is_cross=True)

        assert success is False

    @pytest.mark.asyncio
    async def test_sdk_non_ok_returns_false(self):
        adapter = _build_adapter()
        adapter.exchange.update_leverage = MagicMock(
            return_value={"status": "err", "response": "some error"}
        )
        success = await adapter.set_leverage("BTC", 5, is_cross=True)

        assert success is False
