"""Tests for dynamic precision formatting in HyperliquidAdapter.

Validates that the adapter uses cached meta() precision rules instead of
hardcoded BTC rounding. Covers both perp and spot instruments.
"""

import pytest
from unittest.mock import MagicMock

from interfaces.exchange import Order, OrderSide, OrderType


# ---- Mock SDK objects ----

def _make_meta(universe):
    """Build a meta() response."""
    return {"universe": universe}


class MockInfoMultiAsset:
    """Mock Info that returns multi-asset metadata."""

    def __init__(self, universe, mids):
        self._universe = universe
        self._mids = mids
        self._user_state = {
            "assetPositions": [],
            "crossMarginSummary": {"accountValue": "10000"},
        }

    def meta(self):
        return _make_meta(self._universe)

    def all_mids(self):
        return self._mids

    def user_state(self, address):
        return self._user_state

    def open_orders(self, address):
        return []


class MockExchangeSDK:
    """Captures SDK order() calls."""

    def __init__(self):
        self.wallet = MagicMock()
        self.wallet.address = "0xTest"
        self.order_calls = []

    def order(self, **kwargs):
        self.order_calls.append(kwargs)
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 99}}]}},
        }


def _build_adapter(universe, mids):
    """Create adapter with custom meta and mids."""
    from exchanges.hyperliquid.adapter import HyperliquidAdapter

    adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
    adapter.is_connected = True
    adapter.info = MockInfoMultiAsset(universe, mids)
    adapter.exchange = MockExchangeSDK()
    # Trigger precision cache build
    adapter._build_precision_cache()
    return adapter


# ---- BTC (szDecimals=5, perp: 6 total → 1 price decimal) ----

class TestBTCPrecision:
    """BTC perp: szDecimals=5, max price decimals=1."""

    UNIVERSE = [{"name": "BTC", "szDecimals": 5}]
    MIDS = {"BTC": "67580.5"}

    @pytest.mark.asyncio
    async def test_btc_limit_price_rounded(self):
        """BTC limit price 67580.456 → at most 1 decimal place."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t1", asset="BTC", side=OrderSide.BUY,
            size=0.001, order_type=OrderType.LIMIT, price=67580.456,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # max 1 decimal → 67580.4 or 67580.0  (sig fig limited to 5)
        # With 5 sig figs: 67580 (integer) is fine
        price_str = str(call["limit_px"])
        # Should not have more than 1 decimal place
        if "." in price_str:
            decimals = len(price_str.split(".")[1])
            assert decimals <= 1

    @pytest.mark.asyncio
    async def test_btc_size_rounded_to_5_decimals(self):
        """BTC size should use szDecimals=5."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t2", asset="BTC", side=OrderSide.BUY,
            size=0.0012345678, order_type=OrderType.LIMIT, price=67580.0,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        size_str = str(call["sz"])
        if "." in size_str:
            decimals = len(size_str.split(".")[1])
            assert decimals <= 5


# ---- ETH (szDecimals=4, perp: 6 total → 2 price decimals) ----

class TestETHPrecision:
    """ETH perp: szDecimals=4, max price decimals=2."""

    UNIVERSE = [
        {"name": "BTC", "szDecimals": 5},
        {"name": "ETH", "szDecimals": 4},
    ]
    MIDS = {"BTC": "67580.0", "ETH": "3456.789"}

    @pytest.mark.asyncio
    async def test_eth_limit_price_has_2_decimals(self):
        """ETH limit price should allow up to 2 decimals."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t3", asset="ETH", side=OrderSide.BUY,
            size=0.1, order_type=OrderType.LIMIT, price=3456.789,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        price_str = str(call["limit_px"])
        if "." in price_str:
            decimals = len(price_str.split(".")[1])
            assert decimals <= 2

    @pytest.mark.asyncio
    async def test_eth_size_rounded_to_4_decimals(self):
        """ETH size should use szDecimals=4."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t4", asset="ETH", side=OrderSide.BUY,
            size=0.123456, order_type=OrderType.LIMIT, price=3456.0,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        size_str = str(call["sz"])
        if "." in size_str:
            decimals = len(size_str.split(".")[1])
            assert decimals <= 4

    @pytest.mark.asyncio
    async def test_eth_market_order_slippage_price(self):
        """ETH market buy should use adjusted price with ETH precision."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t5", asset="ETH", side=OrderSide.BUY,
            size=0.1, order_type=OrderType.MARKET, price=None,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # Should be ~3456.789 * 1.01 ≈ 3491.36, not hardcoded int()
        assert call["limit_px"] > 3456.0


# ---- Low-price asset (szDecimals=2, perp: 6 total → 4 price decimals) ----

class TestLowPriceAsset:
    """Asset with szDecimals=2 → max 4 price decimals for perps."""

    UNIVERSE = [
        {"name": "BTC", "szDecimals": 5},
        {"name": "DOGE", "szDecimals": 2},
    ]
    MIDS = {"BTC": "67580.0", "DOGE": "0.14567"}

    @pytest.mark.asyncio
    async def test_doge_price_allows_4_decimals(self):
        """Low price asset should allow up to 4 price decimals."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t6", asset="DOGE", side=OrderSide.BUY,
            size=100, order_type=OrderType.LIMIT, price=0.14567,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # With 4 price decimals allowed, 0.1456 or 0.1457 is valid
        assert call["limit_px"] > 0.1
        assert call["limit_px"] < 1.0

    @pytest.mark.asyncio
    async def test_doge_size_rounded_to_2_decimals(self):
        """DOGE size should use szDecimals=2."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        order = Order(
            id="t7", asset="DOGE", side=OrderSide.BUY,
            size=100.456, order_type=OrderType.LIMIT, price=0.145,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        size_str = str(call["sz"])
        if "." in size_str:
            decimals = len(size_str.split(".")[1])
            assert decimals <= 2


# ---- Precision cache ----

class TestPrecisionCache:
    """Test that the precision cache is built and queried correctly."""

    UNIVERSE = [
        {"name": "BTC", "szDecimals": 5},
        {"name": "ETH", "szDecimals": 4},
        {"name": "SOL", "szDecimals": 3},
    ]
    MIDS = {"BTC": "67580.0", "ETH": "3456.0", "SOL": "145.0"}

    def test_cache_populated_on_build(self):
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        assert "BTC" in adapter._precision_cache
        assert "ETH" in adapter._precision_cache
        assert "SOL" in adapter._precision_cache

    def test_cache_stores_sz_decimals(self):
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        assert adapter._precision_cache["BTC"]["sz_decimals"] == 5
        assert adapter._precision_cache["ETH"]["sz_decimals"] == 4
        assert adapter._precision_cache["SOL"]["sz_decimals"] == 3

    def test_cache_stores_is_spot(self):
        """Meta-derived assets should be marked as perps (not spot)."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        assert adapter._precision_cache["BTC"]["is_spot"] is False

    def test_unknown_asset_uses_safe_defaults(self):
        """Unknown asset should use conservative defaults (2 sz_decimals)."""
        adapter = _build_adapter(self.UNIVERSE, self.MIDS)
        rules = adapter._get_precision_rules("UNKNOWN")
        assert rules["sz_decimals"] == 2
        assert rules["is_spot"] is False


# ---- Close position with dynamic precision ----

class TestClosePositionPrecision:
    """Test close_position uses dynamic precision."""

    UNIVERSE = [
        {"name": "BTC", "szDecimals": 5},
        {"name": "ETH", "szDecimals": 4},
    ]
    MIDS = {"BTC": "67580.0", "ETH": "3456.0"}

    def _build_adapter_with_position(self, asset, size, entry_px):
        positions = [{
            "position": {
                "coin": asset,
                "szi": str(size),
                "entryPx": str(entry_px),
            }
        }]
        from exchanges.hyperliquid.adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
        adapter.is_connected = True
        adapter.info = MockInfoMultiAsset(self.UNIVERSE, self.MIDS)
        adapter.info._user_state = {
            "assetPositions": positions,
            "crossMarginSummary": {"accountValue": "10000"},
        }
        adapter.exchange = MockExchangeSDK()
        adapter._build_precision_cache()
        return adapter

    @pytest.mark.asyncio
    async def test_close_eth_uses_eth_precision(self):
        """Close ETH position should use ETH price decimals, not BTC's."""
        adapter = self._build_adapter_with_position("ETH", 1.0, 3400.0)
        await adapter.close_position("ETH")

        call = adapter.exchange.order_calls[0]
        # ETH: 2 price decimals max. Close sell → 3456 * 0.99 = 3421.44
        price_str = str(call["limit_px"])
        if "." in price_str:
            decimals = len(price_str.split(".")[1])
            assert decimals <= 2

    @pytest.mark.asyncio
    async def test_close_eth_size_uses_4_decimals(self):
        """Close ETH should round size to szDecimals=4."""
        adapter = self._build_adapter_with_position("ETH", 1.23456, 3400.0)
        await adapter.close_position("ETH")

        call = adapter.exchange.order_calls[0]
        size_str = str(call["sz"])
        if "." in size_str:
            decimals = len(size_str.split(".")[1])
            assert decimals <= 4
