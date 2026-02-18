"""Tests for spot market support in HyperliquidAdapter.

Validates that:
- Spot pairs (e.g. "PURR/USDC") are detected and handled differently from perps
- Spot precision uses 8-total-decimal rule (not perp's 6)
- Spot metadata is cached from spot_meta_and_asset_ctxs()
- get_market_price works for spot pairs
- place_order and close_position work for spot pairs
"""

import pytest
from unittest.mock import MagicMock

from interfaces.exchange import Order, OrderSide, OrderType


# ---- Mock spot metadata ----

SPOT_META = {
    "tokens": [
        {"index": 0, "name": "PURR", "szDecimals": 0},
        {"index": 1, "name": "USDC", "szDecimals": 6},
        {"index": 2, "name": "HFUN", "szDecimals": 2},
    ],
    "universe": [
        {"index": 0, "name": "PURR/USDC", "tokens": [0, 1], "isCanonical": True},
        {"index": 1, "name": "HFUN/USDC", "tokens": [2, 1], "isCanonical": True},
    ],
}

SPOT_CTXS = [
    {"midPx": "0.03456", "markPx": "0.0345"},   # PURR/USDC
    {"midPx": "1.234", "markPx": "1.23"},         # HFUN/USDC
]

PERP_UNIVERSE = [
    {"name": "BTC", "szDecimals": 5},
    {"name": "ETH", "szDecimals": 4},
]

MIDS = {"BTC": "67580.0", "ETH": "3456.0"}


class MockInfoSpot:
    """Mock Info that returns both perp and spot metadata."""

    def __init__(self):
        self._user_state = {
            "assetPositions": [],
            "crossMarginSummary": {"accountValue": "10000"},
        }

    def meta(self):
        return {"universe": PERP_UNIVERSE}

    def spot_meta_and_asset_ctxs(self):
        return [SPOT_META, SPOT_CTXS]

    def all_mids(self):
        return MIDS

    def user_state(self, address):
        return self._user_state

    def open_orders(self, address):
        return []


class MockExchangeSDK:
    def __init__(self):
        self.wallet = MagicMock()
        self.wallet.address = "0xTest"
        self.order_calls = []

    def order(self, **kwargs):
        self.order_calls.append(kwargs)
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 55}}]}},
        }


def _build_adapter():
    from exchanges.hyperliquid.adapter import HyperliquidAdapter

    adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
    adapter.is_connected = True
    adapter.info = MockInfoSpot()
    adapter.exchange = MockExchangeSDK()
    # Build both perp and spot caches
    adapter._build_precision_cache()
    adapter._build_spot_precision_cache()
    return adapter


# ---- Detection helpers ----

class TestSpotDetection:
    """Test _is_spot helper."""

    def test_perp_symbol(self):
        from exchanges.hyperliquid.adapter import HyperliquidAdapter
        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
        assert adapter._is_spot("BTC") is False
        assert adapter._is_spot("ETH") is False

    def test_spot_pair_format(self):
        from exchanges.hyperliquid.adapter import HyperliquidAdapter
        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
        assert adapter._is_spot("PURR/USDC") is True
        assert adapter._is_spot("HFUN/USDC") is True

    def test_spot_index_format(self):
        from exchanges.hyperliquid.adapter import HyperliquidAdapter
        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=True)
        assert adapter._is_spot("@0") is True
        assert adapter._is_spot("@1") is True


# ---- Spot precision cache ----

class TestSpotPrecisionCache:
    """Test spot metadata caching."""

    def test_spot_cache_populated(self):
        adapter = _build_adapter()
        assert "PURR/USDC" in adapter._precision_cache
        assert "HFUN/USDC" in adapter._precision_cache

    def test_spot_assets_marked_as_spot(self):
        adapter = _build_adapter()
        assert adapter._precision_cache["PURR/USDC"]["is_spot"] is True
        assert adapter._precision_cache["HFUN/USDC"]["is_spot"] is True

    def test_perp_assets_still_cached(self):
        adapter = _build_adapter()
        assert adapter._precision_cache["BTC"]["is_spot"] is False

    def test_spot_sz_decimals_from_base_token(self):
        adapter = _build_adapter()
        # PURR token has szDecimals=0
        assert adapter._precision_cache["PURR/USDC"]["sz_decimals"] == 0
        # HFUN token has szDecimals=2
        assert adapter._precision_cache["HFUN/USDC"]["sz_decimals"] == 2

    def test_spot_max_price_decimals(self):
        adapter = _build_adapter()
        # PURR/USDC: 8 total - 0 sz_decimals = 8 price decimals
        assert adapter._max_price_decimals("PURR/USDC") == 8
        # HFUN/USDC: 8 total - 2 sz_decimals = 6 price decimals
        assert adapter._max_price_decimals("HFUN/USDC") == 6

    def test_spot_pair_index_mapping(self):
        adapter = _build_adapter()
        assert adapter._spot_pair_to_index.get("PURR/USDC") == 0
        assert adapter._spot_pair_to_index.get("HFUN/USDC") == 1


# ---- Spot market price ----

class TestSpotMarketPrice:
    """Test get_market_price for spot pairs."""

    @pytest.mark.asyncio
    async def test_spot_price_from_ctxs(self):
        adapter = _build_adapter()
        price = await adapter.get_market_price("PURR/USDC")
        assert price == pytest.approx(0.03456)

    @pytest.mark.asyncio
    async def test_perp_price_still_works(self):
        adapter = _build_adapter()
        price = await adapter.get_market_price("BTC")
        assert price == pytest.approx(67580.0)

    @pytest.mark.asyncio
    async def test_unknown_spot_pair_raises(self):
        adapter = _build_adapter()
        with pytest.raises(RuntimeError):
            await adapter.get_market_price("UNKNOWN/USDC")


# ---- Spot order placement ----

class TestSpotOrderPlacement:
    """Test place_order for spot pairs."""

    @pytest.mark.asyncio
    async def test_spot_limit_buy(self):
        adapter = _build_adapter()
        order = Order(
            id="s1", asset="PURR/USDC", side=OrderSide.BUY,
            size=100, order_type=OrderType.LIMIT, price=0.035,
        )
        oid = await adapter.place_order(order)

        assert oid == "55"
        call = adapter.exchange.order_calls[0]
        assert call["name"] == "PURR/USDC"
        assert call["is_buy"] is True

    @pytest.mark.asyncio
    async def test_spot_uses_spot_precision(self):
        """PURR szDecimals=0 → size should be whole numbers."""
        adapter = _build_adapter()
        order = Order(
            id="s2", asset="PURR/USDC", side=OrderSide.BUY,
            size=123.456, order_type=OrderType.LIMIT, price=0.035,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # szDecimals=0 → size rounded to integer
        assert call["sz"] == 123.0

    @pytest.mark.asyncio
    async def test_spot_price_precision_higher_than_perp(self):
        """PURR/USDC: 8 total - 0 sz = 8 price decimals allowed."""
        adapter = _build_adapter()
        order = Order(
            id="s3", asset="PURR/USDC", side=OrderSide.BUY,
            size=100, order_type=OrderType.LIMIT, price=0.03456789,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # Should preserve more decimals than perp would allow
        price_str = str(call["limit_px"])
        if "." in price_str:
            decimals = len(price_str.split(".")[1])
            # Spot with 8 max should allow more than perp's 2
            assert decimals >= 3

    @pytest.mark.asyncio
    async def test_spot_market_order(self):
        adapter = _build_adapter()
        order = Order(
            id="s4", asset="PURR/USDC", side=OrderSide.BUY,
            size=100, order_type=OrderType.MARKET, price=None,
        )
        oid = await adapter.place_order(order)

        assert oid == "55"
        call = adapter.exchange.order_calls[0]
        # Should use IOC with slippage price from spot ctxs
        assert call["limit_px"] > 0.034

    @pytest.mark.asyncio
    async def test_hfun_spot_order(self):
        """HFUN/USDC: szDecimals=2, max price decimals=6."""
        adapter = _build_adapter()
        order = Order(
            id="s5", asset="HFUN/USDC", side=OrderSide.BUY,
            size=10.567, order_type=OrderType.LIMIT, price=1.234567,
        )
        await adapter.place_order(order)

        call = adapter.exchange.order_calls[0]
        # szDecimals=2 → size rounded to 2 decimals
        size_str = str(call["sz"])
        if "." in size_str:
            assert len(size_str.split(".")[1]) <= 2


# ---- Spot market info ----

class TestSpotMarketInfo:
    """Test get_market_info for spot pairs."""

    @pytest.mark.asyncio
    async def test_spot_market_info(self):
        adapter = _build_adapter()
        info = await adapter.get_market_info("PURR/USDC")
        assert info.symbol == "PURR/USDC"
        assert info.base_asset == "PURR"
        assert info.quote_asset == "USDC"
        assert info.size_precision == 0
        # min_order_size for szDecimals=0 → 10^0 = 1
        assert info.min_order_size == 1.0

    @pytest.mark.asyncio
    async def test_spot_market_info_hfun(self):
        adapter = _build_adapter()
        info = await adapter.get_market_info("HFUN/USDC")
        assert info.base_asset == "HFUN"
        assert info.quote_asset == "USDC"
        assert info.size_precision == 2
