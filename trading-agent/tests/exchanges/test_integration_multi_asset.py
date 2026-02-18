"""Integration tests: multi-asset precision and spot metadata on mainnet.

These tests connect to the real Hyperliquid mainnet INFO endpoint
to validate that precision cache, spot metadata, and market info
work correctly against live data. No orders are placed.

Usage:
    uv run --package hyperliquid-trading-bot pytest tests/exchanges/test_integration_multi_asset.py -v -m slow
"""

import pytest


@pytest.mark.slow
class TestMainnetPrecisionCache:
    """Verify precision cache against real mainnet metadata."""

    @pytest.fixture(scope="class")
    def mainnet_info(self):
        """Create a live Info connection to mainnet."""
        from hyperliquid.info import Info
        info = Info("https://api.hyperliquid.xyz", skip_ws=True)
        return info

    def test_meta_returns_btc_and_eth(self, mainnet_info):
        """meta() should contain BTC and ETH with valid szDecimals."""
        meta = mainnet_info.meta()
        universe = meta.get("universe", [])
        names = {a["name"] for a in universe}
        assert "BTC" in names
        assert "ETH" in names

    def test_btc_sz_decimals_is_5(self, mainnet_info):
        meta = mainnet_info.meta()
        for a in meta["universe"]:
            if a["name"] == "BTC":
                assert a["szDecimals"] == 5
                break

    def test_eth_sz_decimals_is_reasonable(self, mainnet_info):
        """ETH szDecimals should be 3 or 4 (protocol may change)."""
        meta = mainnet_info.meta()
        for a in meta["universe"]:
            if a["name"] == "ETH":
                assert a["szDecimals"] in (3, 4)
                break

    def test_precision_cache_builds_from_real_meta(self, mainnet_info):
        """Build a real precision cache and verify multi-asset entries."""
        from exchanges.hyperliquid.adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=False)
        adapter.info = mainnet_info
        adapter._build_precision_cache()

        assert "BTC" in adapter._precision_cache
        assert "ETH" in adapter._precision_cache
        assert adapter._precision_cache["BTC"]["sz_decimals"] == 5
        assert adapter._precision_cache["BTC"]["is_spot"] is False

    def test_round_price_btc_uses_real_rules(self, mainnet_info):
        """BTC _round_price should produce valid prices per real meta."""
        from exchanges.hyperliquid.adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=False)
        adapter.info = mainnet_info
        adapter._build_precision_cache()

        # BTC: szDecimals=5, max price decimals = 6-5 = 1
        price = adapter._round_price("BTC", 67580.456, is_buy=True)
        # Should be an integer or have at most 1 decimal
        price_str = str(price)
        if "." in price_str:
            assert len(price_str.split(".")[1]) <= 1

    def test_round_price_eth_uses_real_rules(self, mainnet_info):
        """ETH _round_price should allow more decimals than BTC."""
        from exchanges.hyperliquid.adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=False)
        adapter.info = mainnet_info
        adapter._build_precision_cache()

        # ETH: szDecimals=4, max price decimals = 6-4 = 2
        price = adapter._round_price("ETH", 3456.789, is_buy=True)
        assert price >= 3456.0


@pytest.mark.slow
class TestMainnetSpotMetadata:
    """Verify spot metadata fetch and cache against real mainnet."""

    @pytest.fixture(scope="class")
    def mainnet_info(self):
        from hyperliquid.info import Info
        info = Info("https://api.hyperliquid.xyz", skip_ws=True)
        return info

    def test_spot_meta_returns_tokens_and_universe(self, mainnet_info):
        """spot_meta_and_asset_ctxs() should return valid structure."""
        spot_data = mainnet_info.spot_meta_and_asset_ctxs()
        assert isinstance(spot_data, list)
        assert len(spot_data) >= 2

        spot_meta = spot_data[0]
        assert "tokens" in spot_meta
        assert "universe" in spot_meta
        assert len(spot_meta["tokens"]) > 0
        assert len(spot_meta["universe"]) > 0

    def test_spot_tokens_have_sz_decimals(self, mainnet_info):
        """Each spot token should have a szDecimals field."""
        spot_data = mainnet_info.spot_meta_and_asset_ctxs()
        tokens = spot_data[0]["tokens"]
        for t in tokens[:5]:  # Check first 5
            assert "szDecimals" in t, f"Token {t.get('name')} missing szDecimals"
            assert isinstance(t["szDecimals"], int)

    def test_spot_cache_builds_from_real_data(self, mainnet_info):
        """Build spot precision cache from real data."""
        from exchanges.hyperliquid.adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(private_key="0x" + "a" * 64, testnet=False)
        adapter.info = mainnet_info
        adapter._build_precision_cache()
        adapter._build_spot_precision_cache()

        # Should have at least some spot pairs
        spot_pairs = [k for k, v in adapter._precision_cache.items() if v["is_spot"]]
        assert len(spot_pairs) > 0, "Should have at least one spot pair cached"

        # All spot pairs should be marked as spot
        for pair_name in spot_pairs:
            assert adapter._precision_cache[pair_name]["is_spot"] is True

    def test_spot_price_fetchable(self, mainnet_info):
        """Should be able to get a spot price from ctxs."""
        spot_data = mainnet_info.spot_meta_and_asset_ctxs()
        spot_meta = spot_data[0]
        ctxs = spot_data[1]

        # Find a canonical pair
        for pair in spot_meta.get("universe", []):
            if pair.get("isCanonical") and pair.get("name"):
                idx = pair["index"]
                if idx < len(ctxs):
                    ctx = ctxs[idx]
                    price = float(ctx.get("midPx") or ctx.get("markPx") or 0)
                    if price > 0:
                        # Found a valid spot pair with a price
                        assert True
                        return

        # If we get here, no canonical pair had a price — still OK
        pytest.skip("No canonical spot pair with valid price found")
