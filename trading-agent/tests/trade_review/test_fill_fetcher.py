"""Tests for fill fetcher (Hyperliquid API interaction)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from trade_review.fill_fetcher import FillFetcher, FillFetcherConfig


# ---------------------------------------------------------------------------
# FillFetcherConfig
# ---------------------------------------------------------------------------
class TestFillFetcherConfig:
    @pytest.fixture(autouse=True)
    def _no_dotenv(self, monkeypatch):
        """Prevent load_dotenv from loading real .env in all config tests."""
        monkeypatch.setattr(
            "trade_review.fill_fetcher.load_dotenv", lambda *a, **kw: None
        )

    def test_from_env_mainnet(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "mainnet")
        monkeypatch.setenv("MAINNET_WALLET_ADDRESS", "0xABC")
        cfg = FillFetcherConfig.from_env()
        assert cfg.network == "mainnet"
        assert cfg.wallet_address == "0xABC"

    def test_from_env_testnet(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "testnet")
        monkeypatch.setenv("TESTNET_WALLET_ADDRESS", "0xDEF")
        cfg = FillFetcherConfig.from_env()
        assert cfg.network == "testnet"
        assert cfg.wallet_address == "0xDEF"

    def test_from_env_defaults_to_mainnet(self, monkeypatch):
        monkeypatch.delenv("HYPERLIQUID_NETWORK", raising=False)
        monkeypatch.setenv("MAINNET_WALLET_ADDRESS", "0x123")
        cfg = FillFetcherConfig.from_env()
        assert cfg.network == "mainnet"

    def test_from_env_falls_back_to_private_key(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "mainnet")
        monkeypatch.delenv("MAINNET_WALLET_ADDRESS", raising=False)
        # Use a deterministic test key (not a real key)
        test_key = "0x" + "ab" * 32
        monkeypatch.setenv("HYPERLIQUID_MAINNET_PRIVATE_KEY", test_key)
        cfg = FillFetcherConfig.from_env()
        assert cfg.network == "mainnet"
        # Wallet address should be derived from the private key
        assert cfg.wallet_address is not None
        assert cfg.wallet_address.startswith("0x")

    def test_from_env_network_override(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "testnet")
        monkeypatch.setenv("MAINNET_WALLET_ADDRESS", "0xMAIN")
        cfg = FillFetcherConfig.from_env(network_override="mainnet")
        assert cfg.network == "mainnet"
        assert cfg.wallet_address == "0xMAIN"

    def test_wallet_override_takes_priority(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "mainnet")
        monkeypatch.setenv("MAINNET_WALLET_ADDRESS", "0xAPI_WALLET")
        cfg = FillFetcherConfig.from_env(wallet_override="0xMAIN_WALLET")
        assert cfg.wallet_address == "0xMAIN_WALLET"

    def test_account_address_env_takes_priority_over_wallet(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "mainnet")
        monkeypatch.setenv("MAINNET_ACCOUNT_ADDRESS", "0xDELEGATED")
        monkeypatch.setenv("MAINNET_WALLET_ADDRESS", "0xAPI_WALLET")
        cfg = FillFetcherConfig.from_env()
        assert cfg.wallet_address == "0xDELEGATED"

    def test_from_env_raises_without_wallet(self, monkeypatch):
        monkeypatch.setenv("HYPERLIQUID_NETWORK", "mainnet")
        monkeypatch.delenv("MAINNET_ACCOUNT_ADDRESS", raising=False)
        monkeypatch.delenv("MAINNET_WALLET_ADDRESS", raising=False)
        monkeypatch.delenv("HYPERLIQUID_MAINNET_PRIVATE_KEY", raising=False)
        with pytest.raises(SystemExit):
            FillFetcherConfig.from_env()


# ---------------------------------------------------------------------------
# FillFetcher
# ---------------------------------------------------------------------------
class TestFillFetcher:
    def _make_fetcher_with_mock(self, fills: list[dict] | None = None):
        """Create a FillFetcher with a mocked Info SDK."""
        cfg = FillFetcherConfig(network="mainnet", wallet_address="0xTEST")
        fetcher = FillFetcher(cfg)
        # Mock the Info SDK
        mock_info = MagicMock()
        mock_info.user_fills_by_time.return_value = fills or []
        fetcher._info = mock_info
        return fetcher

    def test_fetch_filters_by_coin(self):
        fills = [
            {"coin": "BTC", "dir": "Open Long", "px": "100", "sz": "1",
             "time": 1700000000000, "side": "B"},
            {"coin": "SOL", "dir": "Open Long", "px": "50", "sz": "10",
             "time": 1700000001000, "side": "B"},
            {"coin": "BTC", "dir": "Close Long", "px": "105", "sz": "1",
             "time": 1700000060000, "side": "A"},
        ]
        fetcher = self._make_fetcher_with_mock(fills)
        result = fetcher.fetch(symbol="BTC", start_time_ms=1700000000000)
        assert len(result) == 2
        assert all(f["coin"] == "BTC" for f in result)

    def test_fetch_passes_correct_start_time(self):
        fetcher = self._make_fetcher_with_mock([])
        fetcher.fetch(symbol="BTC", start_time_ms=1700000000000)
        fetcher._info.user_fills_by_time.assert_called_once_with(
            "0xTEST", 1700000000000
        )

    def test_fetch_empty_fills(self):
        fetcher = self._make_fetcher_with_mock([])
        result = fetcher.fetch(symbol="BTC", start_time_ms=1700000000000)
        assert result == []

    def test_fetch_filters_by_end_time(self):
        fills = [
            {"coin": "BTC", "dir": "Open Long", "px": "100", "sz": "1",
             "time": 1700000000000, "side": "B"},
            {"coin": "BTC", "dir": "Close Long", "px": "105", "sz": "1",
             "time": 1700000060000, "side": "A"},
            {"coin": "BTC", "dir": "Open Long", "px": "106", "sz": "1",
             "time": 1700000120000, "side": "B"},
        ]
        fetcher = self._make_fetcher_with_mock(fills)
        result = fetcher.fetch(
            symbol="BTC",
            start_time_ms=1700000000000,
            end_time_ms=1700000100000,
        )
        assert len(result) == 2  # third fill excluded by end_time

    def test_fetch_sorts_by_time(self):
        fills = [
            {"coin": "BTC", "dir": "Close Long", "px": "105", "sz": "1",
             "time": 1700000060000, "side": "A"},
            {"coin": "BTC", "dir": "Open Long", "px": "100", "sz": "1",
             "time": 1700000000000, "side": "B"},
        ]
        fetcher = self._make_fetcher_with_mock(fills)
        result = fetcher.fetch(symbol="BTC", start_time_ms=1700000000000)
        assert result[0]["time"] < result[1]["time"]

    def test_connect_uses_wallet_address_directly(self):
        cfg = FillFetcherConfig(network="mainnet", wallet_address="0xDIRECT")
        fetcher = FillFetcher(cfg)
        with patch("trade_review.fill_fetcher.Info") as MockInfo:
            fetcher.connect()
            MockInfo.assert_called_once()
            assert fetcher._wallet_address == "0xDIRECT"
