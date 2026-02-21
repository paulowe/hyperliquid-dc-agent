"""Tests for candle fetcher module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backtesting.candle_fetcher import CandleFetcher, CandleFetcherConfig


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def fetcher(tmp_cache_dir):
    config = CandleFetcherConfig(cache_dir=tmp_cache_dir)
    return CandleFetcher(config)


class TestCachePath:
    def test_cache_path_contains_symbol_interval_days(self, fetcher):
        path = fetcher._cache_path("SOL", "1m", 7)
        assert "SOL" in path.name
        assert "1m" in path.name
        assert "7d" in path.name
        assert path.suffix == ".json"

    def test_different_params_produce_different_paths(self, fetcher):
        p1 = fetcher._cache_path("SOL", "1m", 7)
        p2 = fetcher._cache_path("BTC", "1m", 7)
        p3 = fetcher._cache_path("SOL", "5m", 7)
        assert p1 != p2
        assert p1 != p3


class TestCacheValidity:
    def test_cache_valid_when_fresh(self, fetcher, tmp_cache_dir):
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = tmp_cache_dir / "SOL_1m_7d.json"
        cache_file.write_text("[]")
        assert fetcher._is_cache_valid(cache_file) is True

    def test_cache_invalid_when_missing(self, fetcher, tmp_cache_dir):
        missing = tmp_cache_dir / "nonexistent.json"
        assert fetcher._is_cache_valid(missing) is False

    def test_cache_invalid_when_expired(self, fetcher, tmp_cache_dir):
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = tmp_cache_dir / "SOL_1m_7d.json"
        cache_file.write_text("[]")
        # Set mtime to 25 hours ago (past 24h TTL)
        old_time = time.time() - 25 * 3600
        import os

        os.utime(cache_file, (old_time, old_time))
        assert fetcher._is_cache_valid(cache_file) is False


class TestDeduplication:
    def test_removes_duplicates(self):
        candles = [
            {"t": 100, "c": "1.0"},
            {"t": 100, "c": "1.0"},
            {"t": 200, "c": "2.0"},
        ]
        result = CandleFetcher._deduplicate(candles)
        assert len(result) == 2

    def test_sorts_chronologically(self):
        candles = [
            {"t": 300, "c": "3.0"},
            {"t": 100, "c": "1.0"},
            {"t": 200, "c": "2.0"},
        ]
        result = CandleFetcher._deduplicate(candles)
        assert [c["t"] for c in result] == [100, 200, 300]

    def test_preserves_unique(self):
        candles = [
            {"t": 100, "c": "1.0"},
            {"t": 200, "c": "2.0"},
            {"t": 300, "c": "3.0"},
        ]
        result = CandleFetcher._deduplicate(candles)
        assert len(result) == 3


class TestFetchWithCache:
    def test_uses_cache_when_valid(self, fetcher, tmp_cache_dir):
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = tmp_cache_dir / "SOL_1m_7d.json"
        cached_data = [{"t": 100, "c": "85.0"}]
        cache_file.write_text(json.dumps(cached_data))

        result = fetcher.fetch("SOL", "1m", 7)
        assert result == cached_data

    def test_force_refresh_bypasses_cache(self, fetcher, tmp_cache_dir):
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = tmp_cache_dir / "SOL_1m_7d.json"
        cache_file.write_text(json.dumps([{"t": 100, "c": "85.0"}]))

        # Mock the API to return different data
        api_data = [{"t": 200, "c": "90.0"}]
        with patch.object(fetcher, "_fetch_all_chunks", return_value=api_data):
            result = fetcher.fetch("SOL", "1m", 7, force_refresh=True)
        assert result == api_data


class TestFetchChunk:
    def test_correct_payload_structure(self, fetcher):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"t": 100, "c": "85.0"}]

        with patch("backtesting.candle_fetcher.requests.post", return_value=mock_resp) as mock_post:
            fetcher._fetch_chunk("SOL", "1m", 1000, 2000)
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["type"] == "candleSnapshot"
            assert payload["req"]["coin"] == "SOL"
            assert payload["req"]["interval"] == "1m"
            assert payload["req"]["startTime"] == 1000
            assert payload["req"]["endTime"] == 2000

    def test_retries_on_rate_limit(self, fetcher):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = [{"t": 100, "c": "85.0"}]

        with patch(
            "backtesting.candle_fetcher.requests.post",
            side_effect=[rate_limit_resp, ok_resp],
        ):
            with patch("backtesting.candle_fetcher.time.sleep"):
                result = fetcher._fetch_chunk("SOL", "1m", 1000, 2000)
        assert result == [{"t": 100, "c": "85.0"}]

    def test_returns_empty_after_max_retries(self, fetcher):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        with patch(
            "backtesting.candle_fetcher.requests.post",
            return_value=rate_limit_resp,
        ):
            with patch("backtesting.candle_fetcher.time.sleep"):
                result = fetcher._fetch_chunk("SOL", "1m", 1000, 2000)
        assert result == []
