"""Fetch historical candles from Hyperliquid with disk caching."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests


@dataclass
class CandleFetcherConfig:
    """Configuration for candle fetching."""

    base_url: str = "https://api.hyperliquid.xyz/info"
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "hyperliquid-backtest"
    )
    cache_ttl_hours: int = 24
    chunk_size_days: int = 1
    rate_limit_delay: float = 0.5
    max_retries: int = 5


class CandleFetcher:
    """Fetch historical candles from Hyperliquid with disk caching.

    Candles are fetched in daily chunks from the REST API and cached to disk
    as JSON files. Subsequent calls for the same symbol/interval/days reuse
    the cache until it expires (default 24h).
    """

    def __init__(self, config: CandleFetcherConfig | None = None):
        self._config = config or CandleFetcherConfig()
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 7,
        force_refresh: bool = False,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[dict]:
        """Fetch candles, using cache when available.

        Args:
            symbol: Asset symbol (e.g. "SOL", "BTC").
            interval: Candle interval (e.g. "1m", "5m", "1h").
            days: Number of days of history to fetch.
            force_refresh: Bypass cache even if valid.
            progress_callback: Called with total candle count after each chunk.

        Returns:
            List of candle dicts with keys: t, T, s, i, o, c, h, l, v, n.
        """
        cache_path = self._cache_path(symbol, interval, days)

        if not force_refresh and self._is_cache_valid(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        candles = self._fetch_all_chunks(symbol, interval, days, progress_callback)
        candles = self._deduplicate(candles)

        # Write cache
        with open(cache_path, "w") as f:
            json.dump(candles, f)

        return candles

    def _cache_path(self, symbol: str, interval: str, days: int) -> Path:
        """Compute cache file path."""
        return self._config.cache_dir / f"{symbol}_{interval}_{days}d.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file exists and is not expired."""
        if not cache_path.exists():
            return False
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        return age_hours < self._config.cache_ttl_hours

    def _fetch_all_chunks(
        self,
        symbol: str,
        interval: str,
        days: int,
        progress_callback: Callable[[int], None] | None,
    ) -> list[dict]:
        """Fetch candles from API in daily chunks with rate limiting."""
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        chunk_ms = self._config.chunk_size_days * 24 * 60 * 60 * 1000
        all_candles: list[dict] = []
        cursor = start_ms

        while cursor < end_ms:
            chunk_end = min(cursor + chunk_ms, end_ms)
            chunk = self._fetch_chunk(symbol, interval, cursor, chunk_end)
            if chunk:
                all_candles.extend(chunk)
                if progress_callback:
                    progress_callback(len(all_candles))
            cursor = chunk_end + 1
            time.sleep(self._config.rate_limit_delay)

        return all_candles

    def _fetch_chunk(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[dict]:
        """Fetch a single chunk from Hyperliquid API with retry on rate limit."""
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }

        for attempt in range(self._config.max_retries):
            resp = requests.post(
                self._config.base_url, json=payload, timeout=15
            )
            if resp.status_code == 429:
                wait = 2**attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()

        return []

    @staticmethod
    def _deduplicate(candles: list[dict]) -> list[dict]:
        """Remove duplicate candles by timestamp, sort chronologically."""
        seen: set[int] = set()
        unique: list[dict] = []
        for c in candles:
            t = c["t"]
            if t not in seen:
                seen.add(t)
                unique.append(c)
        unique.sort(key=lambda c: c["t"])
        return unique
