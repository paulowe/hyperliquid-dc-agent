"""RegimeDetector — classifies market as trending, choppy, or quiet.

Uses DC event frequency and directional consistency over a rolling window
to determine the current market regime. Choppy markets (high event rate,
alternating directions) are toxic for DC overshoot strategies.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Classifies market regime using DC event rate and direction consistency."""

    def __init__(
        self,
        lookback_seconds: float = 600.0,
        choppy_rate_threshold: float = 4.0,
        trending_consistency_threshold: float = 0.6,
    ):
        # Rolling window of (timestamp, direction) tuples
        self._events: List[Tuple[float, int]] = []
        self._lookback = lookback_seconds
        # Events/min above this → fast-moving market
        self._choppy_rate = choppy_rate_threshold
        # Direction consistency above this → trending (range 0-1)
        self._trending_consistency = trending_consistency_threshold

    def record_event(self, direction: int, timestamp: float) -> None:
        """Record a DC event.

        Args:
            direction: +1 for up, -1 for down
            timestamp: Event time in seconds
        """
        self._events.append((timestamp, direction))
        self._purge_old(timestamp)

    def classify(self, timestamp: float) -> str:
        """Classify current market regime.

        Returns:
            "trending", "choppy", "neutral", or "quiet"
        """
        self._purge_old(timestamp)

        if len(self._events) < 3:
            return "quiet"

        # Need at least 60s of data to compute meaningful rate
        span = timestamp - self._events[0][0]
        if span < 60:
            return "quiet"

        # Event rate (events per minute)
        rate = len(self._events) / (span / 60.0)

        # Direction consistency: |mean(directions)| → 1.0 = all same, 0.0 = balanced
        directions = [d for _, d in self._events]
        consistency = abs(sum(directions)) / len(directions)

        # Trending: strong directional consistency
        if consistency >= self._trending_consistency:
            return "trending"

        # Choppy: high event rate + no clear direction
        if rate > self._choppy_rate and consistency < self._trending_consistency:
            return "choppy"

        return "neutral"

    def should_trade(self, timestamp: float) -> bool:
        """Whether the current regime allows new trade entries."""
        regime = self.classify(timestamp)
        return regime != "choppy"

    def event_rate(self, timestamp: float) -> float:
        """Current event rate in events per minute."""
        self._purge_old(timestamp)
        if len(self._events) < 2:
            return 0.0
        span = timestamp - self._events[0][0]
        if span < 1.0:
            return 0.0
        return len(self._events) / (span / 60.0)

    def _purge_old(self, current_time: float) -> None:
        """Remove events older than lookback window."""
        cutoff = current_time - self._lookback
        self._events = [(t, d) for t, d in self._events if t >= cutoff]
