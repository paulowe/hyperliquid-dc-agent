"""TrendDirectionFilter — Guard 4: trend bias from DC event directions.

Detects macro trend direction from DC sensor event flow to block or
reduce counter-trend entries.

Inspired by Chen & Tsang Ch 5 (Tracking Regime Changes Using DC Indicators).
Uses TMV-weighted directional bias from sensor DC events to detect trend.

Two bias modes:
- "simple": signed_bias = sum(directions) / len(directions)
  Pure event-count ratio. Fast, works well for clear trends.
- "tmv_weighted": signed_bias = sum(direction_i * tmv_i) / sum(tmv_i)
  Weights each event by its Total price Movement Value (DC end-to-end
  displacement). Larger moves contribute more — captures not just "more
  up events" but "up moves are bigger than down moves". More robust
  in choppy markets with occasional strong directional bursts.

Decision rules (from Ch 5 B-Simple / B-Strict):
- counter_trend_action="block" -> B-Simple: if bias > threshold, block.
- counter_trend_action="reduce" -> B-Strict: requires bias > strict_threshold
  (higher bar) before reducing size. Fewer false trend classifications.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrendDirectionFilter:
    """Detects macro trend direction from DC event flow."""

    def __init__(
        self,
        lookback_seconds: float = 900.0,
        min_events: int = 5,
        min_consistency: float = 0.6,
        counter_trend_action: str = "block",
        counter_trend_size_fraction: float = 0.5,
        bias_mode: str = "tmv_weighted",
        strict_threshold: float = 0.8,
    ):
        """Initialize trend direction filter.

        Args:
            lookback_seconds: Rolling window for trend detection (default 15 min).
            min_events: Minimum sensor events before trend filtering activates.
            min_consistency: Directional bias threshold (0-1) for trend detection.
            counter_trend_action: "block" (B-Simple), "reduce" (B-Strict), or "allow".
            counter_trend_size_fraction: Size multiplier when action="reduce".
            bias_mode: "simple" (count-only) or "tmv_weighted" (Ch 5 TMV).
            strict_threshold: B-Strict confidence bar for "reduce" mode.
        """
        # (timestamp, direction, tmv) tuples
        self._events: List[Tuple[float, int, float]] = []
        self._lookback = lookback_seconds
        self._min_events = min_events
        self._min_consistency = min_consistency
        self._counter_trend_action = counter_trend_action
        self._counter_trend_size_fraction = counter_trend_size_fraction
        self._bias_mode = bias_mode
        self._strict_threshold = strict_threshold

    def record_event(self, direction: int, timestamp: float, tmv: float = 1.0) -> None:
        """Record a DC event direction (+1=up, -1=down) with optional TMV weight.

        Args:
            direction: +1 for up, -1 for down.
            timestamp: Event time in seconds.
            tmv: Total price Movement Value of the DC trend that just completed.
                 For sensor events, abs(end_price - start_price) / start_price.
                 Defaults to 1.0 for backward compatibility (equivalent to simple mode).
        """
        self._events.append((timestamp, direction, tmv))
        self._purge_old(timestamp)

    def bias(self, timestamp: float) -> float:
        """Compute signed directional bias in [-1, +1].

        simple mode:       sum(d_i) / N
        tmv_weighted mode: sum(d_i * tmv_i) / sum(tmv_i)

        Returns 0.0 if no events in window.
        """
        self._purge_old(timestamp)

        if not self._events:
            return 0.0

        if self._bias_mode == "tmv_weighted":
            weighted_sum = sum(d * tmv for _, d, tmv in self._events)
            total_tmv = sum(tmv for _, _, tmv in self._events)
            if total_tmv == 0:
                return 0.0
            return weighted_sum / total_tmv
        else:
            # Simple mode: pure event count ratio
            direction_sum = sum(d for _, d, _ in self._events)
            return direction_sum / len(self._events)

    def dominant_trend(self, timestamp: float) -> Optional[str]:
        """Return 'up', 'down', or None (no clear trend).

        Uses bias() with min_consistency and min_events thresholds.
        """
        self._purge_old(timestamp)

        if len(self._events) < self._min_events:
            return None

        b = self.bias(timestamp)
        if b >= self._min_consistency:
            return "up"
        elif b <= -self._min_consistency:
            return "down"
        return None

    def is_counter_trend(self, side: str, timestamp: float) -> bool:
        """Check if a trade side opposes the dominant trend.

        Returns True if side is counter-trend, False otherwise.
        """
        trend = self.dominant_trend(timestamp)
        if trend is None:
            return False
        if trend == "up" and side == "SHORT":
            return True
        if trend == "down" and side == "LONG":
            return True
        return False

    def should_trade_direction(self, side: str, timestamp: float) -> Tuple[bool, float]:
        """Check if a trade direction is allowed by the trend filter.

        Returns: (allowed: bool, size_multiplier: float)
        - If aligned with trend or no trend: (True, 1.0)
        - If counter-trend and action="block" (B-Simple):
            bias > min_consistency -> (False, 0.0)
        - If counter-trend and action="reduce" (B-Strict):
            bias > strict_threshold -> (True, counter_trend_size_fraction)
            bias > min_consistency but < strict_threshold -> (True, 1.0)
        - If action="allow": (True, 1.0) -- filter disabled
        """
        if self._counter_trend_action == "allow":
            return (True, 1.0)

        trend = self.dominant_trend(timestamp)

        # No trend detected -> allow everything
        if trend is None:
            return (True, 1.0)

        # Check if this is counter-trend
        is_counter = self.is_counter_trend(side, timestamp)
        if not is_counter:
            return (True, 1.0)

        # Counter-trend detected
        if self._counter_trend_action == "block":
            # B-Simple: block if bias exceeds min_consistency (already checked in dominant_trend)
            return (False, 0.0)
        elif self._counter_trend_action == "reduce":
            # B-Strict: only reduce size if bias exceeds strict_threshold
            abs_bias = abs(self.bias(timestamp))
            if abs_bias >= self._strict_threshold:
                return (True, self._counter_trend_size_fraction)
            # Below strict threshold but above min_consistency -> not confident enough to reduce
            return (True, 1.0)

        # Fallback: allow
        return (True, 1.0)

    def get_status(self, timestamp: float) -> Dict:
        """Return current trend state for monitoring/telemetry."""
        self._purge_old(timestamp)

        # TMV breakdown by direction
        up_tmvs = [tmv for _, d, tmv in self._events if d > 0]
        down_tmvs = [tmv for _, d, tmv in self._events if d < 0]

        return {
            "dominant_trend": self.dominant_trend(timestamp),
            "bias": round(self.bias(timestamp), 4),
            "bias_mode": self._bias_mode,
            "event_count": len(self._events),
            "events_in_window": len(self._events),
            "avg_tmv_up": round(sum(up_tmvs) / len(up_tmvs), 6) if up_tmvs else 0.0,
            "avg_tmv_down": round(sum(down_tmvs) / len(down_tmvs), 6) if down_tmvs else 0.0,
            "counter_trend_action": self._counter_trend_action,
            "min_consistency": self._min_consistency,
        }

    def _purge_old(self, current_time: float) -> None:
        """Remove events older than lookback window."""
        cutoff = current_time - self._lookback
        self._events = [(t, d, tmv) for t, d, tmv in self._events if t >= cutoff]
