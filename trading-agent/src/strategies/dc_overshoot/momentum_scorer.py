"""Exponential moving momentum scorer for multi-threshold DC sensor fusion.

Aggregates directional signals from low-threshold DC events into a single
momentum score in [-1.0, +1.0]. Positive = bullish consensus, negative = bearish.

Algorithm:
  On each sensor DC event:
    score = alpha * direction + (1 - alpha) * score

  where direction = +1 (PDCC2_UP) or -1 (PDCC_Down).

  alpha controls responsiveness: 0.3 gives ~5-event effective window.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict


class MomentumScorer:
    """Computes directional momentum score from sensor DC events.

    Uses exponential moving average of directional signals.
    Score range: [-1.0, +1.0].
      +1.0 = all recent sensors say UP (bullish)
      -1.0 = all recent sensors say DOWN (bearish)
       0.0 = mixed (choppy / no consensus)
    """

    # Regime classification thresholds
    _TRENDING_THRESHOLD = 0.5

    def __init__(
        self,
        alpha: float = 0.3,
        event_history_size: int = 50,
    ) -> None:
        """
        Args:
            alpha: EMA smoothing factor. Higher = more weight on recent events.
                   Equivalent to window ~= (2/alpha) - 1 events.
                   0.3 ~= 5-event effective window.
            event_history_size: Max events to retain for event rate computation.
        """
        self._alpha = alpha
        self._score = 0.0
        self._event_count = 0
        # Recent event timestamps for rate computation
        self._event_times: deque[float] = deque(maxlen=event_history_size)

    @property
    def score(self) -> float:
        """Current momentum score in [-1.0, +1.0]."""
        return self._score

    @property
    def abs_score(self) -> float:
        """Absolute momentum score (regime strength indicator)."""
        return abs(self._score)

    @property
    def event_count(self) -> int:
        """Total number of sensor events processed."""
        return self._event_count

    def update(
        self,
        direction: int,
        timestamp: float,
        threshold_key: str = "",
        price: float = 0.0,
    ) -> float:
        """Process a new sensor DC event and return updated score.

        Args:
            direction: +1 for PDCC2_UP, -1 for PDCC_Down.
            timestamp: Epoch seconds of the event.
            threshold_key: Which sensor threshold generated this (for logging).
            price: Price at the event (for logging).

        Returns:
            Updated momentum score.
        """
        self._score = self._alpha * direction + (1.0 - self._alpha) * self._score
        # Clamp to [-1.0, +1.0] to handle floating point drift
        self._score = max(-1.0, min(1.0, self._score))
        self._event_count += 1
        self._event_times.append(timestamp)
        return self._score

    def get_regime(self) -> str:
        """Classify current market regime based on score.

        Returns:
            "trending_up", "trending_down", "choppy", or "neutral"
        """
        if self._event_count == 0:
            return "neutral"
        if self._score > self._TRENDING_THRESHOLD:
            return "trending_up"
        if self._score < -self._TRENDING_THRESHOLD:
            return "trending_down"
        return "choppy"

    def get_event_rate(self, lookback_seconds: float = 300.0) -> float:
        """Compute sensor event frequency over lookback window.

        High frequency = choppy market (many small reversals).
        Low frequency = trending market (sustained direction).

        Returns:
            Events per minute over the lookback window.
        """
        if not self._event_times:
            return 0.0

        latest = self._event_times[-1]
        cutoff = latest - lookback_seconds
        recent_count = sum(1 for t in self._event_times if t > cutoff)
        # Convert to events per minute
        lookback_minutes = lookback_seconds / 60.0
        return recent_count / lookback_minutes if lookback_minutes > 0 else 0.0

    def reset(self) -> None:
        """Reset scorer state."""
        self._score = 0.0
        self._event_count = 0
        self._event_times.clear()

    def get_status(self) -> Dict[str, Any]:
        """Return current state for monitoring/logging."""
        return {
            "score": round(self._score, 4),
            "abs_score": round(abs(self._score), 4),
            "event_count": self._event_count,
            "regime": self.get_regime(),
            "alpha": self._alpha,
        }
