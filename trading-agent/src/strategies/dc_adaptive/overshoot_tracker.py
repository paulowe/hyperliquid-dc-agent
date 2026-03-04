"""OvershootTracker — adapts TP to recent overshoot magnitudes.

Tracks completed DC lifecycle overshoots in a rolling window and computes
the adaptive take-profit as a fraction of the median overshoot. This ensures
TP is always calibrated to what the market is actually delivering.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OvershootTracker:
    """Tracks recent overshoot magnitudes for adaptive TP."""

    def __init__(
        self,
        window_size: int = 20,
        min_samples: int = 5,
        tp_fraction: float = 0.8,
        min_tp_pct: float = 0.003,
        default_tp_pct: float = 0.005,
    ):
        self._overshoots: deque[float] = deque(maxlen=window_size)
        self._min_samples = min_samples
        # Fraction of median overshoot to use as TP
        self._tp_fraction = tp_fraction
        # Floor for adaptive TP (must cover fees)
        self._min_tp = min_tp_pct
        # TP to use before enough samples
        self._default_tp = default_tp_pct

    @property
    def count(self) -> int:
        return len(self._overshoots)

    def record_overshoot(self, magnitude: float) -> None:
        """Record a completed overshoot magnitude (fraction, not percent).

        Args:
            magnitude: Overshoot as a fraction (e.g., 0.01 for 1%)
        """
        if magnitude > 0:
            self._overshoots.append(magnitude)

    def adaptive_tp(self) -> float:
        """Return TP based on recent median overshoot.

        Falls back to default_tp_pct if not enough samples.
        """
        if len(self._overshoots) < self._min_samples:
            return self._default_tp

        p50 = self._compute_median()
        adaptive = p50 * self._tp_fraction
        return max(adaptive, self._min_tp)

    def percentiles(self) -> Optional[Dict[str, Any]]:
        """Return overshoot distribution percentiles, or None if no data."""
        if len(self._overshoots) < 1:
            return None

        sorted_os = sorted(self._overshoots)
        n = len(sorted_os)

        return {
            "p25": self._percentile(sorted_os, 0.25),
            "p50": self._percentile(sorted_os, 0.50),
            "p75": self._percentile(sorted_os, 0.75),
            "p90": self._percentile(sorted_os, 0.90),
            "mean": sum(sorted_os) / n,
            "min": sorted_os[0],
            "max": sorted_os[-1],
            "count": n,
        }

    def _compute_median(self) -> float:
        """Compute median of current overshoots."""
        return self._percentile(sorted(self._overshoots), 0.50)

    @staticmethod
    def _percentile(sorted_values: list, p: float) -> float:
        """Compute percentile from sorted list using linear interpolation."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_values[0]
        idx = p * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac
