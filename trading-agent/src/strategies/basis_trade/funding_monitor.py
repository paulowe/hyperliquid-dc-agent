"""Funding rate monitor — tracks funding history and accumulated P&L.

Queries Hyperliquid for current and historical funding rates,
maintains a rolling window for entry/exit decisions, and tracks
cumulative funding payments received.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FundingSnapshot:
    """A single funding rate observation."""
    timestamp: float  # Unix seconds
    rate: float       # Hourly funding rate (decimal, e.g., 0.0001 = 0.01%)
    mark_price: float
    oracle_price: float = 0.0


@dataclass
class FundingPayment:
    """A funding payment received (or paid) on the perp position."""
    timestamp: float
    rate: float
    notional: float   # Position notional at time of payment
    payment_usd: float  # Positive = received (short when funding > 0)


class FundingMonitor:
    """Tracks funding rates and accumulated payments for basis trade decisions.

    Maintains a rolling window of hourly funding rate observations and
    records all funding payments to compute cumulative P&L.
    """

    def __init__(self, window_hours: int = 24):
        # Rolling window of funding observations
        self._window_hours = window_hours
        self._observations: Deque[FundingSnapshot] = deque()

        # All funding payments received/paid
        self._payments: List[FundingPayment] = []

        # Cumulative funding P&L
        self._cumulative_funding_usd: float = 0.0

        # Track consecutive hours above/below thresholds
        self._consecutive_above: int = 0
        self._consecutive_below: int = 0
        self._last_rate: Optional[float] = None

    def record_observation(self, snapshot: FundingSnapshot) -> None:
        """Record a funding rate observation and update streak counters."""
        self._observations.append(snapshot)
        self._last_rate = snapshot.rate

        # Prune old observations outside the window
        cutoff = snapshot.timestamp - (self._window_hours * 3600)
        while self._observations and self._observations[0].timestamp < cutoff:
            self._observations.popleft()

    def update_streaks(self, rate: float, above_threshold: float,
                       below_threshold: float) -> None:
        """Update consecutive-hour streak counters based on current rate.

        Called once per hour (or per funding observation) to track how many
        consecutive periods funding has been above entry or below exit threshold.
        """
        if rate >= above_threshold:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        if rate <= below_threshold:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

    def record_payment(self, timestamp: float, rate: float,
                       notional: float) -> FundingPayment:
        """Record a funding payment and update cumulative P&L.

        For a short position with positive funding: payment = rate * notional
        (shorts receive funding from longs).

        For a short position with negative funding: payment is negative
        (shorts pay longs).
        """
        # Short position receives funding when rate > 0
        payment_usd = rate * notional
        payment = FundingPayment(
            timestamp=timestamp,
            rate=rate,
            notional=notional,
            payment_usd=payment_usd,
        )
        self._payments.append(payment)
        self._cumulative_funding_usd += payment_usd

        logger.debug(
            "Funding payment: rate=%.6f%% notional=$%.2f payment=$%.6f cumulative=$%.6f",
            rate * 100, notional, payment_usd, self._cumulative_funding_usd,
        )
        return payment

    @property
    def cumulative_funding_usd(self) -> float:
        """Total funding received (positive) or paid (negative) in USD."""
        return self._cumulative_funding_usd

    @property
    def consecutive_above(self) -> int:
        """Number of consecutive observations above the entry threshold."""
        return self._consecutive_above

    @property
    def consecutive_below(self) -> int:
        """Number of consecutive observations below the exit threshold."""
        return self._consecutive_below

    @property
    def last_rate(self) -> Optional[float]:
        """Most recently observed funding rate."""
        return self._last_rate

    @property
    def total_payments(self) -> int:
        """Number of funding payments recorded."""
        return len(self._payments)

    def average_rate(self, hours: int = 0) -> Optional[float]:
        """Average funding rate over the last N hours (0 = all observations)."""
        if not self._observations:
            return None

        if hours <= 0:
            obs = list(self._observations)
        else:
            cutoff = time.time() - (hours * 3600)
            obs = [o for o in self._observations if o.timestamp >= cutoff]

        if not obs:
            return None

        return sum(o.rate for o in obs) / len(obs)

    def current_apr(self) -> Optional[float]:
        """Annualized return based on the most recent funding rate."""
        if self._last_rate is None:
            return None
        return self._last_rate * 24 * 365 * 100

    def get_status(self) -> Dict:
        """Return monitor state for logging/telemetry."""
        avg_rate = self.average_rate()
        return {
            "last_rate": self._last_rate,
            "last_rate_pct": f"{self._last_rate * 100:.6f}%" if self._last_rate else None,
            "current_apr": f"{self.current_apr():.1f}%" if self._last_rate else None,
            "avg_rate": avg_rate,
            "observations": len(self._observations),
            "consecutive_above": self._consecutive_above,
            "consecutive_below": self._consecutive_below,
            "total_payments": len(self._payments),
            "cumulative_funding_usd": round(self._cumulative_funding_usd, 6),
        }
