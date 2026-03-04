"""LossStreakGuard — circuit breaker after consecutive losses.

Prevents the bot from trading during loss spirals. After a configurable
number of consecutive losses, enters an escalating cooldown period.
Resets immediately after any winning trade.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LossStreakGuard:
    """Circuit breaker that pauses trading after consecutive losses."""

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        base_cooldown_seconds: float = 300.0,
    ):
        self._max = max_consecutive_losses
        self._base_cooldown = base_cooldown_seconds
        self._consecutive_losses = 0
        self._cooldown_until = 0.0

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def record_trade(self, is_win: bool, timestamp: float) -> None:
        """Record a completed trade result.

        Args:
            is_win: True if trade was profitable (net P&L > 0)
            timestamp: Time of trade exit
        """
        if is_win:
            self._consecutive_losses = 0
            self._cooldown_until = 0.0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._max:
                cooldown = self._base_cooldown * self._consecutive_losses
                self._cooldown_until = timestamp + cooldown
                logger.info(
                    "LossStreakGuard: %d consecutive losses — cooldown %.0fs",
                    self._consecutive_losses, cooldown,
                )

    def should_trade(self, timestamp: float) -> bool:
        """Whether trading is currently allowed."""
        if timestamp < self._cooldown_until:
            return False
        return True

    def get_status(self, timestamp: float = 0.0) -> Dict[str, Any]:
        """Return current guard state."""
        in_cooldown = timestamp < self._cooldown_until
        remaining = max(0.0, self._cooldown_until - timestamp) if in_cooldown else 0.0
        return {
            "consecutive_losses": self._consecutive_losses,
            "in_cooldown": in_cooldown,
            "cooldown_remaining_s": remaining,
        }
