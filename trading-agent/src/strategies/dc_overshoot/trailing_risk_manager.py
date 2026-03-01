"""Trailing Risk Manager — trailing SL with fixed TP for DC Overshoot strategy.

Core behavior:
- TP stays fixed at initial level and fires when hit (primary profit exit)
- When in a loss: SL remains at initial level (no ratcheting)
- When in profit: SL ratchets toward profit direction (locks in gains)

LONG positions:
- SL starts below entry, ratchets UP as price makes new highs
- TP stays fixed above entry at initial level
- high_water_mark tracks the highest price since entry

SHORT positions:
- SL starts above entry, ratchets DOWN as price makes new lows
- TP stays fixed below entry at initial level
- low_water_mark tracks the lowest price since entry
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from interfaces.strategy import SignalType, TradingSignal

logger = logging.getLogger(__name__)


class TrailingRiskManager:
    """Manages trailing stop loss with fixed take profit for a single position.

    TP stays at its initial level and fires as the primary profit exit.
    SL trailing only activates when the position is profitable.
    When in a loss, SL remains at its initial level.
    """

    def __init__(
        self,
        asset: str,
        initial_stop_loss_pct: float,
        initial_take_profit_pct: float,
        trail_pct: float,
        min_profit_to_trail_pct: float = 0.0,
    ):
        self.asset = asset
        self._sl_pct = initial_stop_loss_pct
        self._tp_pct = initial_take_profit_pct
        self._trail_pct = trail_pct
        # Don't start trailing until profit exceeds this percentage from entry.
        # Default 0.0 = trail immediately (original behavior).
        # Recommended: set to ~sl_pct so SL doesn't ratchet above entry on noise.
        self._min_profit_to_trail = min_profit_to_trail_pct

        # Position state (None = no position)
        self._side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._size: Optional[float] = None

        # SL/TP levels
        self._current_sl: Optional[float] = None
        self._current_tp: Optional[float] = None

        # Water marks
        self._high_water_mark: Optional[float] = None
        self._low_water_mark: Optional[float] = None

    # --- Public properties ---

    @property
    def has_position(self) -> bool:
        return self._side is not None

    @property
    def side(self) -> Optional[str]:
        return self._side

    @property
    def entry_price(self) -> Optional[float]:
        return self._entry_price

    @property
    def size(self) -> Optional[float]:
        return self._size

    @property
    def current_sl_price(self) -> Optional[float]:
        return self._current_sl

    @property
    def current_tp_price(self) -> Optional[float]:
        return self._current_tp

    @property
    def high_water_mark(self) -> Optional[float]:
        return self._high_water_mark

    @property
    def low_water_mark(self) -> Optional[float]:
        return self._low_water_mark

    # --- Position lifecycle ---

    def open_position(self, side: str, entry_price: float, size: float) -> None:
        """Initialize a new position with initial SL/TP levels.

        Args:
            side: "LONG" or "SHORT"
            entry_price: Actual fill price
            size: Position size (always positive)

        Raises:
            RuntimeError: If a position is already open
        """
        if self.has_position:
            raise RuntimeError(
                f"Cannot open {side} position — already open "
                f"{self._side} at {self._entry_price}"
            )

        self._side = side
        self._entry_price = entry_price
        self._size = size

        if side == "LONG":
            self._current_sl = entry_price * (1 - self._sl_pct)
            self._current_tp = entry_price * (1 + self._tp_pct)
            self._high_water_mark = entry_price
            self._low_water_mark = entry_price  # track for MAE
        else:  # SHORT
            self._current_sl = entry_price * (1 + self._sl_pct)
            self._current_tp = entry_price * (1 - self._tp_pct)
            self._low_water_mark = entry_price
            self._high_water_mark = entry_price  # track for MAE

        logger.info(
            "TrailingRM: opened %s %s @ %.2f | SL=%.2f TP=%.2f",
            side, self.asset, entry_price, self._current_sl, self._current_tp,
        )

    def close_position(self) -> None:
        """Clear all position state."""
        self._side = None
        self._entry_price = None
        self._size = None
        self._current_sl = None
        self._current_tp = None
        self._high_water_mark = None
        self._low_water_mark = None

    # --- Tick-by-tick update ---

    def update(self, price: float, timestamp: float) -> List[TradingSignal]:
        """Process a price tick and return exit signals if SL/TP hit.

        Called on every tick. Updates trailing levels and checks for exits.

        Args:
            price: Current market price
            timestamp: Current timestamp

        Returns:
            List of TradingSignal (empty or one CLOSE signal)
        """
        if not self.has_position:
            return []

        if self._side == "LONG":
            return self._update_long(price, timestamp)
        else:
            return self._update_short(price, timestamp)

    def _update_long(self, price: float, timestamp: float) -> List[TradingSignal]:
        """Update trailing levels for a LONG position."""
        # Track both water marks before exit checks so close signal
        # metadata always reflects the trigger price
        if price < self._low_water_mark:
            self._low_water_mark = price
        if price > self._high_water_mark:
            self._high_water_mark = price

        # Check SL first (price <= SL)
        if price <= self._current_sl:
            return [self._make_close_signal("stop_loss", price)]

        # Check TP (price >= TP)
        if price >= self._current_tp:
            return [self._make_close_signal("take_profit", price)]

        # Ratchet SL when price is at a new high and profit exceeds threshold
        profit_pct = (price - self._entry_price) / self._entry_price
        if profit_pct >= self._min_profit_to_trail:
            # Ratchet SL: lock in trail_pct of profit from entry
            # TP stays fixed at initial level (primary profit exit)
            new_sl = self._entry_price + (self._high_water_mark - self._entry_price) * self._trail_pct
            if new_sl > self._current_sl:
                self._current_sl = new_sl
                logger.debug(
                    "TrailingRM LONG: hwm=%.2f SL=%.2f TP=%.2f",
                    self._high_water_mark, self._current_sl, self._current_tp,
                )

        return []

    def _update_short(self, price: float, timestamp: float) -> List[TradingSignal]:
        """Update trailing levels for a SHORT position."""
        # Track both water marks before exit checks so close signal
        # metadata always reflects the trigger price
        if price > self._high_water_mark:
            self._high_water_mark = price
        if price < self._low_water_mark:
            self._low_water_mark = price

        # Check SL first (price >= SL for short)
        if price >= self._current_sl:
            return [self._make_close_signal("stop_loss", price)]

        # Check TP (price <= TP for short)
        if price <= self._current_tp:
            return [self._make_close_signal("take_profit", price)]

        # Ratchet SL when price is at a new low and profit exceeds threshold
        profit_pct = (self._entry_price - price) / self._entry_price
        if profit_pct >= self._min_profit_to_trail:
            # Ratchet SL down: lock in trail_pct of profit from entry
            # TP stays fixed at initial level (primary profit exit)
            new_sl = self._entry_price - (self._entry_price - self._low_water_mark) * self._trail_pct
            if new_sl < self._current_sl:
                self._current_sl = new_sl
                logger.debug(
                    "TrailingRM SHORT: lwm=%.2f SL=%.2f TP=%.2f",
                    self._low_water_mark, self._current_sl, self._current_tp,
                )

        return []

    def _make_close_signal(self, reason: str, trigger_price: float) -> TradingSignal:
        """Create a CLOSE signal with metadata about the exit."""
        pnl_pct = self._compute_pnl_pct(trigger_price)
        return TradingSignal(
            signal_type=SignalType.CLOSE,
            asset=self.asset,
            size=self._size,
            reason=f"trailing_{reason}",
            metadata={
                "trigger_price": trigger_price,
                "entry_price": self._entry_price,
                "side": self._side,
                "pnl_pct": pnl_pct,
                "sl_level": self._current_sl,
                "tp_level": self._current_tp,
                "high_water_mark": self._high_water_mark,
                "low_water_mark": self._low_water_mark,
            },
        )

    def _compute_pnl_pct(self, price: float) -> float:
        """Compute unrealized PnL percentage."""
        if self._side == "LONG":
            return (price - self._entry_price) / self._entry_price
        else:  # SHORT
            return (self._entry_price - price) / self._entry_price

    # --- Status ---

    def get_status(self) -> Dict[str, Any]:
        """Return current state for logging/monitoring."""
        if not self.has_position:
            return {"has_position": False}

        # MFE/MAE: max favorable/adverse excursion as fraction of entry price
        if self._side == "LONG":
            mfe = (self._high_water_mark - self._entry_price) / self._entry_price
            mae = (self._entry_price - self._low_water_mark) / self._entry_price
        else:  # SHORT
            mfe = (self._entry_price - self._low_water_mark) / self._entry_price
            mae = (self._high_water_mark - self._entry_price) / self._entry_price

        return {
            "has_position": True,
            "side": self._side,
            "entry_price": self._entry_price,
            "size": self._size,
            "current_sl": self._current_sl,
            "current_tp": self._current_tp,
            "high_water_mark": self._high_water_mark,
            "low_water_mark": self._low_water_mark,
            "max_favorable_excursion_pct": mfe,
            "max_adverse_excursion_pct": mae,
        }
