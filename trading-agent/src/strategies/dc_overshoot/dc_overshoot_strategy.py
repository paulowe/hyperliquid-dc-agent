"""DC Overshoot Strategy — model-free trading based on DC heuristic.

Core theory: After a Directional Change confirmation event (PDCC), price
tends to overshoot in the same direction by approximately the same threshold.

Entry rules:
  - PDCC_Down (downward confirmation) → SHORT entry
  - PDCC2_UP (upward confirmation) → LONG entry

Risk management:
  - TrailingRiskManager handles SL/TP with greedy trailing
  - When in loss: SL and TP stay at initial levels
  - When in profit: SL ratchets toward profit, TP pushes further

No ML model needed. The threshold itself is the prediction.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
    TradingStrategy,
)
from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_overshoot.config import DCOvershootConfig
from strategies.dc_overshoot.trailing_risk_manager import TrailingRiskManager

logger = logging.getLogger(__name__)


class DCOvershootStrategy(TradingStrategy):
    """Trading strategy exploiting DC overshoot heuristic with trailing risk."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("dc_overshoot", config)
        self._cfg = DCOvershootConfig.from_dict(config)

        # DC detector (reuse from dc_forecast)
        self._detector = LiveDCDetector(
            thresholds=self._cfg.dc_thresholds,
            symbol=self._cfg.symbol,
        )

        # Trailing risk manager
        self._trailing_rm = TrailingRiskManager(
            asset=self._cfg.symbol,
            initial_stop_loss_pct=self._cfg.initial_stop_loss_pct,
            initial_take_profit_pct=self._cfg.initial_take_profit_pct,
            trail_pct=self._cfg.trail_pct,
        )

        # Counters
        self._tick_count = 0
        self._dc_event_count = 0
        self._trade_count = 0
        self._last_entry_time = 0.0

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a price tick: detect DC events, manage trailing, emit signals."""
        price = market_data.price
        ts = market_data.timestamp
        self._tick_count += 1

        signals: List[TradingSignal] = []

        # Step 1: Run DC detection
        dc_events = self._detector.process_tick(price, ts)

        # Step 2: Check trailing risk manager for exits (every tick)
        exit_signals = self._trailing_rm.update(price, ts)
        if exit_signals:
            for sig in exit_signals:
                logger.info(
                    "DC Overshoot EXIT: %s @ %.2f | reason=%s | %s",
                    sig.signal_type.value, price, sig.reason, sig.metadata,
                )
            signals.extend(exit_signals)
            # Close the position tracking
            self._trailing_rm.close_position()

        # Step 3: Check for entry signals from PDCC events
        for event in dc_events:
            self._dc_event_count += 1
            event_type = event["event_type"]

            if self._cfg.log_events:
                logger.info(
                    "DC Event: %s | price=%.2f | start=%.2f → end=%.2f",
                    event_type, price,
                    event.get("start_price", 0), event.get("end_price", 0),
                )

            # Only PDCC events trigger entries (not OSV events)
            if event_type not in ("PDCC_Down", "PDCC2_UP"):
                continue

            # Gate: no entry if position already open
            if self._trailing_rm.has_position:
                logger.debug(
                    "DC Overshoot: skipping %s — position already open", event_type
                )
                continue

            # Gate: cooldown
            if ts - self._last_entry_time < self._cfg.cooldown_seconds:
                logger.debug(
                    "DC Overshoot: skipping %s — cooldown (%.1fs remaining)",
                    event_type,
                    self._cfg.cooldown_seconds - (ts - self._last_entry_time),
                )
                continue

            # Generate entry signal
            size = self._cfg.position_size_usd / price
            if event_type == "PDCC_Down":
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    asset=self._cfg.symbol,
                    size=size,
                    reason=f"dc_overshoot_short: {event_type}",
                    metadata={"dc_event": event},
                )
            else:  # PDCC2_UP
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    asset=self._cfg.symbol,
                    size=size,
                    reason=f"dc_overshoot_long: {event_type}",
                    metadata={"dc_event": event},
                )

            logger.info(
                "DC Overshoot ENTRY: %s %.6f %s @ %.2f | event=%s",
                signal.signal_type.value, size, self._cfg.symbol, price, event_type,
            )
            signals.append(signal)
            self._last_entry_time = ts
            break  # Only one entry per tick

        return signals

    def on_trade_executed(
        self, signal: TradingSignal, executed_price: float, executed_size: float
    ) -> None:
        """Called when an entry signal results in a trade fill.

        Initializes trailing risk manager with actual fill price.
        """
        self._trade_count += 1

        if signal.signal_type == SignalType.SELL:
            side = "SHORT"
        elif signal.signal_type == SignalType.BUY:
            side = "LONG"
        else:
            # CLOSE signal — position was exited, nothing to initialize
            return

        logger.info(
            "DC Overshoot FILL: %s %.6f %s @ %.2f",
            side, executed_size, self._cfg.symbol, executed_price,
        )

        # Initialize trailing RM with actual fill price
        if not self._trailing_rm.has_position:
            self._trailing_rm.open_position(side, executed_price, executed_size)

    def get_status(self) -> Dict[str, Any]:
        """Return strategy state for monitoring."""
        return {
            "name": self.name,
            "active": self.is_active,
            "symbol": self._cfg.symbol,
            "tick_count": self._tick_count,
            "dc_event_count": self._dc_event_count,
            "trade_count": self._trade_count,
            "trailing_rm": self._trailing_rm.get_status(),
            "config": {
                "dc_thresholds": self._cfg.dc_thresholds,
                "sl_pct": self._cfg.initial_stop_loss_pct,
                "tp_pct": self._cfg.initial_take_profit_pct,
                "trail_pct": self._cfg.trail_pct,
                "cooldown_s": self._cfg.cooldown_seconds,
            },
        }

    def start(self) -> None:
        """Activate strategy."""
        super().start()
        logger.info(
            "DC Overshoot strategy started: %s | thresholds=%s | SL=%.4f TP=%.4f trail=%.2f",
            self._cfg.symbol,
            self._cfg.dc_thresholds,
            self._cfg.initial_stop_loss_pct,
            self._cfg.initial_take_profit_pct,
            self._cfg.trail_pct,
        )

    def stop(self) -> None:
        """Deactivate strategy."""
        super().stop()
        logger.info(
            "DC Overshoot strategy stopped: ticks=%d events=%d trades=%d",
            self._tick_count, self._dc_event_count, self._trade_count,
        )
