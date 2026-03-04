"""DC Adaptive Strategy — DC overshoot with three adaptive guards.

Extends the proven DC overshoot entry/exit mechanics with:
1. RegimeDetector — blocks entries in choppy markets
2. OvershootTracker — adapts TP to recent overshoot magnitudes
3. LossStreakGuard — circuit breaker after consecutive losses

The core signal flow is identical to DCOvershootStrategy. The guards
wrap the entry logic to prevent trading in unfavorable conditions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
    TradingStrategy,
)
from strategies.dc_adaptive.config import DCAdaptiveConfig
from strategies.dc_adaptive.loss_streak_guard import LossStreakGuard
from strategies.dc_adaptive.overshoot_tracker import OvershootTracker
from strategies.dc_adaptive.regime_detector import RegimeDetector
from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_overshoot.trailing_risk_manager import TrailingRiskManager

logger = logging.getLogger(__name__)


class DCAdaptiveStrategy(TradingStrategy):
    """DC overshoot strategy with regime detection, adaptive TP, and loss guards."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("dc_adaptive", config)
        self._cfg = DCAdaptiveConfig.from_dict(config)

        # All thresholds for the DC detector (trade + sensor)
        all_thresholds = list(self._cfg.dc_thresholds) + [self._cfg.sensor_threshold]
        self._sensor_key = f"{self._cfg.sensor_threshold[0]}:{self._cfg.sensor_threshold[1]}"
        self._trade_keys = {
            f"{t[0]}:{t[1]}" for t in self._cfg.dc_thresholds
        }

        # DC detector with all thresholds
        self._detector = LiveDCDetector(
            thresholds=all_thresholds,
            symbol=self._cfg.symbol,
        )

        # Trailing risk manager (reused from dc_overshoot)
        self._trailing_rm = TrailingRiskManager(
            asset=self._cfg.symbol,
            initial_stop_loss_pct=self._cfg.initial_stop_loss_pct,
            initial_take_profit_pct=self._cfg.default_tp_pct,
            trail_pct=self._cfg.trail_pct,
            min_profit_to_trail_pct=self._cfg.min_profit_to_trail_pct,
        )

        # === NEW: Three adaptive guards ===

        # Guard 1: Regime detector
        self._regime = RegimeDetector(
            lookback_seconds=self._cfg.lookback_seconds,
            choppy_rate_threshold=self._cfg.choppy_rate_threshold,
            trending_consistency_threshold=self._cfg.trending_consistency_threshold,
        )

        # Guard 2: Overshoot tracker
        self._os_tracker = OvershootTracker(
            window_size=self._cfg.os_window_size,
            min_samples=self._cfg.os_min_samples,
            tp_fraction=self._cfg.tp_fraction,
            min_tp_pct=self._cfg.min_tp_pct,
            default_tp_pct=self._cfg.default_tp_pct,
        )

        # Guard 3: Loss streak guard
        self._loss_guard = LossStreakGuard(
            max_consecutive_losses=self._cfg.max_consecutive_losses,
            base_cooldown_seconds=self._cfg.base_cooldown_seconds,
        )

        # For tracking overshoots: last DC event per direction
        self._last_dc_price: Optional[float] = None
        self._last_dc_direction: Optional[str] = None

        # Optional telemetry callback
        self._on_dc_event: Optional[Callable] = None

        # Counters
        self._tick_count = 0
        self._dc_event_count = 0
        self._trade_count = 0
        self._skipped_choppy = 0
        self._skipped_loss_guard = 0
        self._last_entry_time = 0.0

    def set_dc_event_callback(self, callback: Callable) -> None:
        """Register a callback invoked on every DC event (for telemetry)."""
        self._on_dc_event = callback

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a price tick: detect DC events, apply guards, emit signals."""
        price = market_data.price
        ts = market_data.timestamp
        self._tick_count += 1

        signals: List[TradingSignal] = []

        # Step 1: Run DC detection on all thresholds
        dc_events = self._detector.process_tick(price, ts)

        # Step 2: Route events — sensor vs trade
        trade_events = []
        for event in dc_events:
            event_type = event["event_type"]
            threshold_key = f"{event.get('threshold_down', 0)}:{event.get('threshold_up', 0)}"

            # Fire telemetry callback
            if self._on_dc_event is not None:
                try:
                    self._on_dc_event(event)
                except Exception:
                    pass

            # Sensor events feed the regime detector
            if threshold_key == self._sensor_key:
                if event_type == "PDCC2_UP":
                    self._regime.record_event(+1, ts)
                elif event_type == "PDCC_Down":
                    self._regime.record_event(-1, ts)
            # Trade events queue for entry logic and feed overshoot tracker
            elif threshold_key in self._trade_keys:
                # Track overshoots only from trade-threshold DC events
                # (sensor overshoots are too small and would miscalibrate TP)
                if event_type in ("PDCC_Down", "PDCC2_UP"):
                    self._track_overshoot(event, price)
                trade_events.append(event)

        # Step 3: Check trailing risk manager for exits (every tick)
        exit_signals = self._trailing_rm.update(price, ts)
        if exit_signals:
            for sig in exit_signals:
                # Record trade result in loss guard
                pnl = sig.metadata.get("pnl_pct", 0.0)
                is_win = pnl > 0
                self._loss_guard.record_trade(is_win, ts)

                logger.info(
                    "DC Adaptive EXIT: %s @ %.2f | reason=%s | pnl=%.4f%%",
                    sig.signal_type.value, price, sig.reason, pnl * 100,
                )
            signals.extend(exit_signals)
            self._trailing_rm.close_position()

        # Step 4: Process trade-level PDCC events for entries
        for event in trade_events:
            self._dc_event_count += 1
            event_type = event["event_type"]

            if self._cfg.log_events:
                logger.info(
                    "DC Event: %s | price=%.2f | start=%.2f -> end=%.2f",
                    event_type, price,
                    event.get("start_price", 0), event.get("end_price", 0),
                )

            # Only PDCC events trigger entries
            if event_type not in ("PDCC_Down", "PDCC2_UP"):
                continue

            new_side = "SHORT" if event_type == "PDCC_Down" else "LONG"

            # === GUARD 1: Regime check ===
            if not self._regime.should_trade(ts):
                self._skipped_choppy += 1
                logger.info(
                    "DC Adaptive: SKIPPED %s — choppy regime (rate=%.1f/min)",
                    event_type, self._regime.event_rate(ts),
                )
                continue

            # === GUARD 3: Loss streak check ===
            if not self._loss_guard.should_trade(ts):
                self._skipped_loss_guard += 1
                logger.info(
                    "DC Adaptive: SKIPPED %s — loss streak cooldown (%d losses)",
                    event_type, self._loss_guard.consecutive_losses,
                )
                continue

            # Gate: position already open
            is_reversal = False
            old_size = 0.0
            previous_side = None
            if self._trailing_rm.has_position:
                current_side = self._trailing_rm.side
                if current_side == new_side:
                    logger.debug(
                        "DC Adaptive: skipping %s — already %s",
                        event_type, current_side,
                    )
                    continue
                # Opposing direction → reversal
                logger.info(
                    "DC Adaptive REVERSAL: %s while %s — atomic flip",
                    event_type, current_side,
                )
                is_reversal = True
                old_size = self._trailing_rm.size
                previous_side = current_side
                self._trailing_rm.close_position()

            # Gate: cooldown (skip on reversals)
            if not is_reversal and ts - self._last_entry_time < self._cfg.cooldown_seconds:
                continue

            # Calculate new entry size
            new_entry_size = self._cfg.position_size_usd / price
            order_size = (old_size + new_entry_size) if is_reversal else new_entry_size

            metadata: Dict[str, Any] = {"dc_event": event}
            if is_reversal:
                metadata["reversal"] = True
                metadata["previous_side"] = previous_side
                metadata["new_position_size"] = new_entry_size

            # Add guard metadata
            metadata["regime"] = self._regime.classify(ts)
            metadata["adaptive_tp"] = self._os_tracker.adaptive_tp()
            metadata["consecutive_losses"] = self._loss_guard.consecutive_losses

            if event_type == "PDCC_Down":
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    asset=self._cfg.symbol,
                    size=order_size,
                    reason=f"dc_adaptive_short: {event_type}",
                    metadata=metadata,
                )
            else:
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    asset=self._cfg.symbol,
                    size=order_size,
                    reason=f"dc_adaptive_long: {event_type}",
                    metadata=metadata,
                )

            logger.info(
                "DC Adaptive ENTRY: %s %.6f %s @ %.2f | event=%s%s | regime=%s | tp=%.4f%%",
                signal.signal_type.value, order_size, self._cfg.symbol, price,
                event_type,
                " (reversal)" if is_reversal else "",
                metadata["regime"],
                metadata["adaptive_tp"] * 100,
            )
            signals.append(signal)
            self._last_entry_time = ts
            break  # Only one entry per tick

        return signals

    def on_trade_executed(
        self, signal: TradingSignal, executed_price: float, executed_size: float
    ) -> None:
        """Called when an entry signal results in a trade fill."""
        self._trade_count += 1

        if signal.signal_type == SignalType.SELL:
            side = "SHORT"
        elif signal.signal_type == SignalType.BUY:
            side = "LONG"
        else:
            return

        actual_size = signal.metadata.get("new_position_size", executed_size)

        # === GUARD 2: Use adaptive TP for this position ===
        adaptive_tp = signal.metadata.get("adaptive_tp", self._cfg.default_tp_pct)

        logger.info(
            "DC Adaptive FILL: %s %.6f %s @ %.2f | adaptive_tp=%.4f%%%s",
            side, actual_size, self._cfg.symbol, executed_price,
            adaptive_tp * 100,
            " (reversal)" if signal.metadata.get("reversal") else "",
        )

        # Initialize trailing RM with adaptive TP
        if not self._trailing_rm.has_position:
            # Temporarily update TP before opening position
            old_tp = self._trailing_rm._tp_pct
            self._trailing_rm._tp_pct = adaptive_tp
            self._trailing_rm.open_position(side, executed_price, actual_size)
            self._trailing_rm._tp_pct = old_tp  # restore for next position

    def _track_overshoot(self, event: Dict[str, Any], current_price: float) -> None:
        """Track overshoots from consecutive DC events.

        When we see a DC event in the opposite direction of the previous one,
        the price traveled from the previous DC endpoint to the current one.
        The overshoot is the extension beyond the previous DC confirmation.
        """
        event_type = event["event_type"]
        dc_end_price = event.get("end_price", current_price)

        if event_type == "PDCC_Down":
            direction = "down"
        elif event_type == "PDCC2_UP":
            direction = "up"
        else:
            return

        # If we have a previous DC in the opposite direction, compute overshoot
        if self._last_dc_price is not None and self._last_dc_direction != direction:
            overshoot = abs(dc_end_price - self._last_dc_price) / self._last_dc_price
            if overshoot > 0:
                self._os_tracker.record_overshoot(overshoot)
                logger.debug(
                    "Overshoot tracked: %.4f%% (%s -> %s)",
                    overshoot * 100, self._last_dc_direction, direction,
                )

        self._last_dc_price = dc_end_price
        self._last_dc_direction = direction

    def get_status(self) -> Dict[str, Any]:
        """Return strategy state for monitoring."""
        return {
            "name": self.name,
            "active": self.is_active,
            "symbol": self._cfg.symbol,
            "tick_count": self._tick_count,
            "dc_event_count": self._dc_event_count,
            "trade_count": self._trade_count,
            "skipped_choppy": self._skipped_choppy,
            "skipped_loss_guard": self._skipped_loss_guard,
            "regime": self._regime.classify(time.time()),
            "adaptive_tp": self._os_tracker.adaptive_tp(),
            "overshoot_distribution": self._os_tracker.percentiles(),
            "loss_streak": self._loss_guard.get_status(time.time()),
            "trailing_rm": self._trailing_rm.get_status(),
            "config": {
                "dc_thresholds": self._cfg.dc_thresholds,
                "sensor_threshold": self._cfg.sensor_threshold,
                "sl_pct": self._cfg.initial_stop_loss_pct,
                "default_tp_pct": self._cfg.default_tp_pct,
                "tp_fraction": self._cfg.tp_fraction,
                "trail_pct": self._cfg.trail_pct,
            },
        }

    def start(self) -> None:
        """Activate strategy."""
        super().start()
        logger.info(
            "DC Adaptive strategy started: %s | thresholds=%s | sensor=%s | "
            "SL=%.4f TP_default=%.4f trail=%.2f",
            self._cfg.symbol,
            self._cfg.dc_thresholds,
            self._cfg.sensor_threshold,
            self._cfg.initial_stop_loss_pct,
            self._cfg.default_tp_pct,
            self._cfg.trail_pct,
        )

    def stop(self) -> None:
        """Deactivate strategy."""
        super().stop()
        logger.info(
            "DC Adaptive strategy stopped: ticks=%d events=%d trades=%d "
            "skipped_choppy=%d skipped_loss_guard=%d",
            self._tick_count, self._dc_event_count, self._trade_count,
            self._skipped_choppy, self._skipped_loss_guard,
        )
