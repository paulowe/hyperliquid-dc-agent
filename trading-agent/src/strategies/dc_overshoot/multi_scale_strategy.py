"""Multi-Scale DC Overshoot Strategy.

Uses low-threshold DC events as intelligence sensors to inform high-threshold
trading decisions. The momentum score from sensors must confirm the trade-
threshold signal before an entry is permitted.

Entry rules:
  - Trade-threshold PDCC_Down + momentum < -min_score → SHORT
  - Trade-threshold PDCC2_UP  + momentum > +min_score → LONG
  - If momentum does not confirm → signal is filtered (no trade)

The key insight: low thresholds (0.002-0.004) produce many signals that are
individually unprofitable due to fees, but they contain valuable directional
information that improves high-threshold trade quality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
    TradingStrategy,
)
from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_overshoot.momentum_scorer import MomentumScorer
from strategies.dc_overshoot.multi_scale_config import MultiScaleConfig
from strategies.dc_overshoot.trailing_risk_manager import TrailingRiskManager

logger = logging.getLogger(__name__)


class MultiScaleDCStrategy(TradingStrategy):
    """Multi-scale DC strategy with sensor-informed momentum filtering."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("multi_scale_dc", config)
        self._cfg = MultiScaleConfig.from_dict(config)

        # Single DC detector with ALL thresholds (sensors + trade)
        self._detector = LiveDCDetector(
            thresholds=self._cfg.all_thresholds(),
            symbol=self._cfg.symbol,
        )

        # Momentum scorer (fed by sensor events only)
        self._scorer = MomentumScorer(alpha=self._cfg.momentum_alpha)

        # Trailing risk manager (reused from dc_overshoot)
        self._trailing_rm = TrailingRiskManager(
            asset=self._cfg.symbol,
            initial_stop_loss_pct=self._cfg.initial_stop_loss_pct,
            initial_take_profit_pct=self._cfg.initial_take_profit_pct,
            trail_pct=self._cfg.trail_pct,
            min_profit_to_trail_pct=self._cfg.min_profit_to_trail_pct,
        )

        # Threshold routing sets
        self._sensor_keys = self._cfg.sensor_threshold_keys()
        self._trade_key = self._cfg.trade_threshold_key()

        # Optional telemetry callback: f(event_dict) -> None
        self._on_dc_event = None

        # Counters
        self._tick_count = 0
        self._sensor_event_count = 0
        self._trade_event_count = 0
        self._filtered_count = 0  # Trade signals blocked by momentum filter
        self._trade_count = 0
        self._last_entry_time = 0.0

    def set_dc_event_callback(self, callback) -> None:
        """Register a callback invoked on every DC event (for telemetry)."""
        self._on_dc_event = callback

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a price tick: detect DC events across all thresholds.

        Sensor events update momentum score.
        Trade events are filtered by momentum before emitting entry signals.
        Trailing RM checks for exits on every tick.
        """
        price = market_data.price
        ts = market_data.timestamp
        self._tick_count += 1

        signals: List[TradingSignal] = []

        # Step 1: Run DC detection on ALL thresholds simultaneously
        dc_events = self._detector.process_tick(price, ts)

        # Step 2: Check trailing risk manager for exits (every tick)
        exit_signals = self._trailing_rm.update(price, ts)
        if exit_signals:
            for sig in exit_signals:
                logger.info(
                    "Multi-scale EXIT: %s @ %.2f | reason=%s",
                    sig.signal_type.value, price, sig.reason,
                )
            signals.extend(exit_signals)
            self._trailing_rm.close_position()

        # Step 3: Route events — sensor events → scorer, trade events → entry logic
        trade_events = []
        for event in dc_events:
            event_type = event["event_type"]
            threshold_key = f"{event['threshold_down']}:{event['threshold_up']}"

            # Only PDCC events are relevant (not OSV)
            if event_type not in ("PDCC_Down", "PDCC2_UP"):
                continue

            direction = +1 if event_type == "PDCC2_UP" else -1

            # Fire telemetry callback if registered
            if self._on_dc_event is not None:
                try:
                    is_sensor = threshold_key in self._sensor_keys
                    self._on_dc_event({**event, "is_sensor": is_sensor})
                except Exception:
                    pass  # Telemetry must never crash strategy

            if threshold_key in self._sensor_keys:
                # Sensor event → update momentum scorer
                self._sensor_event_count += 1
                self._scorer.update(direction, ts, threshold_key, price)
                if self._cfg.log_sensor_events:
                    logger.info(
                        "Sensor %s: %s @ %.2f | score=%.3f",
                        threshold_key, event_type, price, self._scorer.score,
                    )

            elif threshold_key == self._trade_key:
                # Trade event → queue for momentum filtering
                self._trade_event_count += 1
                trade_events.append(event)

        # Step 4: Process trade events through momentum filter
        for event in trade_events:
            event_type = event["event_type"]
            new_side = "SHORT" if event_type == "PDCC_Down" else "LONG"
            direction = +1 if event_type == "PDCC2_UP" else -1
            score = self._scorer.score

            # Momentum filter: score must confirm trade direction
            score_confirms = (
                (direction == +1 and score > self._cfg.min_momentum_score)
                or (direction == -1 and score < -self._cfg.min_momentum_score)
            )

            if not score_confirms:
                self._filtered_count += 1
                if self._cfg.log_events:
                    logger.info(
                        "Multi-scale FILTERED: %s @ %.2f | score=%.3f (need %s%.3f)",
                        event_type, price, score,
                        ">" if direction == +1 else "<-",
                        self._cfg.min_momentum_score,
                    )
                continue

            # Gate: position already open
            is_reversal = False
            old_size = 0.0
            previous_side = None
            if self._trailing_rm.has_position:
                current_side = self._trailing_rm.side

                # Same direction → skip
                if current_side == new_side:
                    continue

                # Opposing direction → reversal
                logger.info(
                    "Multi-scale REVERSAL: %s while %s | score=%.3f",
                    event_type, current_side, score,
                )
                is_reversal = True
                old_size = self._trailing_rm.size
                previous_side = current_side
                self._trailing_rm.close_position()

            # Gate: cooldown (skip on reversals)
            if not is_reversal and ts - self._last_entry_time < self._cfg.cooldown_seconds:
                continue

            # Calculate entry size
            new_entry_size = self._cfg.position_size_usd / price
            order_size = (old_size + new_entry_size) if is_reversal else new_entry_size

            metadata: Dict[str, Any] = {
                "dc_event": event,
                "momentum_score": round(score, 4),
                "sensor_event_count": self._sensor_event_count,
                "regime": self._scorer.get_regime(),
            }
            if is_reversal:
                metadata["reversal"] = True
                metadata["previous_side"] = previous_side
                metadata["new_position_size"] = new_entry_size

            if event_type == "PDCC_Down":
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    asset=self._cfg.symbol,
                    size=order_size,
                    reason=f"multi_scale_short: {event_type} (score={score:.3f})",
                    metadata=metadata,
                )
            else:
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    asset=self._cfg.symbol,
                    size=order_size,
                    reason=f"multi_scale_long: {event_type} (score={score:.3f})",
                    metadata=metadata,
                )

            logger.info(
                "Multi-scale ENTRY: %s %.6f %s @ %.2f | score=%.3f | regime=%s%s",
                signal.signal_type.value, order_size, self._cfg.symbol, price,
                score, self._scorer.get_regime(),
                " (reversal)" if is_reversal else "",
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

        logger.info(
            "Multi-scale FILL: %s %.6f %s @ %.2f%s",
            side, actual_size, self._cfg.symbol, executed_price,
            " (reversal)" if signal.metadata.get("reversal") else "",
        )

        if not self._trailing_rm.has_position:
            self._trailing_rm.open_position(side, executed_price, actual_size)

    def get_status(self) -> Dict[str, Any]:
        """Return strategy state for monitoring."""
        return {
            "name": self.name,
            "active": self.is_active,
            "symbol": self._cfg.symbol,
            "tick_count": self._tick_count,
            "sensor_event_count": self._sensor_event_count,
            "trade_event_count": self._trade_event_count,
            "filtered_count": self._filtered_count,
            "trade_count": self._trade_count,
            "scorer": self._scorer.get_status(),
            "trailing_rm": self._trailing_rm.get_status(),
            "config": {
                "sensor_thresholds": self._cfg.sensor_thresholds,
                "trade_threshold": self._cfg.trade_threshold,
                "momentum_alpha": self._cfg.momentum_alpha,
                "min_momentum_score": self._cfg.min_momentum_score,
                "sl_pct": self._cfg.initial_stop_loss_pct,
                "tp_pct": self._cfg.initial_take_profit_pct,
                "trail_pct": self._cfg.trail_pct,
            },
        }

    def start(self) -> None:
        """Activate strategy."""
        super().start()
        logger.info(
            "Multi-scale DC strategy started: %s | sensors=%s | trade=%s | "
            "alpha=%.2f min_score=%.2f | SL=%.4f TP=%.4f trail=%.2f",
            self._cfg.symbol,
            self._cfg.sensor_thresholds,
            self._cfg.trade_threshold,
            self._cfg.momentum_alpha,
            self._cfg.min_momentum_score,
            self._cfg.initial_stop_loss_pct,
            self._cfg.initial_take_profit_pct,
            self._cfg.trail_pct,
        )

    def stop(self) -> None:
        """Deactivate strategy."""
        super().stop()
        logger.info(
            "Multi-scale DC stopped: ticks=%d sensors=%d trades=%d filtered=%d",
            self._tick_count, self._sensor_event_count,
            self._trade_count, self._filtered_count,
        )
