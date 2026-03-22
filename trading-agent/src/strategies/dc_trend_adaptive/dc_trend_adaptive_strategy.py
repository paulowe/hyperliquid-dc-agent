"""DC Trend-Adaptive Strategy — DC Adaptive with Guard 4: Trend Direction Filter.

Extends DCAdaptiveStrategy to block or reduce counter-trend entries
based on macro trend direction detected from DC sensor event flow.

Guard 4 (TrendDirectionFilter) adds:
- TMV-weighted directional bias detection from sensor events
- B-Simple/B-Strict decision rules for counter-trend trades
- Optional position close on trend flip (Ch 6 JC1/JC2 protective close)
- Nuclear long_only / short_only overrides

The core signal flow is inherited from DCAdaptiveStrategy. Guard 4
wraps the entry logic after Guards 1-3 have already filtered.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
)
from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy
from strategies.dc_trend_adaptive.config import DCTrendAdaptiveConfig
from strategies.dc_trend_adaptive.trend_direction_filter import TrendDirectionFilter

logger = logging.getLogger(__name__)


class DCTrendAdaptiveStrategy(DCAdaptiveStrategy):
    """DC Adaptive + Guard 4: Trend Direction Filter.

    Subclasses DCAdaptiveStrategy and overrides generate_signals() to
    insert trend filtering after the existing guards. All regime detection,
    overshoot tracking, and loss streak logic is inherited unchanged.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Override name for monitoring/telemetry
        self.name = "dc_trend_adaptive"

        self._trend_cfg = DCTrendAdaptiveConfig.from_dict(config)

        # Guard 4: Trend direction filter
        self._trend_filter = TrendDirectionFilter(
            lookback_seconds=self._trend_cfg.trend_lookback_seconds,
            min_events=self._trend_cfg.trend_min_events,
            min_consistency=self._trend_cfg.trend_min_consistency,
            counter_trend_action=self._trend_cfg.counter_trend_action,
            counter_trend_size_fraction=self._trend_cfg.counter_trend_size_fraction,
            bias_mode=self._trend_cfg.trend_bias_mode,
            strict_threshold=self._trend_cfg.trend_strict_threshold,
        )

        # Additional counters
        self._skipped_counter_trend = 0
        self._reduced_counter_trend = 0
        self._skipped_long_only = 0
        self._skipped_short_only = 0
        self._trend_flip_closes = 0

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a price tick with all 4 guards + trend flip protection.

        Overrides DCAdaptiveStrategy.generate_signals() to add:
        1. Feeding sensor events to the trend filter (alongside regime detector)
        2. Guard 4 check before entry signals
        3. Close-on-trend-flip protective rule (Ch 6 JC1/JC2)
        4. Nuclear long_only / short_only overrides
        """
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

            # Sensor events feed both regime detector AND trend filter
            if threshold_key == self._sensor_key:
                if event_type == "PDCC2_UP":
                    self._regime.record_event(+1, ts)
                    # Feed trend filter with TMV weight
                    start_p = event.get("start_price", price)
                    end_p = event.get("end_price", price)
                    tmv = abs(end_p - start_p) / start_p if start_p > 0 else 0.0
                    self._trend_filter.record_event(+1, ts, tmv=tmv)
                elif event_type == "PDCC_Down":
                    self._regime.record_event(-1, ts)
                    start_p = event.get("start_price", price)
                    end_p = event.get("end_price", price)
                    tmv = abs(end_p - start_p) / start_p if start_p > 0 else 0.0
                    self._trend_filter.record_event(-1, ts, tmv=tmv)
            # Trade events queue for entry logic and feed overshoot tracker
            elif threshold_key in self._trade_keys:
                if event_type in ("PDCC_Down", "PDCC2_UP"):
                    self._track_overshoot(event, price)
                trade_events.append(event)

        # Step 2.5: Close-on-trend-flip (Ch 6 JC1/JC2 protective close)
        if self._trend_cfg.close_on_trend_flip and self._trailing_rm.has_position:
            pos_side = self._trailing_rm.side
            trend = self._trend_filter.dominant_trend(ts)
            should_close = (
                (trend == "up" and pos_side == "SHORT")
                or (trend == "down" and pos_side == "LONG")
            )
            if should_close:
                logger.info(
                    "TREND FLIP CLOSE: %s position now counter-trend (trend=%s)",
                    pos_side, trend,
                )
                self._trend_flip_closes += 1
                pnl_pct = 0.0
                if pos_side == "LONG":
                    pnl_pct = (price - self._trailing_rm.entry_price) / self._trailing_rm.entry_price
                else:
                    pnl_pct = (self._trailing_rm.entry_price - price) / self._trailing_rm.entry_price

                # Record in loss guard
                self._loss_guard.record_trade(pnl_pct > 0, ts)

                signals.append(TradingSignal(
                    signal_type=SignalType.CLOSE,
                    asset=self._trend_cfg.symbol,
                    size=self._trailing_rm.size,
                    reason="trend_flip_protective",
                    metadata={
                        "pnl_pct": pnl_pct,
                        "dominant_trend": trend,
                        "position_side": pos_side,
                    },
                ))
                self._trailing_rm.close_position()
                return signals

        # Step 3: Check trailing risk manager for exits (every tick)
        exit_signals = self._trailing_rm.update(price, ts)
        if exit_signals:
            for sig in exit_signals:
                pnl = sig.metadata.get("pnl_pct", 0.0)
                is_win = pnl > 0
                self._loss_guard.record_trade(is_win, ts)

                logger.info(
                    "DC TrendAdaptive EXIT: %s @ %.2f | reason=%s | pnl=%.4f%%",
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

            # === NUCLEAR: long_only / short_only ===
            if self._trend_cfg.long_only and new_side == "SHORT":
                self._skipped_long_only += 1
                logger.info("SKIPPED SHORT — long_only mode")
                continue
            if self._trend_cfg.short_only and new_side == "LONG":
                self._skipped_short_only += 1
                logger.info("SKIPPED LONG — short_only mode")
                continue

            # === GUARD 1: Regime check ===
            if not self._regime.should_trade(ts):
                self._skipped_choppy += 1
                logger.info(
                    "DC TrendAdaptive: SKIPPED %s — choppy regime (rate=%.1f/min)",
                    event_type, self._regime.event_rate(ts),
                )
                continue

            # === GUARD 3: Loss streak check ===
            if not self._loss_guard.should_trade(ts):
                self._skipped_loss_guard += 1
                logger.info(
                    "DC TrendAdaptive: SKIPPED %s — loss streak cooldown (%d losses)",
                    event_type, self._loss_guard.consecutive_losses,
                )
                continue

            # === GUARD 4: Trend direction filter ===
            allowed, size_mult = self._trend_filter.should_trade_direction(new_side, ts)
            if not allowed:
                self._skipped_counter_trend += 1
                logger.info(
                    "DC TrendAdaptive: SKIPPED %s — counter-trend (trend=%s, bias=%.2f)",
                    new_side, self._trend_filter.dominant_trend(ts),
                    self._trend_filter.bias(ts),
                )
                continue

            if size_mult < 1.0:
                self._reduced_counter_trend += 1

            # Gate: position already open
            is_reversal = False
            old_size = 0.0
            previous_side = None
            if self._trailing_rm.has_position:
                current_side = self._trailing_rm.side
                if current_side == new_side:
                    logger.debug(
                        "DC TrendAdaptive: skipping %s — already %s",
                        event_type, current_side,
                    )
                    continue
                # Opposing direction -> reversal
                logger.info(
                    "DC TrendAdaptive REVERSAL: %s while %s — atomic flip",
                    event_type, current_side,
                )
                is_reversal = True
                old_size = self._trailing_rm.size
                previous_side = current_side
                self._trailing_rm.close_position()

            # Gate: cooldown (skip on reversals)
            if not is_reversal and ts - self._last_entry_time < self._cfg.cooldown_seconds:
                continue

            # Calculate new entry size with optional trend size reduction
            new_entry_size = self._cfg.position_size_usd / price
            if size_mult < 1.0:
                new_entry_size *= size_mult
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
            metadata["dominant_trend"] = self._trend_filter.dominant_trend(ts)
            metadata["trend_bias"] = self._trend_filter.bias(ts)

            # Mark counter-trend trades for asymmetric SL
            if self._trend_cfg.counter_trend_sl_pct and self._trend_filter.is_counter_trend(new_side, ts):
                metadata["counter_trend_sl"] = self._trend_cfg.counter_trend_sl_pct

            if event_type == "PDCC_Down":
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    asset=self._trend_cfg.symbol,
                    size=order_size,
                    reason=f"dc_trend_adaptive_short: {event_type}",
                    metadata=metadata,
                )
            else:
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    asset=self._trend_cfg.symbol,
                    size=order_size,
                    reason=f"dc_trend_adaptive_long: {event_type}",
                    metadata=metadata,
                )

            logger.info(
                "DC TrendAdaptive ENTRY: %s %.6f %s @ %.2f | event=%s%s | "
                "regime=%s | tp=%.4f%% | trend=%s",
                signal.signal_type.value, order_size, self._trend_cfg.symbol, price,
                event_type,
                " (reversal)" if is_reversal else "",
                metadata["regime"],
                metadata["adaptive_tp"] * 100,
                metadata["dominant_trend"],
            )
            signals.append(signal)
            self._last_entry_time = ts
            break  # Only one entry per tick

        return signals

    def on_trade_executed(
        self, signal: TradingSignal, executed_price: float, executed_size: float
    ) -> None:
        """Called when an entry signal results in a trade fill.

        Extends parent to support asymmetric SL for counter-trend trades.
        """
        self._trade_count += 1

        if signal.signal_type == SignalType.SELL:
            side = "SHORT"
        elif signal.signal_type == SignalType.BUY:
            side = "LONG"
        else:
            return

        actual_size = signal.metadata.get("new_position_size", executed_size)

        # Use adaptive TP
        adaptive_tp = signal.metadata.get("adaptive_tp", self._cfg.default_tp_pct)

        # Check for asymmetric SL on counter-trend trades
        counter_trend_sl = signal.metadata.get("counter_trend_sl")

        logger.info(
            "DC TrendAdaptive FILL: %s %.6f %s @ %.2f | adaptive_tp=%.4f%%%s%s",
            side, actual_size, self._trend_cfg.symbol, executed_price,
            adaptive_tp * 100,
            " (reversal)" if signal.metadata.get("reversal") else "",
            f" (counter-trend SL={counter_trend_sl*100:.2f}%)" if counter_trend_sl else "",
        )

        # Initialize trailing RM with adaptive TP (and optionally tighter SL)
        if not self._trailing_rm.has_position:
            old_tp = self._trailing_rm._tp_pct
            old_sl = self._trailing_rm._sl_pct
            self._trailing_rm._tp_pct = adaptive_tp
            if counter_trend_sl is not None:
                self._trailing_rm._sl_pct = counter_trend_sl
            self._trailing_rm.open_position(side, executed_price, actual_size)
            self._trailing_rm._tp_pct = old_tp
            self._trailing_rm._sl_pct = old_sl

    def get_status(self) -> Dict[str, Any]:
        """Return strategy state for monitoring."""
        status = super().get_status()
        status["name"] = "dc_trend_adaptive"

        # Add trend filter info
        ts = time.time()
        trend_status = self._trend_filter.get_status(ts)
        status["dominant_trend"] = trend_status["dominant_trend"]
        status["trend_bias"] = trend_status["bias"]
        status["trend_bias_mode"] = trend_status["bias_mode"]
        status["trend_events_in_window"] = trend_status["events_in_window"]
        status["avg_tmv_up"] = trend_status["avg_tmv_up"]
        status["avg_tmv_down"] = trend_status["avg_tmv_down"]
        status["skipped_counter_trend"] = self._skipped_counter_trend
        status["reduced_counter_trend"] = self._reduced_counter_trend
        status["skipped_long_only"] = self._skipped_long_only
        status["skipped_short_only"] = self._skipped_short_only
        status["trend_flip_closes"] = self._trend_flip_closes
        status["close_on_trend_flip"] = self._trend_cfg.close_on_trend_flip
        status["counter_trend_action"] = self._trend_cfg.counter_trend_action

        return status

    def start(self) -> None:
        """Activate strategy."""
        super().start()
        logger.info(
            "DC TrendAdaptive overlay: trend_lookback=%.0fs | min_events=%d | "
            "min_consistency=%.2f | bias_mode=%s | action=%s | "
            "close_on_flip=%s | long_only=%s | short_only=%s",
            self._trend_cfg.trend_lookback_seconds,
            self._trend_cfg.trend_min_events,
            self._trend_cfg.trend_min_consistency,
            self._trend_cfg.trend_bias_mode,
            self._trend_cfg.counter_trend_action,
            self._trend_cfg.close_on_trend_flip,
            self._trend_cfg.long_only,
            self._trend_cfg.short_only,
        )

    def stop(self) -> None:
        """Deactivate strategy."""
        logger.info(
            "DC TrendAdaptive stopped: ticks=%d events=%d trades=%d "
            "skipped_choppy=%d skipped_loss_guard=%d skipped_counter_trend=%d "
            "reduced_counter_trend=%d trend_flip_closes=%d",
            self._tick_count, self._dc_event_count, self._trade_count,
            self._skipped_choppy, self._skipped_loss_guard,
            self._skipped_counter_trend, self._reduced_counter_trend,
            self._trend_flip_closes,
        )
        self.is_active = False
