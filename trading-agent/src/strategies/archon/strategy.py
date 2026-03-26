"""Archon Strategy — Claude-augmented DC trading.

Combines DC event detection (timing) with Claude intelligence (decisions)
and trailing risk management (exits).

Signal flow:
  Tick → DC Detector → DC Event → Context Builder → Claude Reasoner → Signal
                                                          ↓
                                              TrailingRiskManager → Exit Signal
"""

from __future__ import annotations

import asyncio
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
from strategies.archon.config import ArchonConfig
from strategies.archon.context import ContextBuilder, DCEvent, TradeResult
from strategies.archon.reasoner import ArchonReasoner, TradeDecision
from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_overshoot.trailing_risk_manager import TrailingRiskManager

logger = logging.getLogger(__name__)


class ArchonStrategy(TradingStrategy):
    """Claude-augmented DC trading strategy.

    Uses DC events for timing, Claude for entry decisions,
    and TrailingRiskManager for exits.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("archon", config)
        self._cfg = ArchonConfig.from_dict(config)

        # DC detector with trade + sensor thresholds
        all_thresholds = [list(self._cfg.dc_threshold), list(self._cfg.sensor_threshold)]
        self._detector = LiveDCDetector(
            thresholds=all_thresholds,
            symbol=self._cfg.symbol,
        )
        self._trade_key = f"{self._cfg.dc_threshold[0]}:{self._cfg.dc_threshold[1]}"
        self._sensor_key = f"{self._cfg.sensor_threshold[0]}:{self._cfg.sensor_threshold[1]}"

        # Trailing risk manager
        self._trailing_rm = TrailingRiskManager(
            asset=self._cfg.symbol,
            initial_stop_loss_pct=self._cfg.initial_stop_loss_pct,
            initial_take_profit_pct=self._cfg.default_tp_pct,
            trail_pct=self._cfg.trail_pct,
            min_profit_to_trail_pct=self._cfg.min_profit_to_trail_pct,
        )

        # Context builder
        self._context = ContextBuilder(
            symbol=self._cfg.symbol,
            max_ticks=self._cfg.context_ticks,
            max_dc_events=self._cfg.context_dc_events,
            max_trades=self._cfg.context_trades,
        )

        # Claude reasoner
        self._reasoner = ArchonReasoner(
            model=self._cfg.model,
            use_ai=self._cfg.use_ai,
            min_confidence=self._cfg.min_confidence,
            max_calls_per_hour=self._cfg.max_calls_per_hour,
            direction_filter=self._cfg.direction_filter,
            system_prompt=self._cfg.system_prompt,
        )

        # Counters
        self._tick_count = 0
        self._dc_event_count = 0
        self._trade_count = 0
        self._skip_count = 0
        self._last_entry_time = 0.0
        self._last_decision: Optional[TradeDecision] = None

        # Callbacks
        self._on_dc_event: Optional[Callable] = None
        self._on_decision: Optional[Callable] = None

        # Pending decisions from async Claude calls
        self._pending_signals: List[TradingSignal] = []

    def set_dc_event_callback(self, callback: Callable) -> None:
        """Register callback for DC event telemetry."""
        self._on_dc_event = callback

    def set_decision_callback(self, callback: Callable) -> None:
        """Register callback for trade decisions (for logging)."""
        self._on_decision = callback

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a price tick synchronously.

        DC detection and exit checks happen synchronously.
        Entry decisions are queued via _process_dc_event (async call in bridge).
        """
        price = market_data.price
        ts = market_data.timestamp
        self._tick_count += 1

        # Record tick for context
        self._context.record_tick(price, ts)

        signals: List[TradingSignal] = []

        # Drain pending signals from async decisions
        if self._pending_signals:
            signals.extend(self._pending_signals)
            self._pending_signals.clear()

        # Step 1: DC detection
        dc_events = self._detector.process_tick(price, ts)

        for event in dc_events:
            event_type = event["event_type"]
            threshold_key = f"{event.get('threshold_down', 0)}:{event.get('threshold_up', 0)}"

            # Telemetry callback
            if self._on_dc_event is not None:
                try:
                    self._on_dc_event(event)
                except Exception:
                    pass

            # Route events
            if threshold_key == self._sensor_key:
                # Sensor events update regime detection
                self._context.record_dc_event(DCEvent(
                    event_type=event_type,
                    price=price,
                    start_price=event.get("start_price", price),
                    end_price=event.get("end_price", price),
                    timestamp=ts,
                    threshold=self._cfg.sensor_threshold[0],
                    is_sensor=True,
                ))

            elif threshold_key == self._trade_key:
                # Trade-level events: only PDCC confirmations trigger analysis
                if event_type in ("PDCC_Down", "PDCC2_UP"):
                    self._dc_event_count += 1
                    self._context.record_dc_event(DCEvent(
                        event_type=event_type,
                        price=price,
                        start_price=event.get("start_price", price),
                        end_price=event.get("end_price", price),
                        timestamp=ts,
                        threshold=self._cfg.dc_threshold[0],
                        is_sensor=False,
                    ))

                    logger.info(
                        "Archon DC Event: %s | price=%.2f | start=%.2f → end=%.2f",
                        event_type, price,
                        event.get("start_price", 0), event.get("end_price", 0),
                    )

                    # Queue async decision (will be processed by bridge)
                    # Store event for bridge to call process_dc_event_async()
                    self._last_trigger_event = event

        # Step 2: Check trailing RM for exits (every tick)
        exit_signals = self._trailing_rm.update(price, ts)
        if exit_signals:
            for sig in exit_signals:
                pnl = sig.metadata.get("pnl_pct", 0.0)
                # Record trade result in context
                entry_price = sig.metadata.get("entry_price", price)
                self._context.record_trade(TradeResult(
                    side=sig.metadata.get("side", ""),
                    entry_price=entry_price,
                    exit_price=price,
                    pnl_pct=pnl,
                    exit_reason=sig.reason,
                    duration_s=ts - self._last_entry_time,
                    timestamp=ts,
                ))
                logger.info(
                    "Archon EXIT: %s @ %.2f | reason=%s | pnl=%.4f%%",
                    sig.signal_type.value, price, sig.reason, pnl * 100,
                )
            signals.extend(exit_signals)
            self._trailing_rm.close_position()

        return signals

    async def process_dc_event_async(
        self, event: Dict[str, Any], price: float, ts: float
    ) -> Optional[TradingSignal]:
        """Process a DC event asynchronously — calls Claude for a decision.

        Called by the bridge when a trade-level DC event occurs.
        Returns a TradingSignal if Claude decides to trade, None otherwise.
        """
        # Build context
        context = self._context.build(
            trigger_event=event,
            position_side=self._trailing_rm.side,
            position_entry=self._trailing_rm.entry_price,
        )

        # Get decision from Claude (or heuristic fallback)
        decision = await self._reasoner.decide(context)
        self._last_decision = decision

        # Log decision
        logger.info(
            "Archon DECISION [%s]: %s (conf=%.2f) — %s",
            decision.source, decision.action, decision.confidence, decision.reasoning,
        )

        # Notify callback
        if self._on_decision is not None:
            try:
                self._on_decision(decision, context)
            except Exception:
                pass

        # Confidence gate — applies to ALL sources (Claude and heuristic)
        if decision.action in ("enter_long", "enter_short"):
            if decision.confidence < self._cfg.min_confidence:
                logger.info(
                    "Archon: SKIPPED %s — confidence %.2f < %.2f threshold (%s)",
                    decision.action, decision.confidence,
                    self._cfg.min_confidence, decision.reasoning,
                )
                self._skip_count += 1
                return None

        # Cooldown check
        if ts - self._last_entry_time < self._cfg.cooldown_seconds:
            if decision.action in ("enter_long", "enter_short"):
                logger.info("Archon: cooldown active, skipping entry")
                self._skip_count += 1
                return None

        # Convert decision to signal
        if decision.action == "enter_long":
            size = self._cfg.position_size_usd / price
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                asset=self._cfg.symbol,
                size=size,
                reason=f"archon_long: {decision.reasoning}",
                metadata={
                    "dc_event": event,
                    "confidence": decision.confidence,
                    "source": decision.source,
                    "adaptive_tp": decision.tp_pct,
                    "adaptive_sl": decision.sl_pct,
                    "regime": context.regime,
                },
            )
            logger.info(
                "Archon ENTRY: LONG %.6f %s @ %.2f | conf=%.2f | tp=%.3f%% sl=%.3f%%",
                size, self._cfg.symbol, price,
                decision.confidence, decision.tp_pct * 100, decision.sl_pct * 100,
            )
            return signal

        elif decision.action == "enter_short":
            size = self._cfg.position_size_usd / price
            signal = TradingSignal(
                signal_type=SignalType.SELL,
                asset=self._cfg.symbol,
                size=size,
                reason=f"archon_short: {decision.reasoning}",
                metadata={
                    "dc_event": event,
                    "confidence": decision.confidence,
                    "source": decision.source,
                    "adaptive_tp": decision.tp_pct,
                    "adaptive_sl": decision.sl_pct,
                    "regime": context.regime,
                },
            )
            logger.info(
                "Archon ENTRY: SHORT %.6f %s @ %.2f | conf=%.2f | tp=%.3f%% sl=%.3f%%",
                size, self._cfg.symbol, price,
                decision.confidence, decision.tp_pct * 100, decision.sl_pct * 100,
            )
            return signal

        elif decision.action == "close" and self._trailing_rm.has_position:
            pnl_pct = self._trailing_rm._compute_pnl_pct(price)
            signal = TradingSignal(
                signal_type=SignalType.CLOSE,
                asset=self._cfg.symbol,
                size=self._trailing_rm.size,
                reason=f"archon_close: {decision.reasoning}",
                metadata={
                    "side": self._trailing_rm.side,
                    "pnl_pct": pnl_pct,
                    "trigger_price": price,
                    "entry_price": self._trailing_rm.entry_price,
                    "confidence": decision.confidence,
                    "source": decision.source,
                },
            )
            # Record trade
            self._context.record_trade(TradeResult(
                side=self._trailing_rm.side,
                entry_price=self._trailing_rm.entry_price,
                exit_price=price,
                pnl_pct=pnl_pct,
                exit_reason="archon_close",
                duration_s=ts - self._last_entry_time,
                timestamp=ts,
            ))
            self._trailing_rm.close_position()
            logger.info(
                "Archon CLOSE: %s @ %.2f | pnl=%.4f%% | reason=%s",
                signal.metadata["side"], price, pnl_pct * 100, decision.reasoning,
            )
            return signal

        self._skip_count += 1
        return None

    def on_trade_executed(
        self, signal: TradingSignal, executed_price: float, executed_size: float,
        timestamp: float = 0.0,
    ) -> None:
        """Called when an entry signal results in a trade fill."""
        self._trade_count += 1
        # Use provided timestamp (from candle) or fallback to wall clock
        self._last_entry_time = timestamp if timestamp > 0 else time.time()

        if signal.signal_type == SignalType.SELL:
            side = "SHORT"
        elif signal.signal_type == SignalType.BUY:
            side = "LONG"
        else:
            return

        # Use adaptive TP and SL from the decision
        adaptive_tp = signal.metadata.get("adaptive_tp", self._cfg.default_tp_pct)
        adaptive_sl = signal.metadata.get("adaptive_sl", self._cfg.initial_stop_loss_pct)

        # Open position in trailing RM with decision-specific TP and SL
        if not self._trailing_rm.has_position:
            old_tp = self._trailing_rm._tp_pct
            old_sl = self._trailing_rm._sl_pct
            self._trailing_rm._tp_pct = adaptive_tp
            self._trailing_rm._sl_pct = adaptive_sl
            self._trailing_rm.open_position(side, executed_price, executed_size)
            self._trailing_rm._tp_pct = old_tp
            self._trailing_rm._sl_pct = old_sl

        logger.info(
            "Archon FILL: %s %.6f %s @ %.2f | tp=%.3f%% sl=%.3f%%",
            side, executed_size, self._cfg.symbol, executed_price,
            adaptive_tp * 100, adaptive_sl * 100,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return strategy state for monitoring."""
        return {
            "name": self.name,
            "active": self.is_active,
            "symbol": self._cfg.symbol,
            "tick_count": self._tick_count,
            "dc_event_count": self._dc_event_count,
            "trade_count": self._trade_count,
            "skip_count": self._skip_count,
            "regime": self._context.get_regime(time.time()),
            "consecutive_losses": self._context.consecutive_losses,
            "trailing_rm": self._trailing_rm.get_status(),
            "reasoner": self._reasoner.get_stats(),
            "last_decision": {
                "action": self._last_decision.action,
                "confidence": self._last_decision.confidence,
                "source": self._last_decision.source,
                "reasoning": self._last_decision.reasoning,
            } if self._last_decision else None,
        }

    def start(self) -> None:
        super().start()
        logger.info(
            "Archon strategy started: %s | threshold=%s | sensor=%s | "
            "direction=%s | model=%s | ai=%s",
            self._cfg.symbol,
            self._cfg.dc_threshold,
            self._cfg.sensor_threshold,
            self._cfg.direction_filter,
            self._cfg.model,
            self._cfg.use_ai,
        )

    def stop(self) -> None:
        super().stop()
        stats = self._reasoner.get_stats()
        logger.info(
            "Archon strategy stopped: ticks=%d events=%d trades=%d skips=%d "
            "| ai_calls=%d (ok=%d fail=%d) heuristic=%d",
            self._tick_count, self._dc_event_count, self._trade_count,
            self._skip_count, stats["ai_calls"], stats["ai_successes"],
            stats["ai_failures"], stats["heuristic_calls"],
        )
