"""Tests for Archon strategy — integration of DC detection + reasoning + risk management."""

import asyncio
import time

import pytest

from interfaces.strategy import MarketData, SignalType
from strategies.archon.strategy import ArchonStrategy


def make_config(**overrides):
    """Build strategy config with test defaults."""
    cfg = {
        "symbol": "BTC",
        "dc_threshold": [0.001, 0.001],  # 0.1% for fast testing
        "sensor_threshold": [0.0005, 0.0005],
        "position_size_usd": 50.0,
        "leverage": 5,
        "initial_stop_loss_pct": 0.003,
        "default_tp_pct": 0.002,
        "trail_pct": 0.5,
        "min_profit_to_trail_pct": 0.001,
        "use_ai": False,  # Use heuristic for tests
        "direction_filter": "long",
        "cooldown_seconds": 0,
        "max_consecutive_losses": 3,
        "context_ticks": 20,
        "context_dc_events": 10,
        "context_trades": 10,
        "lookback_seconds": 600,
        "choppy_rate_threshold": 4.0,
        "trending_consistency_threshold": 0.6,
    }
    cfg.update(overrides)
    return cfg


def md(price, ts=None):
    return MarketData(
        asset="BTC", price=price, volume_24h=0.0,
        timestamp=ts or time.time(),
    )


class TestArchonInit:
    def test_creates_strategy(self):
        strategy = ArchonStrategy(make_config())
        strategy.start()
        assert strategy.is_active
        assert strategy.name == "archon"

    def test_get_status(self):
        strategy = ArchonStrategy(make_config())
        status = strategy.get_status()
        assert "tick_count" in status
        assert "reasoner" in status
        assert status["tick_count"] == 0


class TestDCDetection:
    def test_ticks_increment(self):
        strategy = ArchonStrategy(make_config())
        strategy.start()
        strategy.generate_signals(md(100_000.0, 1000.0), [], 100_000.0)
        strategy.generate_signals(md(100_001.0, 1001.0), [], 100_000.0)
        assert strategy._tick_count == 2

    def test_dc_events_detected_on_price_drop(self):
        strategy = ArchonStrategy(make_config())
        strategy.start()
        price = 100_000.0
        ts = 1000.0
        for i in range(20):
            price -= price * 0.0002  # 0.02% drop each tick
            strategy.generate_signals(md(price, ts), [], 100_000.0)
            ts += 10.0
        # With 0.1% threshold, 20 * 0.02% = 0.4% total should trigger events
        assert strategy._dc_event_count >= 1


class TestAsyncDecisions:
    def test_process_dc_event_long_entry(self):
        strategy = ArchonStrategy(make_config(direction_filter="long"))
        strategy.start()
        # Seed some context
        strategy._context.record_tick(40.0, 999.0)
        strategy._context.record_tick(40.5, 1000.0)

        event = {"event_type": "PDCC2_UP", "price": 40.5,
                 "start_price": 39.8, "end_price": 40.5}
        signal = asyncio.run(
            strategy.process_dc_event_async(event, 40.5, 1001.0)
        )
        assert signal is not None
        assert signal.signal_type == SignalType.BUY

    def test_process_dc_event_skip_short_in_long_only(self):
        strategy = ArchonStrategy(make_config(direction_filter="long"))
        strategy.start()
        strategy._context.record_tick(40.0, 999.0)

        event = {"event_type": "PDCC_Down", "price": 39.5,
                 "start_price": 40.2, "end_price": 39.5}
        signal = asyncio.run(
            strategy.process_dc_event_async(event, 39.5, 1001.0)
        )
        # Should be skip (no position to close) or None
        assert signal is None

    def test_process_dc_event_close_on_reversal(self):
        strategy = ArchonStrategy(make_config(direction_filter="long"))
        strategy.start()
        strategy._context.record_tick(40.0, 999.0)

        # First, enter a LONG position
        entry_event = {"event_type": "PDCC2_UP", "price": 40.0,
                       "start_price": 39.2, "end_price": 40.0}
        entry_signal = asyncio.run(
            strategy.process_dc_event_async(entry_event, 40.0, 1000.0)
        )
        assert entry_signal is not None
        strategy.on_trade_executed(entry_signal, 40.0, entry_signal.size)
        assert strategy._trailing_rm.has_position

        # Now DC Down should close the position
        close_event = {"event_type": "PDCC_Down", "price": 39.5,
                       "start_price": 40.2, "end_price": 39.5}
        close_signal = asyncio.run(
            strategy.process_dc_event_async(close_event, 39.5, 1100.0)
        )
        assert close_signal is not None
        assert close_signal.signal_type == SignalType.CLOSE

    def test_cooldown_blocks_entry(self):
        strategy = ArchonStrategy(make_config(
            direction_filter="long", cooldown_seconds=60,
        ))
        strategy.start()
        strategy._context.record_tick(40.0, 999.0)
        strategy._last_entry_time = time.time()  # just entered

        event = {"event_type": "PDCC2_UP", "price": 40.5,
                 "start_price": 39.8, "end_price": 40.5}
        signal = asyncio.run(
            strategy.process_dc_event_async(event, 40.5, time.time())
        )
        assert signal is None  # cooldown active


class TestTradeExecution:
    def test_on_trade_executed_opens_position(self):
        strategy = ArchonStrategy(make_config())
        strategy.start()
        strategy._context.record_tick(40.0, 999.0)

        event = {"event_type": "PDCC2_UP", "price": 40.0,
                 "start_price": 39.2, "end_price": 40.0}
        signal = asyncio.run(
            strategy.process_dc_event_async(event, 40.0, 1000.0)
        )
        strategy.on_trade_executed(signal, 40.0, signal.size)

        assert strategy._trailing_rm.has_position
        assert strategy._trailing_rm.side == "LONG"
        assert strategy._trade_count == 1

    def test_trailing_stop_exit(self):
        strategy = ArchonStrategy(make_config(
            initial_stop_loss_pct=0.01,
        ))
        strategy.start()
        strategy._context.record_tick(40.0, 999.0)

        # Enter position
        event = {"event_type": "PDCC2_UP", "price": 40.0,
                 "start_price": 39.2, "end_price": 40.0}
        signal = asyncio.run(
            strategy.process_dc_event_async(event, 40.0, 1000.0)
        )
        strategy.on_trade_executed(signal, 40.0, signal.size)

        # Price drops to hit SL (40.0 * 0.99 = 39.6)
        exit_signals = strategy.generate_signals(md(39.5, 1100.0), [], 100_000.0)
        closes = [s for s in exit_signals if s.signal_type == SignalType.CLOSE]
        assert len(closes) >= 1
        assert "stop_loss" in closes[0].reason


class TestStrategyLifecycle:
    def test_start_stop(self):
        strategy = ArchonStrategy(make_config())
        strategy.start()
        assert strategy.is_active
        strategy.stop()
        assert not strategy.is_active

    def test_decision_callback(self):
        decisions = []
        strategy = ArchonStrategy(make_config(direction_filter="long"))
        strategy.start()
        strategy.set_decision_callback(lambda d, c: decisions.append(d))
        strategy._context.record_tick(40.0, 999.0)

        event = {"event_type": "PDCC2_UP", "price": 40.0,
                 "start_price": 39.2, "end_price": 40.0}
        asyncio.run(strategy.process_dc_event_async(event, 40.0, 1000.0))

        assert len(decisions) == 1
        assert decisions[0].action == "enter_long"
