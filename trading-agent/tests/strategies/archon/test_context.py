"""Tests for Archon context builder."""

import time

import pytest

from strategies.archon.context import ContextBuilder, DCEvent, MarketContext, TradeResult


class TestContextBuilder:
    def test_record_tick(self):
        ctx = ContextBuilder("HYPE", max_ticks=5)
        for i in range(3):
            ctx.record_tick(40.0 + i * 0.1, 1000.0 + i)
        assert len(ctx._ticks) == 3

    def test_tick_window_eviction(self):
        ctx = ContextBuilder("HYPE", max_ticks=3)
        for i in range(5):
            ctx.record_tick(40.0 + i, 1000.0 + i)
        assert len(ctx._ticks) == 3
        assert ctx._ticks[0].price == 42.0  # oldest surviving

    def test_record_trade_updates_stats(self):
        ctx = ContextBuilder("HYPE")
        ctx.record_trade(TradeResult(
            side="LONG", entry_price=40.0, exit_price=40.5,
            pnl_pct=0.0125, exit_reason="tp", duration_s=60, timestamp=1000,
        ))
        assert ctx._total_trades == 1
        assert ctx._win_count == 1
        assert ctx._consecutive_losses == 0

    def test_consecutive_losses_tracking(self):
        ctx = ContextBuilder("HYPE")
        ctx.record_trade(TradeResult(
            side="LONG", entry_price=40.0, exit_price=39.5,
            pnl_pct=-0.0125, exit_reason="sl", duration_s=60, timestamp=1000,
        ))
        ctx.record_trade(TradeResult(
            side="LONG", entry_price=40.0, exit_price=39.6,
            pnl_pct=-0.01, exit_reason="sl", duration_s=60, timestamp=1100,
        ))
        assert ctx.consecutive_losses == 2
        assert ctx._loss_count == 2

    def test_win_resets_consecutive_losses(self):
        ctx = ContextBuilder("HYPE")
        ctx.record_trade(TradeResult(
            side="LONG", entry_price=40.0, exit_price=39.5,
            pnl_pct=-0.0125, exit_reason="sl", duration_s=60, timestamp=1000,
        ))
        ctx.record_trade(TradeResult(
            side="LONG", entry_price=40.0, exit_price=40.5,
            pnl_pct=0.0125, exit_reason="tp", duration_s=60, timestamp=1100,
        ))
        assert ctx.consecutive_losses == 0


class TestRegimeDetection:
    def test_quiet_with_few_events(self):
        ctx = ContextBuilder("HYPE")
        now = time.time()
        assert ctx.get_regime(now) == "quiet"

    def test_quiet_with_under_3_events(self):
        ctx = ContextBuilder("HYPE")
        now = time.time()
        ctx._sensor_events.append((now - 10, 1))
        ctx._sensor_events.append((now - 5, -1))
        assert ctx.get_regime(now) == "quiet"

    def test_neutral_with_moderate_events(self):
        ctx = ContextBuilder("HYPE")
        now = time.time()
        # 3 events in 600s = low rate but enough for neutral
        for i in range(4):
            ctx._sensor_events.append((now - 500 + i * 100, 1))
        assert ctx.get_regime(now) == "neutral"

    def test_choppy_with_rapid_alternating(self):
        ctx = ContextBuilder("HYPE")
        now = time.time()
        # Many rapid alternating events
        for i in range(30):
            direction = 1 if i % 2 == 0 else -1
            ctx._sensor_events.append((now - 120 + i * 4, direction))
        regime = ctx.get_regime(now)
        assert regime == "choppy"


class TestBuildContext:
    def test_build_basic(self):
        ctx = ContextBuilder("HYPE")
        ctx.record_tick(40.0, 1000.0)
        ctx.record_tick(40.5, 1001.0)

        trigger = {"event_type": "PDCC2_UP", "price": 40.5,
                    "start_price": 39.7, "end_price": 40.5}
        mc = ctx.build(trigger)

        assert isinstance(mc, MarketContext)
        assert mc.symbol == "HYPE"
        assert mc.current_price == 40.5
        assert mc.price_high == 40.5
        assert mc.price_low == 40.0
        assert mc.has_position is False
        assert mc.trigger_event == trigger

    def test_build_with_position(self):
        ctx = ContextBuilder("HYPE")
        ctx.record_tick(40.0, 1000.0)
        trigger = {"event_type": "PDCC_Down", "price": 40.0,
                    "start_price": 40.5, "end_price": 40.0}
        mc = ctx.build(trigger, position_side="LONG", position_entry=39.0)

        assert mc.has_position is True
        assert mc.position_side == "LONG"
        assert mc.position_entry == 39.0
        assert mc.position_pnl_pct > 0  # 40.0 > 39.0

    def test_to_prompt_text(self):
        ctx = ContextBuilder("HYPE")
        ctx.record_tick(40.0, 1000.0)
        trigger = {"event_type": "PDCC2_UP", "price": 40.0,
                    "start_price": 39.2, "end_price": 40.0}
        mc = ctx.build(trigger)
        text = mc.to_prompt_text()

        assert "HYPE" in text
        assert "40.00" in text
        assert "PDCC2_UP" in text
        assert "FLAT" in text
