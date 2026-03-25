"""Tests for Archon reasoner — heuristic logic and Claude parsing."""

import pytest

from strategies.archon.context import MarketContext
from strategies.archon.reasoner import ArchonReasoner, TradeDecision


def make_context(**overrides) -> MarketContext:
    """Build a MarketContext with test defaults."""
    defaults = {
        "current_price": 40.0,
        "timestamp": 1000.0,
        "symbol": "HYPE",
        "price_high": 40.5,
        "price_low": 39.5,
        "price_mean": 40.0,
        "price_trend_pct": 0.5,
        "tick_count": 60,
        "trigger_event": {"event_type": "PDCC2_UP", "price": 40.0,
                          "start_price": 39.2, "end_price": 40.0},
        "recent_dc_events": [],
        "dc_up_count": 3,
        "dc_down_count": 2,
        "avg_overshoot_pct": 1.2,
        "regime": "quiet",
        "sensor_event_rate": 1.5,
        "recent_trades": [],
        "total_trades": 10,
        "win_count": 6,
        "loss_count": 4,
        "consecutive_losses": 0,
        "net_pnl_pct": 0.5,
        "has_position": False,
        "position_side": None,
        "position_entry": None,
        "position_pnl_pct": None,
    }
    defaults.update(overrides)
    return MarketContext(**defaults)


class TestHeuristicDecision:
    """Test the rule-based fallback logic."""

    def test_long_entry_on_dc_up(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="long")
        ctx = make_context(
            trigger_event={"event_type": "PDCC2_UP"},
            regime="quiet",
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "enter_long"
        assert decision.confidence >= 0.55

    def test_skip_short_in_long_only_mode(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="long")
        ctx = make_context(
            trigger_event={"event_type": "PDCC_Down"},
            regime="quiet",
            has_position=False,
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "skip"

    def test_close_on_opposing_dc_event(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="long")
        ctx = make_context(
            trigger_event={"event_type": "PDCC_Down"},
            has_position=True,
            position_side="LONG",
            position_entry=39.5,
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "close"

    def test_skip_same_direction_when_in_position(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="long")
        ctx = make_context(
            trigger_event={"event_type": "PDCC2_UP"},
            has_position=True,
            position_side="LONG",
            position_entry=39.5,
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "skip"

    def test_skip_in_choppy_regime(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="both")
        ctx = make_context(
            trigger_event={"event_type": "PDCC2_UP"},
            regime="choppy",
            sensor_event_rate=5.0,
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "skip"

    def test_skip_after_consecutive_losses(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="both")
        ctx = make_context(
            trigger_event={"event_type": "PDCC2_UP"},
            consecutive_losses=4,  # loss guard escalates: 3 for <3, 4 for >=3
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "skip"

    def test_short_entry_in_both_mode(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="both")
        ctx = make_context(
            trigger_event={"event_type": "PDCC_Down"},
            regime="quiet",
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.action == "enter_short"

    def test_boost_confidence_with_trend(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="long")
        ctx = make_context(
            trigger_event={"event_type": "PDCC2_UP"},
            price_trend_pct=1.0,  # strong uptrend
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.confidence >= 0.6  # base 0.55 + 0.10 trend boost

    def test_adaptive_tp_from_overshoot(self):
        reasoner = ArchonReasoner(use_ai=False, direction_filter="long")
        ctx = make_context(
            trigger_event={"event_type": "PDCC2_UP"},
            avg_overshoot_pct=2.0,  # 2% avg overshoot
        )
        decision = reasoner._decide_heuristic(ctx)
        assert decision.tp_pct == pytest.approx(0.008, abs=0.002)


class TestClaudeResponseParsing:
    """Test parsing of Claude's JSON responses."""

    def test_parse_valid_json(self):
        reasoner = ArchonReasoner(direction_filter="both")
        ctx = make_context()
        response = '{"action": "enter_long", "confidence": 0.8, "reasoning": "Strong uptrend", "tp_pct": 0.01, "sl_pct": 0.015}'
        decision = reasoner._parse_claude_response(response, ctx)
        assert decision.action == "enter_long"
        assert decision.confidence == 0.8
        assert decision.tp_pct == 0.01

    def test_parse_json_in_code_block(self):
        reasoner = ArchonReasoner(direction_filter="both")
        ctx = make_context()
        response = '```json\n{"action": "skip", "confidence": 0.3, "reasoning": "Unclear"}\n```'
        decision = reasoner._parse_claude_response(response, ctx)
        assert decision.action == "skip"

    def test_direction_filter_overrides_claude(self):
        reasoner = ArchonReasoner(direction_filter="long")
        ctx = make_context()
        response = '{"action": "enter_short", "confidence": 0.9, "reasoning": "Short signal"}'
        decision = reasoner._parse_claude_response(response, ctx)
        assert decision.action == "skip"  # filtered out

    def test_low_confidence_becomes_skip(self):
        reasoner = ArchonReasoner(direction_filter="both", min_confidence=0.6)
        ctx = make_context()
        response = '{"action": "enter_long", "confidence": 0.4, "reasoning": "Weak signal"}'
        decision = reasoner._parse_claude_response(response, ctx)
        assert decision.action == "skip"

    def test_close_action_passes_through(self):
        reasoner = ArchonReasoner(direction_filter="long")
        ctx = make_context()
        response = '{"action": "close", "confidence": 0.7, "reasoning": "Take profit"}'
        decision = reasoner._parse_claude_response(response, ctx)
        assert decision.action == "close"


class TestRateLimiting:
    def test_rate_limit_check(self):
        reasoner = ArchonReasoner(max_calls_per_hour=2)
        assert reasoner._check_rate_limit() is True
        reasoner._call_timestamps = [1000.0, 1001.0]  # expired
        assert reasoner._check_rate_limit() is True  # old timestamps pruned

    def test_rate_limit_blocks(self):
        import time
        reasoner = ArchonReasoner(max_calls_per_hour=2)
        now = time.time()
        reasoner._call_timestamps = [now - 10, now - 5]
        assert reasoner._check_rate_limit() is False


class TestReasonerStats:
    def test_initial_stats(self):
        reasoner = ArchonReasoner()
        stats = reasoner.get_stats()
        assert stats["ai_calls"] == 0
        assert stats["heuristic_calls"] == 0

    def test_heuristic_increments(self):
        reasoner = ArchonReasoner(use_ai=False)
        ctx = make_context()
        import asyncio
        asyncio.run(reasoner.decide(ctx))
        stats = reasoner.get_stats()
        assert stats["heuristic_calls"] == 1
