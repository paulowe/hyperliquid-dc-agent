"""Tests for Archon prompt loading and rendering."""

import pytest

from strategies.archon.prompts import load_system_prompt, render_decision_prompt


class TestSystemPrompt:
    def test_loads(self):
        prompt = load_system_prompt()
        assert len(prompt) > 100
        assert "Archon" in prompt
        assert "DC" in prompt
        assert "JSON" in prompt

    def test_contains_edge_knowledge(self):
        prompt = load_system_prompt()
        # Should encode our live-learned insights, not generic platitudes
        assert "high in range" in prompt
        assert "0.76%" in prompt or "MFE" in prompt
        assert "38%" in prompt  # 38% of trades never reach 0.5%
        assert "fee" in prompt.lower()


class TestDecisionPrompt:
    def test_renders_basic(self):
        data = {
            "symbol": "HYPE",
            "current_price": 40.50,
            "trigger_event_type": "PDCC2_UP",
            "trigger_start_price": 39.70,
            "trigger_end_price": 40.50,
            "trigger_move_pct": 2.02,
            "price_high": 40.60,
            "price_low": 39.90,
            "price_mean": 40.25,
            "price_trend_pct": 0.85,
            "range_pct": 1.75,
            "range_position": 0.86,
            "tick_count": 60,
            "regime": "quiet",
            "sensor_event_rate": 1.5,
            "dc_event_count": 5,
            "recent_dc_events": [],
            "dc_up_count": 3,
            "dc_down_count": 2,
            "avg_overshoot_pct": 1.2,
            "recent_trades": [],
            "total_trades": 10,
            "win_count": 7,
            "loss_count": 3,
            "win_rate": 70.0,
            "consecutive_losses": 0,
            "net_pnl_pct": 1.5,
            "has_position": False,
            "position_side": None,
            "position_entry": 0,
            "position_pnl_pct": 0,
            "recent_ticks": [40.1, 40.2, 40.3, 40.4, 40.5],
        }
        rendered = render_decision_prompt(data)
        assert "HYPE" in rendered
        assert "40.50" in rendered
        assert "PDCC2_UP" in rendered
        assert "FLAT" in rendered
        assert "quiet" in rendered

    def test_renders_with_position(self):
        data = {
            "symbol": "SOL",
            "current_price": 92.0,
            "trigger_event_type": "PDCC_Down",
            "trigger_start_price": 94.0,
            "trigger_end_price": 92.0,
            "trigger_move_pct": -2.13,
            "price_high": 94.5,
            "price_low": 91.5,
            "price_mean": 93.0,
            "price_trend_pct": -1.5,
            "range_pct": 3.28,
            "range_position": 0.17,
            "tick_count": 30,
            "regime": "neutral",
            "sensor_event_rate": 2.5,
            "dc_event_count": 3,
            "recent_dc_events": [
                {"event_type": "PDCC2_UP", "price": 94.0, "overshoot_pct": 1.5},
            ],
            "dc_up_count": 2,
            "dc_down_count": 1,
            "avg_overshoot_pct": 1.8,
            "recent_trades": [
                {"side": "LONG", "exit_reason": "trailing_take_profit",
                 "pnl_pct": 0.6, "duration_s": 120},
            ],
            "total_trades": 5,
            "win_count": 3,
            "loss_count": 2,
            "win_rate": 60.0,
            "consecutive_losses": 1,
            "net_pnl_pct": 0.3,
            "has_position": True,
            "position_side": "LONG",
            "position_entry": 93.50,
            "position_pnl_pct": -1.6,
            "recent_ticks": [93.0, 92.5, 92.0],
        }
        rendered = render_decision_prompt(data)
        assert "SOL" in rendered
        assert "LONG" in rendered
        assert "93.50" in rendered
        assert "PDCC_Down" in rendered

    def test_renders_recent_ticks(self):
        data = {
            "symbol": "TAO",
            "current_price": 340.0,
            "trigger_event_type": "PDCC2_UP",
            "trigger_start_price": 333.0,
            "trigger_end_price": 340.0,
            "trigger_move_pct": 2.1,
            "price_high": 341.0,
            "price_low": 332.0,
            "price_mean": 336.0,
            "price_trend_pct": 2.4,
            "range_pct": 2.7,
            "range_position": 0.89,
            "tick_count": 60,
            "regime": "trending",
            "sensor_event_rate": 3.0,
            "dc_event_count": 0,
            "recent_dc_events": [],
            "dc_up_count": 0,
            "dc_down_count": 0,
            "avg_overshoot_pct": 0,
            "recent_trades": [],
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "consecutive_losses": 0,
            "net_pnl_pct": 0,
            "has_position": False,
            "position_side": None,
            "position_entry": 0,
            "position_pnl_pct": 0,
            "recent_ticks": [332.0, 334.0, 336.0, 338.0, 340.0],
        }
        rendered = render_decision_prompt(data)
        assert "332.00" in rendered
        assert "340.00" in rendered
