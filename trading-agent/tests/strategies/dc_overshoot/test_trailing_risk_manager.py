"""Tests for TrailingRiskManager — the core trailing SL/TP logic.

Tests cover:
- Position initialization (LONG and SHORT)
- Trailing SL ratcheting when profitable
- Static SL/TP when in a loss
- TP pushing further on new highs/lows
- Stop loss and take profit exit signals
- State management (close, has_position, get_status)
"""

import pytest

from strategies.dc_overshoot.trailing_risk_manager import TrailingRiskManager
from interfaces.strategy import SignalType


# Helper constants
ENTRY_PRICE = 100_000.0  # BTC-like price
SL_PCT = 0.003  # 0.3%
TP_PCT = 0.002  # 0.2%
TRAIL_PCT = 0.5  # Lock in 50% of profit


def make_manager(**kwargs):
    defaults = {
        "asset": "BTC",
        "initial_stop_loss_pct": SL_PCT,
        "initial_take_profit_pct": TP_PCT,
        "trail_pct": TRAIL_PCT,
    }
    defaults.update(kwargs)
    return TrailingRiskManager(**defaults)


class TestOpenPositionLong:
    """Verify LONG position initialization sets correct levels."""

    def test_open_long_sets_initial_sl(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        expected_sl = ENTRY_PRICE * (1 - SL_PCT)  # 99700
        assert mgr.current_sl_price == pytest.approx(expected_sl)

    def test_open_long_sets_initial_tp(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        expected_tp = ENTRY_PRICE * (1 + TP_PCT)  # 100200
        assert mgr.current_tp_price == pytest.approx(expected_tp)

    def test_open_long_sets_high_water_mark(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        assert mgr.high_water_mark == ENTRY_PRICE

    def test_open_long_has_position(self):
        mgr = make_manager()
        assert mgr.has_position is False
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        assert mgr.has_position is True

    def test_open_long_stores_entry_details(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        assert mgr.side == "LONG"
        assert mgr.entry_price == ENTRY_PRICE
        assert mgr.size == 0.001


class TestOpenPositionShort:
    """Verify SHORT position initialization sets correct levels."""

    def test_open_short_sets_initial_sl(self):
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        # Short SL is above entry
        expected_sl = ENTRY_PRICE * (1 + SL_PCT)  # 100300
        assert mgr.current_sl_price == pytest.approx(expected_sl)

    def test_open_short_sets_initial_tp(self):
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        # Short TP is below entry
        expected_tp = ENTRY_PRICE * (1 - TP_PCT)  # 99800
        assert mgr.current_tp_price == pytest.approx(expected_tp)

    def test_open_short_sets_low_water_mark(self):
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        assert mgr.low_water_mark == ENTRY_PRICE

    def test_open_position_when_already_open_raises(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        with pytest.raises(RuntimeError, match="already open"):
            mgr.open_position("SHORT", ENTRY_PRICE, 0.001)


class TestLongTrailingNoExit:
    """LONG position: price moves but doesn't hit SL or TP."""

    def test_flat_price_no_exit(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        signals = mgr.update(ENTRY_PRICE, 1.0)
        assert signals == []

    def test_small_dip_no_exit(self):
        """Price dips slightly but stays above SL."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        # SL at 99700, price at 99800 → no exit
        signals = mgr.update(99_800.0, 1.0)
        assert signals == []

    def test_small_rise_no_exit(self):
        """Price rises slightly but stays below TP."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        # TP at 100200, price at 100100 → no exit
        signals = mgr.update(100_100.0, 1.0)
        assert signals == []


class TestLongStopLoss:
    """LONG position: price drops to hit stop loss."""

    def test_price_hits_initial_sl(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        # SL at 99700, price drops to 99700
        signals = mgr.update(99_700.0, 1.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE
        assert signals[0].asset == "BTC"
        assert "stop_loss" in signals[0].reason

    def test_price_gaps_below_sl(self):
        """Price gaps well below SL — should still trigger."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        signals = mgr.update(99_000.0, 1.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE


class TestLongTakeProfit:
    """LONG position: price rises to hit take profit."""

    def test_price_hits_initial_tp(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        # TP at 100200, price rises to 100200
        signals = mgr.update(100_200.0, 1.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE
        assert "take_profit" in signals[0].reason

    def test_price_gaps_above_tp(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        signals = mgr.update(101_000.0, 1.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE


class TestLongTrailingRatchet:
    """LONG position: SL ratchets up when price makes new highs."""

    def test_new_high_updates_water_mark(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        # Price rises to 100100 (below TP of 100200), update water mark
        mgr.update(100_100.0, 1.0)
        assert mgr.high_water_mark == 100_100.0

    def test_new_high_ratchets_sl_up(self):
        """SL should increase when price moves into profit."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        initial_sl = mgr.current_sl_price  # 99700

        # Price rises to 100100 (profit = 100)
        mgr.update(100_100.0, 1.0)

        # new_sl = entry + (hwm - entry) * trail_pct = 100000 + 100 * 0.5 = 100050
        expected_new_sl = ENTRY_PRICE + (100_100.0 - ENTRY_PRICE) * TRAIL_PCT
        assert mgr.current_sl_price == pytest.approx(expected_new_sl)
        assert mgr.current_sl_price > initial_sl

    def test_sl_never_ratchets_down(self):
        """SL should never decrease, even if price drops after ratchet."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)

        # Ratchet up with price at 100150
        mgr.update(100_150.0, 1.0)
        ratcheted_sl = mgr.current_sl_price

        # Price drops back but stays above SL
        mgr.update(100_050.0, 2.0)
        assert mgr.current_sl_price == ratcheted_sl  # Unchanged

    def test_tp_pushes_higher_on_new_high(self):
        """TP should increase when price makes new highs."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        initial_tp = mgr.current_tp_price  # 100200

        # Price rises to 100150 (below initial TP)
        mgr.update(100_150.0, 1.0)

        # TP should be max(initial_tp, hwm * (1 + tp_pct))
        # = max(100200, 100150 * 1.002) = max(100200, 100350.3) = 100350.3
        expected_tp = 100_150.0 * (1 + TP_PCT)
        assert mgr.current_tp_price == pytest.approx(expected_tp)
        assert mgr.current_tp_price > initial_tp

    def test_multiple_ratchet_steps_create_higher_floor(self):
        """Progressive new highs should keep raising the SL."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)

        # Step 1: price = 100100
        mgr.update(100_100.0, 1.0)
        sl_1 = mgr.current_sl_price

        # Step 2: price = 100150
        mgr.update(100_150.0, 2.0)
        sl_2 = mgr.current_sl_price
        assert sl_2 > sl_1

        # Step 3: price = 100180
        mgr.update(100_180.0, 3.0)
        sl_3 = mgr.current_sl_price
        assert sl_3 > sl_2

    def test_ratcheted_sl_triggers_on_pullback(self):
        """After SL ratchets up, a pullback should trigger exit at higher level."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)

        # Ratchet: price = 100150, new SL = 100000 + 150*0.5 = 100075
        mgr.update(100_150.0, 1.0)
        ratcheted_sl = mgr.current_sl_price  # ~100075

        # Pullback to ratcheted SL level
        signals = mgr.update(ratcheted_sl, 2.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE
        assert "stop_loss" in signals[0].reason

    def test_in_loss_sl_stays_at_initial(self):
        """When price is below entry, SL should not move."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        initial_sl = mgr.current_sl_price  # 99700

        # Price dips to 99800 (in loss, above SL)
        mgr.update(99_800.0, 1.0)
        assert mgr.current_sl_price == initial_sl  # Unchanged


class TestShortStopLoss:
    """SHORT position: price rises to hit stop loss."""

    def test_price_hits_initial_sl(self):
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        # Short SL at 100300
        signals = mgr.update(100_300.0, 1.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE
        assert "stop_loss" in signals[0].reason


class TestShortTakeProfit:
    """SHORT position: price drops to hit take profit."""

    def test_price_hits_initial_tp(self):
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        # Short TP at 99800
        signals = mgr.update(99_800.0, 1.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE
        assert "take_profit" in signals[0].reason


class TestShortTrailingRatchet:
    """SHORT position: SL ratchets down when price makes new lows."""

    def test_new_low_updates_water_mark(self):
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        # Price drops to 99900 (profit for short, above TP of 99800)
        mgr.update(99_900.0, 1.0)
        assert mgr.low_water_mark == 99_900.0

    def test_new_low_ratchets_sl_down(self):
        """SL should decrease when short is profitable."""
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)
        initial_sl = mgr.current_sl_price  # 100300

        # Price drops to 99900 (profit = 100)
        mgr.update(99_900.0, 1.0)

        # new_sl = entry - (entry - lwm) * trail_pct = 100000 - 100*0.5 = 99950
        expected_new_sl = ENTRY_PRICE - (ENTRY_PRICE - 99_900.0) * TRAIL_PCT
        assert mgr.current_sl_price == pytest.approx(expected_new_sl)
        assert mgr.current_sl_price < initial_sl

    def test_short_sl_never_ratchets_up(self):
        """Short SL should never increase (only ratchet down)."""
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)

        # Ratchet down with price at 99850
        mgr.update(99_850.0, 1.0)
        ratcheted_sl = mgr.current_sl_price

        # Price bounces back up but stays below SL
        mgr.update(99_900.0, 2.0)
        assert mgr.current_sl_price == ratcheted_sl  # Unchanged

    def test_short_tp_pushes_lower(self):
        """Short TP should decrease on new lows."""
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)

        # Price drops to 99850
        mgr.update(99_850.0, 1.0)

        # TP = min(initial_tp, lwm * (1 - tp_pct)) = min(99800, 99850*0.998) = min(99800, 99650.3)
        expected_tp = 99_850.0 * (1 - TP_PCT)
        assert mgr.current_tp_price == pytest.approx(expected_tp)

    def test_short_ratcheted_sl_triggers_on_bounce(self):
        """After SL ratchets down for short, bounce should trigger exit."""
        mgr = make_manager()
        mgr.open_position("SHORT", ENTRY_PRICE, 0.001)

        # Ratchet: price = 99850, new SL = 100000 - 150*0.5 = 99925
        mgr.update(99_850.0, 1.0)
        ratcheted_sl = mgr.current_sl_price  # ~99925

        # Bounce to ratcheted SL level
        signals = mgr.update(ratcheted_sl, 2.0)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE


class TestStateManagement:
    """Test close_position, has_position, get_status."""

    def test_close_position_clears_state(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        assert mgr.has_position is True

        mgr.close_position()
        assert mgr.has_position is False

    def test_close_then_reopen(self):
        """After closing, should be able to open a new position."""
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        mgr.close_position()
        # Should not raise
        mgr.open_position("SHORT", 99_000.0, 0.002)
        assert mgr.side == "SHORT"
        assert mgr.entry_price == 99_000.0

    def test_update_without_position_returns_empty(self):
        """update() with no position should return empty list."""
        mgr = make_manager()
        signals = mgr.update(ENTRY_PRICE, 1.0)
        assert signals == []

    def test_get_status_with_position(self):
        mgr = make_manager()
        mgr.open_position("LONG", ENTRY_PRICE, 0.001)
        status = mgr.get_status()
        assert status["has_position"] is True
        assert status["side"] == "LONG"
        assert status["entry_price"] == ENTRY_PRICE
        assert "current_sl" in status
        assert "current_tp" in status
        assert "high_water_mark" in status

    def test_get_status_without_position(self):
        mgr = make_manager()
        status = mgr.get_status()
        assert status["has_position"] is False
