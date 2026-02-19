"""Integration tests: DCOvershootStrategy + TrailingRiskManager working together.

Tests the full lifecycle:
  entry signal → on_trade_executed → trailing SL/TP → exit signal → re-entry

These tests verify the wiring between strategy and risk manager,
not individual component logic (covered in other test files).
"""

import time
import pytest

from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from interfaces.strategy import MarketData, SignalType


def make_config(**overrides):
    cfg = {
        "symbol": "BTC",
        "dc_thresholds": [[0.001, 0.001]],
        "position_size_usd": 100.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.003,   # 0.3% → SL at 99700 for 100k entry
        "initial_take_profit_pct": 0.002,  # 0.2% → TP at 100200 for 100k entry
        "trail_pct": 0.5,
        "cooldown_seconds": 0,
        "max_open_positions": 1,
        "log_events": False,
    }
    cfg.update(overrides)
    return cfg


def md(price, ts):
    return MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=ts)


def trigger_pdcc_down(strategy, start_price=100_000.0, start_ts=1000.0):
    """Drive price down to trigger PDCC_Down, return (sell_signal, last_price, last_ts)."""
    price = start_price
    ts = start_ts
    for i in range(30):
        price *= 0.99985  # ~0.015% per tick, cumulative >0.1% after ~7 ticks
        ts += 1.0
        signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        if sell_signals:
            return sell_signals[0], price, ts

    raise RuntimeError("Failed to trigger PDCC_Down")


def trigger_pdcc2_up(strategy, start_price=100_000.0, start_ts=1000.0):
    """Drive price down then up to trigger PDCC2_UP, return (buy_signal, last_price, last_ts)."""
    price = start_price
    ts = start_ts

    # Phase 1: Down
    for _ in range(15):
        price *= 0.99985
        ts += 1.0
        strategy.generate_signals(md(price, ts), [], 100_000.0)

    # Phase 2: Reverse up
    for i in range(30):
        price *= 1.00015
        ts += 1.0
        signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        if buy_signals:
            return buy_signals[0], price, ts

    raise RuntimeError("Failed to trigger PDCC2_UP")


class TestOnTradeExecutedInitializesTrailing:
    """on_trade_executed should initialize the trailing risk manager."""

    def test_sell_fill_opens_short_position(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signal, price, _ = trigger_pdcc_down(strategy)

        # Before execution, no position in trailing RM
        assert strategy._trailing_rm.has_position is False

        strategy.on_trade_executed(signal, price, signal.size)
        assert strategy._trailing_rm.has_position is True
        assert strategy._trailing_rm.side == "SHORT"

    def test_buy_fill_opens_long_position(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signal, price, _ = trigger_pdcc2_up(strategy)

        strategy.on_trade_executed(signal, price, signal.size)
        assert strategy._trailing_rm.has_position is True
        assert strategy._trailing_rm.side == "LONG"


class TestTrailingExitsOnTick:
    """generate_signals should emit CLOSE when trailing SL/TP hits."""

    def test_short_stop_loss_on_price_rise(self):
        """SHORT entry → price rises → SL hit → CLOSE signal."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signal, entry_price, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)

        # SL for short at entry * (1 + 0.003) = entry * 1.003
        sl_price = entry_price * 1.003

        # Jump price above SL
        ts += 1.0
        signals = strategy.generate_signals(md(sl_price + 1.0, ts), [], 100_000.0)
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(close_signals) == 1
        assert "stop_loss" in close_signals[0].reason

    def test_short_take_profit_on_price_drop(self):
        """SHORT entry → price drops → TP hit → CLOSE signal."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signal, entry_price, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)

        # TP for short at entry * (1 - 0.002) = entry * 0.998
        tp_price = entry_price * 0.998

        ts += 1.0
        signals = strategy.generate_signals(md(tp_price - 1.0, ts), [], 100_000.0)
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(close_signals) == 1
        assert "take_profit" in close_signals[0].reason

    def test_long_stop_loss_on_price_drop(self):
        """LONG entry → price drops → SL hit → CLOSE signal."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signal, entry_price, ts = trigger_pdcc2_up(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)

        # SL for long at entry * (1 - 0.003)
        sl_price = entry_price * 0.997
        ts += 1.0
        signals = strategy.generate_signals(md(sl_price - 1.0, ts), [], 100_000.0)
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(close_signals) == 1
        assert "stop_loss" in close_signals[0].reason


class TestTrailingRatchetAndExit:
    """Test ratcheting SL and then triggering on pullback."""

    def test_short_ratchet_then_exit(self):
        """SHORT: price drops (profit) → SL ratchets down → bounce → exit at ratcheted level."""
        # Set min_profit_to_trail to 0 for this test to ensure ratcheting activates
        strategy = DCOvershootStrategy(make_config(min_profit_to_trail_pct=0.0))
        strategy.start()
        signal, entry_price, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)

        initial_sl = strategy._trailing_rm.current_sl_price

        # Price drops significantly (profit for short), but not to TP
        profit_price = entry_price * 0.9985  # 0.15% drop, above TP but well into profit
        ts += 1.0
        strategy.generate_signals(md(profit_price, ts), [], 100_000.0)

        # SL should have ratcheted down
        ratcheted_sl = strategy._trailing_rm.current_sl_price
        assert ratcheted_sl < initial_sl

        # Price bounces to ratcheted SL → exit
        ts += 1.0
        signals = strategy.generate_signals(md(ratcheted_sl, ts), [], 100_000.0)
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(close_signals) == 1


class TestReentryAfterExit:
    """After position closes, a new PDCC should trigger a fresh entry."""

    def test_reentry_after_close(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # First entry
        signal1, price1, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal1, price1, signal1.size)

        # Force exit by hitting SL
        sl_price = price1 * 1.004  # Well above SL
        ts += 1.0
        exit_signals = strategy.generate_signals(md(sl_price, ts), [], 100_000.0)
        assert any(s.signal_type == SignalType.CLOSE for s in exit_signals)

        # Position should be cleared
        assert strategy._trailing_rm.has_position is False

        # Continue feeding price drops for a new PDCC_Down
        price = sl_price
        new_entry = None
        for _ in range(40):
            price *= 0.99985
            ts += 1.0
            signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
            sells = [s for s in signals if s.signal_type == SignalType.SELL]
            if sells:
                new_entry = sells[0]
                break

        assert new_entry is not None, "Should get a new SELL after position closed"


class TestFullCycle:
    """Entry → trailing → exit → re-entry complete cycle."""

    def test_entry_exit_reentry_cycle(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Entry 1: SHORT
        signal1, price1, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal1, price1, signal1.size)
        assert strategy._trailing_rm.side == "SHORT"

        # TP hit: price drops past TP
        tp_price = price1 * 0.997
        ts += 1.0
        exit_sigs = strategy.generate_signals(md(tp_price, ts), [], 100_000.0)
        assert any(s.signal_type == SignalType.CLOSE for s in exit_sigs)

        # Status should show trade count
        status = strategy.get_status()
        assert status["trade_count"] >= 1
        assert status["trailing_rm"]["has_position"] is False


class TestReversalOnOpposingPDCC:
    """Opposing PDCC while position open → CLOSE + reverse entry."""

    def test_short_reversed_to_long_on_pdcc2_up(self):
        """SHORT open → PDCC2_UP → CLOSE short + BUY (new LONG)."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Open SHORT
        signal, entry_price, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)
        assert strategy._trailing_rm.side == "SHORT"

        # Drive price up to trigger PDCC2_UP (opposing event)
        price = entry_price
        all_signals = []
        for _ in range(30):
            price *= 1.00015
            ts += 1.0
            sigs = strategy.generate_signals(md(price, ts), [], 100_000.0)
            all_signals.extend(sigs)

        # Should have reversal CLOSE + BUY
        close_sigs = [s for s in all_signals if s.signal_type == SignalType.CLOSE
                      and "reversal" in s.reason]
        buy_sigs = [s for s in all_signals if s.signal_type == SignalType.BUY]
        assert len(close_sigs) >= 1, "Reversal CLOSE expected"
        assert len(buy_sigs) >= 1, "BUY entry expected after reversal"

    def test_long_reversed_to_short_on_pdcc_down(self):
        """LONG open → PDCC_Down → CLOSE long + SELL (new SHORT)."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Open LONG
        signal, entry_price, ts = trigger_pdcc2_up(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)
        assert strategy._trailing_rm.side == "LONG"

        # Drive price down to trigger PDCC_Down (opposing event)
        price = entry_price
        all_signals = []
        for _ in range(30):
            price *= 0.99985
            ts += 1.0
            sigs = strategy.generate_signals(md(price, ts), [], 100_000.0)
            all_signals.extend(sigs)

        # Should have reversal CLOSE + SELL
        close_sigs = [s for s in all_signals if s.signal_type == SignalType.CLOSE
                      and "reversal" in s.reason]
        sell_sigs = [s for s in all_signals if s.signal_type == SignalType.SELL]
        assert len(close_sigs) >= 1, "Reversal CLOSE expected"
        assert len(sell_sigs) >= 1, "SELL entry expected after reversal"

    def test_reversal_then_trailing_on_new_position(self):
        """After reversal, trailing RM should track the new position correctly."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Open SHORT
        signal, entry_price, ts = trigger_pdcc_down(strategy)
        strategy.on_trade_executed(signal, entry_price, signal.size)

        # Reverse to LONG via PDCC2_UP
        price = entry_price
        buy_signal = None
        for _ in range(30):
            price *= 1.00015
            ts += 1.0
            sigs = strategy.generate_signals(md(price, ts), [], 100_000.0)
            buys = [s for s in sigs if s.signal_type == SignalType.BUY]
            if buys:
                buy_signal = buys[0]
                break

        assert buy_signal is not None, "Should get BUY signal on reversal"

        # Execute the BUY → trailing RM should now be LONG
        strategy.on_trade_executed(buy_signal, price, buy_signal.size)
        assert strategy._trailing_rm.side == "LONG"

        # Now the LONG's SL should be below current price
        sl = strategy._trailing_rm.current_sl_price
        assert sl < price, "LONG SL should be below entry"

        # Hit the LONG's SL → should get CLOSE
        ts += 1.0
        sigs = strategy.generate_signals(md(sl - 1.0, ts), [], 100_000.0)
        close_sigs = [s for s in sigs if s.signal_type == SignalType.CLOSE]
        assert len(close_sigs) == 1
        assert "stop_loss" in close_sigs[0].reason
