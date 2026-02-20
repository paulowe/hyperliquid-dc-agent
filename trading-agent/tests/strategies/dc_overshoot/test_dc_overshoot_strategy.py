"""Tests for DCOvershootStrategy — entry signal generation.

Tests verify:
- PDCC_Down → SELL (short entry)
- PDCC2_UP → BUY (long entry)
- OSV events → no entry
- Cooldown enforcement
- Position-aware gating (no new entry when position open)
- start/stop/get_status
"""

import time
import pytest

from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from interfaces.strategy import MarketData, Position, SignalType


def make_config(**overrides):
    """Build a strategy config dict with sensible test defaults."""
    cfg = {
        "symbol": "BTC",
        "dc_thresholds": [[0.001, 0.001]],  # 0.1%
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.003,
        "initial_take_profit_pct": 0.002,
        "trail_pct": 0.5,
        "cooldown_seconds": 0,  # No cooldown for most tests
        "max_open_positions": 1,
        "log_events": False,
    }
    cfg.update(overrides)
    return cfg


def make_market_data(price, ts=None):
    return MarketData(
        asset="BTC", price=price, volume_24h=0.0,
        timestamp=ts or time.time(),
    )


def drive_price_down(strategy, start_price=100_000.0, steps=20):
    """Feed ticks that monotonically decrease by >0.1% to trigger PDCC_Down.

    Returns (all_signals, last_price, last_ts).
    """
    price = start_price
    ts = 1000.0
    all_signals = []
    for i in range(steps):
        # Drop 0.015% per tick → after ~7 ticks, cumulative drop exceeds 0.1%
        price *= 0.99985
        ts += 1.0
        md = make_market_data(price, ts)
        signals = strategy.generate_signals(md, [], 100_000.0)
        all_signals.extend(signals)
    return all_signals, price, ts


def drive_price_up(strategy, start_price=100_000.0, steps=20):
    """Feed ticks that monotonically increase by >0.1% to trigger PDCC2_UP.

    Must first establish a downtrend, then reverse.
    Returns (all_signals, last_price, last_ts).
    """
    price = start_price
    ts = 1000.0
    all_signals = []

    # Phase 1: Drive price down to establish downtrend
    for _ in range(15):
        price *= 0.99985
        ts += 1.0
        md = make_market_data(price, ts)
        signals = strategy.generate_signals(md, [], 100_000.0)
        all_signals.extend(signals)

    # Phase 2: Reverse upward to trigger PDCC2_UP
    for _ in range(steps):
        price *= 1.00015
        ts += 1.0
        md = make_market_data(price, ts)
        signals = strategy.generate_signals(md, [], 100_000.0)
        all_signals.extend(signals)

    return all_signals, price, ts


class TestEntrySignalPDCCDown:
    """PDCC_Down events should trigger SELL (short entry)."""

    def test_monotonic_drop_triggers_sell(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signals, _, _ = drive_price_down(strategy, start_price=100_000.0, steps=20)

        # Should have at least one SELL signal from PDCC_Down
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1, f"Expected SELL, got signals: {signals}"
        assert sell_signals[0].asset == "BTC"
        assert sell_signals[0].size > 0


class TestEntrySignalPDCC2UP:
    """PDCC2_UP events should trigger BUY (long entry)."""

    def test_reversal_triggers_buy(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        signals, _, _ = drive_price_up(strategy, start_price=100_000.0, steps=20)

        # Should have BUY signal from PDCC2_UP (and possibly SELL from PDCC_Down first)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1, f"Expected BUY, got signals: {signals}"
        assert buy_signals[0].asset == "BTC"


class TestNoSignalOnOSV:
    """OSV events should NOT trigger entry signals."""

    def test_flat_price_no_signal(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        # Flat price → no DC events at all
        for i in range(50):
            md = make_market_data(100_000.0, 1000.0 + i)
            signals = strategy.generate_signals(md, [], 100_000.0)
            assert signals == [], f"Unexpected signal on flat price: {signals}"


class TestCooldownEnforcement:
    """Second PDCC within cooldown should not trigger a new entry."""

    def test_cooldown_blocks_second_entry(self):
        config = make_config(cooldown_seconds=60.0)  # 60s cooldown
        strategy = DCOvershootStrategy(config)
        strategy.start()

        # Drive first PDCC_Down → SELL
        signals1, price, ts = drive_price_down(strategy, steps=20)
        sell_signals = [s for s in signals1 if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1

        # Simulate trade execution so trailing RM has a position
        strategy.on_trade_executed(sell_signals[0], price, sell_signals[0].size)

        # Close the position so strategy can accept new entries
        strategy._trailing_rm.close_position()

        # Try to trigger another PDCC within cooldown window
        # Continue dropping from current price
        new_signals = []
        for i in range(20):
            price *= 0.99985
            ts += 1.0
            md = make_market_data(price, ts)
            sigs = strategy.generate_signals(md, [], 100_000.0)
            new_signals.extend(sigs)

        # No new entry signals because cooldown hasn't expired
        sell_signals2 = [s for s in new_signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals2) == 0, "Should be blocked by cooldown"


class TestPositionGating:
    """Same-direction PDCC is blocked; opposing PDCC triggers reversal."""

    def test_same_direction_blocked_short(self):
        """SHORT open + continued drops (same-direction PDCC_Down) → no new entry."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Trigger PDCC_Down → SELL
        signals1, price, ts = drive_price_down(strategy, steps=20)
        sell_signals = [s for s in signals1 if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1

        # Execute the trade → SHORT position open
        strategy.on_trade_executed(sell_signals[0], price, sell_signals[0].size)

        # Continue feeding drops — same direction, should not get new SELL
        entry_signals = []
        for i in range(30):
            price *= 0.99985
            ts += 1.0
            md = make_market_data(price, ts)
            sigs = strategy.generate_signals(md, [], 100_000.0)
            entries = [s for s in sigs if s.signal_type == SignalType.SELL]
            entry_signals.extend(entries)

        assert len(entry_signals) == 0, "Same-direction PDCC should be blocked"

    def test_opposing_pdcc_triggers_reversal_short_to_long(self):
        """SHORT open + PDCC2_UP → single BUY with 2x size (atomic flip)."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Step 1: Drive price down → PDCC_Down → SELL
        signals1, price, ts = drive_price_down(strategy, steps=20)
        sell_signals = [s for s in signals1 if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1
        old_size = sell_signals[0].size
        strategy.on_trade_executed(sell_signals[0], price, old_size)
        assert strategy._trailing_rm.side == "SHORT"

        # Step 2: Reverse price upward to trigger PDCC2_UP
        all_signals = []
        for _ in range(30):
            price *= 1.00015
            ts += 1.0
            md = make_market_data(price, ts)
            sigs = strategy.generate_signals(md, [], 100_000.0)
            all_signals.extend(sigs)

        # Should have single BUY (no separate CLOSE)
        close_signals = [s for s in all_signals if s.signal_type == SignalType.CLOSE]
        buy_signals = [s for s in all_signals if s.signal_type == SignalType.BUY]
        assert len(close_signals) == 0, "Reversal should NOT emit separate CLOSE"
        assert len(buy_signals) >= 1, "Should BUY to atomically flip"

        # BUY size = old position + new entry (2x for atomic flip)
        buy = buy_signals[0]
        new_entry_size = 50.0 / price  # position_size_usd / current price
        assert buy.size == pytest.approx(old_size + new_entry_size, rel=0.05)

    def test_opposing_pdcc_triggers_reversal_long_to_short(self):
        """LONG open + PDCC_Down → single SELL with 2x size (atomic flip)."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Step 1: Drive price down then up → PDCC2_UP → BUY
        signals1, price, ts = drive_price_up(strategy, steps=20)
        buy_signals = [s for s in signals1 if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1
        old_size = buy_signals[0].size
        strategy.on_trade_executed(buy_signals[0], price, old_size)
        assert strategy._trailing_rm.side == "LONG"

        # Step 2: Drive price down to trigger PDCC_Down
        all_signals = []
        for _ in range(30):
            price *= 0.99985
            ts += 1.0
            md = make_market_data(price, ts)
            sigs = strategy.generate_signals(md, [], 100_000.0)
            all_signals.extend(sigs)

        # Should have single SELL (no separate CLOSE)
        close_signals = [s for s in all_signals if s.signal_type == SignalType.CLOSE]
        sell_signals = [s for s in all_signals if s.signal_type == SignalType.SELL]
        assert len(close_signals) == 0, "Reversal should NOT emit separate CLOSE"
        assert len(sell_signals) >= 1, "Should SELL to atomically flip"

        # SELL size = old position + new entry
        sell = sell_signals[0]
        new_entry_size = 50.0 / price
        assert sell.size == pytest.approx(old_size + new_entry_size, rel=0.05)

    def test_reversal_signal_has_metadata(self):
        """Reversal BUY/SELL signal should have reversal metadata with new_position_size."""
        strategy = DCOvershootStrategy(make_config())
        strategy.start()

        # Open SHORT
        signals1, price, ts = drive_price_down(strategy, steps=20)
        sell_signals = [s for s in signals1 if s.signal_type == SignalType.SELL]
        old_size = sell_signals[0].size
        strategy.on_trade_executed(sell_signals[0], price, old_size)

        # Reverse to trigger PDCC2_UP
        all_signals = []
        for _ in range(30):
            price *= 1.00015
            ts += 1.0
            md = make_market_data(price, ts)
            sigs = strategy.generate_signals(md, [], 100_000.0)
            all_signals.extend(sigs)

        buy_signals = [s for s in all_signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1
        buy = buy_signals[0]
        # Reversal signal should have metadata
        assert buy.metadata.get("reversal") is True
        assert buy.metadata.get("previous_side") == "SHORT"
        # new_position_size should be the actual position (not the 2x order size)
        new_entry_size = 50.0 / price
        assert buy.metadata.get("new_position_size") == pytest.approx(new_entry_size, rel=0.05)


class TestPositionSizing:
    """Verify position size is computed from config."""

    def test_sell_signal_size_from_price(self):
        config = make_config(position_size_usd=100.0)
        strategy = DCOvershootStrategy(config)
        strategy.start()
        signals, price, _ = drive_price_down(strategy, start_price=100_000.0, steps=20)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1

        # Size should be approximately position_size_usd / price
        expected_size = 100.0 / price
        assert sell_signals[0].size == pytest.approx(expected_size, rel=0.05)


class TestStrategyLifecycle:
    """Test start/stop/get_status."""

    def test_start_sets_active(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        assert strategy.is_active is True

    def test_stop_sets_inactive(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        strategy.stop()
        assert strategy.is_active is False

    def test_get_status_reports_config(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        status = strategy.get_status()
        assert status["name"] == "dc_overshoot"
        assert status["symbol"] == "BTC"
        assert "tick_count" in status
        assert "dc_event_count" in status
        assert "trailing_rm" in status

    def test_get_status_tick_count_increments(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        for i in range(5):
            md = make_market_data(100_000.0, 1000.0 + i)
            strategy.generate_signals(md, [], 100_000.0)
        status = strategy.get_status()
        assert status["tick_count"] == 5
