"""Tests for DCAdaptiveStrategy — DC overshoot with adaptive guards."""

import time

import pytest

from interfaces.strategy import MarketData, SignalType
from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy


def make_config(**overrides):
    """Build a strategy config dict with sensible test defaults."""
    cfg = {
        "symbol": "BTC",
        "dc_thresholds": [[0.001, 0.001]],  # 0.1% trade threshold
        "sensor_threshold": [0.0005, 0.0005],  # 0.05% sensor
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.003,
        "initial_take_profit_pct": 0.002,
        "trail_pct": 0.5,
        "min_profit_to_trail_pct": 0.001,
        "cooldown_seconds": 0,
        "max_open_positions": 1,
        "log_events": False,
        # Regime detector
        "lookback_seconds": 600,
        "choppy_rate_threshold": 4.0,
        "trending_consistency_threshold": 0.6,
        # Overshoot tracker
        "os_window_size": 20,
        "os_min_samples": 5,
        "tp_fraction": 0.8,
        "min_tp_pct": 0.003,
        "default_tp_pct": 0.002,
        # Loss streak guard
        "max_consecutive_losses": 3,
        "base_cooldown_seconds": 300,
    }
    cfg.update(overrides)
    return cfg


def md(price, ts=None):
    return MarketData(
        asset="BTC", price=price, volume_24h=0.0,
        timestamp=ts or time.time(),
    )


def drive_price_down(strategy, start_price=100_000.0, steps=20):
    """Feed ticks that decrease to trigger PDCC_Down."""
    all_signals = []
    drop_per_step = start_price * 0.0002  # 0.02% per step
    price = start_price
    ts = 1000.0
    for i in range(steps):
        price -= drop_per_step
        signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
        all_signals.extend(signals)
        ts += 10.0
    return all_signals, price, ts


def drive_price_up(strategy, start_price=100_000.0, steps=20):
    """Feed ticks that increase to trigger PDCC2_UP."""
    all_signals = []
    rise_per_step = start_price * 0.0002
    price = start_price
    ts = 1000.0
    # First drive down to establish downtrend
    for i in range(steps):
        price -= rise_per_step
        strategy.generate_signals(md(price, ts), [], 100_000.0)
        ts += 10.0
    # Then drive up to trigger PDCC2_UP
    for i in range(steps * 2):
        price += rise_per_step
        signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
        all_signals.extend(signals)
        ts += 10.0
    return all_signals, price, ts


class TestDCAdaptiveInit:
    """Verify strategy initializes correctly."""

    def test_creates_with_default_config(self):
        strategy = DCAdaptiveStrategy(make_config())
        strategy.start()
        assert strategy.is_active is True
        assert strategy.name == "dc_adaptive"

    def test_has_regime_detector(self):
        strategy = DCAdaptiveStrategy(make_config())
        status = strategy.get_status()
        assert "regime" in status

    def test_has_loss_streak_guard(self):
        strategy = DCAdaptiveStrategy(make_config())
        status = strategy.get_status()
        assert "loss_streak" in status


class TestEntrySignals:
    """Verify entry signals are generated from DC events."""

    def test_downtrend_triggers_sell(self):
        strategy = DCAdaptiveStrategy(make_config())
        strategy.start()
        signals, _, _ = drive_price_down(strategy)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) >= 1

    def test_uptrend_triggers_buy(self):
        strategy = DCAdaptiveStrategy(make_config())
        strategy.start()
        signals, _, _ = drive_price_up(strategy)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) >= 1


class TestRegimeGating:
    """Regime detector blocks entries in choppy markets."""

    def test_choppy_regime_blocks_entry(self):
        """When regime is choppy, no entry signals should be emitted."""
        strategy = DCAdaptiveStrategy(make_config(
            sensor_threshold=[0.0003, 0.0003],
            choppy_rate_threshold=2.0,
            trending_consistency_threshold=0.8,
        ))
        strategy.start()

        ts = 0.0
        price = 100_000.0
        # Create choppy regime: rapid alternating sensor events
        # by feeding alternating small up/down moves
        for cycle in range(40):
            # Down move
            for i in range(5):
                price -= 10
                strategy.generate_signals(md(price, ts), [], 100_000.0)
                ts += 1.0
            # Up move
            for i in range(5):
                price += 10
                strategy.generate_signals(md(price, ts), [], 100_000.0)
                ts += 1.0

        # Now try to trigger a real entry with a big drop
        # The regime detector should block it
        signals = []
        for i in range(30):
            price -= 50
            sigs = strategy.generate_signals(md(price, ts), [], 100_000.0)
            signals.extend(sigs)
            ts += 1.0

        entry_signals = [s for s in signals
                         if s.signal_type in (SignalType.BUY, SignalType.SELL)]
        # In a choppy regime, entries should be filtered
        status = strategy.get_status()
        if status["regime"] == "choppy":
            assert len(entry_signals) == 0


class TestLossStreakGating:
    """Loss streak guard blocks entries after consecutive losses."""

    def test_guard_blocks_after_losses(self):
        """After recording max_consecutive_losses, new entries are blocked."""
        strategy = DCAdaptiveStrategy(make_config(
            max_consecutive_losses=2,
            base_cooldown_seconds=600,
        ))
        strategy.start()

        # Simulate 2 losing trades via the guard directly
        strategy._loss_guard.record_trade(is_win=False, timestamp=100.0)
        strategy._loss_guard.record_trade(is_win=False, timestamp=200.0)

        # Now try to generate signals — should be blocked
        assert strategy._loss_guard.should_trade(300.0) is False


class TestAdaptiveTP:
    """Overshoot tracker adapts TP dynamically."""

    def test_tp_starts_at_default(self):
        strategy = DCAdaptiveStrategy(make_config(default_tp_pct=0.005))
        strategy.start()
        assert strategy._os_tracker.adaptive_tp() == 0.005

    def test_tp_adapts_after_enough_overshoots(self):
        strategy = DCAdaptiveStrategy(make_config(
            os_min_samples=3,
            tp_fraction=1.0,
            min_tp_pct=0.0,
            default_tp_pct=0.005,
        ))
        strategy.start()
        # Record some overshoots
        strategy._os_tracker.record_overshoot(0.01)
        strategy._os_tracker.record_overshoot(0.02)
        strategy._os_tracker.record_overshoot(0.03)
        # Median = 0.02
        assert strategy._os_tracker.adaptive_tp() == pytest.approx(0.02)


class TestOnTradeExecuted:
    """Verify on_trade_executed initializes trailing RM."""

    def test_trade_executed_sets_position(self):
        strategy = DCAdaptiveStrategy(make_config())
        strategy.start()
        signals, price, ts = drive_price_down(strategy)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        if sells:
            strategy.on_trade_executed(sells[0], price, sells[0].size)
            assert strategy._trailing_rm.has_position is True


class TestDirectionFilter:
    """Direction filter blocks entries in one direction and converts reversals to closes."""

    def test_long_only_blocks_sell_entries(self):
        """In long-only mode, PDCC_Down should NOT produce SELL entry signals."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="long"))
        strategy.start()
        signals, _, _ = drive_price_down(strategy)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) == 0

    def test_long_only_allows_buy_entries(self):
        """In long-only mode, PDCC2_UP should still produce BUY entry signals."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="long"))
        strategy.start()
        signals, _, _ = drive_price_up(strategy)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) >= 1

    def test_short_only_blocks_buy_entries(self):
        """In short-only mode, PDCC2_UP should NOT produce BUY entry signals."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="short"))
        strategy.start()
        signals, _, _ = drive_price_up(strategy)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) == 0

    def test_short_only_allows_sell_entries(self):
        """In short-only mode, PDCC_Down should still produce SELL entry signals."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="short"))
        strategy.start()
        signals, _, _ = drive_price_down(strategy)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) >= 1

    def test_long_only_closes_on_reversal(self):
        """In long-only mode, PDCC_Down while LONG should emit CLOSE (not SELL)."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="long"))
        strategy.start()

        # First get a BUY entry
        signals, price, ts = drive_price_up(strategy)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) >= 1, "Need at least one BUY to test reversal close"

        # Simulate fill so strategy tracks position
        buy_sig = buys[0]
        strategy.on_trade_executed(buy_sig, price, buy_sig.size)
        assert strategy._trailing_rm.has_position

        # Now drive price down to trigger PDCC_Down — should close, not reverse
        all_signals = []
        for i in range(30):
            price -= price * 0.0002
            ts += 10.0
            sigs = strategy.generate_signals(md(price, ts), [], 100_000.0)
            all_signals.extend(sigs)

        closes = [s for s in all_signals if s.signal_type == SignalType.CLOSE]
        sells = [s for s in all_signals if s.signal_type == SignalType.SELL]
        # Should have a CLOSE from direction filter or from SL/TP
        # Should NOT have new SELL entries
        assert len(sells) == 0, "Long-only should not produce SELL entries on reversal"
        # Position should be closed (either by direction filter close or SL/TP)
        # Check that at least one close was emitted
        assert len(closes) >= 1, "Should have closed position on reversal DC event or SL"

    def test_long_only_reversal_close_has_correct_metadata(self):
        """CLOSE from direction filter should include side, pnl_pct, and entry_price."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="long"))
        strategy.start()

        # Get a BUY entry
        signals, price, ts = drive_price_up(strategy)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        if not buys:
            pytest.skip("No BUY signal generated")

        strategy.on_trade_executed(buys[0], price, buys[0].size)

        # Drive price down to trigger reversal close
        all_signals = []
        for i in range(40):
            price -= price * 0.0002
            ts += 10.0
            sigs = strategy.generate_signals(md(price, ts), [], 100_000.0)
            all_signals.extend(sigs)

        dc_closes = [s for s in all_signals
                     if s.signal_type == SignalType.CLOSE and s.reason == "dc_reversal_close"]
        if dc_closes:
            close = dc_closes[0]
            assert "side" in close.metadata
            assert close.metadata["side"] == "LONG"
            assert "pnl_pct" in close.metadata
            assert "entry_price" in close.metadata

    def test_both_direction_allows_all(self):
        """Default direction_filter='both' allows both BUY and SELL."""
        strategy = DCAdaptiveStrategy(make_config(direction_filter="both"))
        strategy.start()

        # Down
        signals_down, _, _ = drive_price_down(strategy)
        sells = [s for s in signals_down if s.signal_type == SignalType.SELL]
        assert len(sells) >= 1

        # Fresh strategy for up
        strategy2 = DCAdaptiveStrategy(make_config(direction_filter="both"))
        strategy2.start()
        signals_up, _, _ = drive_price_up(strategy2)
        buys = [s for s in signals_up if s.signal_type == SignalType.BUY]
        assert len(buys) >= 1


class TestConfigDirectionFilter:
    """Config validation for direction_filter."""

    def test_default_is_both(self):
        strategy = DCAdaptiveStrategy(make_config())
        assert strategy._cfg.direction_filter == "both"

    def test_long_filter_accepted(self):
        strategy = DCAdaptiveStrategy(make_config(direction_filter="long"))
        assert strategy._cfg.direction_filter == "long"

    def test_short_filter_accepted(self):
        strategy = DCAdaptiveStrategy(make_config(direction_filter="short"))
        assert strategy._cfg.direction_filter == "short"

    def test_invalid_filter_rejected(self):
        with pytest.raises(ValueError, match="direction_filter"):
            DCAdaptiveStrategy(make_config(direction_filter="invalid"))


class TestStrategyLifecycle:
    """Start/stop lifecycle."""

    def test_start_sets_active(self):
        strategy = DCAdaptiveStrategy(make_config())
        strategy.start()
        assert strategy.is_active is True

    def test_stop_clears_active(self):
        strategy = DCAdaptiveStrategy(make_config())
        strategy.start()
        strategy.stop()
        assert strategy.is_active is False
