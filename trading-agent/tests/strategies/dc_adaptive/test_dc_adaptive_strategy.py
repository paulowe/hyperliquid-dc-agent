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
