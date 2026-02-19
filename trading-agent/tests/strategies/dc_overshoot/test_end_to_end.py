"""End-to-end tests: replay synthetic price sequences through DCOvershootStrategy.

These tests construct realistic price sequences (not mainnet — no network required)
that exercise the full entry → trailing → exit pipeline.

Marked as NOT slow (pure computation, no network access).
"""

import pytest

from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from interfaces.strategy import MarketData, SignalType


def make_config(**overrides):
    cfg = {
        "symbol": "BTC",
        "dc_thresholds": [[0.001, 0.001]],  # 0.1%
        "position_size_usd": 100.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.003,
        "initial_take_profit_pct": 0.002,
        "trail_pct": 0.5,
        "cooldown_seconds": 0,
        "max_open_positions": 1,
        "log_events": False,
    }
    cfg.update(overrides)
    return cfg


def md(price, ts):
    return MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=ts)


def generate_v_shape(start=100_000.0, drop_pct=0.002, rise_pct=0.003, ticks_down=20, ticks_up=30):
    """Generate a V-shaped price sequence: drop then rise.

    Returns list of (price, timestamp) tuples.
    """
    ticks = []
    price = start
    ts = 1000.0
    drop_per_tick = drop_pct / ticks_down
    rise_per_tick = rise_pct / ticks_up

    for _ in range(ticks_down):
        price *= (1 - drop_per_tick)
        ts += 1.0
        ticks.append((price, ts))

    for _ in range(ticks_up):
        price *= (1 + rise_per_tick)
        ts += 1.0
        ticks.append((price, ts))

    return ticks


def generate_downtrend(start=100_000.0, drop_pct=0.005, ticks=50):
    """Generate a downtrend: monotonic price decrease."""
    prices = []
    price = start
    ts = 1000.0
    per_tick = drop_pct / ticks

    for _ in range(ticks):
        price *= (1 - per_tick)
        ts += 1.0
        prices.append((price, ts))

    return prices


def generate_zigzag(start=100_000.0, amplitude_pct=0.002, legs=6, ticks_per_leg=15):
    """Generate a zigzag: alternating up and down legs.

    Each leg moves by approximately amplitude_pct.
    """
    prices = []
    price = start
    ts = 1000.0
    per_tick = amplitude_pct / ticks_per_leg

    for leg in range(legs):
        direction = 1 if leg % 2 == 0 else -1
        for _ in range(ticks_per_leg):
            price *= (1 + direction * per_tick)
            ts += 1.0
            prices.append((price, ts))

    return prices


def run_strategy_on_ticks(strategy, ticks):
    """Feed ticks through strategy, collect all signals."""
    all_signals = []
    for price, ts in ticks:
        signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
        for s in signals:
            all_signals.append(s)
            # Simulate trade execution for entry signals
            if s.signal_type in (SignalType.BUY, SignalType.SELL):
                strategy.on_trade_executed(s, price, s.size)
    return all_signals


class TestDowntrendScenario:
    """Downtrend should trigger SHORT entry via PDCC_Down."""

    def test_downtrend_triggers_short_entry(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        ticks = generate_downtrend(drop_pct=0.005, ticks=50)
        signals = run_strategy_on_ticks(strategy, ticks)

        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1, "Downtrend should trigger at least one SHORT entry"

    def test_downtrend_short_tp_keeps_trailing(self):
        """In a sustained downtrend, trailing TP keeps pushing lower (greedy).

        The TP won't be hit because it retreats as the short profits.
        Instead, the SL ratchets down, locking in profit. No exit occurs
        until a reversal hits the ratcheted SL.
        """
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        ticks = generate_downtrend(drop_pct=0.008, ticks=80)
        signals = run_strategy_on_ticks(strategy, ticks)

        # Position should still be open (TP never hit, SL never hit)
        assert strategy._trailing_rm.has_position is True
        # SL should have ratcheted down from initial level
        entry = strategy._trailing_rm.entry_price
        initial_sl = entry * (1 + 0.003)
        assert strategy._trailing_rm.current_sl_price < initial_sl


class TestVShapeScenario:
    """V-shape should trigger SHORT on the way down, then potentially LONG on reversal."""

    def test_v_shape_triggers_entry_and_exit(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        ticks = generate_v_shape(drop_pct=0.004, rise_pct=0.006, ticks_down=30, ticks_up=40)
        signals = run_strategy_on_ticks(strategy, ticks)

        # Should have at least one entry signal
        entries = [s for s in signals if s.signal_type in (SignalType.BUY, SignalType.SELL)]
        assert len(entries) >= 1

        # The reversal should cause a SL exit (price rises against SHORT)
        # or there should be some exit signal
        all_types = [s.signal_type for s in signals]
        assert SignalType.CLOSE in all_types or len(entries) >= 2, (
            "V-shape should cause at least an exit or re-entry"
        )


class TestZigzagScenario:
    """Zigzag should produce multiple entry/exit cycles."""

    def test_zigzag_produces_multiple_trades(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        ticks = generate_zigzag(amplitude_pct=0.003, legs=8, ticks_per_leg=20)
        signals = run_strategy_on_ticks(strategy, ticks)

        entries = [s for s in signals if s.signal_type in (SignalType.BUY, SignalType.SELL)]
        # Zigzag with 8 legs of 0.3% should produce multiple DC events
        assert len(entries) >= 2, f"Expected multiple entries in zigzag, got {len(entries)}"


class TestStrategyStatusAfterReplay:
    """Strategy status should accurately reflect processed data."""

    def test_status_after_downtrend(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        ticks = generate_downtrend(drop_pct=0.005, ticks=50)
        run_strategy_on_ticks(strategy, ticks)

        status = strategy.get_status()
        assert status["tick_count"] == 50
        assert status["dc_event_count"] >= 1
        assert status["trade_count"] >= 1

    def test_status_trade_count_matches_fills(self):
        strategy = DCOvershootStrategy(make_config())
        strategy.start()
        ticks = generate_zigzag(amplitude_pct=0.003, legs=6, ticks_per_leg=20)
        signals = run_strategy_on_ticks(strategy, ticks)

        entries = [s for s in signals if s.signal_type in (SignalType.BUY, SignalType.SELL)]
        status = strategy.get_status()
        assert status["trade_count"] == len(entries)
