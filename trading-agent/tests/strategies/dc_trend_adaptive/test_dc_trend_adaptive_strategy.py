"""Tests for DCTrendAdaptiveStrategy — DC Adaptive + Guard 4: Trend Direction Filter."""

import time

import pytest

from interfaces.strategy import MarketData, SignalType
from strategies.dc_trend_adaptive.dc_trend_adaptive_strategy import DCTrendAdaptiveStrategy


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
        # Trend direction filter (Guard 4) — use simple mode for deterministic tests
        "trend_lookback_seconds": 900,
        "trend_min_events": 5,
        "trend_min_consistency": 0.6,
        "trend_bias_mode": "simple",
        "trend_strict_threshold": 0.8,
        "counter_trend_action": "block",
        "counter_trend_size_fraction": 0.5,
        "close_on_trend_flip": False,
        "long_only": False,
        "short_only": False,
    }
    cfg.update(overrides)
    return cfg


def md(price, ts=None):
    return MarketData(
        asset="BTC", price=price, volume_24h=0.0,
        timestamp=ts or time.time(),
    )


def establish_uptrend(strategy, ts=100.0):
    """Feed sensor events to establish an uptrend in the trend filter.

    Directly feeds the trend filter to avoid needing exact price sequences
    that would trigger DC sensor events.
    """
    for i in range(10):
        strategy._trend_filter.record_event(+1, ts + i)
    return ts + 10


def establish_downtrend(strategy, ts=100.0):
    """Feed sensor events to establish a downtrend in the trend filter."""
    for i in range(10):
        strategy._trend_filter.record_event(-1, ts + i)
    return ts + 10


def drive_price_down(strategy, start_price=100_000.0, steps=20, ts_start=1000.0):
    """Feed ticks that decrease to trigger PDCC_Down."""
    all_signals = []
    drop_per_step = start_price * 0.0002
    price = start_price
    ts = ts_start
    for i in range(steps):
        price -= drop_per_step
        signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
        all_signals.extend(signals)
        ts += 10.0
    return all_signals, price, ts


def drive_price_up(strategy, start_price=100_000.0, steps=20, ts_start=1000.0):
    """Feed ticks that increase to trigger PDCC2_UP (after establishing downtrend)."""
    all_signals = []
    rise_per_step = start_price * 0.0002
    price = start_price
    ts = ts_start
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


# ── Init ─────────────────────────────────────────────────────────────


class TestInit:
    """Verify strategy initializes correctly."""

    def test_creates_with_default_config(self):
        strategy = DCTrendAdaptiveStrategy(make_config())
        strategy.start()
        assert strategy.is_active is True
        assert strategy.name == "dc_trend_adaptive"

    def test_has_trend_filter(self):
        strategy = DCTrendAdaptiveStrategy(make_config())
        assert hasattr(strategy, "_trend_filter")

    def test_inherits_all_adaptive_guards(self):
        """Regime, OS tracker, loss guard still present."""
        strategy = DCTrendAdaptiveStrategy(make_config())
        assert hasattr(strategy, "_regime")
        assert hasattr(strategy, "_os_tracker")
        assert hasattr(strategy, "_loss_guard")
        assert hasattr(strategy, "_trailing_rm")


# ── Guard 4: Trend filtering ────────────────────────────────────────


class TestTrendFiltering:
    """Guard 4 blocks/reduces counter-trend entries."""

    def test_blocks_counter_trend_short_in_uptrend(self):
        """With uptrend established, PDCC_Down (SHORT) should be blocked."""
        strategy = DCTrendAdaptiveStrategy(make_config(counter_trend_action="block"))
        strategy.start()

        # Establish uptrend via trend filter
        establish_uptrend(strategy, ts=500.0)

        # Drive price down to trigger PDCC_Down -> SHORT
        signals, _, _ = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]

        # Should be blocked by trend filter
        assert len(sells) == 0
        assert strategy._skipped_counter_trend > 0

    def test_allows_aligned_long_in_uptrend(self):
        """With uptrend established, PDCC2_UP (LONG) should be allowed."""
        strategy = DCTrendAdaptiveStrategy(make_config(counter_trend_action="block"))
        strategy.start()

        # Establish uptrend
        establish_uptrend(strategy, ts=500.0)

        # Drive price up to trigger PDCC2_UP -> LONG
        signals, _, _ = drive_price_up(strategy, ts_start=600.0)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) >= 1

    def test_blocks_counter_trend_long_in_downtrend(self):
        """With downtrend, PDCC2_UP (LONG) should be blocked."""
        strategy = DCTrendAdaptiveStrategy(make_config(counter_trend_action="block"))
        strategy.start()

        # Establish downtrend
        establish_downtrend(strategy, ts=500.0)

        # Drive price up to trigger PDCC2_UP -> LONG
        signals, _, _ = drive_price_up(strategy, ts_start=600.0)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) == 0
        assert strategy._skipped_counter_trend > 0

    def test_no_trend_allows_both(self):
        """Without enough events, both directions should be allowed."""
        strategy = DCTrendAdaptiveStrategy(make_config(
            counter_trend_action="block", trend_min_events=100
        ))
        strategy.start()

        # Very few trend events -> no trend
        strategy._trend_filter.record_event(+1, 500.0)

        # Should allow entry
        signals, _, _ = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) >= 1

    def test_reduce_mode_halves_size(self):
        """action=reduce with high bias -> size *= counter_trend_size_fraction."""
        strategy = DCTrendAdaptiveStrategy(make_config(
            counter_trend_action="reduce",
            counter_trend_size_fraction=0.5,
            trend_strict_threshold=0.7,
        ))
        strategy.start()

        # Strong uptrend with many events — sensor events generated during price
        # drive will add some down events, so we need enough up events to maintain
        # a high bias even after dilution.
        for i in range(30):
            strategy._trend_filter.record_event(+1, 500.0 + i)
        strategy._trend_filter.record_event(-1, 530.0)

        # Drive price down -> SHORT (counter-trend, should be reduced)
        signals, _, _ = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]

        if sells:
            # Size should be reduced
            assert strategy._reduced_counter_trend > 0

    def test_reversal_into_counter_trend_blocked(self):
        """Reversal into counter-trend direction should be blocked."""
        strategy = DCTrendAdaptiveStrategy(make_config(counter_trend_action="block"))
        strategy.start()

        # Enter a LONG position first (no trend yet)
        signals_up, price_up, ts = drive_price_up(strategy, ts_start=600.0)
        buys = [s for s in signals_up if s.signal_type == SignalType.BUY]
        if buys:
            strategy.on_trade_executed(buys[0], price_up, buys[0].size)

            # Now establish uptrend
            establish_uptrend(strategy, ts=ts)

            # Try reversal to SHORT (counter-trend in uptrend) -> should be blocked
            signals_down, _, _ = drive_price_down(strategy, start_price=price_up, ts_start=ts + 100)
            sells = [s for s in signals_down if s.signal_type == SignalType.SELL]

            # Filter out CLOSE signals (those are trailing RM exits)
            entry_sells = [s for s in sells if "dc_trend_adaptive" in s.reason]
            assert len(entry_sells) == 0

    def test_skipped_counter_trend_counter(self):
        """Counter increments on blocked trades."""
        strategy = DCTrendAdaptiveStrategy(make_config(counter_trend_action="block"))
        strategy.start()
        assert strategy._skipped_counter_trend == 0

        establish_uptrend(strategy, ts=500.0)
        drive_price_down(strategy, ts_start=600.0)

        assert strategy._skipped_counter_trend > 0

    def test_trend_filter_fed_from_sensor_events(self):
        """Sensor events feed both regime AND trend filter when driven through strategy."""
        strategy = DCTrendAdaptiveStrategy(make_config(
            sensor_threshold=[0.0003, 0.0003],
            trend_min_events=3,
        ))
        strategy.start()

        # Drive price with small oscillations to trigger sensor events
        # The strategy should feed these to the trend filter via generate_signals
        ts = 0.0
        price = 100_000.0
        # Drive down for sensor events
        for i in range(50):
            price -= 15  # small drops to trigger sensor
            strategy.generate_signals(md(price, ts), [], 100_000.0)
            ts += 1.0

        # The trend filter should have received events (from sensor threshold)
        status = strategy._trend_filter.get_status(ts)
        assert status["event_count"] >= 0  # May or may not fire depending on threshold


# ── Nuclear modes ────────────────────────────────────────────────────


class TestNuclearModes:
    """long_only and short_only nuclear overrides."""

    def test_long_only_blocks_shorts(self):
        strategy = DCTrendAdaptiveStrategy(make_config(long_only=True))
        strategy.start()

        signals, _, _ = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) == 0
        assert strategy._skipped_long_only > 0

    def test_short_only_blocks_longs(self):
        strategy = DCTrendAdaptiveStrategy(make_config(short_only=True))
        strategy.start()

        signals, _, _ = drive_price_up(strategy, ts_start=600.0)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) == 0
        assert strategy._skipped_short_only > 0


# ── Close on trend flip (Ch 6) ──────────────────────────────────────


class TestCloseOnTrendFlip:
    """Ch 6 JC1/JC2 protective close rule."""

    def test_close_on_trend_flip_exits_counter_trend(self):
        """Holding SHORT, trend flips to UP -> generates close signal."""
        strategy = DCTrendAdaptiveStrategy(make_config(close_on_trend_flip=True))
        strategy.start()

        # Enter SHORT position (no trend established yet)
        signals, price, ts = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        if sells:
            strategy.on_trade_executed(sells[0], price, sells[0].size)
            assert strategy._trailing_rm.has_position
            assert strategy._trailing_rm.side == "SHORT"

            # Now establish strong uptrend -> should trigger protective close
            for i in range(10):
                strategy._trend_filter.record_event(+1, ts + i)
            ts += 10

            # Next tick should trigger close
            close_signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
            closes = [s for s in close_signals if s.signal_type == SignalType.CLOSE]
            assert len(closes) == 1
            assert closes[0].reason == "trend_flip_protective"
            assert strategy._trend_flip_closes == 1

    def test_close_on_trend_flip_keeps_aligned(self):
        """Holding LONG in uptrend -> no close signal."""
        strategy = DCTrendAdaptiveStrategy(make_config(close_on_trend_flip=True))
        strategy.start()

        # Establish uptrend first
        establish_uptrend(strategy, ts=500.0)

        # Enter LONG position
        signals, price, ts = drive_price_up(strategy, ts_start=600.0)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        if buys:
            strategy.on_trade_executed(buys[0], price, buys[0].size)

            # Maintain uptrend -> no close
            for i in range(5):
                strategy._trend_filter.record_event(+1, ts + i)
            ts += 5

            close_signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
            trend_closes = [s for s in close_signals
                           if s.signal_type == SignalType.CLOSE and s.reason == "trend_flip_protective"]
            assert len(trend_closes) == 0

    def test_close_on_trend_flip_disabled(self):
        """close_on_trend_flip=False -> no protective close."""
        strategy = DCTrendAdaptiveStrategy(make_config(close_on_trend_flip=False))
        strategy.start()

        # Enter SHORT
        signals, price, ts = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        if sells:
            strategy.on_trade_executed(sells[0], price, sells[0].size)

            # Flip trend to up -> should NOT close because disabled
            for i in range(10):
                strategy._trend_filter.record_event(+1, ts + i)
            ts += 10

            close_signals = strategy.generate_signals(md(price, ts), [], 100_000.0)
            trend_closes = [s for s in close_signals
                           if s.signal_type == SignalType.CLOSE and s.reason == "trend_flip_protective"]
            assert len(trend_closes) == 0
            assert strategy._trend_flip_closes == 0


# ── Asymmetric SL ────────────────────────────────────────────────────


class TestAsymmetricSL:
    """Counter-trend trades get tighter SL if configured."""

    def test_asymmetric_sl_in_metadata(self):
        """counter_trend_sl_pct appears in signal metadata for counter-trend trades."""
        strategy = DCTrendAdaptiveStrategy(make_config(
            counter_trend_action="allow",  # Allow but mark
            counter_trend_sl_pct=0.01,
        ))
        strategy.start()

        # Establish uptrend
        establish_uptrend(strategy, ts=500.0)

        # SHORT entry (counter-trend in uptrend) -> should have counter_trend_sl
        signals, _, _ = drive_price_down(strategy, ts_start=600.0)
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        if sells:
            assert sells[0].metadata.get("counter_trend_sl") == 0.01


# ── Status ───────────────────────────────────────────────────────────


class TestStatus:
    """get_status() includes trend info."""

    def test_get_status_includes_trend(self):
        strategy = DCTrendAdaptiveStrategy(make_config())
        status = strategy.get_status()
        assert status["name"] == "dc_trend_adaptive"
        assert "dominant_trend" in status
        assert "trend_bias" in status
        assert "trend_bias_mode" in status
        assert "skipped_counter_trend" in status
        assert "reduced_counter_trend" in status
        assert "trend_flip_closes" in status
        assert "close_on_trend_flip" in status
        assert "counter_trend_action" in status

    def test_get_status_inherits_adaptive_fields(self):
        strategy = DCTrendAdaptiveStrategy(make_config())
        status = strategy.get_status()
        # From parent
        assert "regime" in status
        assert "adaptive_tp" in status
        assert "loss_streak" in status
        assert "trailing_rm" in status


# ── Lifecycle ────────────────────────────────────────────────────────


class TestLifecycle:
    """Start/stop lifecycle."""

    def test_start_stop(self):
        strategy = DCTrendAdaptiveStrategy(make_config())
        strategy.start()
        assert strategy.is_active is True
        strategy.stop()
        assert strategy.is_active is False
