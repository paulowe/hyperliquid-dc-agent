"""Tests for TrendDirectionFilter — Guard 4: trend bias from DC event directions."""

import pytest

from strategies.dc_trend_adaptive.trend_direction_filter import TrendDirectionFilter


# ── Basic bias tests ────────────────────────────────────────────────


class TestNoTrendDetected:
    """No trend when insufficient data."""

    def test_no_events_returns_none(self):
        f = TrendDirectionFilter()
        assert f.dominant_trend(1000.0) is None

    def test_few_events_returns_none(self):
        """Below min_events threshold -> no trend."""
        f = TrendDirectionFilter(min_events=5)
        for i in range(4):
            f.record_event(+1, 100.0 + i)
        assert f.dominant_trend(104.0) is None

    def test_balanced_returns_none(self):
        """50/50 up/down -> no trend."""
        f = TrendDirectionFilter(min_events=4)
        for i in range(5):
            f.record_event(+1, 100.0 + i * 2)
            f.record_event(-1, 101.0 + i * 2)
        assert f.dominant_trend(110.0) is None


class TestUptrendDetected:
    """Detect uptrend from biased events."""

    def test_uptrend_detected(self):
        f = TrendDirectionFilter(min_events=5, min_consistency=0.6)
        # 8 up, 2 down -> bias = 0.6 -> uptrend
        for i in range(8):
            f.record_event(+1, 100.0 + i)
        for i in range(2):
            f.record_event(-1, 108.0 + i)
        assert f.dominant_trend(110.0) == "up"

    def test_bias_value_uptrend(self):
        """Simple bias = sum(d) / N."""
        f = TrendDirectionFilter(bias_mode="simple", min_events=5)
        for i in range(8):
            f.record_event(+1, 100.0 + i)
        for i in range(2):
            f.record_event(-1, 108.0 + i)
        assert f.bias(110.0) == pytest.approx(0.6)


class TestDowntrendDetected:
    """Detect downtrend from biased events."""

    def test_downtrend_detected(self):
        f = TrendDirectionFilter(min_events=5, min_consistency=0.6)
        for i in range(8):
            f.record_event(-1, 100.0 + i)
        for i in range(2):
            f.record_event(+1, 108.0 + i)
        assert f.dominant_trend(110.0) == "down"


class TestLookbackWindow:
    """Events outside lookback window are purged."""

    def test_lookback_window_purge(self):
        f = TrendDirectionFilter(lookback_seconds=100.0, min_events=3, min_consistency=0.6)
        # Old uptrend events
        for i in range(10):
            f.record_event(+1, 100.0 + i)
        # Recent downtrend events
        for i in range(10):
            f.record_event(-1, 300.0 + i)
        # Only recent events should matter
        assert f.dominant_trend(310.0) == "down"

    def test_trend_transitions(self):
        """Events shift from up to down -> trend flips."""
        f = TrendDirectionFilter(lookback_seconds=50.0, min_events=3, min_consistency=0.6)
        # Uptrend
        for i in range(8):
            f.record_event(+1, 100.0 + i)
        assert f.dominant_trend(108.0) == "up"

        # Much later: downtrend
        for i in range(8):
            f.record_event(-1, 200.0 + i)
        assert f.dominant_trend(208.0) == "down"


class TestMinConsistencyBoundary:
    """Edge cases around min_consistency threshold."""

    def test_exactly_at_threshold(self):
        """bias = min_consistency -> up."""
        f = TrendDirectionFilter(bias_mode="simple", min_events=5, min_consistency=0.6)
        # 8 up, 2 down -> bias = 6/10 = 0.6
        for i in range(8):
            f.record_event(+1, 100.0 + i)
        for i in range(2):
            f.record_event(-1, 108.0 + i)
        assert f.dominant_trend(110.0) == "up"

    def test_just_below_threshold(self):
        """bias < min_consistency -> None."""
        f = TrendDirectionFilter(bias_mode="simple", min_events=5, min_consistency=0.6)
        # 7 up, 3 down -> bias = 4/10 = 0.4 < 0.6
        for i in range(7):
            f.record_event(+1, 100.0 + i)
        for i in range(3):
            f.record_event(-1, 107.0 + i)
        assert f.dominant_trend(110.0) is None


# ── TMV-weighted mode (Ch 5) ────────────────────────────────────────


class TestTMVWeightedMode:
    """TMV-weighted bias mode from Chen & Tsang Ch 5."""

    def test_tmv_weighted_large_up_moves_detect_uptrend(self):
        """3 up (tmv=2%) + 3 down (tmv=0.5%) -> up even though count is 50/50."""
        f = TrendDirectionFilter(
            bias_mode="tmv_weighted", min_events=5, min_consistency=0.5
        )
        # Up events with large TMV
        for i in range(3):
            f.record_event(+1, 100.0 + i, tmv=0.02)
        # Down events with small TMV
        for i in range(3):
            f.record_event(-1, 103.0 + i, tmv=0.005)
        # TMV bias = (3*0.02 - 3*0.005) / (3*0.02 + 3*0.005) = 0.045/0.075 = 0.6
        assert f.dominant_trend(106.0) == "up"

    def test_tmv_weighted_equal_tmv_same_as_simple(self):
        """When all TMVs equal, tmv_weighted == simple."""
        f_simple = TrendDirectionFilter(bias_mode="simple", min_events=3)
        f_tmv = TrendDirectionFilter(bias_mode="tmv_weighted", min_events=3)

        events = [(+1, 100.0), (+1, 101.0), (+1, 102.0), (-1, 103.0), (-1, 104.0)]
        for d, ts in events:
            f_simple.record_event(d, ts, tmv=1.0)
            f_tmv.record_event(d, ts, tmv=1.0)

        assert f_simple.bias(105.0) == pytest.approx(f_tmv.bias(105.0))

    def test_tmv_weighted_small_counter_moves_ignored(self):
        """Many small counter-trend events don't flip bias when few large trend events dominate."""
        f = TrendDirectionFilter(
            bias_mode="tmv_weighted", min_events=5, min_consistency=0.5
        )
        # 2 large up moves
        f.record_event(+1, 100.0, tmv=0.03)
        f.record_event(+1, 101.0, tmv=0.03)
        # 5 tiny down moves
        for i in range(5):
            f.record_event(-1, 102.0 + i, tmv=0.002)

        # TMV: up = 0.06, down = 0.01
        # bias = (0.06 - 0.01) / (0.06 + 0.01) = 0.05/0.07 = 0.714
        assert f.dominant_trend(107.0) == "up"

    def test_simple_mode_ignores_tmv(self):
        """bias_mode='simple' treats all events equally regardless of tmv."""
        f = TrendDirectionFilter(
            bias_mode="simple", min_events=5, min_consistency=0.5
        )
        # 3 up with huge TMV, 4 down with tiny TMV
        for i in range(3):
            f.record_event(+1, 100.0 + i, tmv=0.05)
        for i in range(4):
            f.record_event(-1, 103.0 + i, tmv=0.001)

        # Simple: bias = (3 - 4) / 7 = -0.143 -> no trend (or slight down)
        assert f.bias(107.0) == pytest.approx(-1 / 7)
        assert f.dominant_trend(107.0) is None


# ── Decision rules (B-Simple / B-Strict) ────────────────────────────


class TestShouldTradeDirection:
    """B-Simple and B-Strict decision rules."""

    def _make_uptrend_filter(self, **kwargs):
        """Create filter with established uptrend."""
        defaults = dict(
            bias_mode="simple", min_events=5, min_consistency=0.6,
            counter_trend_action="block",
        )
        defaults.update(kwargs)
        f = TrendDirectionFilter(**defaults)
        # 9 up, 1 down -> bias = 0.8
        for i in range(9):
            f.record_event(+1, 100.0 + i)
        f.record_event(-1, 109.0)
        return f

    def test_allows_aligned_trade(self):
        """Long in uptrend -> (True, 1.0)."""
        f = self._make_uptrend_filter()
        allowed, mult = f.should_trade_direction("LONG", 110.0)
        assert allowed is True
        assert mult == 1.0

    def test_blocks_counter_b_simple(self):
        """Short in uptrend + action=block -> (False, 0.0)."""
        f = self._make_uptrend_filter(counter_trend_action="block")
        allowed, mult = f.should_trade_direction("SHORT", 110.0)
        assert allowed is False
        assert mult == 0.0

    def test_reduces_counter_b_strict_above_strict(self):
        """Short in uptrend + action=reduce + bias > strict_threshold -> reduced size."""
        f = self._make_uptrend_filter(
            counter_trend_action="reduce",
            strict_threshold=0.7,
            counter_trend_size_fraction=0.5,
        )
        # bias = 0.8 > strict_threshold 0.7
        allowed, mult = f.should_trade_direction("SHORT", 110.0)
        assert allowed is True
        assert mult == 0.5

    def test_b_strict_no_reduction_below_strict_threshold(self):
        """bias > min_consistency but < strict_threshold -> (True, 1.0)."""
        f = TrendDirectionFilter(
            bias_mode="simple", min_events=5, min_consistency=0.6,
            counter_trend_action="reduce", strict_threshold=0.9,
            counter_trend_size_fraction=0.5,
        )
        # 8 up, 2 down -> bias = 0.6 >= min_consistency but < 0.9
        for i in range(8):
            f.record_event(+1, 100.0 + i)
        for i in range(2):
            f.record_event(-1, 108.0 + i)

        allowed, mult = f.should_trade_direction("SHORT", 110.0)
        assert allowed is True
        assert mult == 1.0  # Not confident enough to reduce

    def test_allows_all_when_allow(self):
        """action=allow -> always (True, 1.0)."""
        f = self._make_uptrend_filter(counter_trend_action="allow")
        allowed, mult = f.should_trade_direction("SHORT", 110.0)
        assert allowed is True
        assert mult == 1.0

    def test_no_trend_allows_all(self):
        """No clear trend -> (True, 1.0) for both sides."""
        f = TrendDirectionFilter(min_events=5)
        # Only 3 events -> no trend
        f.record_event(+1, 100.0)
        f.record_event(-1, 101.0)
        f.record_event(+1, 102.0)
        allowed_long, mult_long = f.should_trade_direction("LONG", 103.0)
        allowed_short, mult_short = f.should_trade_direction("SHORT", 103.0)
        assert allowed_long is True and mult_long == 1.0
        assert allowed_short is True and mult_short == 1.0


# ── Counter-trend detection ─────────────────────────────────────────


class TestIsCounterTrend:
    """is_counter_trend() helper."""

    def test_short_in_uptrend_is_counter(self):
        f = TrendDirectionFilter(bias_mode="simple", min_events=3, min_consistency=0.5)
        for i in range(5):
            f.record_event(+1, 100.0 + i)
        assert f.is_counter_trend("SHORT", 105.0) is True

    def test_long_in_uptrend_not_counter(self):
        f = TrendDirectionFilter(bias_mode="simple", min_events=3, min_consistency=0.5)
        for i in range(5):
            f.record_event(+1, 100.0 + i)
        assert f.is_counter_trend("LONG", 105.0) is False

    def test_no_trend_not_counter(self):
        f = TrendDirectionFilter(min_events=5)
        assert f.is_counter_trend("SHORT", 100.0) is False


# ── Status / telemetry ──────────────────────────────────────────────


class TestGetStatus:
    """Status dict for monitoring/telemetry."""

    def test_get_status_has_expected_keys(self):
        f = TrendDirectionFilter()
        status = f.get_status(100.0)
        assert "dominant_trend" in status
        assert "bias" in status
        assert "bias_mode" in status
        assert "event_count" in status
        assert "events_in_window" in status
        assert "avg_tmv_up" in status
        assert "avg_tmv_down" in status

    def test_get_status_tmv_breakdown(self):
        """avg_tmv_up > avg_tmv_down in uptrend with larger up moves."""
        f = TrendDirectionFilter(bias_mode="tmv_weighted", min_events=3)
        for i in range(5):
            f.record_event(+1, 100.0 + i, tmv=0.02)
        for i in range(2):
            f.record_event(-1, 105.0 + i, tmv=0.005)

        status = f.get_status(107.0)
        assert status["avg_tmv_up"] > status["avg_tmv_down"]
        assert status["avg_tmv_up"] == pytest.approx(0.02)
        assert status["avg_tmv_down"] == pytest.approx(0.005)
