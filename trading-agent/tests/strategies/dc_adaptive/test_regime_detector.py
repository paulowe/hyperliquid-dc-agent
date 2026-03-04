"""Tests for RegimeDetector — classifies market as trending, choppy, or quiet."""

import pytest

from strategies.dc_adaptive.regime_detector import RegimeDetector


class TestRegimeDetectorInit:
    """Verify initial state."""

    def test_default_regime_is_quiet(self):
        rd = RegimeDetector()
        assert rd.classify(1000.0) == "quiet"

    def test_should_trade_when_quiet(self):
        rd = RegimeDetector()
        assert rd.should_trade(1000.0) is True


class TestQuietRegime:
    """Quiet = too few events to classify."""

    def test_fewer_than_3_events_is_quiet(self):
        rd = RegimeDetector(lookback_seconds=600)
        rd.record_event(+1, 100.0)
        rd.record_event(-1, 200.0)
        assert rd.classify(300.0) == "quiet"

    def test_events_too_close_together_is_quiet(self):
        """If all events happen within < 60s, classify as quiet (not enough data)."""
        rd = RegimeDetector(lookback_seconds=600)
        for i in range(5):
            rd.record_event(+1, 100.0 + i * 10)
        assert rd.classify(140.0) == "quiet"


class TestChoppyRegime:
    """Choppy = high event rate with alternating directions."""

    def test_alternating_directions_high_rate(self):
        """Rapid +1/-1 alternation at high rate → choppy."""
        rd = RegimeDetector(
            lookback_seconds=600,
            choppy_rate_threshold=4.0,
            trending_consistency_threshold=0.6,
        )
        t = 0.0
        # 30 events in 5 minutes = 6/min, alternating direction
        for i in range(30):
            direction = +1 if i % 2 == 0 else -1
            rd.record_event(direction, t)
            t += 10.0  # 10 sec apart
        assert rd.classify(t) == "choppy"

    def test_choppy_blocks_trading(self):
        rd = RegimeDetector(
            lookback_seconds=600,
            choppy_rate_threshold=4.0,
            trending_consistency_threshold=0.6,
        )
        t = 0.0
        for i in range(30):
            direction = +1 if i % 2 == 0 else -1
            rd.record_event(direction, t)
            t += 10.0
        assert rd.should_trade(t) is False


class TestTrendingRegime:
    """Trending = events mostly agree on direction."""

    def test_consistent_direction_is_trending(self):
        """All same direction → trending."""
        rd = RegimeDetector(
            lookback_seconds=600,
            choppy_rate_threshold=4.0,
            trending_consistency_threshold=0.6,
        )
        t = 0.0
        for i in range(10):
            rd.record_event(+1, t)
            t += 30.0  # 2/min, moderate rate
        assert rd.classify(t) == "trending"

    def test_mostly_same_direction_is_trending(self):
        """7 up, 3 down = 0.4 net consistency = 0.4, but let's check threshold."""
        rd = RegimeDetector(
            lookback_seconds=600,
            trending_consistency_threshold=0.6,
        )
        t = 0.0
        # 8 up, 2 down = abs(8-2)/10 = 0.6 consistency
        for i in range(10):
            direction = +1 if i < 8 else -1
            rd.record_event(direction, t)
            t += 30.0
        assert rd.classify(t) == "trending"

    def test_trending_allows_trading(self):
        rd = RegimeDetector(trending_consistency_threshold=0.6)
        t = 0.0
        for i in range(10):
            rd.record_event(+1, t)
            t += 30.0
        assert rd.should_trade(t) is True


class TestNeutralRegime:
    """Neutral = moderate activity but no clear direction or chop."""

    def test_moderate_mixed_is_neutral(self):
        """Moderate rate, partially mixed directions → neutral."""
        rd = RegimeDetector(
            lookback_seconds=600,
            choppy_rate_threshold=4.0,
            trending_consistency_threshold=0.6,
        )
        t = 0.0
        # 6 events in 5 min = 1.2/min (below choppy threshold)
        # 4 up, 2 down = consistency 0.33 (below trending threshold)
        for i, d in enumerate([+1, +1, -1, +1, -1, +1]):
            rd.record_event(d, t)
            t += 50.0
        assert rd.classify(t) == "neutral"

    def test_neutral_allows_trading(self):
        rd = RegimeDetector()
        t = 0.0
        for i, d in enumerate([+1, +1, -1, +1, -1, +1]):
            rd.record_event(d, t)
            t += 50.0
        assert rd.should_trade(t) is True


class TestLookbackPurge:
    """Old events outside the lookback window are purged."""

    def test_old_events_purged(self):
        rd = RegimeDetector(lookback_seconds=60)
        # Record choppy events
        for i in range(20):
            rd.record_event(+1 if i % 2 == 0 else -1, float(i * 5))
        # Should be choppy at t=100
        # But at t=200 (all events older than 60s are purged), should be quiet
        assert rd.classify(200.0) == "quiet"

    def test_recent_events_survive_purge(self):
        rd = RegimeDetector(lookback_seconds=300)
        # Old choppy events
        for i in range(10):
            rd.record_event(+1 if i % 2 == 0 else -1, float(i * 5))
        # New trending events
        t = 500.0
        for i in range(10):
            rd.record_event(+1, t)
            t += 20.0
        assert rd.classify(t) == "trending"


class TestEventRate:
    """Event rate computation."""

    def test_event_rate_computation(self):
        rd = RegimeDetector(lookback_seconds=600)
        t = 0.0
        for i in range(12):
            rd.record_event(+1, t)
            t += 30.0  # 2 events/min
        rate = rd.event_rate(t)
        assert rate == pytest.approx(2.0, rel=0.1)

    def test_event_rate_zero_when_empty(self):
        rd = RegimeDetector()
        assert rd.event_rate(100.0) == 0.0
