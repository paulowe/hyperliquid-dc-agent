"""Tests for EMA-based momentum scorer."""

from __future__ import annotations

import pytest

from strategies.dc_overshoot.momentum_scorer import MomentumScorer


# ---------------------------------------------------------------------------
# Basic EMA behavior
# ---------------------------------------------------------------------------
class TestInitialState:
    def test_initial_score_is_zero(self):
        scorer = MomentumScorer(alpha=0.3)
        assert scorer.score == 0.0

    def test_initial_event_count_is_zero(self):
        scorer = MomentumScorer(alpha=0.3)
        assert scorer.event_count == 0


class TestSingleEvent:
    def test_single_up_event_positive(self):
        scorer = MomentumScorer(alpha=0.3)
        scorer.update(direction=+1, timestamp=1.0)
        assert scorer.score > 0.0

    def test_single_down_event_negative(self):
        scorer = MomentumScorer(alpha=0.3)
        scorer.update(direction=-1, timestamp=1.0)
        assert scorer.score < 0.0

    def test_single_event_equals_alpha(self):
        """First event: score = alpha * direction + (1-alpha) * 0 = alpha."""
        scorer = MomentumScorer(alpha=0.3)
        scorer.update(direction=+1, timestamp=1.0)
        assert scorer.score == pytest.approx(0.3)


class TestEMAConvergence:
    def test_consistent_direction_approaches_one(self):
        """Many +1 events should push score close to +1.0."""
        scorer = MomentumScorer(alpha=0.3)
        for i in range(50):
            scorer.update(direction=+1, timestamp=float(i))
        assert scorer.score > 0.95

    def test_consistent_negative_approaches_minus_one(self):
        """Many -1 events should push score close to -1.0."""
        scorer = MomentumScorer(alpha=0.3)
        for i in range(50):
            scorer.update(direction=-1, timestamp=float(i))
        assert scorer.score < -0.95

    def test_alternating_directions_stays_near_zero(self):
        """Alternating +1, -1 keeps score near 0."""
        scorer = MomentumScorer(alpha=0.3)
        for i in range(100):
            direction = +1 if i % 2 == 0 else -1
            scorer.update(direction=direction, timestamp=float(i))
        assert abs(scorer.score) < 0.35


class TestScoreBounds:
    def test_score_never_exceeds_one(self):
        scorer = MomentumScorer(alpha=0.9)  # Aggressive alpha
        for i in range(1000):
            scorer.update(direction=+1, timestamp=float(i))
        assert scorer.score <= 1.0

    def test_score_never_below_minus_one(self):
        scorer = MomentumScorer(alpha=0.9)
        for i in range(1000):
            scorer.update(direction=-1, timestamp=float(i))
        assert scorer.score >= -1.0


class TestAlphaResponsiveness:
    def test_higher_alpha_responds_faster(self):
        """Higher alpha should flip the score faster after direction change."""
        slow = MomentumScorer(alpha=0.1)
        fast = MomentumScorer(alpha=0.5)

        # Build up positive momentum
        for i in range(20):
            slow.update(direction=+1, timestamp=float(i))
            fast.update(direction=+1, timestamp=float(i))

        # Now flip to negative
        for i in range(5):
            slow.update(direction=-1, timestamp=float(20 + i))
            fast.update(direction=-1, timestamp=float(20 + i))

        # Fast scorer should have dropped more
        assert fast.score < slow.score


class TestRecentEventsDominate:
    def test_recent_direction_overrides_history(self):
        """After many +1 events, a burst of -1 events should flip the score."""
        scorer = MomentumScorer(alpha=0.3)

        # 20 bullish events
        for i in range(20):
            scorer.update(direction=+1, timestamp=float(i))
        assert scorer.score > 0.9

        # 15 bearish events should flip it
        for i in range(15):
            scorer.update(direction=-1, timestamp=float(20 + i))
        assert scorer.score < 0.0


# ---------------------------------------------------------------------------
# Event rate and regime detection
# ---------------------------------------------------------------------------
class TestEventRate:
    def test_event_rate_empty(self):
        scorer = MomentumScorer(alpha=0.3)
        assert scorer.get_event_rate(lookback_seconds=300.0) == 0.0

    def test_event_rate_computation(self):
        """10 events in 60 seconds = 10 events/min."""
        scorer = MomentumScorer(alpha=0.3)
        for i in range(10):
            scorer.update(direction=+1, timestamp=float(i * 6))  # Every 6s
        # All events within 60s window, rate = 10 events / 1 min = 10.0
        rate = scorer.get_event_rate(lookback_seconds=60.0)
        assert rate == pytest.approx(10.0)

    def test_event_rate_excludes_old_events(self):
        """Events outside lookback window should not count."""
        scorer = MomentumScorer(alpha=0.3)
        # 5 old events
        for i in range(5):
            scorer.update(direction=+1, timestamp=float(i))
        # 5 recent events
        for i in range(5):
            scorer.update(direction=-1, timestamp=1000.0 + float(i * 10))
        # Only the 5 recent events are within 60s of the latest
        rate = scorer.get_event_rate(lookback_seconds=60.0)
        assert rate == pytest.approx(5.0)


class TestRegime:
    def test_trending_up(self):
        scorer = MomentumScorer(alpha=0.3)
        for i in range(20):
            scorer.update(direction=+1, timestamp=float(i))
        assert scorer.get_regime() == "trending_up"

    def test_trending_down(self):
        scorer = MomentumScorer(alpha=0.3)
        for i in range(20):
            scorer.update(direction=-1, timestamp=float(i))
        assert scorer.get_regime() == "trending_down"

    def test_choppy(self):
        scorer = MomentumScorer(alpha=0.3)
        for i in range(40):
            direction = +1 if i % 2 == 0 else -1
            scorer.update(direction=direction, timestamp=float(i))
        assert scorer.get_regime() in ("choppy", "neutral")

    def test_neutral_on_empty(self):
        scorer = MomentumScorer(alpha=0.3)
        assert scorer.get_regime() == "neutral"


# ---------------------------------------------------------------------------
# Reset and status
# ---------------------------------------------------------------------------
class TestResetAndStatus:
    def test_reset_clears_state(self):
        scorer = MomentumScorer(alpha=0.3)
        for i in range(10):
            scorer.update(direction=+1, timestamp=float(i))
        assert scorer.score != 0.0
        scorer.reset()
        assert scorer.score == 0.0
        assert scorer.event_count == 0

    def test_get_status_returns_dict(self):
        scorer = MomentumScorer(alpha=0.3)
        scorer.update(direction=+1, timestamp=1.0)
        status = scorer.get_status()
        assert "score" in status
        assert "event_count" in status
        assert "regime" in status
        assert status["event_count"] == 1

    def test_event_count_increments(self):
        scorer = MomentumScorer(alpha=0.3)
        for i in range(7):
            scorer.update(direction=+1, timestamp=float(i))
        assert scorer.event_count == 7
