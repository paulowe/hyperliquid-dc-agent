"""Tests for LossStreakGuard — circuit breaker after consecutive losses."""

import pytest

from strategies.dc_adaptive.loss_streak_guard import LossStreakGuard


class TestLossStreakGuardInit:
    """Verify initial state."""

    def test_allows_trading_initially(self):
        g = LossStreakGuard()
        assert g.should_trade(0.0) is True

    def test_zero_consecutive_losses(self):
        g = LossStreakGuard()
        assert g.consecutive_losses == 0


class TestWinResets:
    """Winning trades reset the loss counter."""

    def test_win_resets_counter(self):
        g = LossStreakGuard(max_consecutive_losses=3)
        g.record_trade(is_win=False, timestamp=100.0)
        g.record_trade(is_win=False, timestamp=200.0)
        assert g.consecutive_losses == 2
        g.record_trade(is_win=True, timestamp=300.0)
        assert g.consecutive_losses == 0

    def test_win_after_cooldown_resets(self):
        g = LossStreakGuard(max_consecutive_losses=2, base_cooldown_seconds=60)
        g.record_trade(is_win=False, timestamp=100.0)
        g.record_trade(is_win=False, timestamp=200.0)
        assert g.should_trade(200.0) is False
        g.record_trade(is_win=True, timestamp=500.0)
        assert g.should_trade(500.0) is True


class TestCooldownActivation:
    """Cooldown triggers after max consecutive losses."""

    def test_cooldown_after_max_losses(self):
        g = LossStreakGuard(max_consecutive_losses=3, base_cooldown_seconds=300)
        for i in range(3):
            g.record_trade(is_win=False, timestamp=float(i * 100))
        # Should be in cooldown now
        assert g.should_trade(300.0) is False

    def test_not_in_cooldown_below_max(self):
        g = LossStreakGuard(max_consecutive_losses=3, base_cooldown_seconds=300)
        g.record_trade(is_win=False, timestamp=100.0)
        g.record_trade(is_win=False, timestamp=200.0)
        assert g.should_trade(300.0) is True


class TestCooldownDuration:
    """Cooldown duration escalates with loss count."""

    def test_cooldown_escalates(self):
        g = LossStreakGuard(max_consecutive_losses=3, base_cooldown_seconds=300)
        # 3 losses → cooldown = 300 * 3 = 900s
        for i in range(3):
            g.record_trade(is_win=False, timestamp=float(i * 10))
        assert g.should_trade(30.0) is False  # t=30, cooldown until 30+900=930
        assert g.should_trade(800.0) is False  # still in cooldown
        assert g.should_trade(1000.0) is True  # past cooldown

    def test_additional_loss_extends_cooldown(self):
        g = LossStreakGuard(max_consecutive_losses=3, base_cooldown_seconds=60)
        for i in range(4):
            g.record_trade(is_win=False, timestamp=float(i * 10))
        # 4 losses → cooldown = 60 * 4 = 240s from t=30
        assert g.should_trade(100.0) is False
        assert g.should_trade(300.0) is True


class TestCooldownExpiry:
    """Trading resumes after cooldown expires."""

    def test_trading_resumes_after_cooldown(self):
        g = LossStreakGuard(max_consecutive_losses=2, base_cooldown_seconds=60)
        g.record_trade(is_win=False, timestamp=100.0)
        g.record_trade(is_win=False, timestamp=200.0)
        # cooldown = 60 * 2 = 120s from t=200, expires at t=320
        assert g.should_trade(250.0) is False
        assert g.should_trade(319.0) is False
        assert g.should_trade(320.0) is True


class TestStatus:
    """Status reporting."""

    def test_status_no_cooldown(self):
        g = LossStreakGuard(max_consecutive_losses=3)
        s = g.get_status()
        assert s["consecutive_losses"] == 0
        assert s["in_cooldown"] is False

    def test_status_in_cooldown(self):
        g = LossStreakGuard(max_consecutive_losses=2, base_cooldown_seconds=60)
        g.record_trade(is_win=False, timestamp=100.0)
        g.record_trade(is_win=False, timestamp=200.0)
        s = g.get_status(timestamp=210.0)
        assert s["consecutive_losses"] == 2
        assert s["in_cooldown"] is True
        assert s["cooldown_remaining_s"] == pytest.approx(110.0, abs=1)
