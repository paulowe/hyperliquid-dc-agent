"""Tests for FundingMonitor."""

import time
import pytest
from strategies.basis_trade.funding_monitor import (
    FundingMonitor,
    FundingSnapshot,
    FundingPayment,
)


class TestFundingSnapshot:
    """Test FundingSnapshot data class."""

    def test_basic(self):
        snap = FundingSnapshot(timestamp=1000.0, rate=0.0001, mark_price=36.5)
        assert snap.timestamp == 1000.0
        assert snap.rate == 0.0001
        assert snap.mark_price == 36.5
        assert snap.oracle_price == 0.0


class TestFundingMonitor:
    """Test funding rate tracking and streak detection."""

    def _make_monitor(self, window_hours=24):
        return FundingMonitor(window_hours=window_hours)

    def test_initial_state(self):
        fm = self._make_monitor()
        assert fm.last_rate is None
        assert fm.consecutive_above == 0
        assert fm.consecutive_below == 0
        assert fm.total_payments == 0
        assert fm.cumulative_funding_usd == 0.0

    def test_record_observation(self):
        fm = self._make_monitor()
        snap = FundingSnapshot(timestamp=1000.0, rate=0.0002, mark_price=36.5)
        fm.record_observation(snap)
        assert fm.last_rate == 0.0002

    def test_streak_above(self):
        fm = self._make_monitor()
        threshold = 0.0001

        # 3 consecutive observations above threshold
        for i in range(3):
            snap = FundingSnapshot(timestamp=1000 + i * 3600, rate=0.0002, mark_price=36.5)
            fm.record_observation(snap)
            fm.update_streaks(snap.rate, above_threshold=threshold, below_threshold=-0.00005)

        assert fm.consecutive_above == 3
        assert fm.consecutive_below == 0

    def test_streak_resets_on_dip(self):
        fm = self._make_monitor()
        threshold = 0.0001

        # 2 above, then 1 below, then 1 above
        rates = [0.0002, 0.0003, 0.00005, 0.0002]
        for i, rate in enumerate(rates):
            snap = FundingSnapshot(timestamp=1000 + i * 3600, rate=rate, mark_price=36.5)
            fm.record_observation(snap)
            fm.update_streaks(rate, above_threshold=threshold, below_threshold=-0.00005)

        assert fm.consecutive_above == 1  # Reset after dip
        assert fm.consecutive_below == 0

    def test_streak_below(self):
        fm = self._make_monitor()

        # 4 consecutive below exit threshold
        for i in range(4):
            rate = -0.0001
            fm.update_streaks(rate, above_threshold=0.0001, below_threshold=-0.00005)

        assert fm.consecutive_below == 4
        assert fm.consecutive_above == 0

    def test_record_payment_positive(self):
        fm = self._make_monitor()
        payment = fm.record_payment(timestamp=1000.0, rate=0.0002, notional=50.0)

        assert payment.payment_usd == pytest.approx(0.01)  # 0.0002 * 50
        assert fm.cumulative_funding_usd == pytest.approx(0.01)
        assert fm.total_payments == 1

    def test_record_payment_negative(self):
        fm = self._make_monitor()
        payment = fm.record_payment(timestamp=1000.0, rate=-0.0001, notional=50.0)

        assert payment.payment_usd == pytest.approx(-0.005)  # shorts pay when negative
        assert fm.cumulative_funding_usd == pytest.approx(-0.005)

    def test_cumulative_payments(self):
        fm = self._make_monitor()

        # 3 payments: +0.01, +0.01, -0.005
        fm.record_payment(timestamp=1000.0, rate=0.0002, notional=50.0)
        fm.record_payment(timestamp=4600.0, rate=0.0002, notional=50.0)
        fm.record_payment(timestamp=8200.0, rate=-0.0001, notional=50.0)

        assert fm.cumulative_funding_usd == pytest.approx(0.015)
        assert fm.total_payments == 3

    def test_average_rate_all(self):
        fm = self._make_monitor()
        now = time.time()

        rates = [0.0001, 0.0002, 0.0003]
        for i, rate in enumerate(rates):
            snap = FundingSnapshot(timestamp=now + i * 3600, rate=rate, mark_price=36.5)
            fm.record_observation(snap)

        avg = fm.average_rate()
        assert avg == pytest.approx(0.0002)

    def test_average_rate_empty(self):
        fm = self._make_monitor()
        assert fm.average_rate() is None

    def test_current_apr(self):
        fm = self._make_monitor()
        snap = FundingSnapshot(timestamp=1000.0, rate=0.0001, mark_price=36.5)
        fm.record_observation(snap)

        apr = fm.current_apr()
        # 0.0001 * 24 * 365 * 100 = 87.6%
        assert abs(apr - 87.6) < 0.1

    def test_current_apr_none_when_empty(self):
        fm = self._make_monitor()
        assert fm.current_apr() is None

    def test_window_pruning(self):
        fm = self._make_monitor(window_hours=2)
        base_ts = 100000.0

        # Add 5 observations, 1 hour apart
        for i in range(5):
            snap = FundingSnapshot(
                timestamp=base_ts + i * 3600,
                rate=0.0001 * (i + 1),
                mark_price=36.5,
            )
            fm.record_observation(snap)

        # Window is 2 hours, so only the last ~2 should remain
        # Last observation is at base_ts + 4*3600
        # Cutoff is base_ts + 4*3600 - 2*3600 = base_ts + 2*3600
        # Observations at base_ts, base_ts+3600 should be pruned
        # Remaining: base_ts+2*3600, base_ts+3*3600, base_ts+4*3600
        assert len(fm._observations) == 3

    def test_get_status(self):
        fm = self._make_monitor()
        snap = FundingSnapshot(timestamp=1000.0, rate=0.0002, mark_price=36.5)
        fm.record_observation(snap)
        fm.record_payment(timestamp=1000.0, rate=0.0002, notional=50.0)

        status = fm.get_status()
        assert status["last_rate"] == 0.0002
        assert status["observations"] == 1
        assert status["total_payments"] == 1
        assert status["cumulative_funding_usd"] == pytest.approx(0.01)
