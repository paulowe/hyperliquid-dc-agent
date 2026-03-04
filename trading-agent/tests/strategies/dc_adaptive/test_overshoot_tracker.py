"""Tests for OversshootTracker — adapts TP to recent overshoot magnitudes."""

import pytest

from strategies.dc_adaptive.overshoot_tracker import OvershootTracker


class TestOvershootTrackerInit:
    """Verify initial state."""

    def test_default_tp_when_no_samples(self):
        ot = OvershootTracker(default_tp_pct=0.005)
        assert ot.adaptive_tp() == 0.005

    def test_count_starts_at_zero(self):
        ot = OvershootTracker()
        assert ot.count == 0


class TestAdaptiveTP:
    """TP adapts based on rolling median overshoot."""

    def test_uses_default_below_min_samples(self):
        ot = OvershootTracker(min_samples=5, default_tp_pct=0.005)
        for i in range(4):
            ot.record_overshoot(0.02)  # 2% overshoots
        # Still below min_samples
        assert ot.adaptive_tp() == 0.005

    def test_adapts_at_min_samples(self):
        ot = OvershootTracker(min_samples=5, tp_fraction=0.8, default_tp_pct=0.005)
        for i in range(5):
            ot.record_overshoot(0.01)  # 1% overshoots
        # p50 = 0.01, adaptive = 0.01 * 0.8 = 0.008
        assert ot.adaptive_tp() == pytest.approx(0.008)

    def test_adapts_to_changing_conditions(self):
        ot = OvershootTracker(window_size=5, min_samples=3, tp_fraction=1.0)
        # Small overshoots
        for _ in range(5):
            ot.record_overshoot(0.005)
        assert ot.adaptive_tp() == pytest.approx(0.005)
        # Now bigger overshoots push out old ones
        for _ in range(5):
            ot.record_overshoot(0.02)
        assert ot.adaptive_tp() == pytest.approx(0.02)

    def test_tp_has_minimum_floor(self):
        ot = OvershootTracker(min_samples=3, tp_fraction=0.8, min_tp_pct=0.003)
        for _ in range(5):
            ot.record_overshoot(0.001)  # Tiny overshoots
        # p50 * 0.8 = 0.0008 → clamped to 0.003 floor
        assert ot.adaptive_tp() == 0.003


class TestMedianComputation:
    """Verify median is correctly computed."""

    def test_odd_number_of_samples(self):
        ot = OvershootTracker(window_size=10, min_samples=1, tp_fraction=1.0,
                              min_tp_pct=0.0)
        for v in [0.01, 0.02, 0.03, 0.04, 0.05]:
            ot.record_overshoot(v)
        # Median of [0.01, 0.02, 0.03, 0.04, 0.05] = 0.03
        assert ot.adaptive_tp() == pytest.approx(0.03)

    def test_even_number_of_samples(self):
        ot = OvershootTracker(window_size=10, min_samples=1, tp_fraction=1.0,
                              min_tp_pct=0.0)
        for v in [0.01, 0.02, 0.03, 0.04]:
            ot.record_overshoot(v)
        # Median of [0.01, 0.02, 0.03, 0.04] = (0.02+0.03)/2 = 0.025
        tp = ot.adaptive_tp()
        assert tp == pytest.approx(0.025)

    def test_single_sample(self):
        ot = OvershootTracker(window_size=10, min_samples=1, tp_fraction=1.0,
                              min_tp_pct=0.0)
        ot.record_overshoot(0.015)
        assert ot.adaptive_tp() == pytest.approx(0.015)


class TestRollingWindow:
    """Window evicts oldest samples."""

    def test_window_evicts_oldest(self):
        ot = OvershootTracker(window_size=3, min_samples=1, tp_fraction=1.0,
                              min_tp_pct=0.0)
        ot.record_overshoot(0.01)
        ot.record_overshoot(0.02)
        ot.record_overshoot(0.03)
        assert ot.count == 3
        # Add one more, evicts 0.01
        ot.record_overshoot(0.04)
        assert ot.count == 3
        # Median of [0.02, 0.03, 0.04] = 0.03
        assert ot.adaptive_tp() == pytest.approx(0.03)


class TestPercentiles:
    """Access to percentile distributions."""

    def test_percentiles_empty(self):
        ot = OvershootTracker()
        assert ot.percentiles() is None

    def test_percentiles_with_data(self):
        ot = OvershootTracker(min_samples=1)
        for v in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
            ot.record_overshoot(v)
        p = ot.percentiles()
        assert p is not None
        assert p["p50"] == pytest.approx(0.055)
        assert p["p25"] < p["p50"] < p["p75"]
        assert p["count"] == 10
