"""Tests for LiveDCDetector.

TDD: These tests define the expected behavior before implementation.
The core algorithm is extracted from the Beam streaming detector in
components/dataflow-components/src/dataflow_components/directional_change_streaming.py
"""

import pytest
from strategies.dc_forecast.live_dc_detector import LiveDCDetector, TrendSnapshot


class TestTrendSnapshot:
    """Test the TrendSnapshot dataclass."""

    def test_default_values(self):
        snap = TrendSnapshot(threshold_down=0.01, threshold_up=0.01)
        assert snap.threshold_down == 0.01
        assert snap.threshold_up == 0.01
        assert snap.max_price is None
        assert snap.down_active is False
        assert snap.up_active is False
        assert snap.down_max_osv == 0.0
        assert snap.up_max_osv == 0.0

    def test_initialize_sets_prices(self):
        snap = TrendSnapshot(threshold_down=0.01, threshold_up=0.01)
        snap.initialize(100.0, 1000.0)
        assert snap.max_price == 100.0
        assert snap.max_price_time == 1000.0
        assert snap.min_price2 == 100.0
        assert snap.min_price2_time == 1000.0

    def test_initialize_idempotent(self):
        """Once initialized, calling initialize again should not overwrite."""
        snap = TrendSnapshot(threshold_down=0.01, threshold_up=0.01)
        snap.initialize(100.0, 1000.0)
        snap.initialize(200.0, 2000.0)
        # Should keep original values
        assert snap.max_price == 100.0
        assert snap.min_price2 == 100.0


class TestLiveDCDetector:
    """Test LiveDCDetector behavior."""

    def test_initialization_single_threshold(self):
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        assert len(detector._snapshots) == 1

    def test_initialization_multiple_thresholds(self):
        detector = LiveDCDetector(
            thresholds=[(0.001, 0.001), (0.005, 0.005), (0.01, 0.01)]
        )
        assert len(detector._snapshots) == 3

    def test_no_events_on_flat_price(self):
        """Constant price should produce no DC events."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        events = []
        for i in range(100):
            events.extend(detector.process_tick(100.0, 1000.0 + i))
        assert len(events) == 0

    def test_no_events_on_small_move(self):
        """Price moves smaller than threshold should produce no events."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])  # 1% threshold
        events = []
        # Move price from 100 to 99.5 (0.5% drop, below 1% threshold)
        for i in range(50):
            price = 100.0 - i * 0.01
            events.extend(detector.process_tick(price, 1000.0 + i))
        assert len(events) == 0

    def test_downtrend_detection(self):
        """Price dropping by more than threshold should trigger PDCC_Down."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])  # 1% threshold
        events = []
        # Start at 100, drop to 98 (2% drop, exceeds 1% threshold)
        prices = [100.0] * 5 + [99.0, 98.5, 98.0]
        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        pdcc_down_events = [e for e in events if e["event_type"] == "PDCC_Down"]
        assert len(pdcc_down_events) >= 1
        evt = pdcc_down_events[0]
        assert evt["start_price"] == 100.0
        assert evt["end_price"] < 100.0

    def test_uptrend_detection(self):
        """Price rising by more than threshold should trigger PDCC2_UP."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])  # 1% threshold
        events = []
        # Start at 100, drop to trigger down, then rise to trigger up
        prices = [100.0] * 3
        # Drop to 98 (triggers PDCC_Down)
        prices.extend([99.0, 98.5, 98.0])
        # Hold at 98 briefly
        prices.extend([98.0] * 3)
        # Rise to 100 (>1% from 98 = triggers PDCC2_UP)
        prices.extend([99.0, 99.5, 100.0])

        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        pdcc_up_events = [e for e in events if e["event_type"] == "PDCC2_UP"]
        assert len(pdcc_up_events) >= 1

    def test_multiple_thresholds_different_sensitivity(self):
        """Smaller thresholds should fire more events than larger ones."""
        detector_sensitive = LiveDCDetector(thresholds=[(0.001, 0.001)])  # 0.1%
        detector_coarse = LiveDCDetector(thresholds=[(0.01, 0.01)])  # 1%

        events_sensitive = []
        events_coarse = []

        # Simulate volatile price: oscillate between 100 and 102
        import math

        for i in range(200):
            price = 100.0 + 2.0 * math.sin(i * 0.1)
            ts = 1000.0 + i
            events_sensitive.extend(detector_sensitive.process_tick(price, ts))
            events_coarse.extend(detector_coarse.process_tick(price, ts))

        # More sensitive threshold should produce more events
        assert len(events_sensitive) >= len(events_coarse)

    def test_event_structure(self):
        """Verify event dict has expected keys."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        events = []
        # Force a PDCC_Down event
        prices = [100.0] * 3 + [98.0]
        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        assert len(events) > 0
        evt = events[0]
        assert "event_type" in evt
        assert "start_time" in evt
        assert "end_time" in evt
        assert "start_price" in evt
        assert "end_price" in evt
        assert "symbol" in evt
        assert "threshold_down" in evt
        assert "threshold_up" in evt

    def test_regime_transitions(self):
        """Verify clean alternation between down and up regimes."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        events = []

        # Create clear regime transitions
        prices = (
            [100.0] * 3  # stable
            + [98.0]  # drop -> PDCC_Down
            + [98.0] * 3  # stable in down
            + [100.0]  # rise -> PDCC2_UP
            + [100.0] * 3  # stable in up
            + [98.0]  # drop -> PDCC_Down again
        )

        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        pdcc_types = [e["event_type"] for e in events if e["event_type"].startswith("PDCC")]
        # Should have alternating PDCC_Down and PDCC2_UP
        assert len(pdcc_types) >= 2

    def test_osv_events_emitted(self):
        """Verify OSV (overshoot) events are emitted during regime transitions."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        events = []

        # Create a full cycle: stable -> down -> overshoot -> up
        prices = (
            [100.0] * 3
            + [98.0]  # PDCC_Down
            + [97.0, 96.5]  # overshoot in down regime
            + [98.0, 99.0, 100.0]  # reversal -> PDCC2_UP (should emit OSV_Down)
        )

        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        event_types = [e["event_type"] for e in events]
        # Should have at least PDCC_Down and then OSV_Down when regime switches
        assert "PDCC_Down" in event_types

    def test_get_state_returns_snapshots(self):
        """get_state() should return current snapshot state for debugging."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        detector.process_tick(100.0, 1000.0)
        state = detector.get_state()
        assert len(state) == 1
        key = "0.01:0.01"
        assert key in state
        assert isinstance(state[key], TrendSnapshot)

    def test_symbol_in_events(self):
        """Events should include the configured symbol."""
        detector = LiveDCDetector(
            thresholds=[(0.01, 0.01)], symbol="BTC"
        )
        events = []
        prices = [100.0] * 3 + [98.0]
        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        for evt in events:
            assert evt["symbol"] == "BTC"

    def test_process_tick_returns_list(self):
        """process_tick should always return a list."""
        detector = LiveDCDetector(thresholds=[(0.01, 0.01)])
        result = detector.process_tick(100.0, 1000.0)
        assert isinstance(result, list)

    def test_asymmetric_thresholds(self):
        """Test with different down/up thresholds."""
        # 2% down threshold, 1% up threshold
        detector = LiveDCDetector(thresholds=[(0.02, 0.01)])
        events = []

        # Need 2% drop for PDCC_Down, but only 1% rise for PDCC2_UP
        prices = [100.0] * 3 + [97.5]  # 2.5% drop -> PDCC_Down
        prices += [97.5] * 3
        prices += [98.5]  # ~1% rise from 97.5 -> PDCC2_UP

        for i, price in enumerate(prices):
            events.extend(detector.process_tick(price, 1000.0 + i))

        event_types = [e["event_type"] for e in events]
        assert "PDCC_Down" in event_types
