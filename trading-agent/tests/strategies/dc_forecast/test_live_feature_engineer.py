"""Tests for LiveFeatureEngineer.

TDD: Tests define expected behavior for tick-by-tick DC feature computation.
Logic adapted from the batch FeatureEngineer in
components/vertex-components/src/vertex_components/feature_engineering.py
"""

import pytest
from strategies.dc_forecast.live_feature_engineer import LiveFeatureEngineer


class TestLiveFeatureEngineer:
    """Test tick-by-tick DC feature computation."""

    def test_initial_state_no_regime(self):
        """Before any PDCC event, all features should be zero."""
        fe = LiveFeatureEngineer()
        features = fe.process_tick(100.0, 1000.0, dc_events=[])
        assert features["PDCC_Down"] == 0
        assert features["PDCC2_UP"] == 0
        assert features["OSV_Down"] == 0.0
        assert features["OSV_Up"] == 0.0
        assert features["regime_up"] == 0
        assert features["regime_down"] == 0

    def test_pdcc_down_sets_regime(self):
        """After PDCC_Down event, regime should switch to down."""
        fe = LiveFeatureEngineer()
        dc_events = [
            {
                "event_type": "PDCC_Down",
                "start_price": 100.0,
                "end_price": 98.0,
                "start_time": 999.0,
                "end_time": 1000.0,
            }
        ]
        features = fe.process_tick(98.0, 1000.0, dc_events=dc_events)
        assert features["PDCC_Down"] == 1
        assert features["PDCC2_UP"] == 0
        assert features["regime_down"] == 1
        assert features["regime_up"] == 0

    def test_pdcc_up_sets_regime(self):
        """After PDCC2_UP event, regime should switch to up."""
        fe = LiveFeatureEngineer()
        dc_events = [
            {
                "event_type": "PDCC2_UP",
                "start_price": 98.0,
                "end_price": 100.0,
                "start_time": 999.0,
                "end_time": 1000.0,
            }
        ]
        features = fe.process_tick(100.0, 1000.0, dc_events=dc_events)
        assert features["PDCC2_UP"] == 1
        assert features["PDCC_Down"] == 0
        assert features["regime_up"] == 1
        assert features["regime_down"] == 0

    def test_regime_persists_between_ticks(self):
        """Regime should persist on subsequent ticks without PDCC events."""
        fe = LiveFeatureEngineer()
        # First tick: PDCC_Down
        dc_events = [
            {
                "event_type": "PDCC_Down",
                "end_price": 98.0,
                "start_price": 100.0,
                "start_time": 999.0,
                "end_time": 1000.0,
            }
        ]
        fe.process_tick(98.0, 1000.0, dc_events=dc_events)

        # Subsequent ticks with no events should still be in down regime
        features = fe.process_tick(97.5, 1001.0, dc_events=[])
        assert features["regime_down"] == 1
        assert features["regime_up"] == 0
        assert features["PDCC_Down"] == 0  # PDCC flag is only 1 on the event tick
        assert features["PDCC2_UP"] == 0

    def test_osv_down_computation(self):
        """In down regime, OSV_Down = (anchor - price) / anchor * 100."""
        fe = LiveFeatureEngineer()
        # Set down regime with anchor at 100
        dc_events = [
            {
                "event_type": "PDCC_Down",
                "end_price": 100.0,
                "start_price": 102.0,
                "start_time": 999.0,
                "end_time": 1000.0,
            }
        ]
        fe.process_tick(100.0, 1000.0, dc_events=dc_events)

        # Price drops to 98 -> OSV_Down = (100 - 98) / 100 * 100 = 2.0
        features = fe.process_tick(98.0, 1001.0, dc_events=[])
        assert abs(features["OSV_Down"] - 2.0) < 0.001
        assert features["OSV_Up"] == 0.0

    def test_osv_up_computation(self):
        """In up regime, OSV_Up = (price - anchor) / anchor * 100."""
        fe = LiveFeatureEngineer()
        # Set up regime with anchor at 100
        dc_events = [
            {
                "event_type": "PDCC2_UP",
                "end_price": 100.0,
                "start_price": 98.0,
                "start_time": 999.0,
                "end_time": 1000.0,
            }
        ]
        fe.process_tick(100.0, 1000.0, dc_events=dc_events)

        # Price rises to 102 -> OSV_Up = (102 - 100) / 100 * 100 = 2.0
        features = fe.process_tick(102.0, 1001.0, dc_events=[])
        assert abs(features["OSV_Up"] - 2.0) < 0.001
        assert features["OSV_Down"] == 0.0

    def test_regime_switch_down_to_up(self):
        """Verify clean transition from down to up regime."""
        fe = LiveFeatureEngineer()
        # Enter down regime
        fe.process_tick(
            98.0,
            1000.0,
            dc_events=[{"event_type": "PDCC_Down", "end_price": 98.0, "start_price": 100.0, "start_time": 999.0, "end_time": 1000.0}],
        )
        assert fe._current_regime == "down"

        # Switch to up regime
        features = fe.process_tick(
            101.0,
            1002.0,
            dc_events=[{"event_type": "PDCC2_UP", "end_price": 101.0, "start_price": 98.0, "start_time": 1001.0, "end_time": 1002.0}],
        )
        assert fe._current_regime == "up"
        assert features["regime_up"] == 1
        assert features["regime_down"] == 0
        assert features["PDCC2_UP"] == 1

    def test_osv_zero_outside_regime(self):
        """OSV should be 0 when no regime is active."""
        fe = LiveFeatureEngineer()
        features = fe.process_tick(100.0, 1000.0, dc_events=[])
        assert features["OSV_Down"] == 0.0
        assert features["OSV_Up"] == 0.0

    def test_osv_negative_clamped_to_actual(self):
        """OSV can be negative (retracement within regime). We report actual value."""
        fe = LiveFeatureEngineer()
        # Set up regime with anchor at 100
        fe.process_tick(
            100.0,
            1000.0,
            dc_events=[{"event_type": "PDCC2_UP", "end_price": 100.0, "start_price": 98.0, "start_time": 999.0, "end_time": 1000.0}],
        )

        # Price drops below anchor -> negative OSV_Up
        features = fe.process_tick(99.0, 1001.0, dc_events=[])
        assert features["OSV_Up"] < 0  # -1.0%
        assert abs(features["OSV_Up"] - (-1.0)) < 0.001

    def test_feature_keys_complete(self):
        """All 6 DC features should be present in output."""
        fe = LiveFeatureEngineer()
        features = fe.process_tick(100.0, 1000.0, dc_events=[])
        expected_keys = {
            "PDCC_Down",
            "PDCC2_UP",
            "OSV_Down",
            "OSV_Up",
            "regime_up",
            "regime_down",
        }
        assert set(features.keys()) == expected_keys

    def test_multiple_events_same_tick(self):
        """If both PDCC_Down and OSV_Up arrive on same tick, last PDCC wins regime."""
        fe = LiveFeatureEngineer()
        dc_events = [
            {"event_type": "OSV_Up", "end_price": 99.0, "start_price": 100.0, "start_time": 999.0, "end_time": 1000.0},
            {"event_type": "PDCC_Down", "end_price": 98.0, "start_price": 100.0, "start_time": 999.0, "end_time": 1000.0},
        ]
        features = fe.process_tick(98.0, 1000.0, dc_events=dc_events)
        # PDCC_Down should win as the regime-setting event
        assert features["regime_down"] == 1
        assert features["PDCC_Down"] == 1
