"""Tests for RollingBuffer."""

import numpy as np
import pytest
from strategies.dc_forecast.rolling_buffer import RollingBuffer


class TestRollingBuffer:
    """Test fixed-size rolling buffer for feature vectors."""

    FEATURE_NAMES = ["PRICE_std", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"]

    def test_not_ready_initially(self):
        buf = RollingBuffer(window_size=50, feature_names=self.FEATURE_NAMES)
        assert buf.is_ready() is False

    def test_not_ready_partial_fill(self):
        buf = RollingBuffer(window_size=50, feature_names=self.FEATURE_NAMES)
        for i in range(49):
            buf.append({name: float(i) for name in self.FEATURE_NAMES})
        assert buf.is_ready() is False

    def test_ready_after_window_fills(self):
        buf = RollingBuffer(window_size=50, feature_names=self.FEATURE_NAMES)
        for i in range(50):
            buf.append({name: float(i) for name in self.FEATURE_NAMES})
        assert buf.is_ready() is True

    def test_window_shape(self):
        buf = RollingBuffer(window_size=50, feature_names=self.FEATURE_NAMES)
        for i in range(50):
            buf.append({name: float(i) for name in self.FEATURE_NAMES})
        window = buf.get_window()
        assert isinstance(window, np.ndarray)
        assert window.shape == (50, 7)

    def test_feature_ordering(self):
        """Features should appear in the order specified by feature_names."""
        names = ["A", "B", "C"]
        buf = RollingBuffer(window_size=2, feature_names=names)
        buf.append({"A": 1.0, "B": 2.0, "C": 3.0})
        buf.append({"A": 4.0, "B": 5.0, "C": 6.0})
        window = buf.get_window()
        # First row: [1, 2, 3], second row: [4, 5, 6]
        np.testing.assert_array_equal(window[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(window[1], [4.0, 5.0, 6.0])

    def test_rolling_behavior_oldest_dropped(self):
        """After exceeding window_size, oldest entries are dropped."""
        buf = RollingBuffer(window_size=3, feature_names=["X"])
        buf.append({"X": 1.0})
        buf.append({"X": 2.0})
        buf.append({"X": 3.0})
        buf.append({"X": 4.0})  # Should drop 1.0
        window = buf.get_window()
        np.testing.assert_array_equal(window.flatten(), [2.0, 3.0, 4.0])

    def test_still_ready_after_overflow(self):
        buf = RollingBuffer(window_size=3, feature_names=["X"])
        for i in range(10):
            buf.append({"X": float(i)})
        assert buf.is_ready() is True

    def test_get_window_not_ready_raises(self):
        """get_window() should raise when buffer is not full."""
        buf = RollingBuffer(window_size=50, feature_names=["X"])
        buf.append({"X": 1.0})
        with pytest.raises(ValueError, match="not ready"):
            buf.get_window()

    def test_current_size(self):
        buf = RollingBuffer(window_size=50, feature_names=["X"])
        assert buf.current_size == 0
        buf.append({"X": 1.0})
        assert buf.current_size == 1
        for i in range(100):
            buf.append({"X": float(i)})
        assert buf.current_size == 50  # Capped at window_size

    def test_dtype_float32(self):
        """Window should be float32 for TF model compatibility."""
        buf = RollingBuffer(window_size=2, feature_names=["X"])
        buf.append({"X": 1.0})
        buf.append({"X": 2.0})
        window = buf.get_window()
        assert window.dtype == np.float32
