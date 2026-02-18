"""Rolling buffer for windowed feature vectors.

Maintains a fixed-size deque of feature dicts, providing a NumPy array
window suitable for model inference when full.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np


class RollingBuffer:
    """Fixed-size rolling buffer of feature vectors for windowed model input.

    Usage:
        buf = RollingBuffer(window_size=50, feature_names=["PRICE_std", ...])
        buf.append({"PRICE_std": 0.5, ...})
        if buf.is_ready():
            window = buf.get_window()  # shape (50, n_features), dtype float32
    """

    def __init__(self, window_size: int, feature_names: List[str]) -> None:
        """
        Args:
            window_size: Number of ticks to keep in the buffer.
            feature_names: Ordered list of feature names for column ordering.
        """
        self._window_size = window_size
        self._feature_names = list(feature_names)
        self._buffer: deque = deque(maxlen=window_size)

    def append(self, features: Dict[str, float]) -> None:
        """Append a feature vector to the buffer.

        Args:
            features: Dict mapping feature names to float values.
        """
        # Extract values in the correct column order
        row = [float(features[name]) for name in self._feature_names]
        self._buffer.append(row)

    def is_ready(self) -> bool:
        """Return True when the buffer has enough entries for a full window."""
        return len(self._buffer) >= self._window_size

    def get_window(self) -> np.ndarray:
        """Return the current buffer as a NumPy array.

        Returns:
            Array of shape (window_size, n_features), dtype float32.

        Raises:
            ValueError: If buffer is not full yet.
        """
        if not self.is_ready():
            raise ValueError(
                f"Buffer not ready: {len(self._buffer)}/{self._window_size} ticks. "
                "Call is_ready() before get_window()."
            )
        return np.array(list(self._buffer), dtype=np.float32)

    @property
    def current_size(self) -> int:
        """Number of entries currently in the buffer."""
        return len(self._buffer)

    @property
    def window_size(self) -> int:
        """Target window size."""
        return self._window_size

    @property
    def feature_names(self) -> List[str]:
        """Ordered feature names."""
        return list(self._feature_names)
