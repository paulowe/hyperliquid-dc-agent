"""Live Feature Engineer for real-time DC feature computation.

Stateful tick-by-tick adaptation of the batch FeatureEngineer in
components/vertex-components/src/vertex_components/feature_engineering.py.

For each tick, computes 6 DC features:
    - PDCC_Down: 1 if a PDCC_Down event fired on this tick, else 0
    - PDCC2_UP: 1 if a PDCC2_UP event fired on this tick, else 0
    - OSV_Down: overshoot magnitude (%) if in down regime, else 0.0
    - OSV_Up: overshoot magnitude (%) if in up regime, else 0.0
    - regime_up: 1 if current regime is "up", else 0
    - regime_down: 1 if current regime is "down", else 0
"""

from __future__ import annotations

from typing import Dict, List, Optional


class LiveFeatureEngineer:
    """Tick-by-tick DC feature computation.

    Maintains the current regime state (up/down/none) and anchor price
    from the most recent PDCC event. Computes instantaneous overshoot
    magnitude relative to the anchor.
    """

    def __init__(self) -> None:
        self._current_regime: Optional[str] = None  # "up", "down", or None
        self._anchor_price: Optional[float] = None
        self._anchor_time: Optional[float] = None

    def process_tick(
        self,
        price: float,
        timestamp: float,
        dc_events: List[Dict],
    ) -> Dict[str, float]:
        """Compute DC features for a single tick.

        Args:
            price: Current price.
            timestamp: Epoch seconds of the tick.
            dc_events: DC events emitted by LiveDCDetector for this tick.

        Returns:
            Dict with keys: PDCC_Down, PDCC2_UP, OSV_Down, OSV_Up,
            regime_up, regime_down.
        """
        # Track which PDCC events fired on this tick
        pdcc_down_fired = False
        pdcc_up_fired = False

        # Process DC events to update regime state
        for event in dc_events:
            event_type = event.get("event_type", "")

            if event_type == "PDCC_Down":
                self._current_regime = "down"
                self._anchor_price = event.get("end_price", price)
                self._anchor_time = timestamp
                pdcc_down_fired = True

            elif event_type == "PDCC2_UP":
                self._current_regime = "up"
                self._anchor_price = event.get("end_price", price)
                self._anchor_time = timestamp
                pdcc_up_fired = True

        # Compute regime indicators
        regime_up = 1 if self._current_regime == "up" else 0
        regime_down = 1 if self._current_regime == "down" else 0

        # Compute overshoot magnitudes
        osv_down = 0.0
        osv_up = 0.0

        if self._anchor_price is not None and self._anchor_price != 0.0:
            if self._current_regime == "down":
                # OSV_Down = (anchor - price) / anchor * 100
                # Positive when price is below anchor (overshoot continues)
                # Negative when price retraces above anchor
                osv_down = (self._anchor_price - price) / self._anchor_price * 100.0

            elif self._current_regime == "up":
                # OSV_Up = (price - anchor) / anchor * 100
                # Positive when price is above anchor (overshoot continues)
                # Negative when price retraces below anchor
                osv_up = (price - self._anchor_price) / self._anchor_price * 100.0

        return {
            "PDCC_Down": 1 if pdcc_down_fired else 0,
            "PDCC2_UP": 1 if pdcc_up_fired else 0,
            "OSV_Down": osv_down,
            "OSV_Up": osv_up,
            "regime_up": regime_up,
            "regime_down": regime_down,
        }
