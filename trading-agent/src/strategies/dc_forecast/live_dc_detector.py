"""Live Directional Change Detector for real-time price feeds.

Pure-Python, in-process DC detector extracted from the Apache Beam streaming
pipeline in components/dataflow-components. The core algorithm (TrendSnapshot
and _update_snapshot) is copied verbatim to avoid pulling Beam as a dependency.

Usage:
    detector = LiveDCDetector(thresholds=[(0.001, 0.001), (0.01, 0.01)])
    for price, timestamp in price_stream:
        events = detector.process_tick(price, timestamp)
        for event in events:
            print(event)  # {"event_type": "PDCC_Down", "start_price": ..., ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


@dataclass
class TrendSnapshot:
    """Serializable snapshot of the directional change state for one threshold.

    Copied from directional_change_streaming.py (lines 27-62). This dataclass
    has zero external dependencies.
    """

    threshold_down: float
    threshold_up: float
    # Down-trend tracking
    max_price: Optional[float] = None
    max_price_time: Optional[float] = None  # epoch seconds
    down_active: bool = False
    down_pdcc_price: Optional[float] = None
    down_pdcc_time: Optional[float] = None
    down_min_price: Optional[float] = None
    down_min_time: Optional[float] = None
    down_osv_price: Optional[float] = None
    down_osv_time: Optional[float] = None
    down_max_osv: float = 0.0
    # Up-trend tracking
    min_price2: Optional[float] = None
    min_price2_time: Optional[float] = None
    up_active: bool = False
    up_pdcc_price: Optional[float] = None
    up_pdcc_time: Optional[float] = None
    up_max_price: Optional[float] = None
    up_max_time: Optional[float] = None
    up_osv_price: Optional[float] = None
    up_osv_time: Optional[float] = None
    up_max_osv: float = 0.0

    def initialize(self, price: float, event_ts: float) -> None:
        if self.max_price is None:
            self.max_price = price
            self.max_price_time = event_ts
        if self.min_price2 is None:
            self.min_price2 = price
            self.min_price2_time = event_ts


class LiveDCDetector:
    """In-process directional change detector for real-time tick data.

    Maintains one TrendSnapshot per threshold pair. On each tick, updates
    all snapshots and collects emitted DC events.
    """

    def __init__(
        self,
        thresholds: List[Tuple[float, float]],
        symbol: str = "UNKNOWN",
    ) -> None:
        """
        Args:
            thresholds: List of (down_threshold, up_threshold) pairs.
            symbol: Instrument symbol for event metadata.
        """
        self._symbol = symbol
        self._snapshots: Dict[str, TrendSnapshot] = {}
        for down, up in thresholds:
            key = f"{down}:{up}"
            self._snapshots[key] = TrendSnapshot(
                threshold_down=down, threshold_up=up
            )

    def process_tick(self, price: float, timestamp: float) -> List[Dict]:
        """Feed a single price tick and return any DC events detected.

        Args:
            price: Current price (e.g., midprice).
            timestamp: Epoch seconds of the tick.

        Returns:
            List of event dicts with keys: event_type, start_time, end_time,
            start_price, end_price, symbol, threshold_down, threshold_up.
        """
        all_events: List[Dict] = []

        for key, snapshot in self._snapshots.items():
            snapshot.initialize(price, timestamp)
            events = self._update_snapshot(snapshot, price, timestamp)
            for evt in events:
                evt["symbol"] = self._symbol
                evt["threshold_down"] = snapshot.threshold_down
                evt["threshold_up"] = snapshot.threshold_up
            all_events.extend(events)

        return all_events

    def get_state(self) -> Dict[str, TrendSnapshot]:
        """Return current snapshot state for debugging/inspection."""
        return dict(self._snapshots)

    # ------------------------------------------------------------------
    # Core algorithm copied from directional_change_streaming.py:184-338
    # ------------------------------------------------------------------
    @staticmethod
    def _update_snapshot(
        snapshot: TrendSnapshot, price: float, event_ts: float
    ) -> List[Dict]:
        events: List[Dict] = []

        # Section A: Detect start of downward trend
        if not snapshot.down_active and snapshot.max_price is not None:
            price_drop = snapshot.max_price - price
            if (
                snapshot.max_price > price
                and price_drop >= snapshot.max_price * snapshot.threshold_down
            ):
                events.append(
                    LiveDCDetector._make_event(
                        event_type="PDCC_Down",
                        start_time=snapshot.max_price_time,
                        end_time=event_ts,
                        start_price=snapshot.max_price,
                        end_price=price,
                    )
                )
                snapshot.down_active = True
                snapshot.down_pdcc_price = price
                snapshot.down_pdcc_time = event_ts
                snapshot.down_min_price = price
                snapshot.down_min_time = event_ts
                snapshot.down_osv_price = price
                snapshot.down_osv_time = event_ts
                snapshot.down_max_osv = 0.0
                snapshot.up_active = False
                snapshot.up_pdcc_price = None
                snapshot.up_pdcc_time = None
                snapshot.up_max_price = None
                snapshot.up_max_time = None
                if snapshot.up_osv_time is not None and snapshot.up_pdcc_time is not None:
                    events.append(
                        LiveDCDetector._make_event(
                            event_type="OSV_Up",
                            start_time=snapshot.up_pdcc_time,
                            end_time=snapshot.up_osv_time,
                            start_price=snapshot.up_pdcc_price,
                            end_price=snapshot.up_osv_price,
                            extra={"max_osv": snapshot.up_max_osv},
                        )
                    )
                snapshot.up_osv_price = None
                snapshot.up_osv_time = None
                snapshot.up_max_osv = 0.0
                snapshot.min_price2 = None
                snapshot.min_price2_time = None

            if snapshot.max_price is None or price > snapshot.max_price:
                snapshot.max_price = price
                snapshot.max_price_time = event_ts
        elif not snapshot.down_active:
            snapshot.max_price = price
            snapshot.max_price_time = event_ts

        # Section B: Overshoot tracking while in down trend
        if snapshot.down_active and snapshot.down_pdcc_price:
            if (
                snapshot.down_min_price is None
                or price <= snapshot.down_min_price
                or snapshot.up_active
            ):
                snapshot.down_min_price = price
                snapshot.down_min_time = event_ts
                snapshot.down_osv_price = price
                snapshot.down_osv_time = event_ts

            if price > snapshot.down_min_price:
                overshoot = (snapshot.down_pdcc_price - price) / (
                    snapshot.down_pdcc_price * snapshot.threshold_down
                )
                snapshot.down_max_osv = max(snapshot.down_max_osv, overshoot)
                if overshoot < 0 and snapshot.up_active:
                    snapshot.down_active = False
                    snapshot.down_min_price = None
                    snapshot.down_min_time = None
                    snapshot.down_pdcc_price = None
                    snapshot.down_pdcc_time = None
                    snapshot.max_price = price
                    snapshot.max_price_time = event_ts

        # Section C: Detect start of upward trend
        if not snapshot.up_active and snapshot.min_price2 is not None:
            price_rise = price - snapshot.min_price2
            if (
                price > snapshot.min_price2
                and price_rise >= snapshot.min_price2 * snapshot.threshold_up
            ):
                events.append(
                    LiveDCDetector._make_event(
                        event_type="PDCC2_UP",
                        start_time=snapshot.min_price2_time,
                        end_time=event_ts,
                        start_price=snapshot.min_price2,
                        end_price=price,
                    )
                )
                snapshot.up_active = True
                snapshot.up_pdcc_price = price
                snapshot.up_pdcc_time = event_ts
                snapshot.up_max_price = price
                snapshot.up_max_time = event_ts
                snapshot.up_osv_price = price
                snapshot.up_osv_time = event_ts
                snapshot.up_max_osv = 0.0
                if (
                    snapshot.down_pdcc_time is not None
                    and snapshot.down_osv_time is not None
                ):
                    events.append(
                        LiveDCDetector._make_event(
                            event_type="OSV_Down",
                            start_time=snapshot.down_pdcc_time,
                            end_time=snapshot.down_osv_time,
                            start_price=snapshot.down_pdcc_price,
                            end_price=snapshot.down_osv_price,
                            extra={"max_osv": snapshot.down_max_osv},
                        )
                    )
                snapshot.down_active = False
                snapshot.down_pdcc_price = None
                snapshot.down_pdcc_time = None
                snapshot.down_min_price = None
                snapshot.down_min_time = None
                snapshot.down_osv_price = None
                snapshot.down_osv_time = None
                snapshot.down_max_osv = 0.0
                snapshot.max_price = None
                snapshot.max_price_time = None

            if snapshot.min_price2 is None or price < snapshot.min_price2:
                snapshot.min_price2 = price
                snapshot.min_price2_time = event_ts
        elif not snapshot.up_active:
            snapshot.min_price2 = price
            snapshot.min_price2_time = event_ts

        # Section D: Overshoot tracking while in up trend
        if snapshot.up_active and snapshot.up_pdcc_price:
            if price >= snapshot.up_max_price or snapshot.down_active:
                snapshot.up_max_price = price
                snapshot.up_max_time = event_ts
                snapshot.up_osv_price = price
                snapshot.up_osv_time = event_ts
            elif price < snapshot.up_max_price:
                overshoot = (price - snapshot.up_pdcc_price) / (
                    snapshot.up_pdcc_price * snapshot.threshold_up
                )
                snapshot.up_max_osv = max(snapshot.up_max_osv, overshoot)
                if overshoot < 0 and snapshot.down_active:
                    snapshot.up_active = False
                    snapshot.up_pdcc_price = None
                    snapshot.up_pdcc_time = None
                    snapshot.up_max_price = None
                    snapshot.up_max_time = None
                    snapshot.min_price2 = price
                    snapshot.min_price2_time = event_ts

        return [evt for evt in events if evt["start_time"] and evt["end_time"]]

    @staticmethod
    def _make_event(
        *,
        event_type: str,
        start_time: Optional[float],
        end_time: Optional[float],
        start_price: Optional[float],
        end_price: Optional[float],
        extra: Optional[Dict] = None,
    ) -> Dict:
        payload = {
            "event_type": event_type,
            "start_time": start_time,
            "end_time": end_time,
            "start_price": start_price,
            "end_price": end_price,
        }
        if extra:
            payload.update(extra)
        return payload
