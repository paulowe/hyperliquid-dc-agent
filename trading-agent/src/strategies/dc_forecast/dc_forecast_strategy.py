"""DC Forecast Trading Strategy.

Implements TradingStrategy interface using Directional Change detection
and TensorFlow model inference for price forecasting on Hyperliquid.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
    TradingStrategy,
)
from strategies.dc_forecast.live_dc_detector import LiveDCDetector

logger = logging.getLogger(__name__)


class DCForecastStrategy(TradingStrategy):
    """Trading strategy driven by Directional Change events and ML forecasts.

    Phase 1: Detects DC events on live midprice data and logs them.
    Phase 2: Adds feature engineering + model inference on PDCC events.
    Phase 3: Generates BUY/SELL signals based on predictions.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("dc_forecast", config)

        # DC detector setup
        dc_thresholds = config.get("dc_thresholds", [(0.001, 0.001)])
        # Normalize thresholds: accept list of lists or list of tuples
        self._thresholds = [
            (float(t[0]), float(t[1])) if isinstance(t, (list, tuple)) else (float(t), float(t))
            for t in dc_thresholds
        ]
        symbol = config.get("symbol", "BTC")
        self._symbol = symbol
        self._log_dc_events = config.get("log_dc_events", True)

        # Create DC detector
        self._dc_detector = LiveDCDetector(
            thresholds=self._thresholds,
            symbol=symbol,
        )

        # Event counters for status
        self._total_events = 0
        self._pdcc_down_count = 0
        self._pdcc_up_count = 0
        self._tick_count = 0

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a tick through the DC detector and optionally generate signals.

        Phase 1: Logs DC events, returns empty signal list.
        """
        if market_data.asset != self._symbol:
            return []

        self._tick_count += 1

        # Feed tick to DC detector
        dc_events = self._dc_detector.process_tick(
            price=market_data.price,
            timestamp=market_data.timestamp,
        )

        # Log any DC events
        for event in dc_events:
            self._total_events += 1
            event_type = event["event_type"]

            if event_type == "PDCC_Down":
                self._pdcc_down_count += 1
            elif event_type == "PDCC2_UP":
                self._pdcc_up_count += 1

            if self._log_dc_events:
                logger.info(
                    "[DC Event] %s | price=%.2f | start_price=%.4f → end_price=%.4f | "
                    "threshold=(%.4f, %.4f) | tick #%d",
                    event_type,
                    market_data.price,
                    event.get("start_price", 0),
                    event.get("end_price", 0),
                    event.get("threshold_down", 0),
                    event.get("threshold_up", 0),
                    self._tick_count,
                )

        # Phase 1: No trading signals yet
        return []

    def get_status(self) -> Dict[str, Any]:
        """Return strategy status and DC detector state."""
        state = self._dc_detector.get_state()
        regime_info = {}
        for key, snap in state.items():
            regime_info[key] = {
                "down_active": snap.down_active,
                "up_active": snap.up_active,
                "max_price": snap.max_price,
                "min_price2": snap.min_price2,
            }

        return {
            "name": self.name,
            "active": self.is_active,
            "symbol": self._symbol,
            "thresholds": self._thresholds,
            "tick_count": self._tick_count,
            "total_dc_events": self._total_events,
            "pdcc_down_count": self._pdcc_down_count,
            "pdcc_up_count": self._pdcc_up_count,
            "regimes": regime_info,
        }

    def start(self) -> None:
        """Called when strategy starts."""
        super().start()
        logger.info(
            "[DCForecast] Strategy started | symbol=%s | thresholds=%s",
            self._symbol,
            self._thresholds,
        )

    def stop(self) -> None:
        """Called when strategy stops."""
        super().stop()
        logger.info(
            "[DCForecast] Strategy stopped | ticks=%d | events=%d (down=%d, up=%d)",
            self._tick_count,
            self._total_events,
            self._pdcc_down_count,
            self._pdcc_up_count,
        )
