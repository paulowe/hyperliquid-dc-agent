"""DC Forecast Trading Strategy.

Implements TradingStrategy interface using Directional Change detection
and TensorFlow model inference for price forecasting on Hyperliquid.

Signal flow:
    tick → DC detector → feature engineer → rolling buffer
    On PDCC event (if model loaded + buffer ready):
        buffer window → model.predict() → predicted_price_std
        inverse scale → predicted_price → delta vs current price
        delta > threshold → BUY signal
        delta < -threshold → SELL signal
    Position-aware: respects max position size, emits CLOSE before reversal.
    Cooldown: suppresses signals within cooldown_seconds of last signal.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
    TradingStrategy,
)
from strategies.dc_forecast.config import DCForecastConfig
from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_forecast.live_feature_engineer import LiveFeatureEngineer
from strategies.dc_forecast.rolling_buffer import RollingBuffer
from strategies.dc_forecast.model_loader import ScalerParams

logger = logging.getLogger(__name__)


class DCForecastStrategy(TradingStrategy):
    """Trading strategy driven by Directional Change events and ML forecasts.

    Phase 1 (no model): Detects DC events on live midprice data and logs them.
    Phase 2 (model loaded): Runs inference on PDCC events, logs predictions.
    Phase 3 (model + signals): Generates BUY/SELL/CLOSE trading signals.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("dc_forecast", config)

        # Parse config into dataclass
        self._config = DCForecastConfig.from_dict(config)

        # DC detector setup
        self._symbol = self._config.symbol
        self._log_dc_events = self._config.log_dc_events

        # Create DC detector
        self._dc_detector = LiveDCDetector(
            thresholds=self._config.dc_thresholds,
            symbol=self._symbol,
        )

        # Feature engineer for DC features
        self._feature_engineer = LiveFeatureEngineer()

        # Rolling buffer for windowed model input
        self._buffer = RollingBuffer(
            window_size=self._config.window_size,
            feature_names=self._config.feature_names,
        )

        # Model and scaler (None = Phase 1 mode, injected or loaded for Phase 2+)
        self._model: Any = None
        self._scaler_params: Optional[ScalerParams] = None

        # Counters and tracking
        self._total_events = 0
        self._pdcc_down_count = 0
        self._pdcc_up_count = 0
        self._tick_count = 0
        self._trades_executed = 0
        self._last_signal_time: Optional[float] = None
        self._last_prediction: Optional[float] = None

    @property
    def has_model(self) -> bool:
        """True if a model is loaded and ready for inference."""
        return self._model is not None and self._scaler_params is not None

    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        """Process a tick through DC detector and optionally generate signals.

        If no model is loaded (Phase 1), only logs DC events.
        If model is loaded (Phase 2+), runs inference on PDCC events and
        generates BUY/SELL/CLOSE signals based on predictions.
        """
        if market_data.asset != self._symbol:
            return []

        self._tick_count += 1
        price = market_data.price
        timestamp = market_data.timestamp

        # Feed tick to DC detector
        dc_events = self._dc_detector.process_tick(price, timestamp)

        # Compute DC features
        features = self._feature_engineer.process_tick(price, timestamp, dc_events)

        # Build feature vector and append to rolling buffer
        feature_vector = self._build_feature_vector(price, features)
        self._buffer.append(feature_vector)

        # Track DC events
        pdcc_fired = False
        for event in dc_events:
            self._total_events += 1
            event_type = event["event_type"]

            if event_type == "PDCC_Down":
                self._pdcc_down_count += 1
                pdcc_fired = True
            elif event_type == "PDCC2_UP":
                self._pdcc_up_count += 1
                pdcc_fired = True

            if self._log_dc_events:
                logger.info(
                    "[DC Event] %s | price=%.2f | start_price=%.4f -> end_price=%.4f | "
                    "threshold=(%.4f, %.4f) | tick #%d",
                    event_type,
                    price,
                    event.get("start_price", 0),
                    event.get("end_price", 0),
                    event.get("threshold_down", 0),
                    event.get("threshold_up", 0),
                    self._tick_count,
                )

        # Phase 1: No model → no signals
        if not self.has_model:
            return []

        # Phase 2+: Run inference on PDCC events when buffer is ready
        if not pdcc_fired or not self._buffer.is_ready():
            return []

        # Check cooldown
        if self._is_in_cooldown(timestamp):
            logger.debug("[DCForecast] Signal suppressed by cooldown at tick #%d", self._tick_count)
            return []

        # Run model inference
        prediction_std = self._run_inference()
        if prediction_std is None:
            return []

        self._last_prediction = prediction_std

        # Convert prediction to price delta
        predicted_price = self._scaler_params.inverse_scale_feature("PRICE", prediction_std)
        delta_pct = (predicted_price - price) / price * 100.0

        logger.info(
            "[DCForecast] Prediction | tick #%d | price=%.2f | predicted=%.2f | "
            "delta=%.4f%% | threshold=%.4f%%",
            self._tick_count,
            price,
            predicted_price,
            delta_pct,
            self._config.signal_threshold_pct,
        )

        # Generate signals based on prediction
        signals = self._generate_signals_from_prediction(
            price=price,
            predicted_price=predicted_price,
            delta_pct=delta_pct,
            timestamp=timestamp,
            positions=positions,
            balance=balance,
        )

        if signals:
            self._last_signal_time = timestamp

        return signals

    def _build_feature_vector(self, price: float, dc_features: Dict[str, float]) -> Dict[str, float]:
        """Build the full feature vector for the rolling buffer.

        Maps raw features to the standardized names expected by the buffer
        (matching feature_names from config). If a scaler is available, scales
        continuous features; otherwise uses raw values.
        """
        if self._scaler_params is not None:
            # Build raw feature dict for scaler
            raw = {
                "PRICE": price,
                "PDCC_Down": dc_features["PDCC_Down"],
                "OSV_Down": dc_features["OSV_Down"],
                "PDCC2_UP": dc_features["PDCC2_UP"],
                "OSV_Up": dc_features["OSV_Up"],
                "regime_up": dc_features["regime_up"],
                "regime_down": dc_features["regime_down"],
            }
            return self._scaler_params.apply_scaling(raw)
        else:
            # No scaler: use raw values with standardized names
            return {
                "PRICE_std": price,
                "PDCC_Down": dc_features["PDCC_Down"],
                "OSV_Down_std": dc_features["OSV_Down"],
                "PDCC2_UP": dc_features["PDCC2_UP"],
                "OSV_Up_std": dc_features["OSV_Up"],
                "regime_up": dc_features["regime_up"],
                "regime_down": dc_features["regime_down"],
            }

    def _run_inference(self) -> Optional[float]:
        """Run model inference on the current buffer window.

        Returns:
            Predicted PRICE_std value, or None if inference fails.
        """
        try:
            window = self._buffer.get_window()  # shape (window_size, n_features)
            # Model expects (batch, time_steps, features)
            input_array = window[np.newaxis, :, :]  # shape (1, window_size, n_features)
            prediction = self._model.predict(input_array, verbose=0)
            # Prediction shape: (1, 1) → extract scalar
            return float(prediction[0, 0])
        except Exception as e:
            logger.error("[DCForecast] Inference failed: %s", e)
            return None

    def _is_in_cooldown(self, current_time: float) -> bool:
        """Check if we're within the cooldown period since last signal."""
        if self._last_signal_time is None:
            return False
        elapsed = current_time - self._last_signal_time
        return elapsed < self._config.cooldown_seconds

    def _generate_signals_from_prediction(
        self,
        price: float,
        predicted_price: float,
        delta_pct: float,
        timestamp: float,
        positions: List[Position],
        balance: float,
    ) -> List[TradingSignal]:
        """Convert a model prediction into trading signals.

        Handles position awareness, close-on-reversal, and max position sizing.
        """
        signals: List[TradingSignal] = []

        threshold = self._config.signal_threshold_pct

        # Check if prediction exceeds threshold
        if abs(delta_pct) < threshold:
            return []

        # Determine desired direction
        want_buy = delta_pct > 0
        want_sell = delta_pct < 0

        # Find existing position for this asset
        current_pos = self._find_position(positions)
        current_size = current_pos.size if current_pos else 0.0
        current_value = abs(current_size * price)
        is_long = current_size > 0
        is_short = current_size < 0

        # Signal metadata for debugging
        metadata = {
            "predicted_price_std": self._last_prediction,
            "predicted_price": predicted_price,
            "prediction_delta_pct": delta_pct,
            "tick": self._tick_count,
        }

        # Close-on-reversal: if we hold a position opposite to desired direction
        if want_buy and is_short:
            signals.append(TradingSignal(
                signal_type=SignalType.CLOSE,
                asset=self._symbol,
                size=abs(current_size),
                price=None,  # Market order
                reason=f"Close short before BUY (predicted +{delta_pct:.2f}%)",
                metadata=metadata,
            ))
            current_value = 0.0  # Position will be closed
        elif want_sell and is_long:
            signals.append(TradingSignal(
                signal_type=SignalType.CLOSE,
                asset=self._symbol,
                size=abs(current_size),
                price=None,
                reason=f"Close long before SELL (predicted {delta_pct:.2f}%)",
                metadata=metadata,
            ))
            current_value = 0.0

        # Calculate position size respecting max_position_size_usd
        max_usd = self._config.max_position_size_usd
        remaining_usd = max(0.0, max_usd - current_value)

        if remaining_usd <= 0:
            return signals  # Already at max, only return CLOSE if any

        target_usd = min(self._config.position_size_usd, remaining_usd)
        order_size = target_usd / price if price > 0 else 0.0

        if order_size <= 0:
            return signals

        if want_buy:
            signals.append(TradingSignal(
                signal_type=SignalType.BUY,
                asset=self._symbol,
                size=order_size,
                price=None,  # Market order
                reason=f"DC forecast BUY: predicted +{delta_pct:.2f}%",
                metadata=metadata,
            ))
        elif want_sell:
            signals.append(TradingSignal(
                signal_type=SignalType.SELL,
                asset=self._symbol,
                size=order_size,
                price=None,
                reason=f"DC forecast SELL: predicted {delta_pct:.2f}%",
                metadata=metadata,
            ))

        return signals

    def _find_position(self, positions: List[Position]) -> Optional[Position]:
        """Find the position for this strategy's symbol."""
        for pos in positions:
            if pos.asset == self._symbol:
                return pos
        return None

    def on_trade_executed(
        self, signal: TradingSignal, executed_price: float, executed_size: float
    ) -> None:
        """Track executed trades for monitoring."""
        self._trades_executed += 1
        logger.info(
            "[DCForecast] Trade executed | %s %s %.6f @ %.2f | total_trades=%d",
            signal.signal_type.value,
            signal.asset,
            executed_size,
            executed_price,
            self._trades_executed,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return strategy status including DC detector state and signal stats."""
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
            "thresholds": self._config.dc_thresholds,
            "tick_count": self._tick_count,
            "total_dc_events": self._total_events,
            "pdcc_down_count": self._pdcc_down_count,
            "pdcc_up_count": self._pdcc_up_count,
            "regimes": regime_info,
            "has_model": self.has_model,
            "buffer_ready": self._buffer.is_ready(),
            "buffer_fill": f"{self._buffer.current_size}/{self._buffer.window_size}",
            "trades_executed": self._trades_executed,
            "last_prediction": self._last_prediction,
            "cooldown_seconds": self._config.cooldown_seconds,
        }

    def start(self) -> None:
        """Called when strategy starts."""
        super().start()
        logger.info(
            "[DCForecast] Strategy started | symbol=%s | thresholds=%s | "
            "model=%s | window=%d",
            self._symbol,
            self._config.dc_thresholds,
            "loaded" if self.has_model else "none",
            self._config.window_size,
        )

    def stop(self) -> None:
        """Called when strategy stops."""
        super().stop()
        logger.info(
            "[DCForecast] Strategy stopped | ticks=%d | events=%d (down=%d, up=%d) | "
            "trades=%d",
            self._tick_count,
            self._total_events,
            self._pdcc_down_count,
            self._pdcc_up_count,
            self._trades_executed,
        )
