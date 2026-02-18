"""Integration test: real mainnet WebSocket data through DC forecast pipeline.

Connects to Hyperliquid mainnet, collects real BTC midprice ticks,
and validates the full pipeline (DC detection, feature engineering,
rolling buffer) works correctly on live data. No orders are placed.

Marked as slow since it requires network access and takes 30-60s.

Usage:
    # Run only this test
    uv run pytest tests/strategies/dc_forecast/test_integration_mainnet.py -v -m slow

    # Skip slow tests in normal runs
    uv run pytest tests/ -m "not slow"
"""

import asyncio
import json
import time
import logging
import pytest

from strategies.dc_forecast.dc_forecast_strategy import DCForecastStrategy
from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_forecast.live_feature_engineer import LiveFeatureEngineer
from strategies.dc_forecast.rolling_buffer import RollingBuffer
from strategies.dc_forecast.config import DEFAULT_FEATURE_NAMES
from interfaces.strategy import MarketData

logger = logging.getLogger(__name__)

MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
TICK_TIMEOUT_SECONDS = 120  # Max time to collect ticks
TARGET_TICKS = 60  # Collect 60 ticks (~60 seconds of data)


async def collect_mainnet_ticks(n_ticks: int, timeout: float) -> list:
    """Collect real BTC midprice ticks from mainnet WebSocket.

    Returns list of (price, timestamp) tuples.
    """
    import websockets

    ticks = []
    start = time.time()

    async with websockets.connect(MAINNET_WS_URL) as ws:
        subscribe_msg = {"method": "subscribe", "subscription": {"type": "allMids"}}
        await ws.send(json.dumps(subscribe_msg))

        async for message in ws:
            if time.time() - start > timeout:
                break

            data = json.loads(message)
            if data.get("channel") != "allMids":
                continue

            mids = data.get("data", {}).get("mids", {})
            if "BTC" not in mids:
                continue

            price = float(mids["BTC"])
            ts = time.time()
            ticks.append((price, ts))

            if len(ticks) >= n_ticks:
                break

    return ticks


@pytest.mark.slow
class TestMainnetIntegration:
    """Tests that require real network access to Hyperliquid mainnet."""

    @pytest.fixture(scope="class")
    def mainnet_ticks(self):
        """Collect real ticks once for all tests in this class."""
        try:
            ticks = asyncio.get_event_loop().run_until_complete(
                collect_mainnet_ticks(TARGET_TICKS, TICK_TIMEOUT_SECONDS)
            )
        except Exception:
            # Create new event loop if the default one is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ticks = loop.run_until_complete(
                collect_mainnet_ticks(TARGET_TICKS, TICK_TIMEOUT_SECONDS)
            )
        return ticks

    def test_collects_ticks_from_mainnet(self, mainnet_ticks):
        """Verify we can connect and receive BTC price data."""
        assert len(mainnet_ticks) >= 10, (
            f"Expected at least 10 ticks, got {len(mainnet_ticks)}. "
            "Is mainnet WebSocket accessible?"
        )

    def test_prices_are_reasonable(self, mainnet_ticks):
        """BTC prices should be in a reasonable range."""
        for price, ts in mainnet_ticks:
            assert 10000 < price < 500000, f"BTC price {price} seems unreasonable"
            assert ts > 1700000000, f"Timestamp {ts} seems too old"

    def test_dc_detector_on_real_data(self, mainnet_ticks):
        """DC detector should process real ticks without errors."""
        detector = LiveDCDetector(
            thresholds=[(0.001, 0.001), (0.005, 0.005)],
            symbol="BTC",
        )

        total_events = 0
        for price, ts in mainnet_ticks:
            events = detector.process_tick(price, ts)
            total_events += len(events)

            # Verify event structure
            for event in events:
                assert "event_type" in event
                assert event["event_type"] in ("PDCC_Down", "PDCC2_UP", "OSV_Down", "OSV_Up")
                assert "start_price" in event
                assert "end_price" in event
                assert event["symbol"] == "BTC"

        # State should be consistent
        state = detector.get_state()
        for key, snap in state.items():
            assert snap.max_price is not None or snap.min_price2 is not None

    def test_feature_engineer_on_real_data(self, mainnet_ticks):
        """Feature engineer should compute valid features for real data."""
        detector = LiveDCDetector(thresholds=[(0.001, 0.001)], symbol="BTC")
        engineer = LiveFeatureEngineer()

        for price, ts in mainnet_ticks:
            events = detector.process_tick(price, ts)
            features = engineer.process_tick(price, ts, events)

            # Features should be complete
            assert set(features.keys()) == {
                "PDCC_Down", "PDCC2_UP", "OSV_Down", "OSV_Up",
                "regime_up", "regime_down"
            }

            # Binary indicators should be 0 or 1
            assert features["PDCC_Down"] in (0, 1)
            assert features["PDCC2_UP"] in (0, 1)
            assert features["regime_up"] in (0, 1)
            assert features["regime_down"] in (0, 1)

            # Regimes should be mutually exclusive
            assert not (features["regime_up"] == 1 and features["regime_down"] == 1), (
                "regime_up and regime_down should not both be 1"
            )

    def test_rolling_buffer_fills_on_real_data(self, mainnet_ticks):
        """Rolling buffer should fill and produce valid windows."""
        detector = LiveDCDetector(thresholds=[(0.001, 0.001)], symbol="BTC")
        engineer = LiveFeatureEngineer()
        buffer = RollingBuffer(
            window_size=min(20, len(mainnet_ticks)),
            feature_names=list(DEFAULT_FEATURE_NAMES),
        )

        for price, ts in mainnet_ticks:
            events = detector.process_tick(price, ts)
            features = engineer.process_tick(price, ts, events)

            # Build feature vector
            fv = {
                "PRICE_std": price,  # Raw price (no scaler in this test)
                "PDCC_Down": features["PDCC_Down"],
                "OSV_Down_std": features["OSV_Down"],
                "PDCC2_UP": features["PDCC2_UP"],
                "OSV_Up_std": features["OSV_Up"],
                "regime_up": features["regime_up"],
                "regime_down": features["regime_down"],
            }
            buffer.append(fv)

        # Buffer should be full
        assert buffer.is_ready(), f"Buffer should be ready: {buffer.current_size}/{buffer.window_size}"

        # Window should have correct shape
        window = buffer.get_window()
        assert window.shape == (buffer.window_size, len(DEFAULT_FEATURE_NAMES))

        # No NaN values
        import numpy as np
        assert not np.isnan(window).any(), "Window should not contain NaN values"
        assert not np.isinf(window).any(), "Window should not contain Inf values"

    def test_full_strategy_on_real_data(self, mainnet_ticks):
        """Full DCForecastStrategy should process real data without errors."""
        config = {
            "symbol": "BTC",
            "dc_thresholds": [(0.001, 0.001), (0.005, 0.005)],
            "window_size": min(30, len(mainnet_ticks)),
            "log_dc_events": False,
            "cooldown_seconds": 0,
        }
        strategy = DCForecastStrategy(config)
        strategy.start()

        for price, ts in mainnet_ticks:
            md = MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=ts)
            # Phase 1: should return empty signals (no model)
            signals = strategy.generate_signals(md, [], 100000.0)
            assert signals == [], "Phase 1 should not generate signals"

        status = strategy.get_status()
        assert status["tick_count"] == len(mainnet_ticks)
        assert status["has_model"] is False
        assert status["buffer_ready"] is True

        strategy.stop()
