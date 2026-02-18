"""Standalone demo: DC detection on live Hyperliquid midprice data.

Connects to a Hyperliquid WebSocket (mainnet or testnet), subscribes to
allMids for BTC, feeds midprice ticks through the LiveDCDetector and
LiveFeatureEngineer, and logs DC events with full feature vectors.

Uses mainnet by default for active price action (testnet prices are static).

Usage:
    # Mainnet (default) - for observing live DC events
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/testnet_demo.py

    # Testnet - for testing
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/testnet_demo.py --testnet

    # Custom max ticks
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/testnet_demo.py --max-ticks 2000
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.dc_forecast.live_dc_detector import LiveDCDetector
from strategies.dc_forecast.live_feature_engineer import LiveFeatureEngineer
from strategies.dc_forecast.rolling_buffer import RollingBuffer
from strategies.dc_forecast.config import DEFAULT_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Mainnet endpoints (active market data)
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
# Testnet endpoints (quiet, for trading tests)
TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"

SYMBOL = "BTC"
THRESHOLDS = [(0.001, 0.001), (0.005, 0.005)]  # 0.1% and 0.5%


def parse_args():
    parser = argparse.ArgumentParser(description="DC detection on live Hyperliquid data")
    parser.add_argument("--testnet", action="store_true", help="Use testnet instead of mainnet")
    parser.add_argument("--max-ticks", type=int, default=1000, help="Stop after N ticks (default: 1000)")
    parser.add_argument("--symbol", default="BTC", help="Symbol to track (default: BTC)")
    parser.add_argument("--window-size", type=int, default=50, help="Rolling buffer window size (default: 50)")
    return parser.parse_args()


async def main():
    import websockets

    args = parse_args()
    ws_url = TESTNET_WS_URL if args.testnet else MAINNET_WS_URL
    symbol = args.symbol
    max_ticks = args.max_ticks
    network = "testnet" if args.testnet else "mainnet"

    logger.info("Connecting to %s (%s)...", ws_url, network)

    dc_detector = LiveDCDetector(thresholds=THRESHOLDS, symbol=symbol)
    feature_engineer = LiveFeatureEngineer()

    # Rolling buffer to demonstrate windowed feature accumulation
    # Use PRICE_std as placeholder (raw price) + 6 DC features
    feature_names = list(DEFAULT_FEATURE_NAMES)
    buffer = RollingBuffer(window_size=args.window_size, feature_names=feature_names)

    tick_count = 0
    event_count = 0
    start_time = time.time()

    async with websockets.connect(ws_url) as ws:
        # Subscribe to allMids
        subscribe_msg = {"method": "subscribe", "subscription": {"type": "allMids"}}
        await ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to allMids. Waiting for %s prices on %s...", symbol, network)

        async for message in ws:
            data = json.loads(message)
            if data.get("channel") != "allMids":
                continue

            mids = data.get("data", {}).get("mids", {})
            if symbol not in mids:
                continue

            price = float(mids[symbol])
            ts = time.time()
            tick_count += 1

            # Run DC detection
            dc_events = dc_detector.process_tick(price, ts)

            # Compute features
            features = feature_engineer.process_tick(price, ts, dc_events)

            # Build feature vector for rolling buffer (PRICE_std uses raw price as placeholder)
            feature_vector = {"PRICE_std": price}
            for k in ["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"]:
                feature_vector[k] = features[k]
            # Map OSV features to their _std names (raw values, no scaler yet)
            feature_vector["OSV_Down_std"] = features["OSV_Down"]
            feature_vector["OSV_Up_std"] = features["OSV_Up"]
            buffer.append(feature_vector)

            # Log every 50 ticks
            if tick_count % 50 == 0:
                state = dc_detector.get_state()
                regimes = {
                    k: ("DOWN" if s.down_active else "UP" if s.up_active else "NONE")
                    for k, s in state.items()
                }
                elapsed = ts - start_time
                logger.info(
                    "Tick #%d | price=%.2f | events=%d | buffer=%d/%d | "
                    "elapsed=%.0fs | regimes=%s | features=%s",
                    tick_count,
                    price,
                    event_count,
                    buffer.current_size,
                    buffer.window_size,
                    elapsed,
                    regimes,
                    {k: round(v, 4) for k, v in features.items()},
                )

            # Log DC events immediately when they occur
            for event in dc_events:
                event_count += 1
                logger.info(
                    "*** DC EVENT #%d | %s | price=%.2f | start=%.2f -> end=%.2f | "
                    "threshold=(%.4f, %.4f) | buffer_ready=%s",
                    event_count,
                    event["event_type"],
                    price,
                    event.get("start_price", 0),
                    event.get("end_price", 0),
                    event.get("threshold_down", 0),
                    event.get("threshold_up", 0),
                    buffer.is_ready(),
                )

            if tick_count >= max_ticks:
                elapsed = time.time() - start_time
                logger.info(
                    "Reached %d ticks in %.0fs. Total DC events: %d. "
                    "Buffer: %d/%d. Stopping.",
                    max_ticks,
                    elapsed,
                    event_count,
                    buffer.current_size,
                    buffer.window_size,
                )
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
