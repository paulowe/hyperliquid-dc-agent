"""Standalone demo: DC detection on live Hyperliquid testnet midprice data.

Connects to the Hyperliquid testnet WebSocket, subscribes to allMids for BTC,
feeds midprice ticks through the LiveDCDetector, and logs DC events.

Usage:
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/testnet_demo.py
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
SYMBOL = "BTC"
THRESHOLDS = [(0.001, 0.001), (0.005, 0.005)]  # 0.1% and 0.5%
MAX_TICKS = 500  # Stop after this many ticks (for demo)


async def main():
    import websockets

    logger.info("Connecting to %s...", WS_URL)

    dc_detector = LiveDCDetector(thresholds=THRESHOLDS, symbol=SYMBOL)
    feature_engineer = LiveFeatureEngineer()
    tick_count = 0
    event_count = 0

    async with websockets.connect(WS_URL) as ws:
        # Subscribe to allMids
        subscribe_msg = {"method": "subscribe", "subscription": {"type": "allMids"}}
        await ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to allMids. Waiting for %s prices...", SYMBOL)

        async for message in ws:
            data = json.loads(message)
            if data.get("channel") != "allMids":
                continue

            mids = data.get("data", {}).get("mids", {})
            if SYMBOL not in mids:
                continue

            price = float(mids[SYMBOL])
            ts = time.time()
            tick_count += 1

            # Run DC detection
            dc_events = dc_detector.process_tick(price, ts)

            # Compute features
            features = feature_engineer.process_tick(price, ts, dc_events)

            # Log every 50 ticks
            if tick_count % 50 == 0:
                state = dc_detector.get_state()
                regimes = {
                    k: ("DOWN" if s.down_active else "UP" if s.up_active else "NONE")
                    for k, s in state.items()
                }
                logger.info(
                    "Tick #%d | price=%.2f | events_total=%d | regimes=%s | features=%s",
                    tick_count,
                    price,
                    event_count,
                    regimes,
                    {k: round(v, 4) for k, v in features.items()},
                )

            # Log DC events
            for event in dc_events:
                event_count += 1
                logger.info(
                    "DC EVENT #%d | %s | price=%.2f | start=%.2f -> end=%.2f | "
                    "threshold=(%.4f, %.4f)",
                    event_count,
                    event["event_type"],
                    price,
                    event.get("start_price", 0),
                    event.get("end_price", 0),
                    event.get("threshold_down", 0),
                    event.get("threshold_up", 0),
                )

            if tick_count >= MAX_TICKS:
                logger.info(
                    "Reached %d ticks. Total DC events: %d. Stopping.",
                    MAX_TICKS,
                    event_count,
                )
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
