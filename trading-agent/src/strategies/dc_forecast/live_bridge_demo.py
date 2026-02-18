"""Live bridge demo: mainnet data → DC detection → mock model → testnet orders.

Reads live BTC midprice from Hyperliquid mainnet, runs the DC forecast
strategy with a mock model, and places actual orders on testnet.

This validates the full production pipeline end-to-end:
    mainnet WS → allMids → DCForecastStrategy → TradingSignal
    → HyperliquidAdapter.place_order() → testnet order book

Usage:
    # Set testnet private key first
    export HYPERLIQUID_TESTNET_PRIVATE_KEY=0x...

    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/live_bridge_demo.py

    # Dry-run mode (no orders placed, just logs what would happen)
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/live_bridge_demo.py --dry-run

    # With custom threshold (more sensitive = more events)
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/live_bridge_demo.py --threshold 0.0005
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.dc_forecast.dc_forecast_strategy import DCForecastStrategy
from strategies.dc_forecast.model_loader import ScalerParams
from strategies.dc_forecast.config import DEFAULT_FEATURE_NAMES
from interfaces.strategy import MarketData, Position, SignalType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Endpoints
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"


class MockModel:
    """Mock model that predicts based on recent price momentum.

    Uses a simple heuristic: if price is trending down from max,
    predict further down (sell). If trending up from min, predict
    further up (buy). Returns scaled prediction values.
    """

    def __init__(self, momentum_sensitivity: float = 1.5):
        self._sensitivity = momentum_sensitivity

    def predict(self, x, verbose=0):
        """x shape: (1, window_size, n_features). Returns (1, 1)."""
        # Use the PRICE_std column (index 0) to compute momentum
        prices = x[0, :, 0]  # shape (window_size,)
        # Simple momentum: compare last price to mean of window
        mean_price = np.mean(prices)
        last_price = prices[-1]
        momentum = (last_price - mean_price) / (np.std(prices) + 1e-8)
        # Scale by sensitivity to produce clear BUY/SELL signals
        prediction = momentum * self._sensitivity
        return np.array([[prediction]], dtype=np.float32)


def build_mock_scaler() -> ScalerParams:
    """Build a ScalerParams that uses current BTC price range."""
    return ScalerParams(
        feature_order=["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
        mean=np.array([65000.0, 0.0, 0.0], dtype=np.float64),
        scale=np.array([5000.0, 1.0, 1.0], dtype=np.float64),
        continuous_cols=["PRICE", "OSV_Down", "OSV_Up"],
        indicator_cols=["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"],
        std_feature_order=list(DEFAULT_FEATURE_NAMES),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Live bridge: mainnet data → testnet orders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log signals but don't place real orders")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="DC threshold (default: 0.001 = 0.1%%)")
    parser.add_argument("--max-ticks", type=int, default=2000,
                        help="Stop after N ticks (default: 2000)")
    parser.add_argument("--position-size-usd", type=float, default=10.0,
                        help="Order size in USD (default: 10)")
    parser.add_argument("--private-key", type=str, default=None,
                        help="Testnet private key (or set HYPERLIQUID_TESTNET_PRIVATE_KEY)")
    return parser.parse_args()


async def main():
    import websockets

    args = parse_args()

    # Build strategy with mock model
    config = {
        "symbol": "BTC",
        "dc_thresholds": [(args.threshold, args.threshold)],
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "window_size": 50,
        "signal_threshold_pct": 0.05,  # 0.05% threshold for signals
        "position_size_usd": args.position_size_usd,
        "max_position_size_usd": args.position_size_usd * 3,
        "log_dc_events": True,
        "cooldown_seconds": 60,  # 1 minute between signals
    }

    strategy = DCForecastStrategy(config)
    strategy._model = MockModel(momentum_sensitivity=1.5)
    strategy._scaler_params = build_mock_scaler()
    strategy.start()

    # Optionally set up testnet exchange
    exchange = None
    if not args.dry_run:
        private_key = args.private_key or os.environ.get("HYPERLIQUID_TESTNET_PRIVATE_KEY")
        if not private_key:
            logger.warning("No private key provided. Running in dry-run mode.")
            args.dry_run = True
        else:
            try:
                from exchanges.hyperliquid.adapter import HyperliquidAdapter
                exchange = HyperliquidAdapter(private_key=private_key, testnet=True)
                if not await exchange.connect():
                    logger.error("Failed to connect to testnet exchange. Running dry-run.")
                    args.dry_run = True
                    exchange = None
            except Exception as e:
                logger.error("Exchange init failed: %s. Running dry-run.", e)
                args.dry_run = True

    mode = "DRY-RUN" if args.dry_run else "LIVE (testnet orders)"
    logger.info("=== Live Bridge Demo ===")
    logger.info("Mode: %s", mode)
    logger.info("Data source: mainnet (%s)", MAINNET_WS_URL)
    logger.info("DC threshold: %.4f (%.2f%%)", args.threshold, args.threshold * 100)
    logger.info("Position size: $%.2f", args.position_size_usd)
    logger.info("========================")

    # Connect to mainnet WebSocket for price data
    logger.info("Connecting to mainnet WebSocket...")

    tick_count = 0
    signal_count = 0
    order_count = 0
    positions: list = []
    start_time = time.time()

    async with websockets.connect(MAINNET_WS_URL) as ws:
        subscribe_msg = {"method": "subscribe", "subscription": {"type": "allMids"}}
        await ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to allMids on mainnet. Waiting for BTC prices...")

        async for message in ws:
            data = json.loads(message)
            if data.get("channel") != "allMids":
                continue

            mids = data.get("data", {}).get("mids", {})
            if "BTC" not in mids:
                continue

            price = float(mids["BTC"])
            ts = time.time()
            tick_count += 1

            # Feed through strategy
            md = MarketData(asset="BTC", price=price, volume_24h=0.0, timestamp=ts)
            signals = strategy.generate_signals(md, positions, 100000.0)

            # Log periodic status
            if tick_count % 100 == 0:
                elapsed = ts - start_time
                status = strategy.get_status()
                logger.info(
                    "Status | ticks=%d | events=%d | signals=%d | orders=%d | "
                    "buffer=%s | elapsed=%.0fs | price=%.2f",
                    tick_count,
                    status["total_dc_events"],
                    signal_count,
                    order_count,
                    status["buffer_fill"],
                    elapsed,
                    price,
                )

            # Process signals
            for sig in signals:
                signal_count += 1
                logger.info(
                    ">>> SIGNAL #%d | %s %s %.6f @ market | reason: %s",
                    signal_count,
                    sig.signal_type.value.upper(),
                    sig.asset,
                    sig.size,
                    sig.reason,
                )

                if sig.metadata:
                    logger.info(
                        "    Prediction: price_std=%.4f | delta=%.4f%%",
                        sig.metadata.get("predicted_price_std", 0),
                        sig.metadata.get("prediction_delta_pct", 0),
                    )

                if not args.dry_run and exchange:
                    try:
                        if sig.signal_type in (SignalType.BUY, SignalType.SELL):
                            from interfaces.exchange import Order, OrderSide, OrderType
                            order = Order(
                                id=f"dc_{int(ts * 1000)}",
                                asset=sig.asset,
                                side=OrderSide.BUY if sig.signal_type == SignalType.BUY else OrderSide.SELL,
                                size=sig.size,
                                order_type=OrderType.MARKET,
                                price=None,
                                created_at=ts,
                            )
                            oid = await exchange.place_order(order)
                            order_count += 1
                            logger.info("    ORDER PLACED on testnet | oid=%s", oid)
                            strategy.on_trade_executed(sig, price, sig.size)

                        elif sig.signal_type == SignalType.CLOSE:
                            success = await exchange.close_position(sig.asset)
                            if success:
                                order_count += 1
                                logger.info("    POSITION CLOSED on testnet")
                    except Exception as e:
                        logger.error("    Order failed: %s", e)
                else:
                    logger.info("    [dry-run] Would place %s order", sig.signal_type.value)

            # Refresh positions from exchange (if connected)
            if exchange and tick_count % 50 == 0:
                try:
                    positions = await exchange.get_positions()
                except Exception:
                    pass

            if tick_count >= args.max_ticks:
                elapsed = time.time() - start_time
                status = strategy.get_status()
                logger.info("=== Demo Complete ===")
                logger.info("Ticks: %d | Duration: %.0fs", tick_count, elapsed)
                logger.info("DC events: %d (down=%d, up=%d)",
                            status["total_dc_events"],
                            status["pdcc_down_count"],
                            status["pdcc_up_count"])
                logger.info("Signals generated: %d | Orders placed: %d",
                            signal_count, order_count)
                logger.info("Trades executed: %d", status["trades_executed"])
                break

    # Cleanup
    strategy.stop()
    if exchange:
        await exchange.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
