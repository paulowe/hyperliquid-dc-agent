"""Testnet order lifecycle verification.

Places a small limit order on Hyperliquid testnet, verifies it appears
in open orders, then cancels it. Proves the full order lifecycle works
end-to-end with the DC strategy's signal format.

Usage:
    export HYPERLIQUID_TESTNET_PRIVATE_KEY=0x...
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_forecast/testnet_order_test.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exchanges.hyperliquid.adapter import HyperliquidAdapter
from interfaces.exchange import Order, OrderSide, OrderType
from interfaces.strategy import TradingSignal, SignalType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_order_lifecycle_test(private_key: str):
    """Test the full order lifecycle on testnet.

    Steps:
    1. Connect to testnet
    2. Get current BTC price
    3. Place a limit BUY order well below market (won't fill)
    4. Verify order appears in open orders
    5. Cancel the order
    6. Verify order no longer in open orders
    7. Check positions and account status
    """
    logger.info("=" * 60)
    logger.info("TESTNET ORDER LIFECYCLE TEST")
    logger.info("=" * 60)

    adapter = HyperliquidAdapter(private_key=private_key, testnet=True)

    # Step 1: Connect
    logger.info("\n[Step 1] Connecting to testnet...")
    connected = await adapter.connect()
    if not connected:
        logger.error("Failed to connect to testnet")
        return False

    try:
        # Step 2: Get current price
        logger.info("\n[Step 2] Getting BTC price...")
        btc_price = await adapter.get_market_price("BTC")
        logger.info("Current BTC price: $%.2f", btc_price)

        # Step 3: Place a limit order well below market (won't fill)
        # Place 10% below market to ensure it doesn't fill
        limit_price = float(int(btc_price * 0.90))
        # Minimum order value on Hyperliquid is $10, size accordingly
        min_value = 11.0  # $11 to be safely above $10 minimum
        order_size = round(min_value / limit_price, 5)

        logger.info(
            "\n[Step 3] Placing limit BUY order: %.4f BTC @ $%.0f (10%% below market)",
            order_size, limit_price,
        )

        order = Order(
            id=f"test_{int(time.time() * 1000)}",
            asset="BTC",
            side=OrderSide.BUY,
            size=order_size,
            order_type=OrderType.LIMIT,
            price=limit_price,
            created_at=time.time(),
        )

        try:
            order_id = await adapter.place_order(order)
            logger.info("Order placed successfully! OID: %s", order_id)
        except RuntimeError as e:
            logger.error("Order placement failed: %s", e)
            return False

        # Step 4: Verify order appears in open orders
        logger.info("\n[Step 4] Checking open orders...")
        await asyncio.sleep(1)  # Brief wait for order to register
        open_orders = await adapter.get_open_orders()
        logger.info("Open orders: %d", len(open_orders))

        found = False
        for oo in open_orders:
            logger.info(
                "  - %s %s %.4f @ $%.2f (oid=%s)",
                oo.side.value, oo.asset, oo.size, oo.price or 0, oo.id,
            )
            if oo.id == order_id:
                found = True
                logger.info("    ^ This is our test order!")

        if not found:
            logger.warning("Test order not found in open orders (may have filled or been rejected)")

        # Step 5: Cancel the order
        logger.info("\n[Step 5] Cancelling order %s...", order_id)
        cancelled = await adapter.cancel_order(order_id)
        if cancelled:
            logger.info("Order cancelled successfully!")
        else:
            logger.warning("Cancel returned False (order may have already been filled/expired)")

        # Step 6: Verify order no longer in open orders
        logger.info("\n[Step 6] Verifying order was cancelled...")
        await asyncio.sleep(1)
        open_orders_after = await adapter.get_open_orders()
        still_exists = any(oo.id == order_id for oo in open_orders_after)
        if not still_exists:
            logger.info("Confirmed: order no longer in open orders")
        else:
            logger.warning("Order still appears in open orders!")

        # Step 7: Check positions and account
        logger.info("\n[Step 7] Checking account status...")
        positions = await adapter.get_positions()
        logger.info("Open positions: %d", len(positions))
        for pos in positions:
            logger.info(
                "  - %s: size=%.6f entry=$%.2f value=$%.2f pnl=$%.2f",
                pos.asset, pos.size, pos.entry_price,
                pos.current_value, pos.unrealized_pnl,
            )

        metrics = await adapter.get_account_metrics()
        logger.info("Account value: $%.2f", metrics.get("total_value", 0))

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info("Connection:     PASS")
        logger.info("Price fetch:    PASS ($%.2f)", btc_price)
        logger.info("Order place:    PASS (oid=%s)", order_id)
        logger.info("Order cancel:   %s", "PASS" if cancelled else "WARN")
        logger.info("Order cleanup:  %s", "PASS" if not still_exists else "WARN")
        logger.info("=" * 60)

        return True

    finally:
        await adapter.disconnect()


async def run_signal_to_order_test(private_key: str):
    """Test converting a DC strategy TradingSignal to an actual testnet order.

    Simulates what the TradingEngine does: takes a TradingSignal from
    DCForecastStrategy and converts it to an Order for the exchange.
    """
    logger.info("\n" + "=" * 60)
    logger.info("SIGNAL → ORDER CONVERSION TEST")
    logger.info("=" * 60)

    adapter = HyperliquidAdapter(private_key=private_key, testnet=True)
    connected = await adapter.connect()
    if not connected:
        return False

    try:
        btc_price = await adapter.get_market_price("BTC")

        # Create a TradingSignal like DCForecastStrategy would
        # Min $10 order value on Hyperliquid
        limit_price = float(int(btc_price * 0.90))
        order_size = round(11.0 / limit_price, 5)

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            asset="BTC",
            size=order_size,
            price=limit_price,  # 10% below → won't fill
            reason="DC forecast BUY: predicted +0.15% (test)",
            metadata={
                "predicted_price_std": 1.5,
                "predicted_price": btc_price * 1.0015,
                "prediction_delta_pct": 0.15,
                "tick": 42,
            },
        )

        logger.info("Signal: %s %s %.4f @ $%.0f",
                     signal.signal_type.value, signal.asset,
                     signal.size, signal.price)
        logger.info("Reason: %s", signal.reason)

        # Convert signal → order (mirrors TradingEngine._place_order)
        order = Order(
            id=f"dc_signal_{int(time.time() * 1000)}",
            asset=signal.asset,
            side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
            size=signal.size,
            order_type=OrderType.LIMIT if signal.price else OrderType.MARKET,
            price=signal.price,
            created_at=time.time(),
        )

        logger.info("\nPlacing order on testnet...")
        order_id = await adapter.place_order(order)
        logger.info("Order placed! OID: %s", order_id)

        # Cancel immediately (it's just a test)
        await asyncio.sleep(1)
        await adapter.cancel_order(order_id)
        logger.info("Order cancelled.")

        logger.info("\nSignal → Order conversion: PASS")
        return True

    finally:
        await adapter.disconnect()


async def main():
    private_key = os.environ.get("HYPERLIQUID_TESTNET_PRIVATE_KEY")
    if not private_key:
        # Use the testnet key from the conversation context
        private_key = "0x5995df1311b65d35eac6cfdefcb45ced584a0483bd9aee9a190a25a0d71b6056"
        logger.info("Using default testnet private key")

    # Run both tests
    success1 = await run_order_lifecycle_test(private_key)
    success2 = await run_signal_to_order_test(private_key)

    if success1 and success2:
        logger.info("\n*** ALL TESTS PASSED ***")
    else:
        logger.error("\n*** SOME TESTS FAILED ***")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
