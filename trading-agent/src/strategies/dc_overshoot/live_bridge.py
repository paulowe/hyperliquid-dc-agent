"""Live bridge: mainnet BTC price data → DCOvershootStrategy → testnet trades.

Connects to Hyperliquid mainnet WebSocket for real BTC midprice ticks,
runs the DC Overshoot strategy, and executes trades on testnet.

Usage:
    # Set env vars first (or use .env file)
    export HYPERLIQUID_TESTNET_PRIVATE_KEY=0x...

    # Run the bridge
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_overshoot/live_bridge.py

    # Custom duration (default: 30 minutes)
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_overshoot/live_bridge.py --duration 60

    # Observe only (no trades)
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_overshoot/live_bridge.py --observe-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from interfaces.strategy import MarketData, SignalType
from interfaces.exchange import Order, OrderSide, OrderType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Force unbuffered output
for handler in logging.root.handlers:
    if hasattr(handler, 'stream'):
        handler.stream = sys.stdout
logger = logging.getLogger(__name__)

# Mainnet WebSocket for price data (active market)
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"

SYMBOL = "BTC"


def parse_args():
    parser = argparse.ArgumentParser(
        description="DC Overshoot: mainnet data → testnet trades"
    )
    parser.add_argument(
        "--duration", type=int, default=30,
        help="Run duration in minutes (default: 30)",
    )
    parser.add_argument(
        "--observe-only", action="store_true",
        help="Observe DC events and signals without placing trades",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.001,
        help="DC threshold (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--position-size", type=float, default=50.0,
        help="Position size in USD (default: 50)",
    )
    parser.add_argument(
        "--sl-pct", type=float, default=0.003,
        help="Initial stop loss %% as decimal (default: 0.003 = 0.3%%)",
    )
    parser.add_argument(
        "--tp-pct", type=float, default=0.002,
        help="Initial take profit %% as decimal (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--trail-pct", type=float, default=0.5,
        help="Trail lock-in %% (default: 0.5 = 50%% of profit)",
    )
    parser.add_argument(
        "--leverage", type=int, default=3,
        help="Leverage (default: 3)",
    )
    parser.add_argument(
        "--account-address", type=str, default="",
        help="Main wallet address if using API wallet delegation",
    )
    return parser.parse_args()


async def create_testnet_adapter(private_key: str, leverage: int, account_address: str = ""):
    """Create and connect a testnet adapter for placing trades.

    If account_address is provided, the API wallet (from private_key) will
    trade on behalf of the main wallet (account_address). This requires
    prior delegation setup on Hyperliquid.
    """
    from exchanges.hyperliquid.adapter import HyperliquidAdapter
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from eth_account import Account

    adapter = HyperliquidAdapter(private_key=private_key, testnet=True)

    # Custom connect with account_address support
    wallet = Account.from_key(private_key)
    base_url = "https://api.hyperliquid-testnet.xyz"
    adapter.info = Info(base_url, skip_ws=True)

    if account_address:
        adapter.exchange = Exchange(wallet, base_url, account_address=account_address)
        logger.info("API wallet %s trading on behalf of %s", wallet.address, account_address)
    else:
        adapter.exchange = Exchange(wallet, base_url)

    adapter.is_connected = True
    adapter._build_precision_cache()

    # Set leverage for BTC
    ok = await adapter.set_leverage(SYMBOL, leverage, is_cross=True)
    if ok:
        logger.info("Leverage set to %dx cross for %s", leverage, SYMBOL)
    else:
        logger.warning("Failed to set leverage — using existing setting")

    return adapter


async def execute_signal(adapter, signal, current_price: float) -> bool:
    """Execute a trading signal on testnet. Returns True if successful."""
    try:
        if signal.signal_type == SignalType.CLOSE:
            logger.info(
                "CLOSING position: %s | reason=%s",
                SYMBOL, signal.reason,
            )
            ok = await adapter.close_position(SYMBOL)
            return ok

        elif signal.signal_type in (SignalType.BUY, SignalType.SELL):
            side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            order = Order(
                id=f"bridge_{int(time.time()*1000)}",
                asset=signal.asset,
                side=side,
                order_type=OrderType.MARKET,
                size=signal.size,
                price=current_price,
            )
            logger.info(
                "PLACING %s order: %.6f %s @ ~%.2f | reason=%s",
                side.value, signal.size, signal.asset, current_price, signal.reason,
            )
            oid = await adapter.place_order(order)
            logger.info("Order placed: OID=%s", oid)
            return True

    except Exception as e:
        logger.error("Failed to execute signal: %s", e)
        return False

    return False


async def main():
    import websockets

    args = parse_args()

    # Get testnet private key from env
    private_key = os.environ.get("HYPERLIQUID_TESTNET_PRIVATE_KEY", "")
    if not private_key and not args.observe_only:
        logger.error(
            "HYPERLIQUID_TESTNET_PRIVATE_KEY not set. "
            "Use --observe-only or set the env var."
        )
        sys.exit(1)

    # Build strategy config
    config = {
        "symbol": SYMBOL,
        "dc_thresholds": [[args.threshold, args.threshold]],
        "position_size_usd": args.position_size,
        "max_position_size_usd": args.position_size * 4,
        "initial_stop_loss_pct": args.sl_pct,
        "initial_take_profit_pct": args.tp_pct,
        "trail_pct": args.trail_pct,
        "cooldown_seconds": 10,
        "max_open_positions": 1,
        "log_events": True,
    }
    strategy = DCOvershootStrategy(config)
    strategy.start()

    logger.info("=" * 70)
    logger.info("DC Overshoot Live Bridge")
    logger.info("=" * 70)
    logger.info("Price data : mainnet WebSocket (%s)", MAINNET_WS_URL)
    logger.info("Trading    : %s", "OBSERVE ONLY" if args.observe_only else "testnet")
    logger.info("Symbol     : %s", SYMBOL)
    logger.info("Threshold  : %.4f (%.2f%%)", args.threshold, args.threshold * 100)
    logger.info("Position   : $%.0f USD", args.position_size)
    logger.info("SL/TP      : %.2f%% / %.2f%%", args.sl_pct * 100, args.tp_pct * 100)
    logger.info("Trail      : %.0f%% lock-in", args.trail_pct * 100)
    logger.info("Duration   : %d minutes", args.duration)
    logger.info("=" * 70)

    # Connect testnet adapter (unless observe-only)
    adapter = None
    if not args.observe_only:
        account_addr = args.account_address or os.environ.get("TESTNET_WALLET_ADDRESS", "")
        adapter = await create_testnet_adapter(private_key, args.leverage, account_addr)

        # Show starting account value and positions
        # Use the main wallet address for Info queries when using delegation
        query_addr = account_addr or adapter.exchange.wallet.address
        user_state = adapter.info.user_state(query_addr)
        acct_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
        positions = await adapter.get_positions()
        logger.info("Testnet account value: $%.2f (wallet: %s)", acct_value, query_addr[:10] + "...")
        if positions:
            for p in positions:
                logger.info(
                    "Existing position: %s %.5f @ %.2f (PnL: $%.2f)",
                    p.asset, p.size, p.entry_price, p.unrealized_pnl,
                )
        else:
            logger.info("No existing positions")

    # Track stats
    tick_count = 0
    signal_count = 0
    trade_count = 0
    start_time = time.time()
    end_time = start_time + args.duration * 60

    logger.info("Connecting to mainnet WebSocket for %s prices...", SYMBOL)

    async with websockets.connect(MAINNET_WS_URL) as ws:
        subscribe_msg = {"method": "subscribe", "subscription": {"type": "allMids"}}
        await ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to allMids. Waiting for %s ticks...", SYMBOL)

        async for message in ws:
            if time.time() > end_time:
                logger.info("Duration reached (%d min). Stopping.", args.duration)
                break

            data = json.loads(message)
            if data.get("channel") != "allMids":
                continue

            mids = data.get("data", {}).get("mids", {})
            if SYMBOL not in mids:
                continue

            price = float(mids[SYMBOL])
            ts = time.time()
            tick_count += 1

            # Feed tick to strategy
            md = MarketData(
                asset=SYMBOL, price=price, volume_24h=0.0, timestamp=ts,
            )

            # Get positions for strategy (if adapter available)
            positions = []
            if adapter and tick_count % 10 == 0:
                # Check positions every 10 ticks to avoid rate limiting
                try:
                    positions = await adapter.get_positions()
                except Exception:
                    pass

            balance_val = 100_000.0  # Default; doesn't affect signal generation
            signals = strategy.generate_signals(md, positions, balance_val)

            for signal in signals:
                signal_count += 1
                logger.info(
                    "*** SIGNAL #%d: %s %s | price=%.2f | reason=%s",
                    signal_count, signal.signal_type.value, SYMBOL, price, signal.reason,
                )

                if adapter and not args.observe_only:
                    ok = await execute_signal(adapter, signal, price)
                    if ok:
                        trade_count += 1
                        # Notify strategy of fill
                        if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                            strategy.on_trade_executed(signal, price, signal.size)

            # Status log every 100 ticks
            if tick_count % 100 == 0:
                elapsed = ts - start_time
                rm_status = strategy._trailing_rm.get_status()
                pos_info = ""
                if rm_status.get("has_position"):
                    side = rm_status["side"]
                    entry = rm_status["entry_price"]
                    sl = rm_status["current_sl"]
                    tp = rm_status["current_tp"]
                    pnl_pct = (
                        (price - entry) / entry * 100 if side == "LONG"
                        else (entry - price) / entry * 100
                    )
                    pos_info = (
                        f" | {side} entry={entry:.2f} SL={sl:.2f} TP={tp:.2f} "
                        f"PnL={pnl_pct:+.3f}%"
                    )
                logger.info(
                    "Tick #%d | price=%.2f | signals=%d trades=%d | "
                    "elapsed=%.0fs remaining=%.0fs%s",
                    tick_count, price, signal_count, trade_count,
                    elapsed, end_time - ts, pos_info,
                )

    # Final summary
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Session complete")
    logger.info("=" * 70)
    logger.info("Duration    : %.0f seconds", elapsed)
    logger.info("Ticks       : %d", tick_count)
    logger.info("DC signals  : %d", signal_count)
    logger.info("Trades      : %d", trade_count)

    status = strategy.get_status()
    logger.info("DC events   : %d", status["dc_event_count"])
    logger.info("Trailing RM : %s", status["trailing_rm"])

    if adapter:
        # Show final positions and account value
        query_addr = (
            os.environ.get("TESTNET_WALLET_ADDRESS", "")
            or adapter.exchange.wallet.address
        )
        user_state = adapter.info.user_state(query_addr)
        acct_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
        positions = await adapter.get_positions()
        logger.info("Final account value: $%.2f", acct_value)
        for p in positions:
            logger.info(
                "Final position: %s %.5f @ %.2f (PnL: $%.2f)",
                p.asset, p.size, p.entry_price, p.unrealized_pnl,
            )
        await adapter.disconnect()

    strategy.stop()
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
