"""Live bridge: mainnet BTC price data → DCOvershootStrategy → trades.

Connects to Hyperliquid mainnet WebSocket for real BTC midprice ticks,
runs the DC Overshoot strategy, and executes trades on the configured network.

Network selection is controlled by HYPERLIQUID_NETWORK in .env:
  - "testnet" → trades on testnet (fake money)
  - "mainnet" → trades on mainnet (real money, requires confirmation)

Usage:
    # Configure .env first (set HYPERLIQUID_NETWORK, keys, wallet addresses)
    # Then run:
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_overshoot/live_bridge.py

    # Custom duration and risk params:
    uv run --package hyperliquid-trading-bot python \
        trading-agent/src/strategies/dc_overshoot/live_bridge.py \
        --duration 60 --sl-pct 0.10 --tp-pct 0.10

    # Observe only (no trades):
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
_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

# Load .env from trading-agent/ root
from dotenv import load_dotenv
load_dotenv(_SRC_DIR.parent / ".env")

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

# Price data always comes from mainnet (the only active market)
PRICE_WS_URL = "wss://api.hyperliquid.xyz/ws"

DEFAULT_SYMBOL = "BTC"

# Network-specific constants
NETWORK_CONFIG = {
    "testnet": {
        "base_url": "https://api.hyperliquid-testnet.xyz",
        "key_env": "HYPERLIQUID_TESTNET_PRIVATE_KEY",
        "wallet_env": "TESTNET_WALLET_ADDRESS",
        "label": "testnet",
    },
    "mainnet": {
        "base_url": "https://api.hyperliquid.xyz",
        "key_env": "HYPERLIQUID_MAINNET_PRIVATE_KEY",
        "wallet_env": "MAINNET_WALLET_ADDRESS",
        "label": "MAINNET (real money!)",
    },
}


def get_network() -> str:
    """Read HYPERLIQUID_NETWORK from env, default to testnet."""
    network = os.environ.get("HYPERLIQUID_NETWORK", "testnet").lower().strip()
    if network not in NETWORK_CONFIG:
        logger.error("HYPERLIQUID_NETWORK must be 'testnet' or 'mainnet', got '%s'", network)
        sys.exit(1)
    return network


def parse_args():
    parser = argparse.ArgumentParser(
        description="DC Overshoot: mainnet price data → strategy → trade execution"
    )
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help="Asset to trade (default: BTC). Must match Hyperliquid symbol.",
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
        "--yes", action="store_true",
        help="Skip mainnet confirmation prompt",
    )
    return parser.parse_args()


async def create_adapter(network: str, private_key: str, leverage: int, symbol: str, account_address: str = ""):
    """Create and connect an adapter for the specified network.

    If account_address is provided, the API wallet (from private_key) will
    trade on behalf of the main wallet (account_address). This requires
    prior delegation setup on Hyperliquid.
    """
    from exchanges.hyperliquid.adapter import HyperliquidAdapter
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from eth_account import Account

    net_cfg = NETWORK_CONFIG[network]
    is_testnet = (network == "testnet")
    base_url = net_cfg["base_url"]

    adapter = HyperliquidAdapter(private_key=private_key, testnet=is_testnet)

    # Custom connect with account_address delegation support
    wallet = Account.from_key(private_key)
    adapter.info = Info(base_url, skip_ws=True)

    if account_address:
        adapter.exchange = Exchange(wallet, base_url, account_address=account_address)
        logger.info("API wallet %s trading on behalf of %s", wallet.address, account_address)
    else:
        adapter.exchange = Exchange(wallet, base_url)

    adapter.is_connected = True
    adapter._build_precision_cache()

    # Set leverage for BTC
    ok = await adapter.set_leverage(symbol, leverage, is_cross=True)
    if ok:
        logger.info("Leverage set to %dx cross for %s", leverage, symbol)
    else:
        logger.warning("Failed to set leverage — using existing setting")

    return adapter


async def execute_signal(adapter, signal, current_price: float) -> bool:
    """Execute a trading signal. Returns True if successful."""
    try:
        if signal.signal_type == SignalType.CLOSE:
            logger.info(
                "CLOSING position: %s | reason=%s",
                signal.asset, signal.reason,
            )
            ok = await adapter.close_position(signal.asset)
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
    network = get_network()
    net_cfg = NETWORK_CONFIG[network]
    symbol = args.symbol

    # Get private key for the selected network
    private_key = os.environ.get(net_cfg["key_env"], "")
    if not private_key and not args.observe_only:
        logger.error(
            "%s not set. Set it in .env or use --observe-only.",
            net_cfg["key_env"],
        )
        sys.exit(1)

    # Get wallet address for delegation
    account_address = os.environ.get(net_cfg["wallet_env"], "")

    # Mainnet safety confirmation
    if network == "mainnet" and not args.observe_only and not args.yes:
        print("\n" + "!" * 70)
        print("  WARNING: You are about to trade on MAINNET with REAL MONEY")
        print("!" * 70)
        print(f"  Wallet: {account_address or 'API wallet (no delegation)'}")
        print(f"  SL: {args.sl_pct*100:.1f}%  TP: {args.tp_pct*100:.1f}%  Size: ${args.position_size}")
        print()
        confirm = input("  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
        print()

    # Build strategy config
    config = {
        "symbol": symbol,
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
    logger.info("Price data : mainnet WebSocket (%s)", PRICE_WS_URL)
    logger.info("Trading    : %s", "OBSERVE ONLY" if args.observe_only else net_cfg["label"])
    logger.info("Symbol     : %s", symbol)
    logger.info("Threshold  : %.4f (%.2f%%)", args.threshold, args.threshold * 100)
    logger.info("Position   : $%.0f USD", args.position_size)
    logger.info("SL/TP      : %.2f%% / %.2f%%", args.sl_pct * 100, args.tp_pct * 100)
    logger.info("Trail      : %.0f%% lock-in", args.trail_pct * 100)
    logger.info("Duration   : %d minutes", args.duration)
    logger.info("=" * 70)

    # Connect adapter (unless observe-only)
    adapter = None
    if not args.observe_only:
        adapter = await create_adapter(network, private_key, args.leverage, symbol, account_address)

        # Show starting account value and positions
        query_addr = account_address or adapter.exchange.wallet.address
        user_state = adapter.info.user_state(query_addr)
        acct_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
        positions = await adapter.get_positions()
        logger.info("Account value: $%.2f (wallet: %s)", acct_value, query_addr[:10] + "...")
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

    logger.info("Connecting to mainnet WebSocket for %s prices...", symbol)

    async with websockets.connect(PRICE_WS_URL) as ws:
        subscribe_msg = {"method": "subscribe", "subscription": {"type": "allMids"}}
        await ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to allMids. Waiting for %s ticks...", symbol)

        async for message in ws:
            if time.time() > end_time:
                logger.info("Duration reached (%d min). Stopping.", args.duration)
                break

            data = json.loads(message)
            if data.get("channel") != "allMids":
                continue

            mids = data.get("data", {}).get("mids", {})
            if symbol not in mids:
                continue

            price = float(mids[symbol])
            ts = time.time()
            tick_count += 1

            # Feed tick to strategy
            md = MarketData(
                asset=symbol, price=price, volume_24h=0.0, timestamp=ts,
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
                    signal_count, signal.signal_type.value, symbol, price, signal.reason,
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
        query_addr = account_address or adapter.exchange.wallet.address
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
