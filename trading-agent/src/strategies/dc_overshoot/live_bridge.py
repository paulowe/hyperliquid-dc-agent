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
load_dotenv(_SRC_DIR.parent / ".env", override=True)

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
        "--duration", type=int, default=0,
        help="Run duration in minutes (default: 0 = run forever until Ctrl+C)",
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
        "--min-profit-to-trail-pct", type=float, default=0.001,
        help="Min raw profit before trailing ratchet activates (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--backstop-sl-pct", type=float, default=0.10,
        help="Hard stop-loss on exchange as crash protection (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip mainnet confirmation prompt",
    )
    parser.add_argument(
        "--json-report", type=str, default=None,
        help="Write structured JSON session report to this path at shutdown",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate config for safety/profitability and exit (no trading)",
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

    # Only use delegation if account_address differs from the wallet's own address
    if account_address and account_address.lower() != wallet.address.lower():
        adapter.exchange = Exchange(wallet, base_url, account_address=account_address)
        logger.info("API wallet %s trading on behalf of %s", wallet.address, account_address)
    else:
        adapter.exchange = Exchange(wallet, base_url)
        if account_address:
            logger.info("Wallet %s (no delegation needed)", wallet.address)
        else:
            logger.info("Wallet %s", wallet.address)

    adapter.is_connected = True
    adapter._build_precision_cache()

    # Set leverage for BTC
    ok = await adapter.set_leverage(symbol, leverage, is_cross=True)
    if ok:
        logger.info("Leverage set to %dx cross for %s", leverage, symbol)
    else:
        logger.warning("Failed to set leverage — using existing setting")

    return adapter


async def place_backstop_sl(adapter, symbol: str, side: str, entry_price: float,
                            size: float, backstop_pct: float) -> int | None:
    """Place a trigger stop-loss order on the exchange as a crash-protection backstop.

    Returns the order ID (oid) if successful, None otherwise.
    """
    from hyperliquid.utils.signing import OrderType as HLOrderType

    try:
        # For LONG: sell if price drops below backstop level
        # For SHORT: buy if price rises above backstop level
        if side == "LONG":
            trigger_px = entry_price * (1 - backstop_pct)
            is_buy = False  # Sell to close long
        else:
            trigger_px = entry_price * (1 + backstop_pct)
            is_buy = True   # Buy to close short

        # Round size and trigger price
        rounded_size = float(adapter._round_size(symbol, size))
        rounded_trigger = float(round(trigger_px, 1))

        # limit_px must be on the "worse" side of triggerPx for SL orders.
        # LONG SL (sell): limit below trigger for slippage room.
        # SHORT SL (buy): limit above trigger for slippage room.
        if is_buy:
            limit_px = float(round(rounded_trigger * 1.01, 1))  # 1% above trigger
        else:
            limit_px = float(round(rounded_trigger * 0.99, 1))  # 1% below trigger

        order_type = HLOrderType({"trigger": {
            "triggerPx": rounded_trigger,
            "isMarket": True,
            "tpsl": "sl",
        }})

        result = adapter.exchange.order(
            name=symbol,
            is_buy=is_buy,
            sz=rounded_size,
            limit_px=limit_px,
            order_type=order_type,
            reduce_only=True,
        )

        # Extract order ID from response
        if result.get("status") == "ok":
            statuses = result["response"]["data"]["statuses"]
            for s in statuses:
                if "resting" in s:
                    oid = s["resting"]["oid"]
                    logger.info(
                        "BACKSTOP SL placed: %s %s @ %.2f -> trigger %.2f (%.1f%%) | OID=%s",
                        side, symbol, entry_price, rounded_trigger, backstop_pct * 100, oid,
                    )
                    return oid
                if "filled" in s:
                    # Backstop triggered immediately (price already past trigger)
                    logger.warning("BACKSTOP SL triggered immediately — price already past trigger")
                    return None

        logger.warning("BACKSTOP SL order response unexpected: %s", result)
        return None

    except Exception as e:
        logger.error("Failed to place backstop SL: %s", e)
        return None


async def cancel_backstop_sl(adapter, symbol: str, backstop_oid: int | None) -> None:
    """Cancel the exchange backstop order when position closes normally."""
    if backstop_oid is None:
        return
    try:
        result = adapter.exchange.cancel(symbol, backstop_oid)
        logger.info("BACKSTOP SL cancelled: OID=%s", backstop_oid)
    except Exception as e:
        logger.warning("Failed to cancel backstop SL OID=%s: %s", backstop_oid, e)


async def reconcile_on_reconnect(
    adapter, strategy, symbol: str, backstop_oid: int | None,
) -> int | None:
    """Reconcile strategy state with exchange positions after WS reconnect.

    Handles four scenarios:
    1. Strategy + exchange both have same position → preserve, update water marks
    2. Strategy has position, exchange doesn't → backstop SL likely fired, clear state
    3. Neither has position → no-op
    4. Strategy has no position, exchange does → external position, log and ignore

    Returns the updated backstop_oid (None if position was cleared).
    """
    rm = strategy._trailing_rm

    # Fetch exchange positions via REST (adapter is independent of WS)
    try:
        positions = await adapter.get_positions()
    except Exception as e:
        logger.warning("Reconciliation failed — could not fetch positions: %s", e)
        return backstop_oid  # Keep current state, hope next reconnect works

    # Find position for our symbol
    exchange_pos = None
    for p in positions:
        if p.asset == symbol:
            exchange_pos = p
            break

    strategy_has_pos = rm.has_position
    exchange_has_pos = exchange_pos is not None and abs(exchange_pos.size) > 0

    # Scenario 1: Both have position — verify sides match
    if strategy_has_pos and exchange_has_pos:
        exchange_side = "LONG" if exchange_pos.size > 0 else "SHORT"
        if exchange_side == rm.side:
            # Position survived the outage — update water marks with current price
            try:
                current_price = float(await adapter.get_market_price(symbol))
            except Exception:
                logger.info(
                    "Reconciliation: %s position intact (entry=%.2f, size=%.5f). "
                    "Could not fetch market price for water mark update.",
                    rm.side, rm.entry_price, rm.size,
                )
                return backstop_oid

            # Ratchet water marks in favorable direction only
            if rm.side == "LONG" and rm._high_water_mark is not None:
                if current_price > rm._high_water_mark:
                    rm._high_water_mark = current_price
                    logger.info(
                        "Reconciliation: updated LONG high_water_mark to %.2f",
                        current_price,
                    )
            elif rm.side == "SHORT" and rm._low_water_mark is not None:
                if current_price < rm._low_water_mark:
                    rm._low_water_mark = current_price
                    logger.info(
                        "Reconciliation: updated SHORT low_water_mark to %.2f",
                        current_price,
                    )

            logger.info(
                "Reconciliation: %s position intact (entry=%.2f, size=%.5f)",
                rm.side, rm.entry_price, rm.size,
            )
            return backstop_oid
        else:
            # Side mismatch — clear strategy state
            logger.warning(
                "Reconciliation: side mismatch! Strategy=%s, Exchange=%s. "
                "Clearing strategy state.",
                rm.side, exchange_side,
            )
            rm.close_position()
            return None

    # Scenario 2: Strategy has position, exchange does NOT
    if strategy_has_pos and not exchange_has_pos:
        logger.warning(
            "Reconciliation: strategy had %s position but exchange has none. "
            "Backstop SL likely fired during outage. Clearing strategy state.",
            rm.side,
        )
        rm.close_position()
        return None

    # Scenario 3: Neither has position — clean state
    if not strategy_has_pos and not exchange_has_pos:
        logger.info("Reconciliation: no position on either side. Clean state.")
        return backstop_oid

    # Scenario 4: Exchange has position, strategy doesn't — external position
    if not strategy_has_pos and exchange_has_pos:
        logger.warning(
            "Reconciliation: exchange has %s position (size=%.5f) but strategy "
            "doesn't track it. External position — ignoring.",
            "LONG" if exchange_pos.size > 0 else "SHORT",
            exchange_pos.size,
        )
        return backstop_oid

    return backstop_oid


async def execute_signal(adapter, signal, current_price: float) -> bool:
    """Execute a trading signal. Returns True if successful."""
    try:
        if signal.signal_type == SignalType.CLOSE:
            # Close using signal metadata (side/size) directly instead of
            # adapter.close_position() which relies on unreliable get_positions().
            from hyperliquid.utils.signing import OrderType as HLOrderType

            pos_side = signal.metadata.get("side", "")
            close_size = signal.size
            logger.info(
                "CLOSING %s %s %.6f | reason=%s",
                pos_side, signal.asset, close_size, signal.reason,
            )

            # Close = opposite side of current position
            if pos_side == "LONG":
                is_buy = False  # Sell to close long
            elif pos_side == "SHORT":
                is_buy = True   # Buy to close short
            else:
                # Fallback if metadata missing
                logger.warning("CLOSE signal missing side metadata, falling back to close_position()")
                ok = await adapter.close_position(signal.asset)
                return ok

            # Place reduce-only IOC order directly via exchange SDK
            rounded_size = float(adapter._round_size(signal.asset, close_size))
            market_price = await adapter.get_market_price(signal.asset)
            slippage_price = market_price * (1.01 if is_buy else 0.99)
            limit_price = float(adapter._round_price(signal.asset, slippage_price, is_buy))

            result = adapter.exchange.order(
                name=signal.asset,
                is_buy=is_buy,
                sz=rounded_size,
                limit_px=limit_price,
                order_type=HLOrderType({"limit": {"tif": "Ioc"}}),
                reduce_only=True,
            )

            if result and result.get("status") == "ok":
                statuses = result["response"]["data"]["statuses"]
                for s in statuses:
                    if "filled" in s:
                        logger.info("Close filled: %s", s["filled"])
                        return True
                    if "resting" in s:
                        logger.info("Close order resting: OID=%s", s["resting"]["oid"])
                        return True
                logger.warning("Close order status unexpected: %s", statuses)
                return False
            else:
                logger.error("Close order failed: %s", result)
                return False

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

    # --validate: check config and exit early
    if args.validate:
        from strategies.dc_overshoot.validate import validate_dc_config

        result = validate_dc_config(
            threshold=args.threshold,
            sl_pct=args.sl_pct,
            tp_pct=args.tp_pct,
            backstop_sl_pct=args.backstop_sl_pct,
            leverage=args.leverage,
            position_size_usd=args.position_size,
            trail_pct=args.trail_pct,
            min_profit_to_trail_pct=args.min_profit_to_trail_pct,
        )
        print(result.format())
        sys.exit(1 if result.has_errors else 0)

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
        "min_profit_to_trail_pct": args.min_profit_to_trail_pct,
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
    logger.info("Backstop SL: %.1f%% (hard stop on exchange)", args.backstop_sl_pct * 100)
    logger.info("Duration   : %s", f"{args.duration} minutes" if args.duration > 0 else "unlimited (Ctrl+C to stop)")
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
    backstop_oid = None  # Exchange-level backstop stop-loss order ID
    signal_log = []  # Accumulated signals for --json-report
    start_time = time.time()
    # duration=0 means run forever
    end_time = start_time + args.duration * 60 if args.duration > 0 else float("inf")

    logger.info("Connecting to mainnet WebSocket for %s prices...", symbol)

    # Reconnection state
    reconnect_count = 0
    subscribe_msg = json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}})
    duration_reached = False

    # Hyperliquid JSON heartbeat: server drops connections idle for 60s.
    # Send {"method": "ping"} every 25s as documented by Hyperliquid.
    # The websockets library's frame-level ping_interval additionally provides
    # client-side staleness detection (raises ConnectionClosedError on timeout).
    HL_PING_INTERVAL = 25  # seconds, must be < 60s server idle timeout

    async def hl_heartbeat(ws_conn):
        """Send Hyperliquid JSON pings to prevent server-side idle timeout."""
        try:
            while True:
                await asyncio.sleep(HL_PING_INTERVAL)
                await ws_conn.send(json.dumps({"method": "ping"}))
        except (asyncio.CancelledError, Exception):
            pass  # Task cancelled on disconnect or shutdown — expected

    while not duration_reached:
        # Check if duration has expired before (re)connecting
        if time.time() > end_time:
            logger.info("Duration reached (%d min). Not reconnecting.", args.duration)
            break

        # Reconcile state with exchange before reconnecting (skip first connection)
        if reconnect_count > 0 and adapter is not None:
            logger.info("=" * 40)
            logger.info("RECONNECT #%d — reconciling state...", reconnect_count)
            logger.info("=" * 40)
            backstop_oid = await reconcile_on_reconnect(
                adapter, strategy, symbol, backstop_oid,
            )

        try:
            async with websockets.connect(
                PRICE_WS_URL,
                ping_interval=20,   # WS frame-level keep-alive (client-side staleness detection)
                ping_timeout=20,    # Raises ConnectionClosedError if no pong within 20s
                close_timeout=5,    # Graceful close timeout
            ) as ws:
                await ws.send(subscribe_msg)

                if reconnect_count == 0:
                    logger.info("Subscribed to allMids. Waiting for %s ticks...", symbol)
                else:
                    logger.info(
                        "Reconnected and resubscribed. Strategy state preserved "
                        "(ticks=%d, signals=%d, trades=%d)",
                        tick_count, signal_count, trade_count,
                    )

                # Start Hyperliquid JSON heartbeat as background task
                heartbeat_task = asyncio.create_task(hl_heartbeat(ws))

                try:
                    async for message in ws:
                        if time.time() > end_time:
                            logger.info("Duration reached (%d min). Stopping.", args.duration)
                            duration_reached = True
                            break

                        data = json.loads(message)

                        # Ignore pong responses from our JSON heartbeat
                        if data.get("channel") == "pong":
                            continue

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

                            # Accumulate for --json-report
                            if args.json_report:
                                signal_log.append({
                                    "timestamp": ts,
                                    "type": signal.signal_type.value,
                                    "price": price,
                                    "size": signal.size,
                                    "reason": signal.reason,
                                    "metadata": {
                                        k: v for k, v in (signal.metadata or {}).items()
                                        if k != "dc_event"  # Skip verbose DC event data
                                    },
                                })

                            if adapter and not args.observe_only:
                                # Cancel backstop before closing or reversing
                                needs_cancel = (
                                    backstop_oid is not None
                                    and (
                                        signal.signal_type == SignalType.CLOSE
                                        or signal.metadata.get("reversal")
                                    )
                                )
                                if needs_cancel:
                                    await cancel_backstop_sl(adapter, symbol, backstop_oid)
                                    backstop_oid = None

                                ok = await execute_signal(adapter, signal, price)
                                if ok:
                                    trade_count += 1
                                    # On entry fill: notify strategy + place backstop
                                    if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                                        strategy.on_trade_executed(signal, price, signal.size)
                                        side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
                                        # For reversals, backstop uses new position size (not 2x order size)
                                        backstop_size = signal.metadata.get("new_position_size", signal.size)
                                        backstop_oid = await place_backstop_sl(
                                            adapter, symbol, side, price,
                                            backstop_size, args.backstop_sl_pct,
                                        )

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
                            remaining = f"remaining=%.0fs" % (end_time - ts,) if end_time != float("inf") else "∞"
                            rc_info = f" rc={reconnect_count}" if reconnect_count > 0 else ""
                            logger.info(
                                "Tick #%d | price=%.2f | signals=%d trades=%d | "
                                "elapsed=%.0fs %s%s%s",
                                tick_count, price, signal_count, trade_count,
                                elapsed, remaining, pos_info, rc_info,
                            )
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                # Inner loop exited without duration_reached → WS closed cleanly by server
                if not duration_reached:
                    reconnect_count += 1
                    logger.info(
                        "WebSocket closed by server. Reconnecting immediately...",
                    )

        except websockets.exceptions.ConnectionClosedError as e:
            reconnect_count += 1
            code = e.rcvd.code if e.rcvd else "?"
            reason = e.rcvd.reason if e.rcvd else "?"
            logger.warning(
                "WebSocket disconnected (code=%s, reason=%s). "
                "Reconnecting immediately...",
                code, reason,
            )
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
            reconnect_count += 1
            logger.error(
                "Connection failed: %s. Reconnecting immediately...", e,
            )
        except asyncio.CancelledError:
            # Clean shutdown (Ctrl+C)
            logger.info("Shutdown requested.")
            break
        except Exception as e:
            reconnect_count += 1
            logger.error(
                "Unexpected WS error: %s. Reconnecting immediately...", e,
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
    logger.info("Reconnects  : %d", reconnect_count)

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

    # Write JSON report if requested
    if args.json_report:
        elapsed = time.time() - start_time
        report = {
            "symbol": symbol,
            "threshold": args.threshold,
            "sl_pct": args.sl_pct,
            "tp_pct": args.tp_pct,
            "trail_pct": args.trail_pct,
            "min_profit_to_trail_pct": args.min_profit_to_trail_pct,
            "position_size_usd": args.position_size,
            "leverage": args.leverage,
            "observe_only": args.observe_only,
            "duration_seconds": elapsed,
            "start_time": start_time,
            "end_time": time.time(),
            "tick_count": tick_count,
            "signal_count": signal_count,
            "trade_count": trade_count,
            "dc_event_count": status["dc_event_count"],
            "reconnect_count": reconnect_count,
            "signals": signal_log,
            "strategy_status": status,
        }
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to tmp file, then rename
        tmp_path = report_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(report, f, indent=2)
        tmp_path.rename(report_path)
        logger.info("JSON report written to %s", report_path)

    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
