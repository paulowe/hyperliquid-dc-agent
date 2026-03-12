"""Live bridge: basis trade on Hyperliquid sub-account.

Connects to Hyperliquid mainnet for price data and funding rates,
runs the basis trade strategy (long spot + short perp for funding),
and executes on the configured sub-account via vault_address.

Usage:
    uv run --package hyperliquid-trading-bot python \
        -m strategies.basis_trade.basis_bridge \
        --symbol HYPE --position-size 3 --yes

    # Observe-only (no trades):
    uv run --package hyperliquid-trading-bot python \
        -m strategies.basis_trade.basis_bridge \
        --symbol HYPE --observe-only
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

from strategies.basis_trade.basis_strategy import BasisTradeStrategy, BasisState
from strategies.basis_trade.config import BasisTradeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Force unbuffered output
for handler in logging.root.handlers:
    if hasattr(handler, "stream"):
        handler.stream = sys.stdout
logger = logging.getLogger(__name__)

# Price + funding data from mainnet
MAINNET_API_URL = "https://api.hyperliquid.xyz"
PRICE_WS_URL = "wss://api.hyperliquid.xyz/ws"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Basis trade: long spot + short perp for funding rate arbitrage"
    )
    # Core
    parser.add_argument("--symbol", type=str, default="HYPE", help="Asset to trade")
    parser.add_argument("--spot-pair", type=str, default="@107", help="Spot pair name (default: @107 for HYPE/USDC)")
    parser.add_argument("--position-size", type=float, default=3.0, help="Total capital in USD")
    parser.add_argument("--leverage", type=int, default=10, help="Perp leverage (default: 10)")
    parser.add_argument("--observe-only", action="store_true", help="Monitor funding without trading")
    parser.add_argument("--yes", action="store_true", help="Skip mainnet confirmation prompt")

    # Entry conditions
    parser.add_argument("--min-funding-rate", type=float, default=0.0001,
                        help="Min hourly funding to enter (default: 0.0001 = 0.01%%/h)")
    parser.add_argument("--min-funding-hours", type=int, default=3,
                        help="Consecutive hours above min to trigger entry (default: 3)")

    # Exit conditions
    parser.add_argument("--exit-funding-rate", type=float, default=-0.00005,
                        help="Close if funding below this (default: -0.00005)")
    parser.add_argument("--exit-funding-hours", type=int, default=6,
                        help="Consecutive hours below exit threshold (default: 6)")
    parser.add_argument("--max-hold-hours", type=float, default=0,
                        help="Max hold time in hours (default: 0 = unlimited)")
    parser.add_argument("--target-profit", type=float, default=0.0,
                        help="Close when cumulative funding reaches this USD (default: 0 = disabled)")
    parser.add_argument("--max-loss", type=float, default=0.0,
                        help="Close when cumulative funding loss exceeds this USD (default: 0 = disabled)")

    # Execution
    parser.add_argument("--slippage", type=float, default=0.002,
                        help="Max slippage for market orders (default: 0.002 = 0.2%%)")
    parser.add_argument("--check-interval", type=float, default=300,
                        help="Funding check interval in seconds (default: 300 = 5 min)")

    # Sub-account
    parser.add_argument("--sub-account", type=str, default="",
                        help="Sub-account address (default: from SUB_ACCOUNT_ADDRESS env var)")

    return parser.parse_args()


def get_info():
    """Create a read-only Info connection."""
    from hyperliquid.info import Info
    return Info(MAINNET_API_URL, skip_ws=True)


def get_exchange(vault_address: str = ""):
    """Create Exchange instance for sub-account trading.

    Uses vault_address to route orders through the master wallet's API key
    to the sub-account. The master wallet signs all transactions.
    """
    from hyperliquid.exchange import Exchange
    from eth_account import Account

    private_key = os.environ.get("HYPERLIQUID_MAINNET_PRIVATE_KEY", "")
    if not private_key:
        logger.error("HYPERLIQUID_MAINNET_PRIVATE_KEY not set")
        sys.exit(1)

    wallet = Account.from_key(private_key)

    if vault_address:
        exchange = Exchange(wallet, MAINNET_API_URL, vault_address=vault_address)
        logger.info("Trading on sub-account %s via vault_address", vault_address[:10] + "...")
    else:
        # Fall back to account_address delegation
        account_address = os.environ.get("MAINNET_ACCOUNT_ADDRESS", "")
        if account_address and account_address.lower() != wallet.address.lower():
            exchange = Exchange(wallet, MAINNET_API_URL, account_address=account_address)
            logger.info("Trading on main wallet %s via delegation", account_address[:10] + "...")
        else:
            exchange = Exchange(wallet, MAINNET_API_URL)
            logger.info("Trading on wallet %s", wallet.address[:10] + "...")

    return exchange


def get_funding_rate(info, symbol: str) -> dict:
    """Query current funding rate and prices for a symbol.

    Returns dict with: rate, mark_price, oracle_price, open_interest
    """
    meta_ctxs = info.meta_and_asset_ctxs()
    meta, ctxs = meta_ctxs[0], meta_ctxs[1]

    for i, asset in enumerate(meta["universe"]):
        if asset["name"] == symbol:
            ctx = ctxs[i]
            return {
                "rate": float(ctx.get("funding", "0")),
                "mark_price": float(ctx.get("markPx", "0")),
                "oracle_price": float(ctx.get("oraclePx", "0")),
                "open_interest": float(ctx.get("openInterest", "0")),
                "sz_decimals": asset.get("szDecimals", 2),
            }

    return {"rate": 0, "mark_price": 0, "oracle_price": 0, "open_interest": 0, "sz_decimals": 2}


def get_spot_price(info, spot_pair: str) -> float:
    """Get current spot mid price for the pair."""
    spot_data = info.spot_meta_and_asset_ctxs()
    spot_meta, spot_ctxs = spot_data[0], spot_data[1]

    for pair in spot_meta["universe"]:
        if pair["name"] == spot_pair:
            idx = pair["index"]
            if idx < len(spot_ctxs):
                ctx = spot_ctxs[idx]
                mid = ctx.get("midPx")
                if mid:
                    return float(mid)
                return float(ctx.get("markPx", "0"))

    return 0.0


def place_spot_order(exchange, spot_pair: str, is_buy: bool, size: float,
                     price: float, slippage: float) -> dict:
    """Place a spot market order (IOC with slippage)."""
    from hyperliquid.utils.signing import OrderType as HLOrderType

    # Apply slippage to price for IOC fill
    if is_buy:
        limit_px = round(price * (1 + slippage), 2)
    else:
        limit_px = round(price * (1 - slippage), 2)

    order_type = HLOrderType({"limit": {"tif": "Ioc"}})

    result = exchange.order(
        name=spot_pair,
        is_buy=is_buy,
        sz=size,
        limit_px=limit_px,
        order_type=order_type,
        reduce_only=False,
    )
    return result


def place_perp_order(exchange, symbol: str, is_buy: bool, size: float,
                     price: float, slippage: float) -> dict:
    """Place a perp market order (IOC with slippage)."""
    from hyperliquid.utils.signing import OrderType as HLOrderType

    if is_buy:
        limit_px = round(price * (1 + slippage), 1)
    else:
        limit_px = round(price * (1 - slippage), 1)

    order_type = HLOrderType({"limit": {"tif": "Ioc"}})

    result = exchange.order(
        name=symbol,
        is_buy=is_buy,
        sz=size,
        limit_px=limit_px,
        order_type=order_type,
        reduce_only=False,
    )
    return result


def close_perp_position(exchange, symbol: str, side: str, size: float,
                        price: float, slippage: float) -> dict:
    """Close a perp position with a reduce-only IOC order."""
    from hyperliquid.utils.signing import OrderType as HLOrderType

    # To close a short, we buy; to close a long, we sell
    is_buy = (side == "SHORT")
    if is_buy:
        limit_px = round(price * (1 + slippage), 1)
    else:
        limit_px = round(price * (1 - slippage), 1)

    order_type = HLOrderType({"limit": {"tif": "Ioc"}})

    result = exchange.order(
        name=symbol,
        is_buy=is_buy,
        sz=size,
        limit_px=limit_px,
        order_type=order_type,
        reduce_only=True,
    )
    return result


def check_order_result(result: dict, label: str) -> bool:
    """Check if an order was filled successfully."""
    if result.get("status") != "ok":
        logger.error("%s order failed: %s", label, result)
        return False

    statuses = result.get("response", {}).get("data", {}).get("statuses", [])
    for s in statuses:
        if "filled" in s:
            logger.info("%s order FILLED: OID=%s", label, s["filled"].get("oid"))
            return True
        if "resting" in s:
            logger.warning("%s order RESTING (not filled immediately): OID=%s", label, s["resting"].get("oid"))
            return True
        if "error" in s:
            logger.error("%s order ERROR: %s", label, s["error"])
            return False

    logger.warning("%s order unexpected status: %s", label, statuses)
    return False


async def execute_entry(exchange, info, strategy, args) -> bool:
    """Execute both legs of the basis trade entry.

    1. Buy spot HYPE
    2. Short perp HYPE (equal notional)
    """
    cfg = strategy._cfg
    spot_price = get_spot_price(info, cfg.spot_pair)
    funding = get_funding_rate(info, cfg.symbol)
    perp_price = funding["mark_price"]
    sz_decimals = funding["sz_decimals"]

    if spot_price <= 0 or perp_price <= 0:
        logger.error("Invalid prices: spot=$%.4f perp=$%.4f", spot_price, perp_price)
        return False

    # Calculate position size (equal notional on both legs)
    max_notional = cfg.max_spot_notional()
    size = round(max_notional / spot_price, sz_decimals)

    if size <= 0:
        logger.error("Position size too small: $%.2f / $%.4f = %.6f", max_notional, spot_price, size)
        return False

    logger.info("=" * 50)
    logger.info("ENTERING BASIS TRADE")
    logger.info("  Spot: BUY %.2f %s @ $%.4f ($%.2f)", size, cfg.symbol, spot_price, size * spot_price)
    logger.info("  Perp: SHORT %.2f %s @ $%.4f ($%.2f)", size, cfg.symbol, perp_price, size * perp_price)
    logger.info("  Funding rate: %.6f%%/h (%.1f%% APR)", funding["rate"] * 100, funding["rate"] * 24 * 365 * 100)
    logger.info("=" * 50)

    # Leg 1: Buy spot
    spot_result = place_spot_order(exchange, cfg.spot_pair, is_buy=True,
                                   size=size, price=spot_price, slippage=args.slippage)
    if not check_order_result(spot_result, "SPOT BUY"):
        logger.error("Spot buy failed — aborting entry (no perp leg opened)")
        return False

    # Leg 2: Short perp
    perp_result = place_perp_order(exchange, cfg.symbol, is_buy=False,
                                   size=size, price=perp_price, slippage=args.slippage)
    if not check_order_result(perp_result, "PERP SHORT"):
        logger.error("Perp short failed — SPOT LEG IS OPEN, close manually or retry!")
        # Don't return False here — we still need to track the spot leg
        # The strategy will be in a partial state but at least we track it
        return False

    # Calculate fees (taker 0.035% per side, both legs)
    taker_fee = 0.00035
    spot_fee = size * spot_price * taker_fee
    perp_fee = size * perp_price * taker_fee
    total_fees = spot_fee + perp_fee

    # Notify strategy
    strategy.on_entry_filled(
        spot_price=spot_price, spot_size=size,
        perp_price=perp_price, perp_size=size,
        total_fees=total_fees,
    )

    logger.info("Basis trade OPENED: %.2f %s | spot=$%.4f perp=$%.4f | fees=$%.4f",
                size, cfg.symbol, spot_price, perp_price, total_fees)
    return True


async def execute_exit(exchange, info, strategy, args, reason: str) -> dict:
    """Execute both legs of the basis trade exit.

    1. Sell spot HYPE
    2. Close perp short
    """
    cfg = strategy._cfg
    pos = strategy.position
    if pos is None:
        logger.warning("No position to exit")
        return {}

    spot_price = get_spot_price(info, cfg.spot_pair)
    funding = get_funding_rate(info, cfg.symbol)
    perp_price = funding["mark_price"]

    logger.info("=" * 50)
    logger.info("EXITING BASIS TRADE (reason: %s)", reason)
    logger.info("  Spot: SELL %.2f %s @ $%.4f", pos.spot_size, cfg.symbol, spot_price)
    logger.info("  Perp: CLOSE SHORT %.2f %s @ $%.4f", pos.perp_size, cfg.symbol, perp_price)
    logger.info("=" * 50)

    # Leg 1: Sell spot
    spot_result = place_spot_order(exchange, cfg.spot_pair, is_buy=False,
                                   size=pos.spot_size, price=spot_price, slippage=args.slippage)
    if not check_order_result(spot_result, "SPOT SELL"):
        logger.error("Spot sell failed — position still open!")

    # Leg 2: Close perp short
    perp_result = close_perp_position(exchange, cfg.symbol, side="SHORT",
                                      size=pos.perp_size, price=perp_price, slippage=args.slippage)
    if not check_order_result(perp_result, "PERP CLOSE"):
        logger.error("Perp close failed — position still open!")

    # Calculate exit fees
    taker_fee = 0.00035
    exit_fees = (pos.spot_size * spot_price + pos.perp_size * perp_price) * taker_fee

    # Notify strategy
    summary = strategy.on_exit_filled(
        spot_price=spot_price, perp_price=perp_price,
        total_fees=exit_fees, reason=reason,
    )

    logger.info("Basis trade CLOSED: reason=%s | price_pnl=$%.4f | funding=$%.4f | "
                "fees=$%.4f | net=$%.4f",
                reason,
                summary.get("price_pnl", 0),
                summary.get("funding_pnl", 0),
                summary.get("total_fees", 0),
                summary.get("net_pnl", 0))

    return summary


async def main():
    import websockets

    args = parse_args()

    # Build strategy config
    config = BasisTradeConfig.from_dict({
        "symbol": args.symbol,
        "spot_pair": args.spot_pair,
        "position_size_usd": args.position_size,
        "leverage": args.leverage,
        "min_funding_rate": args.min_funding_rate,
        "min_funding_hours": args.min_funding_hours,
        "exit_funding_rate": args.exit_funding_rate,
        "exit_funding_hours": args.exit_funding_hours,
        "max_hold_hours": args.max_hold_hours,
        "target_profit_usd": args.target_profit,
        "max_loss_usd": args.max_loss,
        "slippage_tolerance": args.slippage,
        "check_interval_seconds": args.check_interval,
    })

    strategy = BasisTradeStrategy(config)
    strategy.start()

    # Resolve sub-account address
    sub_account = args.sub_account or os.environ.get("SUB_ACCOUNT_ADDRESS", "")

    # Safety confirmation
    if not args.observe_only and not args.yes:
        print("\n" + "!" * 70)
        print("  WARNING: MAINNET BASIS TRADE — REAL MONEY")
        print("  Strategy: Long spot + short perp (delta-neutral funding arbitrage)")
        print("!" * 70)
        print(f"  Sub-account: {sub_account or 'MAIN ACCOUNT (no sub-account)'}")
        print(f"  Symbol: {args.symbol} | Spot pair: {args.spot_pair}")
        print(f"  Capital: ${args.position_size} | Leverage: {args.leverage}x")
        print(f"  Max notional/leg: ${config.max_spot_notional():.2f}")
        print(f"  Min entry funding: {args.min_funding_rate*100:.4f}%/h ({config.entry_apr():.0f}% APR)")
        print(f"  Exit funding: {args.exit_funding_rate*100:.4f}%/h after {args.exit_funding_hours}h")
        print()
        confirm = input("  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
        print()

    # Create connections
    info = get_info()
    exchange = None
    if not args.observe_only:
        exchange = get_exchange(vault_address=sub_account)

        # Set leverage on perp
        try:
            exchange.update_leverage(args.leverage, args.symbol, is_cross=True)
            logger.info("Leverage set to %dx cross for %s", args.leverage, args.symbol)
        except Exception as e:
            logger.warning("Failed to set leverage: %s", e)

    # Show initial state
    funding = get_funding_rate(info, args.symbol)
    spot_price = get_spot_price(info, args.spot_pair)

    logger.info("=" * 70)
    logger.info("Basis Trade Bridge")
    logger.info("=" * 70)
    logger.info("Mode       : %s", "OBSERVE ONLY" if args.observe_only else "LIVE TRADING")
    logger.info("Symbol     : %s (spot: %s)", args.symbol, args.spot_pair)
    logger.info("Sub-account: %s", sub_account[:10] + "..." if sub_account else "MAIN ACCOUNT")
    logger.info("Capital    : $%.2f | Leverage: %dx | Max/leg: $%.2f",
                args.position_size, args.leverage, config.max_spot_notional())
    logger.info("Spot price : $%.4f", spot_price)
    logger.info("Perp price : $%.4f", funding["mark_price"])
    logger.info("Funding    : %.6f%%/h (%.1f%% APR)",
                funding["rate"] * 100, funding["rate"] * 24 * 365 * 100)
    logger.info("Entry gate : funding >= %.4f%%/h for %d consecutive hours",
                args.min_funding_rate * 100, args.min_funding_hours)
    logger.info("Exit gate  : funding <= %.4f%%/h for %d consecutive hours",
                args.exit_funding_rate * 100, args.exit_funding_hours)
    logger.info("=" * 70)

    # Show sub-account balance
    if sub_account:
        try:
            query_addr = sub_account
            state = info.user_state(query_addr)
            perp_val = float(state.get("marginSummary", {}).get("accountValue", 0))
            from hyperliquid.info import Info as _Info
            spot_state = _Info(MAINNET_API_URL, skip_ws=True).post(
                "/info", {"type": "spotClearinghouseState", "user": query_addr}
            )
            spot_bals = spot_state.get("balances", [])
            usdc = next((float(b["total"]) for b in spot_bals if b["coin"] == "USDC"), 0)
            logger.info("Sub-account: perps=$%.2f spot_USDC=$%.2f", perp_val, usdc)
        except Exception as e:
            logger.warning("Could not query sub-account state: %s", e)

    # Main loop: monitor funding + manage position
    tick_count = 0
    last_funding_check = 0
    last_funding_hour = -1  # Track which hour we last recorded funding
    start_time = time.time()
    reconnect_count = 0

    # WebSocket heartbeat
    HL_PING_INTERVAL = 25

    async def hl_heartbeat(ws_conn):
        try:
            while True:
                await asyncio.sleep(HL_PING_INTERVAL)
                await ws_conn.send(json.dumps({"method": "ping"}))
        except (asyncio.CancelledError, Exception):
            pass

    subscribe_msg = json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}})

    while True:
        try:
            async with websockets.connect(
                PRICE_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                await ws.send(subscribe_msg)

                if reconnect_count == 0:
                    logger.info("Connected to WebSocket. Monitoring %s funding...", args.symbol)
                else:
                    logger.info("Reconnected (attempt #%d). Strategy state preserved.", reconnect_count)

                heartbeat_task = asyncio.create_task(hl_heartbeat(ws))

                try:
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("channel") in ("pong", None):
                            continue
                        if data.get("channel") != "allMids":
                            continue

                        mids = data.get("data", {}).get("mids", {})
                        if args.symbol not in mids:
                            continue

                        price = float(mids[args.symbol])
                        ts = time.time()
                        tick_count += 1

                        # === Periodic funding check ===
                        if ts - last_funding_check >= args.check_interval:
                            last_funding_check = ts

                            try:
                                funding = get_funding_rate(info, args.symbol)
                                rate = funding["rate"]

                                # Only record one observation per hour (funding is hourly)
                                current_hour = int(ts / 3600)
                                if current_hour != last_funding_hour:
                                    strategy.update_funding(
                                        rate=rate,
                                        mark_price=funding["mark_price"],
                                        oracle_price=funding["oracle_price"],
                                        timestamp=ts,
                                    )
                                    last_funding_hour = current_hour
                                    logger.info(
                                        "Funding update: %.6f%%/h (%.1f%% APR) | "
                                        "above=%d below=%d | state=%s",
                                        rate * 100,
                                        rate * 24 * 365 * 100,
                                        strategy.funding_monitor.consecutive_above,
                                        strategy.funding_monitor.consecutive_below,
                                        strategy.state.value,
                                    )

                                # === Entry check ===
                                if not args.observe_only and exchange and strategy.should_enter():
                                    logger.info("Entry conditions met! Funding above threshold for %d hours.",
                                                strategy.funding_monitor.consecutive_above)
                                    ok = await execute_entry(exchange, info, strategy, args)
                                    if ok:
                                        logger.info("Basis trade active. Monitoring funding for exit...")

                                # === Exit check ===
                                if not args.observe_only and exchange:
                                    exit_reason = strategy.should_exit()
                                    if exit_reason:
                                        logger.info("Exit triggered: %s", exit_reason)
                                        summary = await execute_exit(exchange, info, strategy, args, exit_reason)
                                        if summary:
                                            logger.info("Trade complete. Returning to monitoring mode.")

                            except Exception as e:
                                logger.warning("Funding check failed: %s", e)

                        # === Status log every 500 ticks ===
                        if tick_count % 500 == 0:
                            elapsed = ts - start_time
                            status = strategy.get_status()
                            fm = status["funding"]

                            pos_info = ""
                            if strategy.position:
                                pos = strategy.position
                                price_pnl = pos.net_price_pnl(price)
                                fund_pnl = strategy.funding_monitor.cumulative_funding_usd
                                hold_h = pos.hold_hours(ts)
                                pos_info = (f" | ACTIVE {hold_h:.1f}h price_pnl=${price_pnl:.4f} "
                                            f"funding=${fund_pnl:.4f}")

                            logger.info(
                                "Tick #%d | %s=$%.2f | funding=%s APR=%s | "
                                "above=%d below=%d | state=%s | elapsed=%.0fs%s",
                                tick_count, args.symbol, price,
                                fm.get("last_rate_pct", "?"),
                                fm.get("current_apr", "?"),
                                fm.get("consecutive_above", 0),
                                fm.get("consecutive_below", 0),
                                status["state"],
                                elapsed,
                                pos_info,
                            )

                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                reconnect_count += 1
                logger.info("WebSocket closed. Reconnecting...")

        except websockets.exceptions.ConnectionClosedError as e:
            reconnect_count += 1
            logger.warning("WebSocket disconnected: %s. Reconnecting...", e)
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
            reconnect_count += 1
            logger.error("Connection failed: %s. Retrying in 5s...", e)
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Shutdown requested.")
            break
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down.")
            break
        except Exception as e:
            reconnect_count += 1
            logger.error("Unexpected error: %s. Reconnecting...", e)

    # === Shutdown ===
    # If we have an active position, warn (don't auto-close)
    if strategy.state == BasisState.ACTIVE and strategy.position:
        logger.warning("=" * 50)
        logger.warning("POSITION STILL OPEN — not auto-closing on shutdown")
        logger.warning("Spot: %.2f %s | Perp short: %.2f %s",
                        strategy.position.spot_size, args.symbol,
                        strategy.position.perp_size, args.symbol)
        logger.warning("Close manually or restart the bridge.")
        logger.warning("=" * 50)

    # Final summary
    elapsed = time.time() - start_time
    status = strategy.get_status()
    logger.info("=" * 70)
    logger.info("Session complete")
    logger.info("=" * 70)
    logger.info("Duration    : %.0f seconds (%.1f hours)", elapsed, elapsed / 3600)
    logger.info("Ticks       : %d", tick_count)
    logger.info("Reconnects  : %d", reconnect_count)
    logger.info("Trades      : %d opened, %d closed", status["trades_opened"], status["trades_closed"])
    logger.info("Total P&L   : $%.4f", status["total_pnl_usd"])
    logger.info("State       : %s", status["state"])

    fm = status["funding"]
    logger.info("Funding     : %d payments, $%.4f cumulative",
                fm["total_payments"], fm["cumulative_funding_usd"])
    if fm["last_rate"] is not None:
        logger.info("Last rate   : %s (%s)", fm["last_rate_pct"], fm["current_apr"])

    strategy.stop()


if __name__ == "__main__":
    asyncio.run(main())
