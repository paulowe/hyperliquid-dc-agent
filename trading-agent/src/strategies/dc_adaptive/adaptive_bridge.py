"""Live bridge: mainnet price data → DCAdaptiveStrategy → trades.

Connects to Hyperliquid mainnet WebSocket for real midprice ticks,
runs the DC Adaptive strategy (regime detection + adaptive TP + loss guards),
and executes trades on the configured network.

This is a standalone bridge — it does NOT modify live_bridge.py.

Usage:
    uv run --package hyperliquid-trading-bot python \
        -m strategies.dc_adaptive.adaptive_bridge \
        --symbol HYPE --threshold 0.015 \
        --sensor-threshold 0.004 \
        --sl-pct 0.02 --tp-pct 0.005 \
        --trail-pct 0.3 --min-profit-to-trail-pct 0.003 \
        --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 --leverage 10 --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import NamedTuple

# Add src to path for imports
_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

# Load .env from trading-agent/ root
from dotenv import load_dotenv
load_dotenv(_SRC_DIR.parent / ".env", override=True)

from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy
from interfaces.strategy import MarketData, SignalType
from interfaces.exchange import Order, OrderSide, OrderType
from telemetry.collector import NullCollector, TelemetryCollector
from telemetry.events import EventType

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

DEFAULT_SYMBOL = "HYPE"


class BackstopOids(NamedTuple):
    """OIDs for exchange-level backstop orders (SL and TP)."""
    sl_oid: int | None
    tp_oid: int | None


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
        description="DC Adaptive: mainnet price data → regime-aware strategy → trade execution"
    )
    # --- Core trading parameters ---
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help="Asset to trade (default: HYPE). Must match Hyperliquid symbol.",
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
        "--threshold", type=float, default=0.015,
        help="Trade-level DC threshold (default: 0.015 = 1.5%%)",
    )
    parser.add_argument(
        "--position-size", type=float, default=50.0,
        help="Position size in USD (default: 50)",
    )
    parser.add_argument(
        "--sl-pct", type=float, default=0.02,
        help="Initial stop loss %% as decimal (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--tp-pct", type=float, default=0.005,
        help="Default take profit %% before adaptive kicks in (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--trail-pct", type=float, default=0.3,
        help="Trail lock-in %% (default: 0.3 = 30%% of profit)",
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Leverage (default: 10)",
    )
    parser.add_argument(
        "--min-profit-to-trail-pct", type=float, default=0.003,
        help="Min raw profit before trailing ratchet activates (default: 0.003 = 0.3%%)",
    )
    parser.add_argument(
        "--backstop-sl-pct", type=float, default=0.05,
        help="Hard stop-loss on exchange as crash protection (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--backstop-tp-pct", type=float, default=0.05,
        help="Hard take-profit on exchange as crash protection (default: 0.05 = 5%%)",
    )

    # --- Regime detector parameters ---
    parser.add_argument(
        "--sensor-threshold", type=float, default=0.004,
        help="DC threshold for regime sensing (default: 0.004 = 0.4%%)",
    )
    parser.add_argument(
        "--lookback-seconds", type=float, default=600.0,
        help="Regime detector lookback window in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--choppy-rate-threshold", type=float, default=4.0,
        help="Sensor events/min above which market is 'choppy' (default: 4.0)",
    )
    parser.add_argument(
        "--trending-consistency", type=float, default=0.6,
        help="Min directional agreement for 'trending' regime (default: 0.6)",
    )

    # --- Overshoot tracker parameters ---
    parser.add_argument(
        "--tp-fraction", type=float, default=0.4,
        help="Fraction of median overshoot to use as adaptive TP (default: 0.4)",
    )
    parser.add_argument(
        "--min-tp-pct", type=float, default=0.003,
        help="Floor for adaptive TP (default: 0.003 = 0.3%%)",
    )
    parser.add_argument(
        "--os-window-size", type=int, default=20,
        help="Rolling window of overshoots to track (default: 20)",
    )
    parser.add_argument(
        "--os-min-samples", type=int, default=5,
        help="Min overshoot samples before adaptive TP activates (default: 5)",
    )

    # --- Loss streak guard parameters ---
    parser.add_argument(
        "--max-consecutive-losses", type=int, default=3,
        help="Consecutive losses before cooldown triggers (default: 3)",
    )
    parser.add_argument(
        "--base-cooldown-seconds", type=float, default=300.0,
        help="Base cooldown duration multiplier in seconds (default: 300 = 5 min)",
    )

    # --- Operational flags ---
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip mainnet confirmation prompt",
    )
    parser.add_argument(
        "--json-report", type=str, default=None,
        help="Write structured JSON session report to this path at shutdown",
    )
    parser.add_argument(
        "--telemetry", action="store_true",
        help="Enable structured telemetry (writes NDJSON to ~/.cache/hyperliquid-telemetry/)",
    )
    parser.add_argument(
        "--telemetry-dir", type=str, default=None,
        help="Custom telemetry output directory",
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

    # Set leverage
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
        if side == "LONG":
            trigger_px = entry_price * (1 - backstop_pct)
            is_buy = False
        else:
            trigger_px = entry_price * (1 + backstop_pct)
            is_buy = True

        rounded_size = float(adapter._round_size(symbol, size))
        rounded_trigger = float(round(trigger_px, 1))

        if is_buy:
            limit_px = float(round(rounded_trigger * 1.01, 1))
        else:
            limit_px = float(round(rounded_trigger * 0.99, 1))

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
                    logger.warning("BACKSTOP SL triggered immediately — price already past trigger")
                    return None

        logger.warning("BACKSTOP SL order response unexpected: %s", result)
        return None

    except Exception as e:
        logger.error("Failed to place backstop SL: %s", e)
        return None


async def cancel_backstop_sl(adapter, symbol: str, backstop_oid: int | None) -> None:
    """Cancel the exchange backstop SL order when position closes normally."""
    if backstop_oid is None:
        return
    try:
        result = adapter.exchange.cancel(symbol, backstop_oid)
        logger.info("BACKSTOP SL cancelled: OID=%s", backstop_oid)
    except Exception as e:
        logger.warning("Failed to cancel backstop SL OID=%s: %s", backstop_oid, e)


async def place_backstop_tp(adapter, symbol: str, side: str, entry_price: float,
                            size: float, backstop_pct: float) -> int | None:
    """Place a trigger take-profit order on the exchange as a crash-protection backstop.

    Returns the order ID (oid) if successful, None otherwise.
    """
    from hyperliquid.utils.signing import OrderType as HLOrderType

    try:
        if side == "LONG":
            trigger_px = entry_price * (1 + backstop_pct)
            is_buy = False
        else:
            trigger_px = entry_price * (1 - backstop_pct)
            is_buy = True

        rounded_size = float(adapter._round_size(symbol, size))
        rounded_trigger = float(round(trigger_px, 1))

        if is_buy:
            limit_px = float(round(rounded_trigger * 1.01, 1))
        else:
            limit_px = float(round(rounded_trigger * 0.99, 1))

        order_type = HLOrderType({"trigger": {
            "triggerPx": rounded_trigger,
            "isMarket": True,
            "tpsl": "tp",
        }})

        result = adapter.exchange.order(
            name=symbol,
            is_buy=is_buy,
            sz=rounded_size,
            limit_px=limit_px,
            order_type=order_type,
            reduce_only=True,
        )

        if result.get("status") == "ok":
            statuses = result["response"]["data"]["statuses"]
            for s in statuses:
                if "resting" in s:
                    oid = s["resting"]["oid"]
                    logger.info(
                        "BACKSTOP TP placed: %s %s @ %.2f -> trigger %.2f (%.1f%%) | OID=%s",
                        side, symbol, entry_price, rounded_trigger, backstop_pct * 100, oid,
                    )
                    return oid
                if "filled" in s:
                    logger.warning("BACKSTOP TP triggered immediately — price already past trigger")
                    return None

        logger.warning("BACKSTOP TP order response unexpected: %s", result)
        return None

    except Exception as e:
        logger.error("Failed to place backstop TP: %s", e)
        return None


async def cancel_backstop_tp(adapter, symbol: str, backstop_tp_oid: int | None) -> None:
    """Cancel the exchange backstop TP order when position closes normally."""
    if backstop_tp_oid is None:
        return
    try:
        result = adapter.exchange.cancel(symbol, backstop_tp_oid)
        logger.info("BACKSTOP TP cancelled: OID=%s", backstop_tp_oid)
    except Exception as e:
        logger.warning("Failed to cancel backstop TP OID=%s: %s", backstop_tp_oid, e)


async def reconcile_on_reconnect(
    adapter, strategy, symbol: str, backstop_oids: BackstopOids,
) -> BackstopOids:
    """Reconcile strategy state with exchange positions after WS reconnect.

    Handles four scenarios:
    1. Strategy + exchange both have same position → preserve, update water marks
    2. Strategy has position, exchange doesn't → backstop fired, cancel orphans, clear state
    3. Neither has position → no-op
    4. Strategy has no position, exchange does → external position, log and ignore

    Returns the updated BackstopOids (both None if position was cleared).
    """
    rm = strategy._trailing_rm
    cleared = BackstopOids(sl_oid=None, tp_oid=None)

    try:
        positions = await adapter.get_positions()
    except Exception as e:
        logger.warning("Reconciliation failed — could not fetch positions: %s", e)
        return backstop_oids

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
            try:
                current_price = float(await adapter.get_market_price(symbol))
            except Exception:
                logger.info(
                    "Reconciliation: %s position intact (entry=%.2f, size=%.5f). "
                    "Could not fetch market price for water mark update.",
                    rm.side, rm.entry_price, rm.size,
                )
                return backstop_oids

            if rm.side == "LONG" and rm._high_water_mark is not None:
                if current_price > rm._high_water_mark:
                    rm._high_water_mark = current_price
                    logger.info("Reconciliation: updated LONG high_water_mark to %.2f", current_price)
            elif rm.side == "SHORT" and rm._low_water_mark is not None:
                if current_price < rm._low_water_mark:
                    rm._low_water_mark = current_price
                    logger.info("Reconciliation: updated SHORT low_water_mark to %.2f", current_price)

            logger.info(
                "Reconciliation: %s position intact (entry=%.2f, size=%.5f)",
                rm.side, rm.entry_price, rm.size,
            )
            return backstop_oids
        else:
            logger.warning(
                "Reconciliation: side mismatch! Strategy=%s, Exchange=%s. Clearing strategy state.",
                rm.side, exchange_side,
            )
            rm.close_position()
            return cleared

    # Scenario 2: Strategy has position, exchange does NOT
    if strategy_has_pos and not exchange_has_pos:
        logger.warning(
            "Reconciliation: strategy had %s position but exchange has none. "
            "Backstop likely fired during outage. Clearing strategy state.",
            rm.side,
        )
        if backstop_oids.sl_oid is not None:
            await cancel_backstop_sl(adapter, symbol, backstop_oids.sl_oid)
        if backstop_oids.tp_oid is not None:
            await cancel_backstop_tp(adapter, symbol, backstop_oids.tp_oid)
        rm.close_position()
        return cleared

    # Scenario 3: Neither has position — clean state
    if not strategy_has_pos and not exchange_has_pos:
        logger.info("Reconciliation: no position on either side. Clean state.")
        return backstop_oids

    # Scenario 4: Exchange has position, strategy doesn't — external position
    if not strategy_has_pos and exchange_has_pos:
        logger.warning(
            "Reconciliation: exchange has %s position (size=%.5f) but strategy "
            "doesn't track it. External position — ignoring.",
            "LONG" if exchange_pos.size > 0 else "SHORT",
            exchange_pos.size,
        )
        return backstop_oids

    return backstop_oids


async def execute_signal(adapter, signal, current_price: float) -> bool:
    """Execute a trading signal. Returns True if successful."""
    try:
        if signal.signal_type == SignalType.CLOSE:
            from hyperliquid.utils.signing import OrderType as HLOrderType

            pos_side = signal.metadata.get("side", "")
            close_size = signal.size
            logger.info(
                "CLOSING %s %s %.6f | reason=%s",
                pos_side, signal.asset, close_size, signal.reason,
            )

            if pos_side == "LONG":
                is_buy = False
            elif pos_side == "SHORT":
                is_buy = True
            else:
                logger.warning("CLOSE signal missing side metadata, falling back to close_position()")
                ok = await adapter.close_position(signal.asset)
                return ok

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
    # MAINNET_ACCOUNT_ADDRESS is the delegated-to main wallet (has the funds)
    # MAINNET_WALLET_ADDRESS is a legacy alias — check both
    account_address = os.environ.get("MAINNET_ACCOUNT_ADDRESS", "") or os.environ.get(net_cfg["wallet_env"], "")

    # Mainnet safety confirmation
    if network == "mainnet" and not args.observe_only and not args.yes:
        print("\n" + "!" * 70)
        print("  WARNING: You are about to trade on MAINNET with REAL MONEY")
        print("  Strategy: DC Adaptive (regime detection + adaptive TP + loss guards)")
        print("!" * 70)
        print(f"  Wallet: {account_address or 'API wallet (no delegation)'}")
        print(f"  Threshold: {args.threshold*100:.2f}%  Sensor: {args.sensor_threshold*100:.2f}%")
        print(f"  SL: {args.sl_pct*100:.1f}%  TP: {args.tp_pct*100:.1f}% (default, adaptive overrides)")
        print(f"  Size: ${args.position_size}  Leverage: {args.leverage}x")
        print()
        confirm = input("  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
        print()

    # Build strategy config with all adaptive parameters
    config = {
        "symbol": symbol,
        "dc_thresholds": [[args.threshold, args.threshold]],
        "sensor_threshold": [args.sensor_threshold, args.sensor_threshold],
        "position_size_usd": args.position_size,
        "max_position_size_usd": args.position_size * 4,
        "initial_stop_loss_pct": args.sl_pct,
        "initial_take_profit_pct": args.tp_pct,
        "trail_pct": args.trail_pct,
        "min_profit_to_trail_pct": args.min_profit_to_trail_pct,
        "cooldown_seconds": 10,
        "max_open_positions": 1,
        "log_events": True,
        # Regime detector
        "lookback_seconds": args.lookback_seconds,
        "choppy_rate_threshold": args.choppy_rate_threshold,
        "trending_consistency_threshold": args.trending_consistency,
        # Overshoot tracker
        "os_window_size": args.os_window_size,
        "os_min_samples": args.os_min_samples,
        "tp_fraction": args.tp_fraction,
        "min_tp_pct": args.min_tp_pct,
        "default_tp_pct": args.tp_pct,
        # Loss streak guard
        "max_consecutive_losses": args.max_consecutive_losses,
        "base_cooldown_seconds": args.base_cooldown_seconds,
    }
    strategy = DCAdaptiveStrategy(config)
    strategy.start()

    # Initialize telemetry collector
    if args.telemetry:
        telem = TelemetryCollector(
            symbol=symbol,
            bridge_type="dc_adaptive",
            local_dir=Path(args.telemetry_dir) if args.telemetry_dir else None,
        )
        logger.info("Telemetry enabled (session=%s)", telem.session_id)
    else:
        telem = NullCollector()

    # Register DC event callback for telemetry (includes regime updates on sensor events)
    sensor_key = f"{args.sensor_threshold}:{args.sensor_threshold}"

    def _on_dc_event(event: dict) -> None:
        telem.emit(EventType.DC_EVENT, event)
        # Emit regime update when sensor threshold fires
        threshold_key = f"{event.get('threshold_down', 0)}:{event.get('threshold_up', 0)}"
        if threshold_key == sensor_key:
            telem.emit(EventType.MOMENTUM_UPDATE, {
                "regime": strategy._regime.classify(time.time()),
                "event_rate": strategy._regime.event_rate(time.time()),
                "adaptive_tp": strategy._os_tracker.adaptive_tp(),
                "consecutive_losses": strategy._loss_guard.consecutive_losses,
            })

    strategy.set_dc_event_callback(_on_dc_event)

    logger.info("=" * 70)
    logger.info("DC Adaptive Live Bridge")
    logger.info("=" * 70)
    logger.info("Price data : mainnet WebSocket (%s)", PRICE_WS_URL)
    logger.info("Trading    : %s", "OBSERVE ONLY" if args.observe_only else net_cfg["label"])
    logger.info("Symbol     : %s", symbol)
    logger.info("Threshold  : %.4f (%.2f%%)", args.threshold, args.threshold * 100)
    logger.info("Sensor     : %.4f (%.2f%%)", args.sensor_threshold, args.sensor_threshold * 100)
    logger.info("Position   : $%.0f USD @ %dx", args.position_size, args.leverage)
    logger.info("SL/TP      : %.2f%% / %.2f%% (default, adaptive overrides TP)", args.sl_pct * 100, args.tp_pct * 100)
    logger.info("Trail      : %.0f%% lock-in (min profit: %.2f%%)", args.trail_pct * 100, args.min_profit_to_trail_pct * 100)
    logger.info("Backstop   : SL=%.1f%% TP=%.1f%%", args.backstop_sl_pct * 100, args.backstop_tp_pct * 100)
    logger.info("Regime     : lookback=%ds choppy_rate=%.1f trending_consist=%.1f",
                int(args.lookback_seconds), args.choppy_rate_threshold, args.trending_consistency)
    logger.info("OS Tracker : window=%d min_samples=%d tp_fraction=%.1f min_tp=%.2f%%",
                args.os_window_size, args.os_min_samples, args.tp_fraction, args.min_tp_pct * 100)
    logger.info("Loss Guard : max_losses=%d cooldown=%ds",
                args.max_consecutive_losses, int(args.base_cooldown_seconds))
    logger.info("Duration   : %s", f"{args.duration} minutes" if args.duration > 0 else "unlimited (Ctrl+C to stop)")
    logger.info("=" * 70)

    # Emit session start event with full config
    telem.emit(EventType.SESSION_START, {
        "config": config,
        "network": network,
        "observe_only": args.observe_only,
        "leverage": args.leverage,
        "backstop_sl_pct": args.backstop_sl_pct,
        "backstop_tp_pct": args.backstop_tp_pct,
    })

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
    backstop_sl_oid = None
    backstop_tp_oid = None
    current_trade_id = None
    trade_entry_time = None
    signal_log = []
    start_time = time.time()
    end_time = start_time + args.duration * 60 if args.duration > 0 else float("inf")

    logger.info("Connecting to mainnet WebSocket for %s prices...", symbol)

    # Reconnection state
    reconnect_count = 0
    subscribe_msg = json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}})
    duration_reached = False

    # Hyperliquid JSON heartbeat: server drops connections idle for 60s
    HL_PING_INTERVAL = 25

    async def hl_heartbeat(ws_conn):
        """Send Hyperliquid JSON pings to prevent server-side idle timeout."""
        try:
            while True:
                await asyncio.sleep(HL_PING_INTERVAL)
                await ws_conn.send(json.dumps({"method": "ping"}))
        except (asyncio.CancelledError, Exception):
            pass

    while not duration_reached:
        if time.time() > end_time:
            logger.info("Duration reached (%d min). Not reconnecting.", args.duration)
            break

        # Reconcile state with exchange before reconnecting (skip first connection)
        if reconnect_count > 0 and adapter is not None:
            logger.info("=" * 40)
            logger.info("RECONNECT #%d — reconciling state...", reconnect_count)
            logger.info("=" * 40)
            telem.emit(EventType.RECONNECT, {
                "reconnect_number": reconnect_count,
                "tick_count": tick_count,
                "signal_count": signal_count,
                "trade_count": trade_count,
            })
            backstop_oids = await reconcile_on_reconnect(
                adapter, strategy, symbol,
                BackstopOids(sl_oid=backstop_sl_oid, tp_oid=backstop_tp_oid),
            )
            backstop_sl_oid = backstop_oids.sl_oid
            backstop_tp_oid = backstop_oids.tp_oid

        try:
            async with websockets.connect(
                PRICE_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
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

                        # Emit tick telemetry
                        telem.emit(EventType.TICK, {"price": price})

                        # Feed tick to strategy
                        md = MarketData(
                            asset=symbol, price=price, volume_24h=0.0, timestamp=ts,
                        )

                        # Get positions for strategy (if adapter available)
                        positions = []
                        if adapter and tick_count % 10 == 0:
                            try:
                                positions = await adapter.get_positions()
                            except Exception:
                                pass

                        balance_val = 100_000.0
                        signals = strategy.generate_signals(md, positions, balance_val)

                        for signal in signals:
                            signal_count += 1
                            logger.info(
                                "*** SIGNAL #%d: %s %s | price=%.2f | reason=%s | regime=%s tp=%.3f%%",
                                signal_count, signal.signal_type.value, symbol, price,
                                signal.reason,
                                signal.metadata.get("regime", "?"),
                                signal.metadata.get("adaptive_tp", args.tp_pct) * 100,
                            )

                            # Emit signal telemetry with adaptive metadata
                            telem.emit(EventType.SIGNAL, {
                                "signal_type": signal.signal_type.value,
                                "price": price,
                                "size": signal.size,
                                "reason": signal.reason,
                                "is_reversal": signal.metadata.get("reversal", False),
                                "regime": signal.metadata.get("regime"),
                                "adaptive_tp": signal.metadata.get("adaptive_tp"),
                                "consecutive_losses": signal.metadata.get("consecutive_losses"),
                            })

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
                                        if k != "dc_event"
                                    },
                                })

                            if adapter and not args.observe_only:
                                # Cancel backstops before closing or reversing
                                needs_cancel = (
                                    (backstop_sl_oid is not None or backstop_tp_oid is not None)
                                    and (
                                        signal.signal_type == SignalType.CLOSE
                                        or signal.metadata.get("reversal")
                                    )
                                )
                                if needs_cancel:
                                    await cancel_backstop_sl(adapter, symbol, backstop_sl_oid)
                                    await cancel_backstop_tp(adapter, symbol, backstop_tp_oid)
                                    backstop_sl_oid = None
                                    backstop_tp_oid = None

                                ok = await execute_signal(adapter, signal, price)
                                if ok:
                                    trade_count += 1

                                    # Emit FILL for every execution
                                    telem.emit(EventType.FILL, {
                                        "signal_type": signal.signal_type.value,
                                        "price": price,
                                        "size": signal.size,
                                        "reason": signal.reason,
                                    })

                                    # On entry fill: notify strategy + place backstops
                                    if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                                        strategy.on_trade_executed(signal, price, signal.size)
                                        side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
                                        backstop_size = signal.metadata.get("new_position_size", signal.size)
                                        backstop_sl_oid = await place_backstop_sl(
                                            adapter, symbol, side, price,
                                            backstop_size, args.backstop_sl_pct,
                                        )
                                        backstop_tp_oid = await place_backstop_tp(
                                            adapter, symbol, side, price,
                                            backstop_size, args.backstop_tp_pct,
                                        )

                                        # Emit TRADE_ENTRY with DC context + adaptive metadata
                                        current_trade_id = uuid.uuid4().hex[:12]
                                        trade_entry_time = time.time()
                                        dc_event = signal.metadata.get("dc_event", {})
                                        dc_st = dc_event.get("start_time")
                                        dc_et = dc_event.get("end_time")
                                        telem.emit(EventType.TRADE_ENTRY, {
                                            "trade_id": current_trade_id,
                                            "side": side,
                                            "entry_price": price,
                                            "size": backstop_size,
                                            "is_reversal": signal.metadata.get("reversal", False),
                                            "backstop_sl_pct": args.backstop_sl_pct,
                                            "backstop_tp_pct": args.backstop_tp_pct,
                                            "dc_event_type": dc_event.get("event_type"),
                                            "dc_start_price": dc_event.get("start_price"),
                                            "dc_end_price": dc_event.get("end_price"),
                                            "dc_duration_s": (dc_et - dc_st) if dc_st and dc_et else None,
                                            "dc_threshold": dc_event.get("threshold_down") or dc_event.get("threshold_up"),
                                            # Adaptive-specific fields
                                            "regime": signal.metadata.get("regime"),
                                            "adaptive_tp": signal.metadata.get("adaptive_tp"),
                                            "consecutive_losses": signal.metadata.get("consecutive_losses"),
                                        })

                                    elif signal.signal_type == SignalType.CLOSE:
                                        # Emit TRADE_EXIT with risk manager state + MFE/MAE
                                        rm_state = strategy._trailing_rm.get_status()
                                        telem.emit(EventType.TRADE_EXIT, {
                                            "trade_id": current_trade_id,
                                            "exit_price": price,
                                            "exit_reason": signal.reason,
                                            "entry_price": rm_state.get("entry_price"),
                                            "sl_at_exit": rm_state.get("current_sl"),
                                            "tp_at_exit": rm_state.get("current_tp"),
                                            "max_favorable_excursion_pct": rm_state.get("max_favorable_excursion_pct"),
                                            "max_adverse_excursion_pct": rm_state.get("max_adverse_excursion_pct"),
                                            "trade_duration_s": (time.time() - trade_entry_time) if trade_entry_time else None,
                                        })
                                        current_trade_id = None
                                        trade_entry_time = None

                        # Account snapshot every ~3600 ticks (~1 hour)
                        if adapter and tick_count % 3600 == 0:
                            try:
                                query_addr = account_address or adapter.exchange.wallet.address
                                snap_state = adapter.info.user_state(query_addr)
                                snap_margin = snap_state.get("marginSummary", {})
                                telem.emit(EventType.ACCOUNT_SNAPSHOT, {
                                    "account_value": float(snap_margin.get("accountValue", 0)),
                                    "margin_used": float(snap_margin.get("totalMarginUsed", 0)),
                                    "withdrawable": float(snap_margin.get("totalRawUsd", 0)),
                                })
                            except Exception:
                                pass

                        # Status log every 100 ticks (with adaptive guard info)
                        if tick_count % 100 == 0:
                            elapsed = ts - start_time
                            rm_status = strategy._trailing_rm.get_status()
                            regime = strategy._regime.classify(ts)
                            adaptive_tp = strategy._os_tracker.adaptive_tp()
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
                                "Tick #%d | price=%.2f | regime=%s tp=%.3f%% | "
                                "skip_chop=%d skip_loss=%d | signals=%d trades=%d | "
                                "elapsed=%.0fs %s%s%s",
                                tick_count, price, regime, adaptive_tp * 100,
                                strategy._skipped_choppy, strategy._skipped_loss_guard,
                                signal_count, trade_count,
                                elapsed, remaining, pos_info, rc_info,
                            )
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                if not duration_reached:
                    reconnect_count += 1
                    logger.info("WebSocket closed by server. Reconnecting immediately...")

        except websockets.exceptions.ConnectionClosedError as e:
            reconnect_count += 1
            code = e.rcvd.code if e.rcvd else "?"
            reason = e.rcvd.reason if e.rcvd else "?"
            logger.warning(
                "WebSocket disconnected (code=%s, reason=%s). Reconnecting immediately...",
                code, reason,
            )
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
            reconnect_count += 1
            logger.error("Connection failed: %s. Reconnecting immediately...", e)
        except asyncio.CancelledError:
            logger.info("Shutdown requested.")
            break
        except Exception as e:
            reconnect_count += 1
            logger.error("Unexpected WS error: %s. Reconnecting immediately...", e)

    # Final summary
    elapsed = time.time() - start_time
    status = strategy.get_status()

    # Emit session end event with adaptive-specific summary
    telem.emit(EventType.SESSION_END, {
        "duration_seconds": elapsed,
        "tick_count": tick_count,
        "signal_count": signal_count,
        "trade_count": trade_count,
        "dc_event_count": status["dc_event_count"],
        "reconnect_count": reconnect_count,
        "skipped_choppy": status["skipped_choppy"],
        "skipped_loss_guard": status["skipped_loss_guard"],
        "final_regime": status["regime"],
        "final_adaptive_tp": status["adaptive_tp"],
        "overshoot_distribution": status.get("overshoot_distribution"),
    })
    telem.close()

    logger.info("=" * 70)
    logger.info("Session complete")
    logger.info("=" * 70)
    logger.info("Duration        : %.0f seconds", elapsed)
    logger.info("Ticks           : %d", tick_count)
    logger.info("DC events       : %d", status["dc_event_count"])
    logger.info("Signals         : %d", signal_count)
    logger.info("Trades          : %d", trade_count)
    logger.info("Reconnects      : %d", reconnect_count)
    logger.info("Skipped (choppy): %d", status["skipped_choppy"])
    logger.info("Skipped (loss)  : %d", status["skipped_loss_guard"])
    logger.info("Final regime    : %s", status["regime"])
    logger.info("Adaptive TP     : %.3f%%", status["adaptive_tp"] * 100)
    os_dist = status.get("overshoot_distribution")
    if os_dist and os_dist.get("count", 0) > 0:
        logger.info("Overshoot p50   : %.3f%% (n=%d)", os_dist["p50"] * 100, os_dist["count"])
    logger.info("Loss streak     : %s", status["loss_streak"])
    logger.info("Trailing RM     : %s", status["trailing_rm"])

    if adapter:
        try:
            query_addr = account_address or adapter.exchange.wallet.address
            user_state = adapter.info.user_state(query_addr)
            acct_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
            logger.info("Final account   : $%.2f", acct_value)
        except Exception:
            pass

    # Write JSON report if requested
    if args.json_report:
        report = {
            "mode": "dc_adaptive",
            "symbol": symbol,
            "network": network,
            "duration_seconds": elapsed,
            "config": config,
            "leverage": args.leverage,
            "backstop_sl_pct": args.backstop_sl_pct,
            "backstop_tp_pct": args.backstop_tp_pct,
            "tick_count": tick_count,
            "signal_count": signal_count,
            "trade_count": trade_count,
            "reconnect_count": reconnect_count,
            "strategy_status": status,
            "signals": signal_log,
        }
        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report written to %s", args.json_report)

    strategy.stop()
    if adapter:
        await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
