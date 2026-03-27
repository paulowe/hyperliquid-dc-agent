"""Multi-asset DC scanner for Archon strategy.

Monitors multiple symbols simultaneously via a single WebSocket,
runs independent DC detectors per symbol, and selects the highest-
confidence trade opportunity across all assets.

Only one position at a time (across all symbols). When flat, the
scanner picks the best signal. When in a position, it manages that
position and ignores other symbols.

Usage:
    uv run --package hyperliquid-trading-bot python \
        -m strategies.archon.scanner \
        --symbols HYPE,SOL,TAO,SUI,DOGE \
        --observe-only --no-ai
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

_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv
load_dotenv(_SRC_DIR.parent / ".env", override=True)

from strategies.archon.strategy import ArchonStrategy
from interfaces.strategy import MarketData, SignalType

logger = logging.getLogger(__name__)

PRICE_WS_URL = "wss://api.hyperliquid.xyz/ws"


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for handler in logging.root.handlers:
        if hasattr(handler, 'stream'):
            handler.stream = sys.stdout


def parse_args():
    parser = argparse.ArgumentParser(description="Archon multi-asset DC scanner")
    parser.add_argument("--symbols", default="HYPE,SOL,TAO,SUI,DOGE",
                        help="Comma-separated symbols to scan")
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Default DC threshold (overridden by --thresholds)")
    parser.add_argument("--thresholds", type=str, default="",
                        help="Per-symbol thresholds. Symmetric: 'SOL:0.012'. "
                             "Asymmetric: 'SOL:0.010/0.015' (down/up)")
    parser.add_argument("--sensor-threshold", type=float, default=0.004)
    parser.add_argument("--position-size", type=float, default=10.0)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--sl-pct", type=float, default=0.015)
    parser.add_argument("--tp-pct", type=float, default=0.008)
    parser.add_argument("--trail-pct", type=float, default=0.35)
    parser.add_argument("--min-profit-to-trail-pct", type=float, default=0.002)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--vault-address", default="")
    parser.add_argument("--duration", type=int, default=0)
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


async def main():
    import websockets

    _setup_logging()
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    # Parse per-symbol thresholds: "SOL:0.012" or asymmetric "SOL:0.010/0.015"
    symbol_thresholds: dict[str, tuple[float, float]] = {}
    if args.thresholds:
        for pair in args.thresholds.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2:
                sym = parts[0].strip()
                vals = parts[1].strip().split("/")
                if len(vals) == 2:
                    # Asymmetric: down/up
                    symbol_thresholds[sym] = (float(vals[0]), float(vals[1]))
                else:
                    # Symmetric
                    v = float(vals[0])
                    symbol_thresholds[sym] = (v, v)

    # Create one strategy instance per symbol
    strategies: dict[str, ArchonStrategy] = {}
    for sym in symbols:
        thresh_pair = symbol_thresholds.get(sym, (args.threshold, args.threshold))
        # Sensor threshold scales with average trade threshold (roughly 1/5)
        avg_thresh = (thresh_pair[0] + thresh_pair[1]) / 2
        sensor = avg_thresh * 0.2 if sym in symbol_thresholds else args.sensor_threshold
        config = {
            "symbol": sym,
            "dc_threshold": list(thresh_pair),
            "sensor_threshold": [sensor, sensor],
            "position_size_usd": args.position_size,
            "leverage": args.leverage,
            "initial_stop_loss_pct": args.sl_pct,
            "default_tp_pct": args.tp_pct,
            "trail_pct": args.trail_pct,
            "min_profit_to_trail_pct": args.min_profit_to_trail_pct,
            "min_confidence": args.min_confidence,
            "use_ai": not args.no_ai,
            "model": args.model,
            "direction_filter": "both",
            "cooldown_seconds": 30,
            "max_consecutive_losses": 4,
        }
        strat = ArchonStrategy(config)
        strat.start()
        strategies[sym] = strat

    # Track which symbol we're in (only one position at a time)
    active_symbol: str | None = None
    active_strategy: ArchonStrategy | None = None

    # Stats
    tick_counts: dict[str, int] = {s: 0 for s in symbols}
    total_signals = 0
    total_trades = 0
    total_skips = 0
    trade_log: list[dict] = []
    start_time = time.time()
    end_time = start_time + args.duration * 60 if args.duration > 0 else float("inf")

    mode = "OBSERVE ONLY" if args.observe_only else "LIVE"
    ai_mode = "HEURISTIC" if args.no_ai else f"Claude ({args.model})"

    logger.info("=" * 70)
    logger.info("Archon Multi-Asset Scanner")
    logger.info("=" * 70)
    logger.info("Mode       : %s", mode)
    logger.info("Intelligence: %s (min conf: %.0f%%)", ai_mode, args.min_confidence * 100)
    logger.info("Symbols    : %s", ", ".join(symbols))
    if symbol_thresholds:
        parts = []
        for s in symbols:
            t = symbol_thresholds.get(s, (args.threshold, args.threshold))
            if t[0] == t[1]:
                parts.append(f"{s}={t[0]*100:.1f}%")
            else:
                parts.append(f"{s}=↓{t[0]*100:.1f}%/↑{t[1]*100:.1f}%")
        logger.info("Thresholds : %s", " | ".join(parts))
    else:
        logger.info("Threshold  : %.2f%% (all symbols)", args.threshold * 100)
    logger.info("SL/TP      : %.2f%% / %.2f%% | Trail: %.0f%%",
                args.sl_pct * 100, args.tp_pct * 100, args.trail_pct * 100)
    logger.info("Position   : $%.0f @ %dx | Direction: BOTH (Claude decides)", args.position_size, args.leverage)
    logger.info("=" * 70)

    # Initialize exchange adapter for live trading
    adapter = None
    if not args.observe_only:
        private_key = os.environ.get("HYPERLIQUID_MAINNET_PRIVATE_KEY", "")
        if not private_key:
            logger.error("HYPERLIQUID_MAINNET_PRIVATE_KEY not set — cannot trade live")
            sys.exit(1)

        from exchanges.hyperliquid.adapter import HyperliquidAdapter
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange
        from eth_account import Account

        wallet = Account.from_key(private_key)
        base_url = "https://api.hyperliquid.xyz"
        adapter = HyperliquidAdapter(private_key=private_key, testnet=False)
        adapter.info = Info(base_url, skip_ws=True)

        if args.vault_address:
            adapter.exchange = Exchange(wallet, base_url, vault_address=args.vault_address)
            logger.info("SUB-ACCOUNT: orders → %s", args.vault_address[:14] + "...")
        else:
            account_addr = os.environ.get("MAINNET_ACCOUNT_ADDRESS", "")
            if account_addr and account_addr.lower() != wallet.address.lower():
                adapter.exchange = Exchange(wallet, base_url, account_address=account_addr)
            else:
                adapter.exchange = Exchange(wallet, base_url)

        adapter.is_connected = True
        adapter._build_precision_cache()

        # Set leverage for all symbols
        for sym in symbols:
            try:
                result = adapter.exchange.update_leverage(args.leverage, sym, is_cross=True)
                logger.info("Leverage %dx set for %s", args.leverage, sym)
            except Exception as e:
                logger.warning("Failed to set leverage for %s: %s", sym, e)

    reconnect_count = 0
    subscribe_msg = json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}})

    HL_PING_INTERVAL = 25
    duration_reached = False

    async def hl_heartbeat(ws_conn):
        try:
            while True:
                await asyncio.sleep(HL_PING_INTERVAL)
                await ws_conn.send(json.dumps({"method": "ping"}))
        except (asyncio.CancelledError, Exception):
            pass

    while not duration_reached:
        if time.time() > end_time:
            break

        try:
            async with websockets.connect(
                PRICE_WS_URL, ping_interval=20, ping_timeout=20, close_timeout=5,
            ) as ws:
                await ws.send(subscribe_msg)
                if reconnect_count == 0:
                    logger.info("Connected. Scanning %d symbols...", len(symbols))
                else:
                    logger.info("Reconnected (#%d).", reconnect_count)

                heartbeat_task = asyncio.create_task(hl_heartbeat(ws))

                try:
                    async for message in ws:
                        if time.time() > end_time:
                            duration_reached = True
                            break

                        data = json.loads(message)
                        if data.get("channel") in ("pong", None):
                            continue
                        if data.get("channel") != "allMids":
                            continue

                        mids = data.get("data", {}).get("mids", {})
                        ts = time.time()

                        # Feed ticks to all strategies
                        for sym in symbols:
                            if sym not in mids:
                                continue

                            price = float(mids[sym])
                            tick_counts[sym] = tick_counts.get(sym, 0) + 1
                            strat = strategies[sym]

                            md = MarketData(asset=sym, price=price, volume_24h=0.0, timestamp=ts)
                            signals = strat.generate_signals(md, [], 100_000.0)

                            # Check for DC events needing async processing
                            if hasattr(strat, '_last_trigger_event') and strat._last_trigger_event is not None:
                                event = strat._last_trigger_event
                                strat._last_trigger_event = None

                                # Only process entry if we're flat or this is our active symbol
                                if active_symbol is None or active_symbol == sym:
                                    entry_signal = await strat.process_dc_event_async(event, price, ts)
                                    if entry_signal:
                                        signals.append(entry_signal)
                                else:
                                    logger.info(
                                        "Scanner: %s DC event skipped — in %s position",
                                        sym, active_symbol,
                                    )
                                    strat._skip_count += 1

                            # Process signals
                            for signal in signals:
                                total_signals += 1

                                if signal.signal_type == SignalType.CLOSE:
                                    pnl = signal.metadata.get("pnl_pct", 0)
                                    logger.info(
                                        "*** %s EXIT: %s @ %.2f | %s | pnl=%+.3f%%",
                                        sym, signal.metadata.get("side", "?"),
                                        price, signal.reason, pnl * 100,
                                    )

                                    # Live: close position on exchange
                                    if adapter and active_symbol == sym:
                                        try:
                                            close_size = signal.size
                                            close_side = signal.metadata.get("side", "LONG")
                                            # To close a LONG, sell; to close a SHORT, buy
                                            is_buy = close_side == "SHORT"
                                            result = adapter.exchange.market_close(sym)
                                            logger.info("Close order result: %s", result)
                                        except Exception as e:
                                            logger.error("CLOSE ORDER FAILED: %s", e)

                                    trade_log.append({
                                        "symbol": sym, "action": "close",
                                        "price": price, "pnl_pct": pnl,
                                        "reason": signal.reason, "time": ts,
                                    })
                                    active_symbol = None
                                    active_strategy = None

                                elif signal.signal_type in (SignalType.BUY, SignalType.SELL):
                                    side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
                                    conf = signal.metadata.get("confidence", 0)
                                    logger.info(
                                        "*** %s ENTRY: %s @ %.2f | conf=%.2f | %s",
                                        sym, side, price, conf, signal.reason,
                                    )

                                    if args.observe_only:
                                        strat.on_trade_executed(signal, price, signal.size, timestamp=ts)
                                        total_trades += 1
                                        active_symbol = sym
                                        active_strategy = strat
                                    elif adapter:
                                        # Live: place market order on exchange
                                        try:
                                            is_buy = signal.signal_type == SignalType.BUY
                                            sz = signal.size
                                            # Round size to exchange precision
                                            sz_info = adapter.info.meta_and_asset_ctxs()
                                            sz_decimals = 4  # default
                                            if sz_info and len(sz_info) >= 2:
                                                for i, asset in enumerate(sz_info[0].get("universe", [])):
                                                    if asset["name"] == sym:
                                                        sz_decimals = asset.get("szDecimals", 4)
                                                        break
                                            sz = round(sz, sz_decimals)

                                            result = adapter.exchange.market_open(
                                                sym, is_buy, sz, None, 0.01
                                            )
                                            logger.info("Order result: %s", result)

                                            # Check if fill happened
                                            if result.get("status") == "ok":
                                                fills = result.get("response", {}).get("data", {}).get("statuses", [])
                                                if fills and "filled" in str(fills[0]):
                                                    fill_price = float(fills[0].get("filled", {}).get("avgPx", price))
                                                    strat.on_trade_executed(signal, fill_price, sz, timestamp=ts)
                                                    total_trades += 1
                                                    active_symbol = sym
                                                    active_strategy = strat
                                                    logger.info(
                                                        "*** FILLED: %s %s %.6f @ %.2f",
                                                        sym, side, sz, fill_price,
                                                    )
                                                else:
                                                    logger.warning("Order ok but no fill: %s", fills)
                                            else:
                                                logger.error("Order rejected: %s", result)
                                        except Exception as e:
                                            logger.error("ORDER FAILED: %s — %s", sym, e)

                                    trade_log.append({
                                        "symbol": sym, "action": side,
                                        "price": price, "confidence": conf,
                                        "reason": signal.reason, "time": ts,
                                    })

                        # Status log every 100 ticks (based on first symbol)
                        first_count = tick_counts.get(symbols[0], 0)
                        if first_count > 0 and first_count % 100 == 0:
                            elapsed = ts - start_time
                            pos_info = ""
                            if active_symbol and active_strategy:
                                rm = active_strategy._trailing_rm.get_status()
                                if rm.get("has_position"):
                                    p = float(mids.get(active_symbol, 0))
                                    entry = rm["entry_price"]
                                    side = rm["side"]
                                    pnl_pct = ((p - entry) / entry * 100 if side == "LONG"
                                               else (entry - p) / entry * 100)
                                    pos_info = f" | {active_symbol} {side} {entry:.2f} PnL={pnl_pct:+.2f}%"

                            # Prices summary
                            price_str = " ".join(
                                f"{s}=${float(mids.get(s, 0)):.1f}" for s in symbols[:4]
                            )
                            rc = f" rc={reconnect_count}" if reconnect_count > 0 else ""

                            logger.info(
                                "Scan #%d | %s | signals=%d trades=%d skips=%d "
                                "| %.0fs%s%s",
                                first_count, price_str,
                                total_signals, total_trades,
                                sum(s._skip_count for s in strategies.values()),
                                elapsed, pos_info, rc,
                            )

                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                if not duration_reached:
                    reconnect_count += 1

        except asyncio.CancelledError:
            break
        except Exception as e:
            reconnect_count += 1
            logger.error("Connection error: %s. Reconnecting...", e)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Scanner Summary — %.1f hours", elapsed / 3600)
    logger.info("=" * 70)
    for sym in symbols:
        s = strategies[sym]
        logger.info("  %s: dc=%d trades=%d skips=%d", sym, s._dc_event_count, s._trade_count, s._skip_count)
    logger.info("Total: signals=%d trades=%d", total_signals, total_trades)

    if trade_log:
        log_path = f"/tmp/archon_scanner_{int(start_time)}.json"
        with open(log_path, "w") as f:
            json.dump(trade_log, f, indent=2)
        logger.info("Trade log: %s", log_path)

    for strat in strategies.values():
        strat.stop()
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
