"""Live bridge for Archon strategy.

Connects to Hyperliquid WebSocket, feeds ticks through the Archon strategy,
and either observes or executes trade decisions.

Usage:
    # Observe-only (no trades, log what Claude would do)
    uv run --package hyperliquid-trading-bot python \
        -m strategies.archon.bridge --symbol HYPE --observe-only

    # Live trading on sub-account
    uv run --package hyperliquid-trading-bot python \
        -m strategies.archon.bridge --symbol HYPE \
        --vault-address 0xe614... --yes
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

# Add src to path
_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv
load_dotenv(_SRC_DIR.parent / ".env", override=True)

from strategies.archon.strategy import ArchonStrategy
from interfaces.strategy import MarketData, SignalType

logger = logging.getLogger(__name__)

PRICE_WS_URL = "wss://api.hyperliquid.xyz/ws"

NETWORK_CONFIG = {
    "mainnet": {
        "base_url": "https://api.hyperliquid.xyz",
        "key_env": "HYPERLIQUID_MAINNET_PRIVATE_KEY",
    },
}


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
    parser = argparse.ArgumentParser(description="Archon: Claude-augmented DC trading")

    # Core
    parser.add_argument("--symbol", default="HYPE")
    parser.add_argument("--observe-only", action="store_true",
                        help="Log decisions without placing trades")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run duration in minutes (0 = forever)")

    # DC thresholds
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--sensor-threshold", type=float, default=0.004)

    # Position management
    parser.add_argument("--position-size", type=float, default=10.0)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--sl-pct", type=float, default=0.015)
    parser.add_argument("--tp-pct", type=float, default=0.008)
    parser.add_argument("--trail-pct", type=float, default=0.35)
    parser.add_argument("--min-profit-to-trail-pct", type=float, default=0.002)
    parser.add_argument("--backstop-sl-pct", type=float, default=0.04)
    parser.add_argument("--backstop-tp-pct", type=float, default=0.04)

    # Claude intelligence
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable Claude AI, use heuristic only")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--max-calls-per-hour", type=int, default=30)

    # Direction
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--long-only", action="store_true")
    direction.add_argument("--short-only", action="store_true")

    # Sub-account
    parser.add_argument("--vault-address", default="")

    # Compounding
    parser.add_argument("--compound", action="store_true")
    parser.add_argument("--compound-fraction", type=float, default=0.9)

    # Operational
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--telemetry", action="store_true")

    return parser.parse_args()


async def main():
    import websockets

    _setup_logging()
    args = parse_args()

    symbol = args.symbol
    private_key = os.environ.get("HYPERLIQUID_MAINNET_PRIVATE_KEY", "")

    if not private_key and not args.observe_only:
        logger.error("HYPERLIQUID_MAINNET_PRIVATE_KEY not set. Use --observe-only.")
        sys.exit(1)

    direction = "long" if args.long_only else ("short" if args.short_only else "long")

    # Build strategy config
    config = {
        "symbol": symbol,
        "dc_threshold": [args.threshold, args.threshold],
        "sensor_threshold": [args.sensor_threshold, args.sensor_threshold],
        "position_size_usd": args.position_size,
        "leverage": args.leverage,
        "initial_stop_loss_pct": args.sl_pct,
        "default_tp_pct": args.tp_pct,
        "trail_pct": args.trail_pct,
        "min_profit_to_trail_pct": args.min_profit_to_trail_pct,
        "backstop_sl_pct": args.backstop_sl_pct,
        "backstop_tp_pct": args.backstop_tp_pct,
        "model": args.model,
        "use_ai": not args.no_ai,
        "min_confidence": args.min_confidence,
        "max_calls_per_hour": args.max_calls_per_hour,
        "direction_filter": direction,
        "cooldown_seconds": 30,
        "max_consecutive_losses": 4,
    }

    strategy = ArchonStrategy(config)
    strategy.start()

    # Decision log for observe-only analysis
    decision_log = []

    def on_decision(decision, context):
        decision_log.append({
            "timestamp": time.time(),
            "action": decision.action,
            "confidence": decision.confidence,
            "source": decision.source,
            "reasoning": decision.reasoning,
            "tp_pct": decision.tp_pct,
            "sl_pct": decision.sl_pct,
            "price": context.current_price,
            "regime": context.regime,
            "trend_pct": context.price_trend_pct,
        })

    strategy.set_decision_callback(on_decision)

    # Print banner
    mode = "OBSERVE ONLY" if args.observe_only else "LIVE TRADING"
    ai_mode = "HEURISTIC" if args.no_ai else f"Claude ({args.model})"
    logger.info("=" * 70)
    logger.info("Archon — Claude-Augmented DC Trading")
    logger.info("=" * 70)
    logger.info("Mode       : %s", mode)
    logger.info("Intelligence: %s (min confidence: %.0f%%)", ai_mode, args.min_confidence * 100)
    logger.info("Symbol     : %s", symbol)
    logger.info("Direction  : %s", direction.upper())
    logger.info("Threshold  : %.2f%% | Sensor: %.2f%%", args.threshold * 100, args.sensor_threshold * 100)
    logger.info("SL/TP      : %.2f%% / %.2f%% | Trail: %.0f%%",
                args.sl_pct * 100, args.tp_pct * 100, args.trail_pct * 100)
    logger.info("Position   : $%.0f @ %dx", args.position_size, args.leverage)
    if args.vault_address:
        logger.info("Account    : SUB-ACCOUNT %s", args.vault_address[:14] + "...")
    logger.info("=" * 70)

    # Track state
    tick_count = 0
    signal_count = 0
    trade_count = 0
    start_time = time.time()
    end_time = start_time + args.duration * 60 if args.duration > 0 else float("inf")

    reconnect_count = 0
    subscribe_msg = json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}})

    HL_PING_INTERVAL = 25

    async def hl_heartbeat(ws_conn):
        try:
            while True:
                await asyncio.sleep(HL_PING_INTERVAL)
                await ws_conn.send(json.dumps({"method": "ping"}))
        except (asyncio.CancelledError, Exception):
            pass

    duration_reached = False

    while not duration_reached:
        if time.time() > end_time:
            break

        try:
            async with websockets.connect(
                PRICE_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                await ws.send(subscribe_msg)

                if reconnect_count == 0:
                    logger.info("Connected. Waiting for %s ticks...", symbol)
                else:
                    logger.info("Reconnected (#%d). Strategy state preserved.", reconnect_count)

                heartbeat_task = asyncio.create_task(hl_heartbeat(ws))

                try:
                    async for message in ws:
                        if time.time() > end_time:
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

                        # Feed tick to strategy (synchronous — DC detection + exits)
                        md = MarketData(asset=symbol, price=price, volume_24h=0.0, timestamp=ts)
                        signals = strategy.generate_signals(md, [], 100_000.0)

                        # Check if a DC event was detected that needs async processing
                        if hasattr(strategy, '_last_trigger_event') and strategy._last_trigger_event is not None:
                            event = strategy._last_trigger_event
                            strategy._last_trigger_event = None

                            # Async decision from Claude or heuristic
                            entry_signal = await strategy.process_dc_event_async(event, price, ts)
                            if entry_signal:
                                signals.append(entry_signal)

                        # Process signals
                        for signal in signals:
                            signal_count += 1

                            if args.observe_only:
                                logger.info(
                                    "*** OBSERVE #%d: %s %s | price=%.2f | reason=%s",
                                    signal_count, signal.signal_type.value, symbol,
                                    price, signal.reason,
                                )
                            else:
                                logger.info(
                                    "*** SIGNAL #%d: %s %s | price=%.2f | reason=%s",
                                    signal_count, signal.signal_type.value, symbol,
                                    price, signal.reason,
                                )

                            # In observe-only, simulate the fill without executing
                            if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                                if args.observe_only:
                                    strategy.on_trade_executed(signal, price, signal.size)
                                    trade_count += 1
                                # Live execution would go here

                        # Status log every 100 ticks
                        if tick_count % 100 == 0:
                            elapsed = ts - start_time
                            rm = strategy._trailing_rm.get_status()
                            regime = strategy._context.get_regime(ts)
                            pos_info = ""
                            if rm.get("has_position"):
                                side = rm["side"]
                                entry = rm["entry_price"]
                                sl = rm["current_sl"]
                                tp = rm["current_tp"]
                                pnl_pct = (
                                    (price - entry) / entry * 100 if side == "LONG"
                                    else (entry - price) / entry * 100
                                )
                                pos_info = f" | {side} entry={entry:.2f} SL={sl:.2f} TP={tp:.2f} PnL={pnl_pct:+.3f}%"

                            last_dec = ""
                            if strategy._last_decision:
                                d = strategy._last_decision
                                last_dec = f" | last={d.action}({d.source},{d.confidence:.1f})"

                            rc_info = f" rc={reconnect_count}" if reconnect_count > 0 else ""
                            remaining = f"{(end_time - ts)/60:.0f}m" if end_time != float("inf") else "∞"

                            logger.info(
                                "Tick #%d | price=%.2f | regime=%s | signals=%d trades=%d skips=%d "
                                "| elapsed=%.0fs %s%s%s%s",
                                tick_count, price, regime,
                                signal_count, trade_count, strategy._skip_count,
                                elapsed, remaining, pos_info, last_dec, rc_info,
                            )

                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                if not duration_reached:
                    reconnect_count += 1
                    logger.info("WebSocket closed. Reconnecting...")

        except asyncio.CancelledError:
            break
        except Exception as e:
            reconnect_count += 1
            logger.error("Connection error: %s. Reconnecting...", e)

    # Final summary
    elapsed = time.time() - start_time
    status = strategy.get_status()
    strategy.stop()

    logger.info("=" * 70)
    logger.info("Archon Session Summary")
    logger.info("=" * 70)
    logger.info("Duration   : %.0f seconds (%.1f hours)", elapsed, elapsed / 3600)
    logger.info("Ticks      : %d", tick_count)
    logger.info("DC Events  : %d", status["dc_event_count"])
    logger.info("Signals    : %d (trades=%d, skips=%d)", signal_count, trade_count, status["skip_count"])
    logger.info("Reconnects : %d", reconnect_count)
    if status.get("reasoner"):
        r = status["reasoner"]
        logger.info("Reasoner   : AI=%d (ok=%d, fail=%d) Heuristic=%d",
                    r["ai_calls"], r["ai_successes"], r["ai_failures"], r["heuristic_calls"])

    # Dump decision log for analysis
    if decision_log:
        log_path = f"/tmp/archon_decisions_{symbol}_{int(start_time)}.json"
        with open(log_path, "w") as f:
            json.dump(decision_log, f, indent=2)
        logger.info("Decision log: %s (%d decisions)", log_path, len(decision_log))

        # Quick P&L summary from decisions
        entries = [d for d in decision_log if d["action"].startswith("enter_")]
        closes = [d for d in decision_log if d["action"] == "close"]
        skips = [d for d in decision_log if d["action"] == "skip"]
        logger.info("Decisions  : %d entries, %d closes, %d skips", len(entries), len(closes), len(skips))

    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
