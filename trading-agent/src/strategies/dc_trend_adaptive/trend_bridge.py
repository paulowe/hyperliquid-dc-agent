"""Live bridge: mainnet price data -> DCTrendAdaptiveStrategy -> trades.

Connects to Hyperliquid mainnet WebSocket for real midprice ticks,
runs the DC Trend-Adaptive strategy (regime + adaptive TP + loss guards +
trend direction filter), and executes trades on the configured network.

This is a standalone bridge — it does NOT modify adaptive_bridge.py.

Usage:
    uv run --package hyperliquid-trading-bot python \
        -m strategies.dc_trend_adaptive.trend_bridge \
        --symbol HYPE --threshold 0.02 \
        --sensor-threshold 0.004 \
        --sl-pct 0.018 --tp-pct 0.008 \
        --trail-pct 0.5 --min-profit-to-trail-pct 0.008 \
        --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 --leverage 10 \
        --trend-lookback-seconds 900 --trend-min-consistency 0.6 \
        --counter-trend-action block \
        --telemetry --yes
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

from strategies.dc_trend_adaptive.dc_trend_adaptive_strategy import DCTrendAdaptiveStrategy
from interfaces.strategy import MarketData, SignalType
from interfaces.exchange import Order, OrderSide, OrderType
from telemetry.collector import NullCollector, TelemetryCollector
from telemetry.events import EventType

# Reuse bridge infrastructure from adaptive_bridge
from strategies.dc_adaptive.adaptive_bridge import (
    BackstopOids,
    NETWORK_CONFIG,
    PRICE_WS_URL,
    get_network,
    compute_compound_size,
    create_adapter,
    place_backstop_sl,
    place_backstop_tp,
    cancel_backstop_sl,
    cancel_backstop_tp,
    reconcile_on_reconnect,
    execute_signal,
    _setup_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "HYPE"


def parse_args():
    parser = argparse.ArgumentParser(
        description="DC Trend-Adaptive: mainnet price data -> trend-filtered strategy -> trade execution"
    )
    # --- Core trading parameters ---
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help="Asset to trade (default: HYPE).",
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
        "--threshold", type=float, default=0.02,
        help="Trade-level DC threshold (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--position-size", type=float, default=50.0,
        help="Position size in USD (default: 50)",
    )
    parser.add_argument(
        "--sl-pct", type=float, default=0.018,
        help="Initial stop loss %% as decimal (default: 0.018 = 1.8%%)",
    )
    parser.add_argument(
        "--tp-pct", type=float, default=0.008,
        help="Default take profit %% before adaptive kicks in (default: 0.008 = 0.8%%)",
    )
    parser.add_argument(
        "--trail-pct", type=float, default=0.5,
        help="Trail lock-in %% (default: 0.5 = 50%% of profit)",
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Leverage (default: 10)",
    )
    parser.add_argument(
        "--min-profit-to-trail-pct", type=float, default=0.008,
        help="Min raw profit before trailing ratchet activates (default: 0.008 = 0.8%%)",
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
        "--min-tp-pct", type=float, default=0.005,
        help="Floor for adaptive TP (default: 0.005 = 0.5%%)",
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

    # --- Trend direction filter parameters (Guard 4) ---
    parser.add_argument(
        "--trend-lookback-seconds", type=float, default=900.0,
        help="Trend detection window in seconds (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--trend-min-events", type=int, default=5,
        help="Min sensor events before trend filtering activates (default: 5)",
    )
    parser.add_argument(
        "--trend-min-consistency", type=float, default=0.6,
        help="Directional bias threshold for trend detection (default: 0.6)",
    )
    parser.add_argument(
        "--trend-bias-mode", type=str, default="tmv_weighted",
        choices=["simple", "tmv_weighted"],
        help="Bias computation mode (default: tmv_weighted)",
    )
    parser.add_argument(
        "--trend-strict-threshold", type=float, default=0.8,
        help="B-Strict confidence bar for 'reduce' mode (default: 0.8)",
    )
    parser.add_argument(
        "--counter-trend-action", type=str, default="block",
        choices=["block", "reduce", "allow"],
        help="Action for counter-trend trades: block/reduce/allow (default: block)",
    )
    parser.add_argument(
        "--counter-trend-size-fraction", type=float, default=0.5,
        help="Size multiplier when action=reduce (default: 0.5)",
    )
    parser.add_argument(
        "--counter-trend-sl-pct", type=float, default=None,
        help="Tighter SL for counter-trend trades (default: use normal SL)",
    )
    parser.add_argument(
        "--close-on-trend-flip", action="store_true", default=True,
        help="Close position if trend reverses against it (default: true)",
    )
    parser.add_argument(
        "--no-close-on-trend-flip", dest="close_on_trend_flip", action="store_false",
        help="Disable close-on-trend-flip protective rule",
    )
    parser.add_argument(
        "--long-only", action="store_true", default=False,
        help="Skip all short signals (nuclear option)",
    )
    parser.add_argument(
        "--short-only", action="store_true", default=False,
        help="Skip all long signals (nuclear option)",
    )

    # --- Sub-account routing ---
    parser.add_argument(
        "--vault-address", type=str, default="",
        help="Sub-account address to route orders to via vault_address (isolated from master)",
    )

    # --- Compounding ---
    parser.add_argument(
        "--compound", action="store_true",
        help="Enable compounding: scale position size to account equity before each trade",
    )
    parser.add_argument(
        "--compound-fraction", type=float, default=0.9,
        help="Fraction of account equity to use as position size when compounding (default: 0.9 = 90%%)",
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


async def main():
    import websockets

    _setup_logging()
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

    # Resolve vault_address (sub-account) or account_address (delegation)
    vault_address = args.vault_address  # empty string means "don't use sub-account"

    # account_address for API wallet delegation
    account_address = ""
    if not vault_address:
        account_address = os.environ.get("MAINNET_ACCOUNT_ADDRESS", "") or os.environ.get(net_cfg["wallet_env"], "")

    # Mainnet safety confirmation
    if network == "mainnet" and not args.observe_only and not args.yes:
        print("\n" + "!" * 70)
        print("  WARNING: You are about to trade on MAINNET with REAL MONEY")
        print("  Strategy: DC Trend-Adaptive (regime + adaptive TP + loss + TREND FILTER)")
        print("!" * 70)
        target = vault_address or account_address or "API wallet (no delegation)"
        mode = "SUB-ACCOUNT" if vault_address else "DELEGATION" if account_address else "DIRECT"
        print(f"  Wallet: {target} ({mode})")
        print(f"  Threshold: {args.threshold*100:.2f}%  Sensor: {args.sensor_threshold*100:.2f}%")
        print(f"  SL: {args.sl_pct*100:.1f}%  TP: {args.tp_pct*100:.1f}% (default, adaptive overrides)")
        print(f"  Trend filter: {args.counter_trend_action} (bias_mode={args.trend_bias_mode})")
        size_label = f"${args.position_size}" + (" (compound)" if args.compound else "")
        print(f"  Size: {size_label}  Leverage: {args.leverage}x")
        print()
        confirm = input("  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
        print()

    # Build strategy config with all parameters
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
        # Trend direction filter (Guard 4)
        "trend_lookback_seconds": args.trend_lookback_seconds,
        "trend_min_events": args.trend_min_events,
        "trend_min_consistency": args.trend_min_consistency,
        "trend_bias_mode": args.trend_bias_mode,
        "trend_strict_threshold": args.trend_strict_threshold,
        "counter_trend_action": args.counter_trend_action,
        "counter_trend_size_fraction": args.counter_trend_size_fraction,
        "counter_trend_sl_pct": args.counter_trend_sl_pct,
        "close_on_trend_flip": args.close_on_trend_flip,
        "long_only": args.long_only,
        "short_only": args.short_only,
    }
    strategy = DCTrendAdaptiveStrategy(config)
    strategy.start()

    # Initialize telemetry collector
    if args.telemetry:
        telem = TelemetryCollector(
            symbol=symbol,
            bridge_type="dc_trend_adaptive",
            local_dir=Path(args.telemetry_dir) if args.telemetry_dir else None,
        )
        logger.info("Telemetry enabled (session=%s)", telem.session_id)
    else:
        telem = NullCollector()

    # Register DC event callback for telemetry (includes regime + trend updates on sensor events)
    sensor_key = f"{args.sensor_threshold}:{args.sensor_threshold}"

    def _on_dc_event(event: dict) -> None:
        telem.emit(EventType.DC_EVENT, event)
        # Emit momentum update when sensor threshold fires
        threshold_key = f"{event.get('threshold_down', 0)}:{event.get('threshold_up', 0)}"
        if threshold_key == sensor_key:
            ts = time.time()
            trend_status = strategy._trend_filter.get_status(ts)
            telem.emit(EventType.MOMENTUM_UPDATE, {
                "regime": strategy._regime.classify(ts),
                "event_rate": strategy._regime.event_rate(ts),
                "adaptive_tp": strategy._os_tracker.adaptive_tp(),
                "consecutive_losses": strategy._loss_guard.consecutive_losses,
                # Trend filter fields
                "dominant_trend": trend_status["dominant_trend"],
                "trend_bias": trend_status["bias"],
                "trend_bias_mode": trend_status["bias_mode"],
                "trend_avg_tmv_up": trend_status["avg_tmv_up"],
                "trend_avg_tmv_down": trend_status["avg_tmv_down"],
            })

    strategy.set_dc_event_callback(_on_dc_event)

    logger.info("=" * 70)
    logger.info("DC Trend-Adaptive Live Bridge")
    logger.info("=" * 70)
    logger.info("Price data : mainnet WebSocket (%s)", PRICE_WS_URL)
    logger.info("Trading    : %s", "OBSERVE ONLY" if args.observe_only else net_cfg["label"])
    logger.info("Symbol     : %s", symbol)
    logger.info("Threshold  : %.4f (%.2f%%)", args.threshold, args.threshold * 100)
    logger.info("Sensor     : %.4f (%.2f%%)", args.sensor_threshold, args.sensor_threshold * 100)
    compound_label = f" (compound {args.compound_fraction*100:.0f}%%)" if args.compound else ""
    logger.info("Position   : $%.0f USD @ %dx%s", args.position_size, args.leverage, compound_label)
    if vault_address:
        logger.info("Account    : SUB-ACCOUNT %s", vault_address[:14] + "...")
    elif account_address:
        logger.info("Account    : DELEGATION to %s", account_address[:14] + "...")
    logger.info("SL/TP      : %.2f%% / %.2f%% (default, adaptive overrides TP)", args.sl_pct * 100, args.tp_pct * 100)
    logger.info("Trail      : %.0f%% lock-in (min profit: %.2f%%)", args.trail_pct * 100, args.min_profit_to_trail_pct * 100)
    logger.info("Backstop   : SL=%.1f%% TP=%.1f%%", args.backstop_sl_pct * 100, args.backstop_tp_pct * 100)
    logger.info("Regime     : lookback=%ds choppy_rate=%.1f trending_consist=%.1f",
                int(args.lookback_seconds), args.choppy_rate_threshold, args.trending_consistency)
    logger.info("OS Tracker : window=%d min_samples=%d tp_fraction=%.1f min_tp=%.2f%%",
                args.os_window_size, args.os_min_samples, args.tp_fraction, args.min_tp_pct * 100)
    logger.info("Loss Guard : max_losses=%d cooldown=%ds",
                args.max_consecutive_losses, int(args.base_cooldown_seconds))
    logger.info("Trend      : lookback=%ds events=%d consistency=%.2f mode=%s action=%s",
                int(args.trend_lookback_seconds), args.trend_min_events,
                args.trend_min_consistency, args.trend_bias_mode, args.counter_trend_action)
    logger.info("Trend opts : close_on_flip=%s long_only=%s short_only=%s",
                args.close_on_trend_flip, args.long_only, args.short_only)
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
        adapter = await create_adapter(
            network, private_key, args.leverage, symbol,
            account_address=account_address,
            vault_address=vault_address,
        )

        # Query account state
        query_addr = vault_address or account_address or adapter.exchange.wallet.address
        user_state = adapter.info.user_state(query_addr)
        perp_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
        spot_value = 0.0
        try:
            spot_state = adapter.info.spot_user_state(query_addr)
            for bal in spot_state.get("balances", []):
                if bal["coin"] == "USDC":
                    spot_value = float(bal.get("total", 0))
                    break
        except Exception:
            pass
        acct_value = perp_value + spot_value
        positions = await adapter.get_positions()
        logger.info("Account value: $%.2f (perp=$%.2f + spot=$%.2f) (wallet: %s)",
                    acct_value, perp_value, spot_value, query_addr[:10] + "...")
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

    # Hyperliquid JSON heartbeat
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

        # Reconcile state on reconnect
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
                        market_data = MarketData(
                            asset=symbol, price=price, volume_24h=0.0, timestamp=ts,
                        )

                        # Get positions periodically
                        positions = []
                        if adapter and tick_count % 10 == 0:
                            try:
                                positions = await adapter.get_positions()
                            except Exception:
                                pass

                        balance_val = 100_000.0
                        signals = strategy.generate_signals(market_data, positions, balance_val)

                        for signal in signals:
                            signal_count += 1
                            trend = strategy._trend_filter.dominant_trend(ts)
                            trend_bias = strategy._trend_filter.bias(ts)
                            logger.info(
                                "*** SIGNAL #%d: %s %s | price=%.2f | reason=%s | "
                                "regime=%s tp=%.3f%% trend=%s(%.2f)",
                                signal_count, signal.signal_type.value, symbol, price,
                                signal.reason,
                                signal.metadata.get("regime", "?"),
                                signal.metadata.get("adaptive_tp", args.tp_pct) * 100,
                                trend or "none", trend_bias,
                            )

                            # Emit signal telemetry
                            telem.emit(EventType.SIGNAL, {
                                "signal_type": signal.signal_type.value,
                                "price": price,
                                "size": signal.size,
                                "reason": signal.reason,
                                "is_reversal": signal.metadata.get("reversal", False),
                                "regime": signal.metadata.get("regime"),
                                "adaptive_tp": signal.metadata.get("adaptive_tp"),
                                "consecutive_losses": signal.metadata.get("consecutive_losses"),
                                "dominant_trend": signal.metadata.get("dominant_trend"),
                                "trend_bias": signal.metadata.get("trend_bias"),
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

                                # Compounding
                                if args.compound and signal.signal_type in (SignalType.BUY, SignalType.SELL):
                                    try:
                                        eq_state = adapter.info.user_state(query_addr)
                                        perp_equity = float(eq_state.get("marginSummary", {}).get("accountValue", 0))
                                        spot_equity = 0.0
                                        try:
                                            spot_state = adapter.info.spot_user_state(query_addr)
                                            for bal in spot_state.get("balances", []):
                                                if bal["coin"] == "USDC":
                                                    spot_equity = float(bal.get("total", 0))
                                                    break
                                        except Exception:
                                            pass
                                        equity = perp_equity + spot_equity
                                        new_size = compute_compound_size(equity, args.compound_fraction, args.leverage)
                                        old_size = strategy._cfg.position_size_usd
                                        strategy._cfg.position_size_usd = new_size
                                        strategy._cfg.max_position_size_usd = new_size * 4
                                        new_entry_size = new_size / price
                                        if signal.metadata.get("reversal"):
                                            old_pos_size = signal.metadata.get("new_position_size", signal.size)
                                            signal.size = signal.size - old_pos_size + new_entry_size
                                            signal.metadata["new_position_size"] = new_entry_size
                                        else:
                                            signal.size = new_entry_size
                                        logger.info(
                                            "COMPOUND: equity=$%.2f -> position_size=$%.2f (was $%.2f)",
                                            equity, new_size, old_size,
                                        )
                                    except Exception as e:
                                        logger.warning("Compound equity query failed, using last size: %s", e)

                                ok = await execute_signal(adapter, signal, price)
                                if ok:
                                    trade_count += 1

                                    telem.emit(EventType.FILL, {
                                        "signal_type": signal.signal_type.value,
                                        "price": price,
                                        "size": signal.size,
                                        "reason": signal.reason,
                                    })

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
                                            "regime": signal.metadata.get("regime"),
                                            "adaptive_tp": signal.metadata.get("adaptive_tp"),
                                            "consecutive_losses": signal.metadata.get("consecutive_losses"),
                                            "dominant_trend": signal.metadata.get("dominant_trend"),
                                            "trend_bias": signal.metadata.get("trend_bias"),
                                        })

                                    elif signal.signal_type == SignalType.CLOSE:
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
                                query_addr = vault_address or account_address or adapter.exchange.wallet.address
                                snap_state = adapter.info.user_state(query_addr)
                                snap_margin = snap_state.get("marginSummary", {})
                                telem.emit(EventType.ACCOUNT_SNAPSHOT, {
                                    "account_value": float(snap_margin.get("accountValue", 0)),
                                    "margin_used": float(snap_margin.get("totalMarginUsed", 0)),
                                    "withdrawable": float(snap_margin.get("totalRawUsd", 0)),
                                })
                            except Exception:
                                pass

                        # Status log every 100 ticks (with trend info)
                        if tick_count % 100 == 0:
                            elapsed = ts - start_time
                            rm_status = strategy._trailing_rm.get_status()
                            regime = strategy._regime.classify(ts)
                            adaptive_tp = strategy._os_tracker.adaptive_tp()
                            trend = strategy._trend_filter.dominant_trend(ts)
                            trend_bias = strategy._trend_filter.bias(ts)
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
                            remaining = f"remaining=%.0fs" % (end_time - ts,) if end_time != float("inf") else "inf"
                            rc_info = f" rc={reconnect_count}" if reconnect_count > 0 else ""
                            logger.info(
                                "Tick #%d | price=%.2f | trend=%s(%.2f) regime=%s tp=%.3f%% | "
                                "skip_trend=%d skip_chop=%d skip_loss=%d | signals=%d trades=%d | "
                                "elapsed=%.0fs %s%s%s",
                                tick_count, price,
                                (trend or "none").upper(), trend_bias,
                                regime, adaptive_tp * 100,
                                strategy._skipped_counter_trend, strategy._skipped_choppy,
                                strategy._skipped_loss_guard,
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

    # Emit session end event
    telem.emit(EventType.SESSION_END, {
        "duration_seconds": elapsed,
        "tick_count": tick_count,
        "signal_count": signal_count,
        "trade_count": trade_count,
        "dc_event_count": status["dc_event_count"],
        "reconnect_count": reconnect_count,
        "skipped_choppy": status["skipped_choppy"],
        "skipped_loss_guard": status["skipped_loss_guard"],
        "skipped_counter_trend": status["skipped_counter_trend"],
        "reduced_counter_trend": status["reduced_counter_trend"],
        "trend_flip_closes": status["trend_flip_closes"],
        "final_regime": status["regime"],
        "final_adaptive_tp": status["adaptive_tp"],
        "final_trend": status["dominant_trend"],
        "final_trend_bias": status["trend_bias"],
        "overshoot_distribution": status.get("overshoot_distribution"),
    })
    telem.close()

    logger.info("=" * 70)
    logger.info("Session complete")
    logger.info("=" * 70)
    logger.info("Duration            : %.0f seconds", elapsed)
    logger.info("Ticks               : %d", tick_count)
    logger.info("DC events           : %d", status["dc_event_count"])
    logger.info("Signals             : %d", signal_count)
    logger.info("Trades              : %d", trade_count)
    logger.info("Reconnects          : %d", reconnect_count)
    logger.info("Skipped (choppy)    : %d", status["skipped_choppy"])
    logger.info("Skipped (loss guard): %d", status["skipped_loss_guard"])
    logger.info("Skipped (counter)   : %d", status["skipped_counter_trend"])
    logger.info("Reduced (counter)   : %d", status["reduced_counter_trend"])
    logger.info("Trend flip closes   : %d", status["trend_flip_closes"])
    logger.info("Final regime        : %s", status["regime"])
    logger.info("Final trend         : %s (bias=%.3f, mode=%s)",
                status["dominant_trend"], status["trend_bias"], status["trend_bias_mode"])
    logger.info("Adaptive TP         : %.3f%%", status["adaptive_tp"] * 100)
    os_dist = status.get("overshoot_distribution")
    if os_dist and os_dist.get("count", 0) > 0:
        logger.info("Overshoot p50       : %.3f%% (n=%d)", os_dist["p50"] * 100, os_dist["count"])
    logger.info("Loss streak         : %s", status["loss_streak"])
    logger.info("Trailing RM         : %s", status["trailing_rm"])

    if adapter:
        try:
            query_addr = vault_address or account_address or adapter.exchange.wallet.address
            user_state = adapter.info.user_state(query_addr)
            acct_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
            logger.info("Final account       : $%.2f", acct_value)
        except Exception:
            pass

    # Write JSON report if requested
    if args.json_report:
        report = {
            "mode": "dc_trend_adaptive",
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
