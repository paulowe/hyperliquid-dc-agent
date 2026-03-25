"""Backtest Archon strategy on historical candle data.

Feeds historical 1m candles through the Archon strategy in heuristic mode,
records all decisions and simulated trades, and computes performance metrics.

Usage:
    python -m strategies.archon.backtest --symbol HYPE --days 7
    python -m strategies.archon.backtest --symbol HYPE --days 3 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

from backtesting.candle_fetcher import CandleFetcher, CandleFetcherConfig
from backtesting.engine import TradeRecord
from backtesting.metrics import compute_metrics
from interfaces.strategy import MarketData, SignalType
from strategies.archon.strategy import ArchonStrategy


def run_archon_backtest(
    candles: list,
    symbol: str = "HYPE",
    threshold: float = 0.02,
    sl_pct: float = 0.015,
    tp_pct: float = 0.008,
    trail_pct: float = 0.35,
    min_ptt: float = 0.002,
    direction: str = "long",
) -> tuple[list[TradeRecord], list[dict]]:
    """Run Archon backtest on candle data.

    Returns (trades, decisions) where decisions is a log of all
    Claude/heuristic decisions made during the backtest.
    """
    config = {
        "symbol": symbol,
        "dc_threshold": [threshold, threshold],
        "sensor_threshold": [0.004, 0.004],
        "position_size_usd": 50.0,
        "leverage": 5,
        "initial_stop_loss_pct": sl_pct,
        "default_tp_pct": tp_pct,
        "trail_pct": trail_pct,
        "min_profit_to_trail_pct": min_ptt,
        "use_ai": False,
        "direction_filter": direction,
        "cooldown_seconds": 10,
        "max_consecutive_losses": 4,
        "context_ticks": 60,
        "context_dc_events": 10,
        "context_trades": 10,
    }

    strategy = ArchonStrategy(config)
    strategy.start()

    trades = []
    decisions = []
    current_position = None
    taker_fee_pct = 0.00045

    # Track decisions
    def on_decision(decision, context):
        decisions.append({
            "action": decision.action,
            "confidence": decision.confidence,
            "source": decision.source,
            "reasoning": decision.reasoning,
            "price": context.current_price,
            "regime": context.regime,
            "trend_pct": context.price_trend_pct,
            "tp_pct": decision.tp_pct,
            "sl_pct": decision.sl_pct,
        })

    strategy.set_decision_callback(on_decision)

    loop = asyncio.new_event_loop()

    for candle in candles:
        price = float(candle["c"])
        ts = float(candle["t"]) / 1000.0
        md = MarketData(asset=symbol, price=price, volume_24h=0.0, timestamp=ts)

        # Synchronous signals (DC detection + trailing RM exits)
        signals = strategy.generate_signals(md, [], 100_000.0)

        # Process any pending DC events via the reasoner
        if hasattr(strategy, '_last_trigger_event') and strategy._last_trigger_event is not None:
            event = strategy._last_trigger_event
            strategy._last_trigger_event = None
            entry_signal = loop.run_until_complete(
                strategy.process_dc_event_async(event, price, ts)
            )
            if entry_signal:
                signals.append(entry_signal)

        for signal in signals:
            if signal.signal_type == SignalType.CLOSE:
                if current_position:
                    entry_notional = current_position["entry_price"] * current_position["size"]
                    exit_notional = price * current_position["size"]
                    ef = entry_notional * taker_fee_pct
                    xf = exit_notional * taker_fee_pct
                    tf = ef + xf

                    if current_position["side"] == "LONG":
                        pnl_pct = (price - current_position["entry_price"]) / current_position["entry_price"]
                    else:
                        pnl_pct = (current_position["entry_price"] - price) / current_position["entry_price"]

                    pnl_usd = pnl_pct * entry_notional
                    trades.append(TradeRecord(
                        side=current_position["side"],
                        entry_price=current_position["entry_price"],
                        exit_price=price,
                        size=current_position["size"],
                        entry_time=current_position["entry_time"],
                        exit_time=ts,
                        pnl_pct=pnl_pct,
                        pnl_usd=pnl_usd,
                        entry_fee=ef, exit_fee=xf, total_fees=tf,
                        net_pnl_usd=pnl_usd - tf,
                        reason=signal.reason,
                    ))
                    current_position = None

            elif signal.signal_type in (SignalType.BUY, SignalType.SELL):
                side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
                actual_size = signal.size
                current_position = {
                    "side": side,
                    "entry_price": price,
                    "size": actual_size,
                    "entry_time": ts,
                }
                strategy.on_trade_executed(signal, price, actual_size, timestamp=ts)

    loop.close()
    strategy.stop()
    return trades, decisions


def main():
    parser = argparse.ArgumentParser(description="Backtest Archon strategy")
    parser.add_argument("--symbol", default="HYPE")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--sl-pct", type=float, default=0.015)
    parser.add_argument("--tp-pct", type=float, default=0.008)
    parser.add_argument("--trail-pct", type=float, default=0.35)
    parser.add_argument("--direction", default="long",
                        choices=["long", "short", "both"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Fetch candles
    fetcher = CandleFetcher(CandleFetcherConfig())
    candles = fetcher.fetch(args.symbol, "1m", args.days)

    if not candles:
        print("No candles fetched", file=sys.stderr)
        sys.exit(1)

    actual_days = len(candles) / 60 / 24

    # Run backtest
    trades, decisions = run_archon_backtest(
        candles, args.symbol, args.threshold,
        args.sl_pct, args.tp_pct, args.trail_pct,
        direction=args.direction,
    )

    if args.json:
        result = {
            "symbol": args.symbol,
            "candles": len(candles),
            "days": actual_days,
            "trades": len(trades),
            "decisions": len(decisions),
        }
        if trades:
            m = compute_metrics(trades)
            result["metrics"] = {
                "net_pnl": m.net_pnl_usd,
                "win_rate": m.win_rate_net,
                "profit_factor": m.profit_factor,
                "max_drawdown": m.max_drawdown_usd,
                "pnl_per_day": m.net_pnl_per_day,
            }
        result["decision_summary"] = {
            "total": len(decisions),
            "entries": sum(1 for d in decisions if d["action"].startswith("enter_")),
            "closes": sum(1 for d in decisions if d["action"] == "close"),
            "skips": sum(1 for d in decisions if d["action"] == "skip"),
        }
        print(json.dumps(result, indent=2))
        return

    # Human-readable output
    print(f"\n{'='*60}")
    print(f"  Archon Backtest — {args.symbol} ({actual_days:.1f} days)")
    print(f"{'='*60}")
    print(f"  Candles: {len(candles)} | Direction: {args.direction}")
    print(f"  Params: threshold={args.threshold*100:.1f}% SL={args.sl_pct*100:.1f}% "
          f"TP={args.tp_pct*100:.1f}% trail={args.trail_pct*100:.0f}%")

    # Decision summary
    entries = sum(1 for d in decisions if d["action"].startswith("enter_"))
    closes = sum(1 for d in decisions if d["action"] == "close")
    skips = sum(1 for d in decisions if d["action"] == "skip")
    print(f"\n  Decisions: {len(decisions)} total")
    print(f"    Entries: {entries} | Closes: {closes} | Skips: {skips}")

    if not trades:
        print(f"\n  NO COMPLETED TRADES")
        return

    m = compute_metrics(trades)
    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]
    reasons = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    print(f"\n  Trades: {m.total_trades} ({len(longs)} longs, {len(shorts)} shorts)")
    print(f"  Net P&L: ${m.net_pnl_usd:.2f} (${m.net_pnl_per_day:.2f}/day)")
    print(f"  Win Rate: {m.win_rate_net:.1f}%")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Avg Win: ${m.avg_win_net_usd:.3f} | Avg Loss: ${m.avg_loss_net_usd:.3f}")
    print(f"  Max Drawdown: ${m.max_drawdown_usd:.2f}")
    print(f"  Total Fees: ${m.total_fees_usd:.2f}")
    print(f"  Exits: {reasons}")

    # Trade details
    print(f"\n  Trade Details:")
    print(f"  {'#':<3} {'Side':<6} {'Entry':>8} {'Exit':>8} {'PnL%':>8} {'Net$':>8} {'Reason':<25}")
    for i, t in enumerate(trades):
        print(f"  {i+1:<3} {t.side:<6} {t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.pnl_pct*100:>+7.3f}% ${t.net_pnl_usd:>+7.3f} {t.reason:<25}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
