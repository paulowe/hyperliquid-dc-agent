"""Compare DC Adaptive vs DC Overshoot on historical data.

Runs both strategies through the same candle data and compares metrics.
Usage:
    python -m strategies.dc_adaptive.backtest_compare --symbol HYPE --days 3
"""

from __future__ import annotations

import argparse
import json
import sys

from backtesting.candle_fetcher import CandleFetcher, CandleFetcherConfig
from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.metrics import compute_metrics
from interfaces.strategy import MarketData, SignalType
from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy
from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy


def run_strategy_on_candles(strategy, candles, config):
    """Feed candles through a strategy and record trades.

    This is a simplified backtest engine that works with any TradingStrategy.
    """
    from backtesting.engine import TradeRecord

    strategy.start()
    trades = []
    current_position = None
    taker_fee_pct = 0.00035

    for candle in candles:
        price = float(candle["c"])
        ts = float(candle["t"]) / 1000.0
        md = MarketData(asset=config["symbol"], price=price, volume_24h=0.0, timestamp=ts)

        signals = strategy.generate_signals(md, [], 100_000.0)

        for signal in signals:
            if signal.signal_type == SignalType.CLOSE:
                if current_position:
                    # Close position
                    entry_notional = current_position["entry_price"] * current_position["size"]
                    exit_notional = price * current_position["size"]
                    entry_fee = entry_notional * taker_fee_pct
                    exit_fee = exit_notional * taker_fee_pct
                    total_fees = entry_fee + exit_fee

                    if current_position["side"] == "LONG":
                        pnl_pct = (price - current_position["entry_price"]) / current_position["entry_price"]
                    else:
                        pnl_pct = (current_position["entry_price"] - price) / current_position["entry_price"]

                    pnl_usd = pnl_pct * entry_notional
                    net_pnl = pnl_usd - total_fees

                    trades.append(TradeRecord(
                        side=current_position["side"],
                        entry_price=current_position["entry_price"],
                        exit_price=price,
                        size=current_position["size"],
                        entry_time=current_position["entry_time"],
                        exit_time=ts,
                        pnl_pct=pnl_pct,
                        pnl_usd=pnl_usd,
                        entry_fee=entry_fee,
                        exit_fee=exit_fee,
                        total_fees=total_fees,
                        net_pnl_usd=net_pnl,
                        reason=signal.reason,
                    ))
                    current_position = None

            elif signal.signal_type in (SignalType.BUY, SignalType.SELL):
                is_reversal = signal.metadata.get("reversal", False)

                # Close old position on reversal
                if is_reversal and current_position:
                    entry_notional = current_position["entry_price"] * current_position["size"]
                    exit_notional = price * current_position["size"]
                    entry_fee = entry_notional * taker_fee_pct
                    exit_fee = exit_notional * taker_fee_pct
                    total_fees = entry_fee + exit_fee

                    if current_position["side"] == "LONG":
                        pnl_pct = (price - current_position["entry_price"]) / current_position["entry_price"]
                    else:
                        pnl_pct = (current_position["entry_price"] - price) / current_position["entry_price"]

                    pnl_usd = pnl_pct * entry_notional
                    net_pnl = pnl_usd - total_fees

                    trades.append(TradeRecord(
                        side=current_position["side"],
                        entry_price=current_position["entry_price"],
                        exit_price=price,
                        size=current_position["size"],
                        entry_time=current_position["entry_time"],
                        exit_time=ts,
                        pnl_pct=pnl_pct,
                        pnl_usd=pnl_usd,
                        entry_fee=entry_fee,
                        exit_fee=exit_fee,
                        total_fees=total_fees,
                        net_pnl_usd=net_pnl,
                        reason="reversal_close",
                    ))
                    current_position = None

                # Open new position
                side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
                actual_size = signal.metadata.get("new_position_size", signal.size)
                current_position = {
                    "side": side,
                    "entry_price": price,
                    "size": actual_size,
                    "entry_time": ts,
                }
                strategy.on_trade_executed(signal, price, actual_size)

    strategy.stop()
    return trades


def main():
    parser = argparse.ArgumentParser(description="Compare DC Adaptive vs DC Overshoot")
    parser.add_argument("--symbol", default="HYPE")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Fetch candles
    fetcher = CandleFetcher(CandleFetcherConfig())
    candles = fetcher.fetch(args.symbol, "1m", args.days)

    if not candles:
        print("No candles fetched", file=sys.stderr)
        sys.exit(1)

    # === Baseline: DC Overshoot with the config that was running ===
    baseline_config = {
        "symbol": args.symbol,
        "dc_thresholds": [(0.015, 0.015)],
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.02,
        "initial_take_profit_pct": 0.005,
        "trail_pct": 0.3,
        "min_profit_to_trail_pct": 0.003,
        "cooldown_seconds": 10.0,
        "max_open_positions": 1,
        "log_events": False,
    }

    baseline_strategy = DCOvershootStrategy(baseline_config)
    baseline_trades = run_strategy_on_candles(baseline_strategy, candles, baseline_config)

    # === New: DC Adaptive ===
    adaptive_config = {
        "symbol": args.symbol,
        "dc_thresholds": [(0.015, 0.015)],
        "sensor_threshold": (0.004, 0.004),
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.02,
        "initial_take_profit_pct": 0.005,
        "trail_pct": 0.3,
        "min_profit_to_trail_pct": 0.003,
        "cooldown_seconds": 10.0,
        "max_open_positions": 1,
        "log_events": False,
        # Regime detector
        "lookback_seconds": 600,
        "choppy_rate_threshold": 4.0,
        "trending_consistency_threshold": 0.6,
        # Overshoot tracker
        "os_window_size": 20,
        "os_min_samples": 5,
        "tp_fraction": 0.8,
        "min_tp_pct": 0.003,
        "default_tp_pct": 0.005,
        # Loss streak guard
        "max_consecutive_losses": 3,
        "base_cooldown_seconds": 300,
    }

    adaptive_strategy = DCAdaptiveStrategy(adaptive_config)
    adaptive_trades = run_strategy_on_candles(adaptive_strategy, candles, adaptive_config)

    # Compute metrics
    days = args.days

    def metrics_summary(trades, name):
        if not trades:
            return {"name": name, "total_trades": 0, "net_pnl_usd": 0}
        m = compute_metrics(trades, days=days)
        return {
            "name": name,
            "total_trades": m.total_trades,
            "wins": m.wins,
            "losses": m.losses,
            "win_rate_net": m.win_rate_net,
            "gross_pnl_usd": m.gross_pnl_usd,
            "total_fees_usd": m.total_fees_usd,
            "net_pnl_usd": m.net_pnl_usd,
            "profit_factor": m.profit_factor,
            "max_drawdown_usd": m.max_drawdown_usd,
            "net_pnl_per_day": m.net_pnl_per_day,
            "sl_exits": m.sl_exits,
            "tp_exits": m.tp_exits,
            "reversal_exits": m.reversal_exits,
        }

    baseline_metrics = metrics_summary(baseline_trades, "DC Overshoot (baseline)")
    adaptive_metrics = metrics_summary(adaptive_trades, "DC Adaptive (new)")

    if args.json:
        print(json.dumps({
            "baseline": baseline_metrics,
            "adaptive": adaptive_metrics,
            "adaptive_status": adaptive_strategy.get_status(),
        }, indent=2, default=str))
    else:
        print(f"\n{'='*70}")
        print(f"  DC Strategy Comparison — {args.symbol} ({args.days} days)")
        print(f"  Candles: {len(candles)}")
        print(f"{'='*70}\n")

        fmt = "  {:<25} {:>15} {:>15}"
        print(fmt.format("Metric", "DC Overshoot", "DC Adaptive"))
        print(fmt.format("-" * 25, "-" * 15, "-" * 15))

        for key in ["total_trades", "wins", "losses", "win_rate_net",
                     "net_pnl_usd", "gross_pnl_usd", "total_fees_usd",
                     "profit_factor", "max_drawdown_usd", "net_pnl_per_day",
                     "sl_exits", "tp_exits", "reversal_exits"]:
            bv = baseline_metrics.get(key, "N/A")
            av = adaptive_metrics.get(key, "N/A")
            if isinstance(bv, float):
                bv = f"${bv:.2f}" if "usd" in key or "pnl" in key or "drawdown" in key else f"{bv:.2f}"
            if isinstance(av, float):
                av = f"${av:.2f}" if "usd" in key or "pnl" in key or "drawdown" in key else f"{av:.2f}"
            print(fmt.format(key, str(bv), str(av)))

        # Show adaptive strategy stats
        status = adaptive_strategy.get_status()
        print(f"\n  --- DC Adaptive Guard Stats ---")
        print(f"  Signals skipped (choppy):     {status['skipped_choppy']}")
        print(f"  Signals skipped (loss guard):  {status['skipped_loss_guard']}")
        print(f"  Final adaptive TP:             {status['adaptive_tp']*100:.3f}%")
        os_dist = status.get("overshoot_distribution")
        if os_dist:
            print(f"  Overshoot p50:                 {os_dist['p50']*100:.3f}%")
            print(f"  Overshoot p75:                 {os_dist['p75']*100:.3f}%")
            print(f"  Overshoots tracked:            {os_dist['count']}")

        # Net improvement
        improvement = adaptive_metrics.get("net_pnl_usd", 0) - baseline_metrics.get("net_pnl_usd", 0)
        print(f"\n  Net improvement: ${improvement:+.2f}")
        print()


if __name__ == "__main__":
    main()
