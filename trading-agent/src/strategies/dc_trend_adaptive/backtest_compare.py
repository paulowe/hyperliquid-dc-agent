"""Compare DC Trend-Adaptive vs DC Adaptive vs DC Overshoot on historical data.

Three-way comparison with directional breakdown showing how the trend
filter reduces counter-trend losses.

Usage:
    python -m strategies.dc_trend_adaptive.backtest_compare --symbol HYPE --days 14
"""

from __future__ import annotations

import argparse
import json
import sys

from backtesting.candle_fetcher import CandleFetcher, CandleFetcherConfig
from backtesting.metrics import compute_metrics
from strategies.dc_adaptive.backtest_compare import run_strategy_on_candles
from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy
from strategies.dc_overshoot.dc_overshoot_strategy import DCOvershootStrategy
from strategies.dc_trend_adaptive.dc_trend_adaptive_strategy import DCTrendAdaptiveStrategy


def directional_breakdown(trades):
    """Compute per-direction metrics from a trade list."""
    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]

    def summarize(trade_list):
        if not trade_list:
            return {"count": 0, "net_pnl": 0.0, "wins": 0, "losses": 0}
        wins = sum(1 for t in trade_list if t.net_pnl_usd > 0)
        losses = sum(1 for t in trade_list if t.net_pnl_usd <= 0)
        net = sum(t.net_pnl_usd for t in trade_list)
        return {"count": len(trade_list), "net_pnl": net, "wins": wins, "losses": losses}

    return {"LONG": summarize(longs), "SHORT": summarize(shorts)}


def main():
    parser = argparse.ArgumentParser(
        description="Compare DC Trend-Adaptive vs DC Adaptive vs DC Overshoot"
    )
    parser.add_argument("--symbol", default="HYPE")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--json", action="store_true")
    # Trend filter params
    parser.add_argument("--trend-lookback-seconds", type=float, default=900.0)
    parser.add_argument("--trend-min-events", type=int, default=5)
    parser.add_argument("--trend-min-consistency", type=float, default=0.6)
    parser.add_argument("--trend-bias-mode", default="tmv_weighted")
    parser.add_argument("--counter-trend-action", default="block")
    parser.add_argument("--close-on-trend-flip", action="store_true", default=True)
    parser.add_argument("--no-close-on-trend-flip", dest="close_on_trend_flip", action="store_false")
    args = parser.parse_args()

    # Fetch candles
    fetcher = CandleFetcher(CandleFetcherConfig())
    candles = fetcher.fetch(args.symbol, "1m", args.days)

    if not candles:
        print("No candles fetched", file=sys.stderr)
        sys.exit(1)

    # Shared base config
    base_config = {
        "symbol": args.symbol,
        "dc_thresholds": [(0.02, 0.02)],
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.018,
        "initial_take_profit_pct": 0.008,
        "trail_pct": 0.5,
        "min_profit_to_trail_pct": 0.008,
        "cooldown_seconds": 10.0,
        "max_open_positions": 1,
        "log_events": False,
    }

    # Adaptive config shared
    adaptive_extras = {
        "sensor_threshold": (0.004, 0.004),
        "lookback_seconds": 600,
        "choppy_rate_threshold": 4.0,
        "trending_consistency_threshold": 0.6,
        "os_window_size": 20,
        "os_min_samples": 5,
        "tp_fraction": 0.4,
        "min_tp_pct": 0.005,
        "default_tp_pct": 0.008,
        "max_consecutive_losses": 3,
        "base_cooldown_seconds": 300,
    }

    # === 1. Baseline: DC Overshoot ===
    baseline_config = dict(base_config)
    baseline_strategy = DCOvershootStrategy(baseline_config)
    baseline_trades = run_strategy_on_candles(baseline_strategy, candles, baseline_config)

    # === 2. DC Adaptive ===
    adaptive_config = {**base_config, **adaptive_extras}
    adaptive_strategy = DCAdaptiveStrategy(adaptive_config)
    adaptive_trades = run_strategy_on_candles(adaptive_strategy, candles, adaptive_config)

    # === 3. DC Trend-Adaptive ===
    trend_config = {
        **base_config,
        **adaptive_extras,
        "trend_lookback_seconds": args.trend_lookback_seconds,
        "trend_min_events": args.trend_min_events,
        "trend_min_consistency": args.trend_min_consistency,
        "trend_bias_mode": args.trend_bias_mode,
        "counter_trend_action": args.counter_trend_action,
        "counter_trend_size_fraction": 0.5,
        "close_on_trend_flip": args.close_on_trend_flip,
    }
    trend_strategy = DCTrendAdaptiveStrategy(trend_config)
    trend_trades = run_strategy_on_candles(trend_strategy, candles, trend_config)

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

    baseline_metrics = metrics_summary(baseline_trades, "DC Overshoot")
    adaptive_metrics = metrics_summary(adaptive_trades, "DC Adaptive")
    trend_metrics = metrics_summary(trend_trades, "DC Trend-Adaptive")

    # Directional breakdown
    baseline_dir = directional_breakdown(baseline_trades)
    adaptive_dir = directional_breakdown(adaptive_trades)
    trend_dir = directional_breakdown(trend_trades)

    if args.json:
        print(json.dumps({
            "baseline": {**baseline_metrics, "directional": baseline_dir},
            "adaptive": {**adaptive_metrics, "directional": adaptive_dir},
            "trend_adaptive": {
                **trend_metrics,
                "directional": trend_dir,
                "trend_status": trend_strategy.get_status(),
            },
        }, indent=2, default=str))
    else:
        print(f"\n{'='*80}")
        print(f"  DC Strategy Three-Way Comparison — {args.symbol} ({args.days} days)")
        print(f"  Candles: {len(candles)}")
        print(f"{'='*80}\n")

        fmt = "  {:<25} {:>15} {:>15} {:>18}"
        print(fmt.format("Metric", "DC Overshoot", "DC Adaptive", "DC Trend-Adaptive"))
        print(fmt.format("-" * 25, "-" * 15, "-" * 15, "-" * 18))

        for key in ["total_trades", "wins", "losses", "win_rate_net",
                     "net_pnl_usd", "gross_pnl_usd", "total_fees_usd",
                     "profit_factor", "max_drawdown_usd", "net_pnl_per_day",
                     "sl_exits", "tp_exits", "reversal_exits"]:
            vals = []
            for m in [baseline_metrics, adaptive_metrics, trend_metrics]:
                v = m.get(key, "N/A")
                if isinstance(v, float):
                    v = f"${v:.2f}" if "usd" in key or "pnl" in key or "drawdown" in key else f"{v:.2f}"
                vals.append(str(v))
            print(fmt.format(key, *vals))

        # Directional breakdown
        print(f"\n  --- Directional Breakdown ---")
        dir_fmt = "  {:<25} {:>15} {:>15} {:>18}"
        for direction in ["LONG", "SHORT"]:
            print(f"\n  {direction}:")
            for bd, name in [(baseline_dir, "DC Overshoot"), (adaptive_dir, "DC Adaptive"), (trend_dir, "DC Trend-Adaptive")]:
                d = bd[direction]
                print(f"    {name:<24} trades={d['count']:>3} wins={d['wins']:>3} net=${d['net_pnl']:.2f}")

        # Trend filter stats
        status = trend_strategy.get_status()
        print(f"\n  --- DC Trend-Adaptive Guard Stats ---")
        print(f"  Signals skipped (choppy):        {status['skipped_choppy']}")
        print(f"  Signals skipped (loss guard):     {status['skipped_loss_guard']}")
        print(f"  Signals skipped (counter-trend):  {status['skipped_counter_trend']}")
        print(f"  Signals reduced (counter-trend):  {status['reduced_counter_trend']}")
        print(f"  Trend flip closes:                {status['trend_flip_closes']}")
        print(f"  Final adaptive TP:                {status['adaptive_tp']*100:.3f}%")
        print(f"  Trend bias mode:                  {status['trend_bias_mode']}")
        os_dist = status.get("overshoot_distribution")
        if os_dist:
            print(f"  Overshoot p50:                    {os_dist['p50']*100:.3f}%")
            print(f"  Overshoots tracked:               {os_dist['count']}")

        # Net improvements
        base_pnl = baseline_metrics.get("net_pnl_usd", 0)
        adap_pnl = adaptive_metrics.get("net_pnl_usd", 0)
        trend_pnl = trend_metrics.get("net_pnl_usd", 0)
        print(f"\n  Adaptive vs Baseline:       ${adap_pnl - base_pnl:+.2f}")
        print(f"  Trend-Adaptive vs Baseline: ${trend_pnl - base_pnl:+.2f}")
        print(f"  Trend-Adaptive vs Adaptive: ${trend_pnl - adap_pnl:+.2f}")
        print()


if __name__ == "__main__":
    main()
