"""CLI entry point for backtesting module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src/ to path as fallback (follows live_bridge.py pattern)
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backtesting.candle_fetcher import CandleFetcher, CandleFetcherConfig
from backtesting.engine import BacktestConfig, BacktestEngine, MultiScaleBacktestConfig, MultiScaleBacktestEngine
from backtesting.metrics import compute_metrics
from backtesting.sweep import (
    MultiScaleParameterSweep,
    MultiScaleSweepConfig,
    ParameterSweep,
    SweepConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="DC Overshoot Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single backtest with default params
  python -m backtesting.cli --symbol SOL --days 7

  # Single backtest with custom params
  python -m backtesting.cli --symbol SOL --threshold 0.015 --sl-pct 0.015 --tp-pct 0.005 --days 7

  # Parameter sweep
  python -m backtesting.cli --symbol SOL --sweep --days 7

  # Sweep with JSON output
  python -m backtesting.cli --symbol SOL --sweep --days 7 --json
""",
    )

    # Required
    parser.add_argument("--symbol", default="SOL", help="Trading symbol (default: SOL)")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data (default: 7)")

    # Single backtest params
    parser.add_argument("--threshold", type=float, default=0.004, help="DC threshold (default: 0.004)")
    parser.add_argument("--sl-pct", type=float, default=0.003, help="Stop loss %% (default: 0.003)")
    parser.add_argument("--tp-pct", type=float, default=0.10, help="Take profit %% (default: 0.10)")
    parser.add_argument("--trail-pct", type=float, default=0.5, help="Trail %% (default: 0.5)")
    parser.add_argument("--min-profit-to-trail-pct", type=float, default=0.001, help="Min profit to trail (default: 0.001)")
    parser.add_argument("--position-size", type=float, default=100.0, help="Position size USD (default: 100)")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage (default: 10)")

    # Multi-scale params
    parser.add_argument("--multi-scale", action="store_true", help="Use multi-scale DC strategy")
    parser.add_argument(
        "--sensor-thresholds", type=str, default="0.002,0.004,0.008",
        help="Comma-separated sensor thresholds (default: 0.002,0.004,0.008)",
    )
    parser.add_argument("--trade-threshold", type=float, default=0.015, help="Trade threshold (default: 0.015)")
    parser.add_argument("--momentum-alpha", type=float, default=0.3, help="Momentum EMA alpha (default: 0.3)")
    parser.add_argument("--min-momentum-score", type=float, default=0.3, help="Min momentum score (default: 0.3)")

    # Mode flags
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep instead of single backtest")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output results as JSON")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh candle cache")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top sweep results to show (default: 20)")

    return parser


def run_single(args: argparse.Namespace, candles: list[dict]) -> None:
    """Run a single backtest and print results."""
    config = BacktestConfig(
        symbol=args.symbol,
        threshold=args.threshold,
        position_size_usd=args.position_size,
        initial_stop_loss_pct=args.sl_pct,
        initial_take_profit_pct=args.tp_pct,
        trail_pct=args.trail_pct,
        min_profit_to_trail_pct=args.min_profit_to_trail_pct,
        leverage=args.leverage,
    )

    engine = BacktestEngine(config)
    result = engine.run(candles)

    if args.json_output:
        output = {
            "symbol": args.symbol,
            "days": args.days,
            "candle_count": result.candle_count,
            "first_price": result.first_price,
            "last_price": result.last_price,
            "price_change_pct": round(result.price_change_pct, 4),
            "total_signals": result.total_signals,
            "config": {
                "threshold": config.threshold,
                "sl_pct": config.initial_stop_loss_pct,
                "tp_pct": config.initial_take_profit_pct,
                "trail_pct": config.trail_pct,
                "min_profit_to_trail": config.min_profit_to_trail_pct,
                "position_size_usd": config.position_size_usd,
                "leverage": config.leverage,
            },
        }

        if result.trades:
            metrics = compute_metrics(result.trades, result.total_signals, float(args.days))
            output["metrics"] = {
                "total_trades": metrics.total_trades,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "win_rate_net": round(metrics.win_rate_net, 2),
                "gross_pnl_usd": round(metrics.gross_pnl_usd, 4),
                "total_fees_usd": round(metrics.total_fees_usd, 4),
                "net_pnl_usd": round(metrics.net_pnl_usd, 4),
                "profit_factor": metrics.profit_factor if metrics.profit_factor != float("inf") else "Infinity",
                "max_drawdown_usd": round(metrics.max_drawdown_usd, 4),
                "max_equity_drawdown_usd": round(metrics.max_equity_drawdown_usd, 4),
                "net_pnl_per_day": round(metrics.net_pnl_per_day, 4),
                "trades_eaten_by_fees": metrics.trades_eaten_by_fees,
            }
        else:
            output["metrics"] = None

        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print(f"DC Overshoot Backtest — {args.symbol}")
    print("=" * 70)
    print(f"  Period:        {args.days} days ({result.candle_count} candles)")
    print(f"  Price:         ${result.first_price:.2f} → ${result.last_price:.2f} ({result.price_change_pct:+.2f}%)")
    print(f"  Threshold:     {config.threshold}")
    print(f"  SL/TP:         {config.initial_stop_loss_pct*100:.2f}% / {config.initial_take_profit_pct*100:.1f}%")
    print(f"  Trail:         {config.trail_pct} (min profit: {config.min_profit_to_trail_pct})")
    print(f"  Position:      ${config.position_size_usd:.0f} @ {config.leverage}x")
    print(f"  Total signals: {result.total_signals}")
    print()

    if not result.trades:
        print("  No completed trades.")
        return

    metrics = compute_metrics(result.trades, result.total_signals, float(args.days))
    print(f"  Trades:        {metrics.total_trades} ({metrics.wins}W / {metrics.losses}L)")
    print(f"  Win rate:      {metrics.win_rate_net:.1f}% net | {metrics.win_rate_gross:.1f}% gross")
    print(f"  Gross P&L:     ${metrics.gross_pnl_usd:+.2f}")
    print(f"  Total fees:    ${metrics.total_fees_usd:.2f} ({metrics.fee_pct_of_gross:.1f}% of gross)")
    print(f"  Net P&L:       ${metrics.net_pnl_usd:+.2f}")
    print(f"  Net P&L/day:   ${metrics.net_pnl_per_day:+.2f}")
    pf_str = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "inf"
    print(f"  Profit factor: {pf_str}")
    print(f"  Max drawdown:  ${metrics.max_drawdown_usd:.2f} (realized) / ${metrics.max_equity_drawdown_usd:.2f} (equity)")
    print(f"  Avg hold:      {metrics.avg_hold_seconds:.0f}s")
    print(f"  Fee-eaten:     {metrics.trades_eaten_by_fees} trades")
    print(f"  Exits:         SL={metrics.sl_exits} TP={metrics.tp_exits} Rev={metrics.reversal_exits}")


def run_sweep(args: argparse.Namespace, candles: list[dict]) -> None:
    """Run parameter sweep and print results."""
    sweep_config = SweepConfig(
        symbol=args.symbol,
        position_size_usd=args.position_size,
        leverage=args.leverage,
    )

    total_combos = len(sweep_config.combinations())

    if not args.json_output:
        print("=" * 70)
        print(f"DC Overshoot Parameter Sweep — {args.symbol}")
        print("=" * 70)
        print(f"  Testing {total_combos} parameter combinations...")
        print()

    def progress(current: int, total: int) -> None:
        if not args.json_output and current % 100 == 0:
            print(f"  Progress: {current}/{total}...")

    sweep = ParameterSweep(sweep_config)
    results = sweep.run(candles, days=float(args.days), progress_callback=progress)

    if args.json_output:
        data = {
            "symbol": args.symbol,
            "days": args.days,
            "total_combinations": total_combos,
            "valid_results": len(results),
            "profitable_count": sum(1 for r in results if r.net_pnl_usd > 0),
            "results": ParameterSweep.results_to_json(results[:args.top_n]),
            "patterns": ParameterSweep.analyze_patterns(results),
        }
        print(json.dumps(data, indent=2))
        return

    # Human-readable output
    profitable = [r for r in results if r.net_pnl_usd > 0]
    print(f"\n  {len(results)} valid configurations (>= {sweep_config.min_trades} trades)")
    print(f"  {len(profitable)} profitable configurations")
    print()

    if results:
        print(ParameterSweep.format_results(results, top_n=args.top_n))

    # Pattern analysis
    if results:
        patterns = ParameterSweep.analyze_patterns(results)
        print(f"\n{'='*70}")
        print("PATTERN ANALYSIS")
        print(f"{'='*70}")
        for label, data in [("Profitable", patterns["profitable"]), ("Unprofitable", patterns["unprofitable"])]:
            if data:
                print(f"  {label} ({patterns[label.lower() + '_count']}x):")
                for k, v in data.items():
                    print(f"    {k}: {v:.4f}")

    # Best config command
    if results:
        best = results[0]
        print(f"\n{'='*70}")
        print("BEST CONFIG — Live trading command:")
        print(f"{'='*70}")
        print(f"  uv run --package hyperliquid-trading-bot python \\")
        print(f"    trading-agent/src/strategies/dc_overshoot/live_bridge.py \\")
        print(f"    --symbol {args.symbol} --threshold {best.config.threshold} \\")
        print(f"    --position-size {args.position_size:.0f} \\")
        print(f"    --sl-pct {best.config.initial_stop_loss_pct} \\")
        print(f"    --tp-pct {best.config.initial_take_profit_pct} \\")
        print(f"    --trail-pct {best.config.trail_pct} \\")
        print(f"    --min-profit-to-trail-pct {best.config.min_profit_to_trail_pct} \\")
        print(f"    --backstop-sl-pct 0.05 --leverage {args.leverage} --yes")


def _parse_sensor_thresholds(s: str) -> list[tuple[float, float]]:
    """Parse comma-separated sensor thresholds into list of (down, up) tuples."""
    return [(float(v), float(v)) for v in s.split(",")]


def run_multi_scale_single(args: argparse.Namespace, candles: list[dict]) -> None:
    """Run a single multi-scale backtest and print results."""
    sensor_thresholds = _parse_sensor_thresholds(args.sensor_thresholds)
    config = MultiScaleBacktestConfig(
        symbol=args.symbol,
        sensor_thresholds=sensor_thresholds,
        trade_threshold=args.trade_threshold,
        momentum_alpha=args.momentum_alpha,
        min_momentum_score=args.min_momentum_score,
        position_size_usd=args.position_size,
        initial_stop_loss_pct=args.sl_pct,
        initial_take_profit_pct=args.tp_pct,
        trail_pct=args.trail_pct,
        min_profit_to_trail_pct=args.min_profit_to_trail_pct,
        leverage=args.leverage,
    )

    engine = MultiScaleBacktestEngine(config)
    result = engine.run(candles)

    if args.json_output:
        output = {
            "mode": "multi_scale",
            "symbol": args.symbol,
            "days": args.days,
            "candle_count": result.candle_count,
            "first_price": result.first_price,
            "last_price": result.last_price,
            "price_change_pct": round(result.price_change_pct, 4),
            "total_signals": result.total_signals,
            "filtered_signals": result.filtered_signals,
            "sensor_events": result.sensor_events,
            "trade_events": result.trade_events,
            "config": {
                "sensor_thresholds": sensor_thresholds,
                "trade_threshold": config.trade_threshold,
                "momentum_alpha": config.momentum_alpha,
                "min_momentum_score": config.min_momentum_score,
                "sl_pct": config.initial_stop_loss_pct,
                "tp_pct": config.initial_take_profit_pct,
                "trail_pct": config.trail_pct,
                "min_profit_to_trail": config.min_profit_to_trail_pct,
                "position_size_usd": config.position_size_usd,
                "leverage": config.leverage,
            },
        }

        if result.trades:
            metrics = compute_metrics(result.trades, result.total_signals, float(args.days))
            output["metrics"] = {
                "total_trades": metrics.total_trades,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "win_rate_net": round(metrics.win_rate_net, 2),
                "gross_pnl_usd": round(metrics.gross_pnl_usd, 4),
                "total_fees_usd": round(metrics.total_fees_usd, 4),
                "net_pnl_usd": round(metrics.net_pnl_usd, 4),
                "profit_factor": metrics.profit_factor if metrics.profit_factor != float("inf") else "Infinity",
                "max_drawdown_usd": round(metrics.max_drawdown_usd, 4),
                "max_equity_drawdown_usd": round(metrics.max_equity_drawdown_usd, 4),
                "net_pnl_per_day": round(metrics.net_pnl_per_day, 4),
                "trades_eaten_by_fees": metrics.trades_eaten_by_fees,
            }
        else:
            output["metrics"] = None

        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print(f"Multi-Scale DC Backtest — {args.symbol}")
    print("=" * 70)
    print(f"  Period:          {args.days} days ({result.candle_count} candles)")
    print(f"  Price:           ${result.first_price:.2f} → ${result.last_price:.2f} ({result.price_change_pct:+.2f}%)")
    print(f"  Sensors:         {sensor_thresholds}")
    print(f"  Trade threshold: {config.trade_threshold}")
    print(f"  Momentum:        alpha={config.momentum_alpha} min_score={config.min_momentum_score}")
    print(f"  SL/TP:           {config.initial_stop_loss_pct*100:.2f}% / {config.initial_take_profit_pct*100:.1f}%")
    print(f"  Trail:           {config.trail_pct} (min profit: {config.min_profit_to_trail_pct})")
    print(f"  Position:        ${config.position_size_usd:.0f} @ {config.leverage}x")
    print(f"  Sensor events:   {result.sensor_events}")
    print(f"  Trade events:    {result.trade_events}")
    print(f"  Filtered:        {result.filtered_signals}")
    print()

    if not result.trades:
        print("  No completed trades.")
        return

    metrics = compute_metrics(result.trades, result.total_signals, float(args.days))
    print(f"  Trades:          {metrics.total_trades} ({metrics.wins}W / {metrics.losses}L)")
    print(f"  Win rate:        {metrics.win_rate_net:.1f}% net | {metrics.win_rate_gross:.1f}% gross")
    print(f"  Gross P&L:       ${metrics.gross_pnl_usd:+.2f}")
    print(f"  Total fees:      ${metrics.total_fees_usd:.2f} ({metrics.fee_pct_of_gross:.1f}% of gross)")
    print(f"  Net P&L:         ${metrics.net_pnl_usd:+.2f}")
    print(f"  Net P&L/day:     ${metrics.net_pnl_per_day:+.2f}")
    pf_str = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "inf"
    print(f"  Profit factor:   {pf_str}")
    print(f"  Max drawdown:    ${metrics.max_drawdown_usd:.2f} (realized) / ${metrics.max_equity_drawdown_usd:.2f} (equity)")
    print(f"  Avg hold:        {metrics.avg_hold_seconds:.0f}s")
    print(f"  Fee-eaten:       {metrics.trades_eaten_by_fees} trades")
    print(f"  Exits:           SL={metrics.sl_exits} TP={metrics.tp_exits} Rev={metrics.reversal_exits}")


def run_multi_scale_sweep(args: argparse.Namespace, candles: list[dict]) -> None:
    """Run multi-scale parameter sweep and print results."""
    sensor_thresholds = _parse_sensor_thresholds(args.sensor_thresholds)
    sweep_config = MultiScaleSweepConfig(
        symbol=args.symbol,
        position_size_usd=args.position_size,
        leverage=args.leverage,
        sensor_thresholds=sensor_thresholds,
    )

    total_combos = len(sweep_config.combinations())

    if not args.json_output:
        print("=" * 70)
        print(f"Multi-Scale DC Parameter Sweep — {args.symbol}")
        print("=" * 70)
        print(f"  Sensors:  {sensor_thresholds}")
        print(f"  Testing {total_combos} parameter combinations...")
        print()

    def progress(current: int, total: int) -> None:
        if not args.json_output and current % 100 == 0:
            print(f"  Progress: {current}/{total}...")

    sweep = MultiScaleParameterSweep(sweep_config)
    results = sweep.run(candles, days=float(args.days), progress_callback=progress)

    if args.json_output:
        data = {
            "mode": "multi_scale_sweep",
            "symbol": args.symbol,
            "days": args.days,
            "sensor_thresholds": sensor_thresholds,
            "total_combinations": total_combos,
            "valid_results": len(results),
            "profitable_count": sum(1 for r in results if r.net_pnl_usd > 0),
            "results": MultiScaleParameterSweep.results_to_json(results[:args.top_n]),
        }
        print(json.dumps(data, indent=2))
        return

    # Human-readable output
    profitable = [r for r in results if r.net_pnl_usd > 0]
    print(f"\n  {len(results)} valid configurations (>= {sweep_config.min_trades} trades)")
    print(f"  {len(profitable)} profitable configurations")
    print()

    if results:
        print(MultiScaleParameterSweep.format_results(results, top_n=args.top_n))

    # Best config command
    if results:
        best = results[0]
        print(f"\n{'='*70}")
        print("BEST CONFIG — Live trading command:")
        print(f"{'='*70}")
        print(f"  uv run --package hyperliquid-trading-bot python \\")
        print(f"    trading-agent/src/strategies/dc_overshoot/multi_scale_bridge.py \\")
        print(f"    --symbol {args.symbol} \\")
        print(f"    --sensor-thresholds {args.sensor_thresholds} \\")
        print(f"    --trade-threshold {best.config.trade_threshold} \\")
        print(f"    --momentum-alpha {best.config.momentum_alpha} \\")
        print(f"    --min-momentum-score {best.config.min_momentum_score} \\")
        print(f"    --position-size {args.position_size:.0f} \\")
        print(f"    --sl-pct {best.config.initial_stop_loss_pct} \\")
        print(f"    --tp-pct {best.config.initial_take_profit_pct} \\")
        print(f"    --trail-pct {best.config.trail_pct} \\")
        print(f"    --min-profit-to-trail-pct {best.config.min_profit_to_trail_pct} \\")
        print(f"    --backstop-sl-pct 0.05 --backstop-tp-pct 0.05 --leverage {args.leverage} --yes")


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Fetch candles
    if not args.json_output:
        print(f"Fetching {args.days}d of {args.symbol} 1m candles...")
    fetcher = CandleFetcher(CandleFetcherConfig())
    candles = fetcher.fetch(args.symbol, "1m", args.days, force_refresh=args.force_refresh)

    if not candles:
        print("ERROR: No candle data fetched.", file=sys.stderr)
        sys.exit(1)

    if not args.json_output:
        first_price = float(candles[0]["c"])
        last_price = float(candles[-1]["c"])
        print(f"  {len(candles)} candles | ${first_price:.2f} → ${last_price:.2f}")
        print()

    # Route to correct mode
    if args.multi_scale:
        if args.sweep:
            run_multi_scale_sweep(args, candles)
        else:
            run_multi_scale_single(args, candles)
    elif args.sweep:
        run_sweep(args, candles)
    else:
        run_single(args, candles)


if __name__ == "__main__":
    main()
