"""Compare Archon vs DC Adaptive vs DC Overshoot on historical data.

Runs all three strategies through the same candle data and produces
a side-by-side performance comparison.

Usage:
    python -m strategies.archon.compare --symbol HYPE --days 7
    python -m strategies.archon.compare --symbol SOL --days 3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

from backtesting.candle_fetcher import CandleFetcher, CandleFetcherConfig
from backtesting.metrics import compute_metrics
from strategies.archon.backtest import run_archon_backtest
from strategies.dc_adaptive.backtest_compare import run_strategy_on_candles
from strategies.dc_adaptive.dc_adaptive_strategy import DCAdaptiveStrategy


def run_comparison(candles, symbol: str, days: float, as_json: bool = False):
    """Run all strategies and compare results."""

    # === Strategy 1: Archon (long-only, enhanced heuristic) ===
    archon_trades, archon_decisions = run_archon_backtest(
        candles, symbol=symbol, threshold=0.02,
        sl_pct=0.015, tp_pct=0.008, trail_pct=0.35,
        min_ptt=0.002, direction="long",
    )

    # === Strategy 2: DC Adaptive (both directions, current params) ===
    adaptive_config = {
        "symbol": symbol,
        "dc_thresholds": [[0.02, 0.02]],
        "sensor_threshold": [0.004, 0.004],
        "position_size_usd": 50.0,
        "max_position_size_usd": 200.0,
        "initial_stop_loss_pct": 0.015,
        "initial_take_profit_pct": 0.008,
        "trail_pct": 0.35,
        "min_profit_to_trail_pct": 0.002,
        "cooldown_seconds": 10,
        "max_open_positions": 1,
        "log_events": False,
        "lookback_seconds": 600,
        "choppy_rate_threshold": 4.0,
        "trending_consistency_threshold": 0.6,
        "os_window_size": 20,
        "os_min_samples": 5,
        "tp_fraction": 0.4,
        "min_tp_pct": 0.003,
        "default_tp_pct": 0.008,
        "max_consecutive_losses": 3,
        "base_cooldown_seconds": 300,
    }
    adaptive_strategy = DCAdaptiveStrategy(adaptive_config)
    adaptive_trades = run_strategy_on_candles(adaptive_strategy, candles, adaptive_config)

    # === Strategy 3: DC Adaptive long-only (direction filter) ===
    adaptive_long_config = dict(adaptive_config)
    adaptive_long_config["direction_filter"] = "long"
    adaptive_long_strategy = DCAdaptiveStrategy(adaptive_long_config)
    adaptive_long_trades = run_strategy_on_candles(adaptive_long_strategy, candles, adaptive_long_config)

    results = []
    for label, trades in [
        ("Archon (long, heuristic)", archon_trades),
        ("DC Adaptive (both)", adaptive_trades),
        ("DC Adaptive (long-only)", adaptive_long_trades),
    ]:
        if trades:
            m = compute_metrics(trades)
            longs = sum(1 for t in trades if t.side == "LONG")
            shorts = sum(1 for t in trades if t.side == "SHORT")
            results.append({
                "strategy": label,
                "trades": m.total_trades,
                "longs": longs,
                "shorts": shorts,
                "net_pnl": m.net_pnl_usd,
                "win_rate": m.win_rate_net,
                "profit_factor": m.profit_factor,
                "max_drawdown": m.max_drawdown_usd,
                "pnl_per_day": m.net_pnl_per_day,
                "total_fees": m.total_fees_usd,
                "sl_exits": m.sl_exits,
                "tp_exits": m.tp_exits,
            })
        else:
            results.append({
                "strategy": label,
                "trades": 0,
                "net_pnl": 0,
            })

    if as_json:
        output = {
            "symbol": symbol,
            "days": days,
            "candles": len(candles),
            "strategies": results,
            "archon_decisions": {
                "total": len(archon_decisions),
                "entries": sum(1 for d in archon_decisions if d["action"].startswith("enter_")),
                "closes": sum(1 for d in archon_decisions if d["action"] == "close"),
                "skips": sum(1 for d in archon_decisions if d["action"] == "skip"),
            },
        }
        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    print(f"\n{'='*75}")
    print(f"  Strategy Comparison — {symbol} ({days:.1f} days, {len(candles)} candles)")
    print(f"{'='*75}")

    header = f"  {'Strategy':<30} {'Trades':>6} {'Net$':>7} {'WR%':>5} {'PF':>6} {'$/day':>7} {'MaxDD':>7} {'SL':>3}/{' TP':>3}"
    print(header)
    print("  " + "-" * 73)

    for r in results:
        if r["trades"] > 0:
            print(f"  {r['strategy']:<30} {r['trades']:>6} ${r['net_pnl']:>5.2f} "
                  f"{r['win_rate']:>5.1f} {r['profit_factor']:>6.2f} "
                  f"${r['pnl_per_day']:>5.2f} ${r['max_drawdown']:>5.2f} "
                  f"{r.get('sl_exits',0):>3}/ {r.get('tp_exits',0):>3}")
        else:
            print(f"  {r['strategy']:<30}    --- no trades ---")

    print(f"\n  Archon decisions: {len(archon_decisions)} total "
          f"({sum(1 for d in archon_decisions if d['action'].startswith('enter_'))} entries, "
          f"{sum(1 for d in archon_decisions if d['action'] == 'skip')} skips)")
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="Compare Archon vs DC Adaptive")
    parser.add_argument("--symbol", default="HYPE")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    fetcher = CandleFetcher(CandleFetcherConfig())
    candles = fetcher.fetch(args.symbol, "1m", args.days)

    if not candles:
        print("No candles fetched", file=sys.stderr)
        sys.exit(1)

    days = len(candles) / 60 / 24
    run_comparison(candles, args.symbol, days, args.json)


if __name__ == "__main__":
    main()
