"""CLI entry point for trade review module."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src/ to path as fallback (follows backtesting/cli.py pattern)
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backtesting.engine import TradeRecord
from backtesting.metrics import compute_metrics
from trade_review.fill_fetcher import FillFetcher, FillFetcherConfig
from trade_review.fill_pairer import FillPairer


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for trade review CLI."""
    parser = argparse.ArgumentParser(
        description="Review live trade history from Hyperliquid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Review last 24h of HYPE trades
  python -m trade_review.cli --symbol HYPE

  # Review last 7 days of SOL trades with JSON output
  python -m trade_review.cli --symbol SOL --days 7 --json

  # Review testnet trades
  python -m trade_review.cli --symbol BTC --hours 48 --network testnet
""",
    )

    parser.add_argument(
        "--symbol", required=True,
        help="Trading symbol (e.g., HYPE, SOL, BTC)",
    )
    parser.add_argument(
        "--hours", type=float, default=24.0,
        help="Hours of history to review (default: 24)",
    )
    parser.add_argument(
        "--days", type=float, default=None,
        help="Days of history to review (overrides --hours)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON for Claude to parse programmatically",
    )
    parser.add_argument(
        "--network", choices=["mainnet", "testnet"], default=None,
        help="Network to query (default: from HYPERLIQUID_NETWORK env)",
    )
    parser.add_argument(
        "--wallet", type=str, default=None,
        help="Wallet address to query fills for (for delegation setups where "
             "fills are on the main wallet, not the API wallet)",
    )

    return parser


def _trades_to_dicts(trades: list[TradeRecord]) -> list[dict[str, Any]]:
    """Convert TradeRecords to JSON-serializable dicts with ISO timestamps."""
    result = []
    for t in trades:
        result.append({
            "side": t.side,
            "entry_price": round(t.entry_price, 6),
            "exit_price": round(t.exit_price, 6),
            "size": round(t.size, 6),
            "entry_time_iso": datetime.fromtimestamp(
                t.entry_time, tz=timezone.utc
            ).isoformat(),
            "exit_time_iso": datetime.fromtimestamp(
                t.exit_time, tz=timezone.utc
            ).isoformat(),
            "hold_seconds": round(t.exit_time - t.entry_time, 1),
            "pnl_pct": round(t.pnl_pct, 6),
            "pnl_usd": round(t.pnl_usd, 4),
            "fees_usd": round(t.total_fees, 4),
            "net_pnl_usd": round(t.net_pnl_usd, 4),
            "reason": t.reason,
        })
    return result


def format_json(
    symbol: str,
    period_hours: float,
    network: str,
    wallet: str,
    fills: list[dict[str, Any]],
) -> str:
    """Format trade review as JSON string."""
    pairer = FillPairer()
    trades = pairer.process_fills(fills)

    output: dict[str, Any] = {
        "symbol": symbol,
        "period_hours": period_hours,
        "network": network,
        "wallet": wallet,
        "total_fills": len(fills),
        "trades": _trades_to_dicts(trades),
        "metrics": None,
        "open_position": pairer.open_position,
    }

    if trades:
        days = period_hours / 24.0
        metrics = compute_metrics(trades, total_signals=len(fills), days=days)
        pf = metrics.profit_factor
        output["metrics"] = {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate_net": round(metrics.win_rate_net, 2),
            "win_rate_gross": round(metrics.win_rate_gross, 2),
            "gross_pnl_usd": round(metrics.gross_pnl_usd, 4),
            "total_fees_usd": round(metrics.total_fees_usd, 4),
            "net_pnl_usd": round(metrics.net_pnl_usd, 4),
            "fee_pct_of_gross": round(metrics.fee_pct_of_gross, 2),
            "profit_factor": pf if pf != float("inf") else "Infinity",
            "max_drawdown_usd": round(metrics.max_drawdown_usd, 4),
            "avg_hold_seconds": round(metrics.avg_hold_seconds, 1),
            "net_pnl_per_day": round(metrics.net_pnl_per_day, 4),
            "trades_eaten_by_fees": metrics.trades_eaten_by_fees,
            "sl_exits": metrics.sl_exits,
            "tp_exits": metrics.tp_exits,
            "reversal_exits": metrics.reversal_exits,
        }

    return json.dumps(output, indent=2)


def format_human_readable(
    symbol: str,
    period_hours: float,
    network: str,
    wallet: str,
    fills: list[dict[str, Any]],
) -> str:
    """Format trade review as human-readable string."""
    pairer = FillPairer()
    trades = pairer.process_fills(fills)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"DC Overshoot Trade Review -- {symbol} ({network})")
    lines.append("=" * 70)

    # Mask wallet address for display
    if len(wallet) > 10:
        wallet_display = wallet[:6] + "..." + wallet[-4:]
    else:
        wallet_display = wallet

    lines.append(f"  Period:        {period_hours:.1f} hours ({len(fills)} fills)")
    lines.append(f"  Wallet:        {wallet_display}")
    lines.append(f"  Network:       {network}")
    lines.append("")

    if not trades:
        lines.append("  No completed trades in this period.")
        if pairer.open_position:
            pos = pairer.open_position
            lines.append(
                f"  Open position: {pos['side']} "
                f"{pos['remaining_size']:.4f} units @ ${pos['entry_price']:.2f}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    days = period_hours / 24.0
    metrics = compute_metrics(trades, total_signals=len(fills), days=days)

    lines.append(f"  Trades:        {metrics.total_trades} ({metrics.wins}W / {metrics.losses}L)")
    lines.append(f"  Win rate:      {metrics.win_rate_net:.1f}% net | {metrics.win_rate_gross:.1f}% gross")
    lines.append(f"  Gross P&L:     ${metrics.gross_pnl_usd:+.2f}")
    lines.append(f"  Total fees:    ${metrics.total_fees_usd:.2f} ({metrics.fee_pct_of_gross:.1f}% of gross)")
    lines.append(f"  Net P&L:       ${metrics.net_pnl_usd:+.2f}")
    lines.append(f"  Net P&L/day:   ${metrics.net_pnl_per_day:+.2f}")
    pf_str = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "inf"
    lines.append(f"  Profit factor: {pf_str}")
    lines.append(f"  Max drawdown:  ${metrics.max_drawdown_usd:.2f}")

    # Format hold time
    avg_hold = metrics.avg_hold_seconds
    if avg_hold >= 3600:
        hold_str = f"{avg_hold:.0f}s ({avg_hold/3600:.1f}h)"
    elif avg_hold >= 60:
        hold_str = f"{avg_hold:.0f}s ({avg_hold/60:.1f}m)"
    else:
        hold_str = f"{avg_hold:.0f}s"
    lines.append(f"  Avg hold:      {hold_str}")
    lines.append(f"  Fee-eaten:     {metrics.trades_eaten_by_fees} trades")
    lines.append(f"  Exits:         SL={metrics.sl_exits} TP={metrics.tp_exits} Rev={metrics.reversal_exits}")

    # Individual trades table
    lines.append("")
    lines.append("  --- Individual Trades ---")
    for i, t in enumerate(trades, 1):
        hold = t.exit_time - t.entry_time
        entry_dt = datetime.fromtimestamp(t.entry_time, tz=timezone.utc)
        lines.append(
            f"  #{i:<3} {t.side:<6} "
            f"entry=${t.entry_price:<10.4f} "
            f"exit=${t.exit_price:<10.4f} "
            f"sz={t.size:<8.2f} "
            f"hold={hold:>6.0f}s "
            f"net=${t.net_pnl_usd:>+8.4f} "
            f"[{t.reason}] "
            f"{entry_dt:%H:%M}"
        )

    # Open position
    if pairer.open_position:
        pos = pairer.open_position
        lines.append("")
        lines.append(
            f"  Open position: {pos['side']} "
            f"{pos['remaining_size']:.4f} units @ ${pos['entry_price']:.2f}"
        )
    else:
        lines.append("")
        lines.append("  Open position: none")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_review(args: argparse.Namespace) -> None:
    """Fetch fills, pair into trades, compute metrics, print results."""
    # Resolve time range
    if args.days is not None:
        period_hours = args.days * 24.0
    else:
        period_hours = args.hours

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(period_hours * 3600 * 1000)

    # Connect to Hyperliquid
    config = FillFetcherConfig.from_env(
        network_override=args.network,
        wallet_override=args.wallet,
    )
    fetcher = FillFetcher(config)
    fetcher.connect()

    # Fetch fills
    fills = fetcher.fetch(
        symbol=args.symbol,
        start_time_ms=start_ms,
        end_time_ms=now_ms,
    )

    wallet = config.wallet_address or ""

    if args.json_output:
        print(format_json(
            symbol=args.symbol,
            period_hours=period_hours,
            network=config.network,
            wallet=wallet,
            fills=fills,
        ))
    else:
        print(format_human_readable(
            symbol=args.symbol,
            period_hours=period_hours,
            network=config.network,
            wallet=wallet,
            fills=fills,
        ))


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_review(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
