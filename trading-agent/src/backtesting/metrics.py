"""Compute summary statistics from a list of completed trades."""

from __future__ import annotations

from dataclasses import dataclass

from backtesting.engine import TradeRecord


@dataclass
class BacktestMetrics:
    """Computed statistics from a list of trades."""

    total_trades: int
    total_signals: int
    wins: int
    losses: int
    win_rate_net: float
    win_rate_gross: float
    gross_pnl_usd: float
    total_fees_usd: float
    net_pnl_usd: float
    fee_pct_of_gross: float
    avg_win_net_usd: float
    avg_loss_net_usd: float
    avg_hold_seconds: float
    max_drawdown_usd: float
    profit_factor: float
    trades_eaten_by_fees: int
    per_trade_avg_fee: float
    net_pnl_per_day: float
    sl_exits: int
    tp_exits: int
    reversal_exits: int


def compute_metrics(
    trades: list[TradeRecord],
    total_signals: int = 0,
    days: float = 7.0,
) -> BacktestMetrics:
    """Compute summary metrics from a list of completed trades.

    Args:
        trades: List of TradeRecord objects.
        total_signals: Total signals generated (for reporting).
        days: Number of days of data (for per-day calculations).

    Returns:
        BacktestMetrics with all computed statistics.

    Raises:
        ValueError: If trades list is empty.
    """
    if not trades:
        raise ValueError("Cannot compute metrics from empty trade list")

    wins = [t for t in trades if t.net_pnl_usd > 0]
    losses = [t for t in trades if t.net_pnl_usd <= 0]
    gross_wins = [t for t in trades if t.pnl_usd > 0]
    fee_eaten = [t for t in trades if t.pnl_usd > 0 and t.net_pnl_usd <= 0]

    sl_exits = sum(1 for t in trades if "stop_loss" in t.reason)
    tp_exits = sum(1 for t in trades if "take_profit" in t.reason)
    reversal_exits = sum(1 for t in trades if "reversal" in t.reason)

    gross_pnl = sum(t.pnl_usd for t in trades)
    total_fees = sum(t.total_fees for t in trades)
    net_pnl = sum(t.net_pnl_usd for t in trades)

    # Profit factor: sum of winning net / |sum of losing net|
    winning_sum = sum(t.net_pnl_usd for t in wins)
    losing_sum = abs(sum(t.net_pnl_usd for t in losses))
    profit_factor = winning_sum / losing_sum if losing_sum > 0 else float("inf")

    # Max drawdown on cumulative net P&L
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum_pnl += t.net_pnl_usd
        peak = max(peak, cum_pnl)
        dd = peak - cum_pnl
        max_dd = max(max_dd, dd)

    # Hold times
    hold_times = [t.exit_time - t.entry_time for t in trades]
    avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0.0

    # Averages
    avg_win = sum(t.net_pnl_usd for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.net_pnl_usd for t in losses) / len(losses) if losses else 0.0

    n = len(trades)
    return BacktestMetrics(
        total_trades=n,
        total_signals=total_signals,
        wins=len(wins),
        losses=len(losses),
        win_rate_net=len(wins) / n * 100,
        win_rate_gross=len(gross_wins) / n * 100,
        gross_pnl_usd=gross_pnl,
        total_fees_usd=total_fees,
        net_pnl_usd=net_pnl,
        fee_pct_of_gross=(total_fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0.0,
        avg_win_net_usd=avg_win,
        avg_loss_net_usd=avg_loss,
        avg_hold_seconds=avg_hold,
        max_drawdown_usd=max_dd,
        profit_factor=profit_factor,
        trades_eaten_by_fees=len(fee_eaten),
        per_trade_avg_fee=total_fees / n,
        net_pnl_per_day=net_pnl / days if days > 0 else 0.0,
        sl_exits=sl_exits,
        tp_exits=tp_exits,
        reversal_exits=reversal_exits,
    )
