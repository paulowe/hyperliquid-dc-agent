"""Tests for metrics computation."""

from __future__ import annotations

import pytest

from backtesting.engine import TradeRecord
from backtesting.metrics import BacktestMetrics, compute_metrics


def make_trade(
    pnl_usd: float = 0.0,
    net_pnl_usd: float | None = None,
    fees: float = 0.01,
    reason: str = "trailing_stop_loss",
    side: str = "LONG",
    entry_time: float = 1000.0,
    exit_time: float = 2000.0,
) -> TradeRecord:
    """Create a TradeRecord with sensible defaults for testing."""
    if net_pnl_usd is None:
        net_pnl_usd = pnl_usd - fees
    return TradeRecord(
        side=side,
        entry_price=100.0,
        exit_price=100.0 + pnl_usd,
        size=0.1,
        entry_time=entry_time,
        exit_time=exit_time,
        pnl_pct=pnl_usd / 100.0,
        pnl_usd=pnl_usd,
        entry_fee=fees / 2,
        exit_fee=fees / 2,
        total_fees=fees,
        net_pnl_usd=net_pnl_usd,
        reason=reason,
    )


class TestEmptyTrades:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics([])


class TestAllWinning:
    def test_win_rate_100(self):
        trades = [make_trade(pnl_usd=1.0, fees=0.01) for _ in range(3)]
        m = compute_metrics(trades)
        assert m.total_trades == 3
        assert m.wins == 3
        assert m.losses == 0
        assert m.win_rate_net == 100.0
        assert m.max_drawdown_usd == 0.0

    def test_profit_factor_inf(self):
        trades = [make_trade(pnl_usd=1.0, fees=0.01) for _ in range(3)]
        m = compute_metrics(trades)
        assert m.profit_factor == float("inf")


class TestAllLosing:
    def test_win_rate_0(self):
        trades = [make_trade(pnl_usd=-1.0, fees=0.01) for _ in range(3)]
        m = compute_metrics(trades)
        assert m.wins == 0
        assert m.losses == 3
        assert m.win_rate_net == 0.0

    def test_negative_net_pnl(self):
        trades = [make_trade(pnl_usd=-1.0, fees=0.01) for _ in range(3)]
        m = compute_metrics(trades)
        assert m.net_pnl_usd < 0


class TestMixedTrades:
    def test_correct_counts(self):
        trades = [
            make_trade(pnl_usd=2.0, fees=0.01),
            make_trade(pnl_usd=-1.0, fees=0.01),
            make_trade(pnl_usd=0.5, fees=0.01),
        ]
        m = compute_metrics(trades)
        assert m.wins == 2
        assert m.losses == 1
        assert m.win_rate_net == pytest.approx(66.667, rel=0.01)

    def test_gross_net_relationship(self):
        trades = [
            make_trade(pnl_usd=2.0, fees=0.05),
            make_trade(pnl_usd=-1.0, fees=0.05),
        ]
        m = compute_metrics(trades)
        assert m.net_pnl_usd == pytest.approx(m.gross_pnl_usd - m.total_fees_usd)


class TestFeeEatenTrades:
    def test_counts_fee_eaten(self):
        """Trades profitable before fees but negative after should be counted."""
        trades = [
            # pnl_usd=0.005 but fees=0.01 → net = -0.005
            make_trade(pnl_usd=0.005, fees=0.01),
            # pnl_usd=2.0, fees=0.01 → net = 1.99
            make_trade(pnl_usd=2.0, fees=0.01),
        ]
        m = compute_metrics(trades)
        assert m.trades_eaten_by_fees == 1


class TestMaxDrawdown:
    def test_drawdown_calculation(self):
        """Known sequence: +10, -20, +5 → peak=10, trough=-10, dd=20."""
        trades = [
            make_trade(pnl_usd=10.0, fees=0.0, net_pnl_usd=10.0),
            make_trade(pnl_usd=-20.0, fees=0.0, net_pnl_usd=-20.0),
            make_trade(pnl_usd=5.0, fees=0.0, net_pnl_usd=5.0),
        ]
        m = compute_metrics(trades)
        # Peak = 10, then drops to -10, dd = 20
        assert m.max_drawdown_usd == pytest.approx(20.0)

    def test_no_drawdown_on_monotonic_gains(self):
        trades = [
            make_trade(pnl_usd=5.0, fees=0.0, net_pnl_usd=5.0),
            make_trade(pnl_usd=5.0, fees=0.0, net_pnl_usd=5.0),
        ]
        m = compute_metrics(trades)
        assert m.max_drawdown_usd == 0.0


class TestHoldTime:
    def test_avg_hold_time(self):
        trades = [
            make_trade(entry_time=100.0, exit_time=200.0),  # 100s
            make_trade(entry_time=300.0, exit_time=600.0),  # 300s
        ]
        m = compute_metrics(trades)
        assert m.avg_hold_seconds == pytest.approx(200.0)


class TestExitReasons:
    def test_classifies_reasons(self):
        trades = [
            make_trade(reason="trailing_stop_loss"),
            make_trade(reason="trailing_stop_loss"),
            make_trade(reason="trailing_take_profit"),
            make_trade(reason="reversal_close"),
        ]
        m = compute_metrics(trades)
        assert m.sl_exits == 2
        assert m.tp_exits == 1
        assert m.reversal_exits == 1


class TestPerDayCalculation:
    def test_net_pnl_per_day(self):
        trades = [make_trade(pnl_usd=7.0, fees=0.0, net_pnl_usd=7.0)]
        m = compute_metrics(trades, days=7.0)
        assert m.net_pnl_per_day == pytest.approx(1.0)
