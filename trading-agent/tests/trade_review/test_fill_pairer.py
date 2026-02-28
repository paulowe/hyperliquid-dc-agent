"""Tests for fill pairing algorithm that converts raw Hyperliquid fills to TradeRecords."""

from __future__ import annotations

import pytest

from backtesting.engine import TradeRecord
from trade_review.fill_pairer import pair_fills_to_trades, FillPairer


TAKER_FEE_PCT = 0.00035  # 0.035% per side


def make_fill(
    coin: str = "HYPE",
    side: str = "B",
    dir: str = "Open Long",
    px: str = "25.50",
    sz: str = "10.0",
    time: int = 1700000000000,
    oid: int = 100,
    closed_pnl: str = "0.0",
    start_position: str = "0.0",
) -> dict:
    """Create a fill dict matching Hyperliquid API format."""
    return {
        "coin": coin,
        "side": side,
        "dir": dir,
        "px": px,
        "sz": sz,
        "time": time,
        "oid": oid,
        "closedPnl": closed_pnl,
        "startPosition": start_position,
        "crossed": True,
        "hash": f"0x{'a' * 64}",
    }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------
class TestEmptyFills:
    def test_no_fills_returns_empty_trades(self):
        assert pair_fills_to_trades([]) == []

    def test_pairer_class_no_fills(self):
        pairer = FillPairer()
        assert pairer.process_fills([]) == []


# ---------------------------------------------------------------------------
# Single round-trip trades
# ---------------------------------------------------------------------------
class TestSingleRoundTrip:
    def test_open_long_close_long(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 100.0
        assert t.exit_price == 110.0
        assert t.size == 5.0
        assert t.pnl_usd > 0  # profitable long

    def test_open_short_close_short(self):
        fills = [
            make_fill(dir="Open Short", side="A", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Short", side="B", px="90.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t.side == "SHORT"
        assert t.entry_price == 100.0
        assert t.exit_price == 90.0
        assert t.pnl_usd > 0  # profitable short

    def test_losing_long_trade(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="95.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t.pnl_usd < 0  # losing long

    def test_losing_short_trade(self):
        fills = [
            make_fill(dir="Open Short", side="A", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Short", side="B", px="105.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t.pnl_usd < 0  # losing short


# ---------------------------------------------------------------------------
# Fee accounting
# ---------------------------------------------------------------------------
class TestFeeAccounting:
    def test_fees_computed_from_notional(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="10.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="101.00", sz="10.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        t = trades[0]
        # entry_fee = 100 * 10 * 0.00035 = 0.35
        assert abs(t.entry_fee - 0.35) < 0.001
        # exit_fee = 101 * 10 * 0.00035 = 0.3535
        assert abs(t.exit_fee - 0.3535) < 0.001

    def test_net_pnl_equals_gross_minus_fees(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="10.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="101.00", sz="10.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        t = trades[0]
        assert abs(t.net_pnl_usd - (t.pnl_usd - t.total_fees)) < 0.001

    def test_custom_fee_rate(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="10.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="101.00", sz="10.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills, taker_fee_pct=0.001)
        t = trades[0]
        # entry_fee = 100 * 10 * 0.001 = 1.0
        assert abs(t.entry_fee - 1.0) < 0.001

    def test_total_fees_is_sum(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="10.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="101.00", sz="10.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        t = trades[0]
        assert abs(t.total_fees - (t.entry_fee + t.exit_fee)) < 0.001


# ---------------------------------------------------------------------------
# PnL percentage
# ---------------------------------------------------------------------------
class TestPnlPct:
    def test_long_pnl_pct(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        # 10% gain
        assert abs(trades[0].pnl_pct - 0.10) < 0.001

    def test_short_pnl_pct(self):
        fills = [
            make_fill(dir="Open Short", side="A", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Short", side="B", px="90.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        # 10% gain
        assert abs(trades[0].pnl_pct - 0.10) < 0.001


# ---------------------------------------------------------------------------
# Multiple round-trips
# ---------------------------------------------------------------------------
class TestMultipleRoundTrips:
    def test_two_consecutive_longs(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000, oid=1),
            make_fill(dir="Close Long", side="A", px="102.00", sz="5.0",
                      time=1700000060000, oid=2),
            make_fill(dir="Open Long", side="B", px="103.00", sz="5.0",
                      time=1700000120000, oid=3),
            make_fill(dir="Close Long", side="A", px="105.00", sz="5.0",
                      time=1700000180000, oid=4),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 2
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 102.0
        assert trades[1].entry_price == 103.0
        assert trades[1].exit_price == 105.0

    def test_long_then_short(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="102.00", sz="5.0",
                      time=1700000060000),
            make_fill(dir="Open Short", side="A", px="102.00", sz="5.0",
                      time=1700000120000),
            make_fill(dir="Close Short", side="B", px="99.00", sz="5.0",
                      time=1700000180000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 2
        assert trades[0].side == "LONG"
        assert trades[1].side == "SHORT"


# ---------------------------------------------------------------------------
# Scale-in (multiple opens before close)
# ---------------------------------------------------------------------------
class TestScaleIn:
    def test_two_opens_one_close_weighted_avg(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000, oid=1),
            make_fill(dir="Open Long", side="B", px="102.00", sz="5.0",
                      time=1700000030000, oid=2),
            make_fill(dir="Close Long", side="A", px="105.00", sz="10.0",
                      time=1700000060000, oid=3),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t.size == 10.0
        # Weighted avg: (100*5 + 102*5) / 10 = 101.0
        assert abs(t.entry_price - 101.0) < 0.001

    def test_scale_in_fee_uses_actual_entry_notional(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Open Long", side="B", px="102.00", sz="5.0",
                      time=1700000030000),
            make_fill(dir="Close Long", side="A", px="105.00", sz="10.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        t = trades[0]
        # Entry fee = sum of each fill's notional * fee
        # (100*5 + 102*5) * 0.00035 = 1010 * 0.00035 = 0.3535
        assert abs(t.entry_fee - 0.3535) < 0.001


# ---------------------------------------------------------------------------
# Partial close
# ---------------------------------------------------------------------------
class TestPartialClose:
    def test_partial_close_emits_trade_for_closed_portion(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="10.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="105.00", sz="5.0",
                      time=1700000060000),
        ]
        pairer = FillPairer()
        trades = pairer.process_fills(fills)
        # Should produce one trade for 5 units (closed portion)
        assert len(trades) == 1
        assert trades[0].size == 5.0
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 105.0
        # Remaining 5 units still open
        assert pairer.open_position is not None
        assert abs(pairer.open_position["remaining_size"] - 5.0) < 0.001

    def test_two_partial_closes(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="10.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="105.00", sz="4.0",
                      time=1700000060000),
            make_fill(dir="Close Long", side="A", px="108.00", sz="6.0",
                      time=1700000120000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 2
        assert trades[0].size == 4.0
        assert trades[0].exit_price == 105.0
        assert trades[1].size == 6.0
        assert trades[1].exit_price == 108.0


# ---------------------------------------------------------------------------
# Reversal detection
# ---------------------------------------------------------------------------
class TestReversalDetection:
    def test_close_long_then_immediate_open_short_is_reversal(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            # Close and immediately open opposite direction (within 2s)
            make_fill(dir="Close Long", side="A", px="102.00", sz="5.0",
                      time=1700000060000),
            make_fill(dir="Open Short", side="A", px="102.00", sz="5.0",
                      time=1700000061000),  # 1 second later
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        assert trades[0].reason == "reversal_close"

    def test_close_long_then_delayed_open_short_not_reversal(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="102.00", sz="5.0",
                      time=1700000060000),
            make_fill(dir="Open Short", side="A", px="102.00", sz="5.0",
                      time=1700000070000),  # 10 seconds later
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        assert trades[0].reason != "reversal_close"


# ---------------------------------------------------------------------------
# Exit reason inference
# ---------------------------------------------------------------------------
class TestExitReasonInference:
    def test_profitable_long_is_take_profit(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert trades[0].reason == "take_profit"

    def test_unprofitable_long_is_stop_loss(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="95.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert trades[0].reason == "stop_loss"

    def test_profitable_short_is_take_profit(self):
        fills = [
            make_fill(dir="Open Short", side="A", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Short", side="B", px="90.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert trades[0].reason == "take_profit"

    def test_unprofitable_short_is_stop_loss(self):
        fills = [
            make_fill(dir="Open Short", side="A", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Short", side="B", px="105.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        assert trades[0].reason == "stop_loss"


# ---------------------------------------------------------------------------
# Flip fills ("Long > Short" / "Short > Long")
# ---------------------------------------------------------------------------
class TestFlipFills:
    def test_long_to_short_flip_produces_close_trade(self):
        """Open Long → Long>Short should close the LONG and open a SHORT."""
        fills = [
            make_fill(dir="Open Long", px="100.00", sz="5.0",
                      time=1700000000000, start_position="0.0"),
            make_fill(dir="Long > Short", px="95.00", sz="10.0",
                      time=1700000060000, start_position="5.0"),
            make_fill(dir="Close Short", px="90.00", sz="5.0",
                      time=1700000120000, start_position="-5.0"),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 2

        # First trade: LONG closed at a loss by the flip
        assert trades[0].side == "LONG"
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 95.0
        assert trades[0].reason == "reversal_close"
        assert trades[0].size == 5.0

        # Second trade: SHORT opened by the flip, closed later
        assert trades[1].side == "SHORT"
        assert trades[1].entry_price == 95.0
        assert trades[1].exit_price == 90.0
        assert trades[1].size == 5.0

    def test_short_to_long_flip_produces_close_trade(self):
        """Open Short → Short>Long should close the SHORT and open a LONG."""
        fills = [
            make_fill(dir="Open Short", px="100.00", sz="5.0",
                      time=1700000000000, start_position="0.0"),
            make_fill(dir="Short > Long", px="105.00", sz="10.0",
                      time=1700000060000, start_position="-5.0"),
            make_fill(dir="Close Long", px="110.00", sz="5.0",
                      time=1700000120000, start_position="5.0"),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 2

        # First trade: SHORT closed at a loss by the flip
        assert trades[0].side == "SHORT"
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 105.0
        assert trades[0].reason == "reversal_close"

        # Second trade: LONG opened by the flip, closed later
        assert trades[1].side == "LONG"
        assert trades[1].entry_price == 105.0
        assert trades[1].exit_price == 110.0

    def test_flip_with_residual_size(self):
        """Flip size > 2x position creates correct residual on new side."""
        fills = [
            make_fill(dir="Open Long", px="100.00", sz="4.0",
                      time=1700000000000, start_position="0.0"),
            # Flip: close 4 LONG + open 8 SHORT = total 12
            make_fill(dir="Long > Short", px="98.00", sz="12.0",
                      time=1700000060000, start_position="4.0"),
        ]
        pairer = FillPairer()
        trades = pairer.process_fills(fills)
        assert len(trades) == 1
        assert trades[0].side == "LONG"
        assert trades[0].size == 4.0

        # New SHORT position with 8.0 remaining
        assert pairer.open_position is not None
        assert pairer.open_position["side"] == "SHORT"
        assert abs(pairer.open_position["remaining_size"] - 8.0) < 0.001

    def test_flip_marks_close_as_reversal(self):
        """The close portion of a flip is always marked as reversal_close."""
        fills = [
            make_fill(dir="Open Long", px="100.00", sz="5.0",
                      time=1700000000000, start_position="0.0"),
            make_fill(dir="Long > Short", px="102.00", sz="10.0",
                      time=1700000060000, start_position="5.0"),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        assert trades[0].reason == "reversal_close"

    def test_flip_without_tracked_position_opens_new(self):
        """A flip fill when no position is tracked should still open the new side."""
        fills = [
            # No prior open — just a flip (e.g., query window started mid-position)
            make_fill(dir="Long > Short", px="100.00", sz="10.0",
                      time=1700000060000, start_position="5.0"),
            make_fill(dir="Close Short", px="95.00", sz="5.0",
                      time=1700000120000, start_position="-5.0"),
        ]
        trades = pair_fills_to_trades(fills)
        # Should produce 1 trade: the SHORT opened by the flip, closed later
        assert len(trades) == 1
        assert trades[0].side == "SHORT"
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 95.0

    def test_real_hype_sequence(self):
        """Replay the actual HYPE fill sequence from 2026-02-28.

        39 raw fills should produce 8 round-trip trades.
        """
        fills = [
            # Trade 1: Open Short → Close Short
            make_fill(dir="Open Short", px="26.621", sz="11.26",
                      time=1740725886000, start_position="0.0"),
            make_fill(dir="Close Short", px="26.561", sz="11.26",
                      time=1740725934000, start_position="-11.26"),
            # Trade 2: Open Long
            make_fill(dir="Open Long", px="26.89", sz="11.15",
                      time=1740726126000, start_position="0.0"),
            # Trade 2 close + Trade 3 open via flip
            make_fill(dir="Long > Short", px="26.505", sz="11.32",
                      time=1740726737000, start_position="11.15"),
            # Trade 3: Close residual Short (0.17)
            make_fill(dir="Close Short", px="26.436", sz="0.17",
                      time=1740726913000, start_position="-0.17"),
            # Trade 4: Open Long → Close Long
            make_fill(dir="Open Long", px="26.541", sz="11.3",
                      time=1740727480000, start_position="0.0"),
            make_fill(dir="Close Long", px="26.337", sz="11.3",
                      time=1740727819000, start_position="11.3"),
            # Trade 5: Open Short → Close Short
            make_fill(dir="Open Short", px="26.172", sz="11.47",
                      time=1740728199000, start_position="0.0"),
            make_fill(dir="Close Short", px="26.381", sz="11.47",
                      time=1740728512000, start_position="-11.47"),
            # Trade 6: Open Long → Close Long
            make_fill(dir="Open Long", px="26.51", sz="10.88",
                      time=1740728799000, start_position="0.0"),
            make_fill(dir="Close Long", px="26.71", sz="10.88",
                      time=1740728964000, start_position="10.88"),
            # Trade 7: Open Short → Close Short
            make_fill(dir="Open Short", px="27.279", sz="10.99",
                      time=1740746214000, start_position="0.0"),
            make_fill(dir="Close Short", px="27.212", sz="10.99",
                      time=1740746252000, start_position="-10.99"),
            # Trade 8: Open Long → Close Long
            make_fill(dir="Open Long", px="27.551", sz="10.89",
                      time=1740752927000, start_position="0.0"),
            make_fill(dir="Close Long", px="27.617", sz="10.89",
                      time=1740752978000, start_position="10.89"),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 8

        # Trade 1: SHORT, profitable
        assert trades[0].side == "SHORT"
        assert trades[0].pnl_usd > 0

        # Trade 2: LONG closed by flip, losing
        assert trades[1].side == "LONG"
        assert trades[1].entry_price == 26.89
        assert abs(trades[1].exit_price - 26.505) < 0.001
        assert trades[1].reason == "reversal_close"
        assert trades[1].pnl_usd < 0

        # Trade 3: SHORT residual from flip (0.17 units)
        assert trades[2].side == "SHORT"
        assert abs(trades[2].size - 0.17) < 0.01

        # Trade 4: LONG, losing
        assert trades[3].side == "LONG"
        assert trades[3].pnl_usd < 0

        # Trade 5: SHORT, losing (price went up against short)
        assert trades[4].side == "SHORT"
        assert trades[4].pnl_usd < 0

        # Trade 6: LONG, profitable
        assert trades[5].side == "LONG"
        assert trades[5].pnl_usd > 0

        # Trade 7: SHORT, profitable
        assert trades[6].side == "SHORT"
        assert trades[6].pnl_usd > 0

        # Trade 8: LONG, profitable
        assert trades[7].side == "LONG"
        assert trades[7].pnl_usd > 0


# ---------------------------------------------------------------------------
# Orphan positions (open without close)
# ---------------------------------------------------------------------------
class TestOrphanPositions:
    def test_open_without_close_produces_no_trade(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
        ]
        trades = pair_fills_to_trades(fills)
        assert trades == []

    def test_pairer_tracks_open_position(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
        ]
        pairer = FillPairer()
        pairer.process_fills(fills)
        assert pairer.open_position is not None
        assert pairer.open_position["side"] == "LONG"
        assert abs(pairer.open_position["remaining_size"] - 5.0) < 0.001


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------
class TestTimestampHandling:
    def test_entry_exit_times_in_seconds(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        trades = pair_fills_to_trades(fills)
        t = trades[0]
        # API times are in ms, TradeRecord should be in seconds
        assert t.entry_time == 1700000000.0
        assert t.exit_time == 1700000060.0

    def test_unsorted_fills_still_pair_correctly(self):
        fills = [
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
        ]
        trades = pair_fills_to_trades(fills)
        assert len(trades) == 1
        assert trades[0].entry_price == 100.0
        assert trades[0].exit_price == 110.0


# ---------------------------------------------------------------------------
# Integration with compute_metrics
# ---------------------------------------------------------------------------
class TestMetricsCompatibility:
    def test_trade_records_work_with_compute_metrics(self):
        """Verify output TradeRecords are compatible with backtesting metrics."""
        from backtesting.metrics import compute_metrics

        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
            make_fill(dir="Open Short", side="A", px="110.00", sz="5.0",
                      time=1700000120000),
            make_fill(dir="Close Short", side="B", px="105.00", sz="5.0",
                      time=1700000180000),
        ]
        trades = pair_fills_to_trades(fills)
        metrics = compute_metrics(trades, total_signals=4, days=1.0)
        assert metrics.total_trades == 2
        assert metrics.wins == 2
        assert metrics.net_pnl_usd > 0
