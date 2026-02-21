"""Tests for ResultCollector — JSON report parsing and simulated PnL."""

import json
import pytest
from pathlib import Path

from agents.observer.collector import ResultCollector, SessionReport, PnLSummary


def make_sample_report(
    symbol="SOL",
    threshold=0.015,
    signal_count=5,
    tick_count=3000,
    signals=None,
):
    """Create a sample JSON report dict."""
    if signals is None:
        signals = [
            {"timestamp": 1000.0, "type": "SELL", "price": 145.0, "size": 0.69, "reason": "dc_overshoot_short: PDCC_Down", "metadata": {}},
            {"timestamp": 1050.0, "type": "CLOSE", "price": 143.5, "size": 0.69, "reason": "take_profit", "metadata": {"side": "SHORT"}},
            {"timestamp": 1100.0, "type": "BUY", "price": 144.0, "size": 0.69, "reason": "dc_overshoot_long: PDCC2_UP", "metadata": {}},
            {"timestamp": 1150.0, "type": "CLOSE", "price": 143.0, "size": 0.69, "reason": "stop_loss", "metadata": {"side": "LONG"}},
            {"timestamp": 1200.0, "type": "SELL", "price": 142.5, "size": 0.70, "reason": "dc_overshoot_short: PDCC_Down", "metadata": {}},
        ]
    return {
        "symbol": symbol,
        "threshold": threshold,
        "sl_pct": 0.015,
        "tp_pct": 0.005,
        "trail_pct": 0.5,
        "min_profit_to_trail_pct": 0.001,
        "position_size_usd": 100.0,
        "leverage": 10,
        "observe_only": True,
        "duration_seconds": 1800.0,
        "start_time": 900.0,
        "end_time": 2700.0,
        "tick_count": tick_count,
        "signal_count": signal_count,
        "trade_count": 0,
        "dc_event_count": 20,
        "signals": signals,
        "strategy_status": {"tick_count": tick_count, "dc_event_count": 20},
    }


@pytest.fixture
def report_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return d


class TestCollectReports:
    """Reading JSON reports from disk."""

    def test_collect_reads_valid_report(self, report_dir):
        # Write a report file
        report_data = make_sample_report()
        report_path = report_dir / "report_abc123.json"
        report_path.write_text(json.dumps(report_data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["abc123"])
        assert len(reports) == 1
        assert reports[0].session_id == "abc123"
        assert reports[0].symbol == "SOL"
        assert reports[0].threshold == 0.015
        assert reports[0].signal_count == 5

    def test_collect_skips_missing_report(self, report_dir):
        collector = ResultCollector(report_dir)
        reports = collector.collect(["nonexistent"])
        assert len(reports) == 0

    def test_collect_multiple_reports(self, report_dir):
        for sid, thresh in [("aaa", 0.004), ("bbb", 0.01), ("ccc", 0.015)]:
            data = make_sample_report(threshold=thresh)
            (report_dir / f"report_{sid}.json").write_text(json.dumps(data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["aaa", "bbb", "ccc"])
        assert len(reports) == 3
        thresholds = {r.threshold for r in reports}
        assert thresholds == {0.004, 0.01, 0.015}

    def test_collect_handles_corrupt_json(self, report_dir):
        (report_dir / "report_bad.json").write_text("not valid json{{{")
        collector = ResultCollector(report_dir)
        reports = collector.collect(["bad"])
        assert len(reports) == 0


class TestSessionReport:
    """SessionReport computed properties."""

    def test_signal_frequency(self, report_dir):
        report_data = make_sample_report(signal_count=12, tick_count=3000)
        report_data["duration_seconds"] = 1800.0
        (report_dir / "report_x.json").write_text(json.dumps(report_data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["x"])
        r = reports[0]
        # 12 signals / 30 minutes = 0.4 per minute
        assert r.signal_frequency_per_min == pytest.approx(0.4, abs=0.01)

    def test_signal_counts_by_type(self, report_dir):
        report_data = make_sample_report()
        (report_dir / "report_y.json").write_text(json.dumps(report_data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["y"])
        r = reports[0]
        assert r.buy_signals == 1
        assert r.sell_signals == 2
        assert r.close_signals == 2


class TestSimulatedPnL:
    """Simulated PnL computation from observed signals."""

    def test_profitable_short_trade(self, report_dir):
        """SHORT entry → price drops → CLOSE = profit."""
        signals = [
            {"timestamp": 1000.0, "type": "SELL", "price": 100.0, "size": 1.0, "reason": "short", "metadata": {}},
            {"timestamp": 1050.0, "type": "CLOSE", "price": 99.0, "size": 1.0, "reason": "take_profit", "metadata": {"side": "SHORT"}},
        ]
        report_data = make_sample_report(signals=signals, signal_count=2)
        (report_dir / "report_p.json").write_text(json.dumps(report_data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["p"])
        pnl = collector.compute_simulated_pnl(reports[0])

        assert pnl.total_trades == 1
        assert pnl.gross_pnl_usd > 0  # Profitable short

    def test_losing_long_trade(self, report_dir):
        """LONG entry → price drops → CLOSE = loss."""
        signals = [
            {"timestamp": 1000.0, "type": "BUY", "price": 100.0, "size": 1.0, "reason": "long", "metadata": {}},
            {"timestamp": 1050.0, "type": "CLOSE", "price": 99.0, "size": 1.0, "reason": "stop_loss", "metadata": {"side": "LONG"}},
        ]
        report_data = make_sample_report(signals=signals, signal_count=2)
        (report_dir / "report_l.json").write_text(json.dumps(report_data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["l"])
        pnl = collector.compute_simulated_pnl(reports[0])

        assert pnl.total_trades == 1
        assert pnl.gross_pnl_usd < 0  # Losing long

    def test_no_signals_returns_zero_pnl(self, report_dir):
        report_data = make_sample_report(signals=[], signal_count=0)
        (report_dir / "report_z.json").write_text(json.dumps(report_data))

        collector = ResultCollector(report_dir)
        reports = collector.collect(["z"])
        pnl = collector.compute_simulated_pnl(reports[0])

        assert pnl.total_trades == 0
        assert pnl.gross_pnl_usd == 0.0
