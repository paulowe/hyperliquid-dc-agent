"""Tests for trade review CLI."""

from __future__ import annotations

import json

import pytest

from trade_review.cli import build_parser, format_human_readable, format_json


def make_fill(
    coin: str = "HYPE",
    side: str = "B",
    dir: str = "Open Long",
    px: str = "25.50",
    sz: str = "10.0",
    time: int = 1700000000000,
) -> dict:
    """Create a fill dict matching Hyperliquid API format."""
    return {
        "coin": coin, "side": side, "dir": dir, "px": px,
        "sz": sz, "time": time, "oid": 100,
        "closedPnl": "0.0", "startPosition": "0.0",
        "crossed": True, "hash": "0x" + "a" * 64,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
class TestBuildParser:
    def test_symbol_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_hours(self):
        parser = build_parser()
        args = parser.parse_args(["--symbol", "HYPE"])
        assert args.hours == 24.0

    def test_days_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--symbol", "HYPE", "--days", "7"])
        assert args.days == 7.0

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--symbol", "HYPE", "--json"])
        assert args.json_output is True

    def test_network_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--symbol", "HYPE", "--network", "testnet"])
        assert args.network == "testnet"

    def test_wallet_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--symbol", "HYPE", "--wallet", "0xABC123"])
        assert args.wallet == "0xABC123"


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------
class TestJsonOutput:
    def test_json_structure_with_trades(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        output = format_json(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=fills,
        )
        data = json.loads(output)
        assert data["symbol"] == "HYPE"
        assert data["period_hours"] == 24.0
        assert data["network"] == "mainnet"
        assert data["total_fills"] == 2
        assert len(data["trades"]) == 1
        assert data["metrics"]["total_trades"] == 1
        assert data["metrics"]["wins"] == 1

    def test_json_structure_no_trades(self):
        output = format_json(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=[],
        )
        data = json.loads(output)
        assert data["total_fills"] == 0
        assert data["trades"] == []
        assert data["metrics"] is None

    def test_json_trade_has_iso_times(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        output = format_json(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=fills,
        )
        data = json.loads(output)
        trade = data["trades"][0]
        assert "entry_time_iso" in trade
        assert "exit_time_iso" in trade

    def test_json_open_position_reported(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
        ]
        output = format_json(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=fills,
        )
        data = json.loads(output)
        assert data["open_position"] is not None
        assert data["open_position"]["side"] == "LONG"


# ---------------------------------------------------------------------------
# Human-readable output format
# ---------------------------------------------------------------------------
class TestHumanReadableOutput:
    def test_contains_symbol(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        output = format_human_readable(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=fills,
        )
        assert "HYPE" in output
        assert "mainnet" in output

    def test_no_trades_message(self):
        output = format_human_readable(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=[],
        )
        assert "No completed trades" in output

    def test_contains_key_metrics(self):
        fills = [
            make_fill(dir="Open Long", side="B", px="100.00", sz="5.0",
                      time=1700000000000),
            make_fill(dir="Close Long", side="A", px="110.00", sz="5.0",
                      time=1700000060000),
        ]
        output = format_human_readable(
            symbol="HYPE", period_hours=24.0, network="mainnet",
            wallet="0xTEST", fills=fills,
        )
        assert "Win rate" in output
        assert "Net P&L" in output
        assert "Profit factor" in output
