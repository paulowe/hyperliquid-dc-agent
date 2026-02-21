"""Tests for observer CLI argument parsing."""

import pytest

from agents.observer.cli import build_parser


class TestBuildParser:
    """CLI argument parser construction."""

    def test_explore_command_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["explore", "--symbol", "SOL"])
        assert args.command == "explore"
        assert args.symbol == "SOL"
        assert args.duration == 30
        assert args.no_ai is False

    def test_explore_custom_duration(self):
        parser = build_parser()
        args = parser.parse_args(["explore", "--symbol", "BTC", "--duration", "10"])
        assert args.duration == 10

    def test_explore_custom_thresholds(self):
        parser = build_parser()
        args = parser.parse_args(["explore", "--thresholds", "0.005", "0.01", "0.02"])
        assert args.thresholds == [0.005, 0.01, 0.02]

    def test_explore_no_ai_flag(self):
        parser = build_parser()
        args = parser.parse_args(["explore", "--no-ai"])
        assert args.no_ai is True

    def test_status_command(self):
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_stop_command(self):
        parser = build_parser()
        args = parser.parse_args(["stop"])
        assert args.command == "stop"

    def test_loop_command_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["loop", "--symbol", "SOL"])
        assert args.command == "loop"
        assert args.symbol == "SOL"
        assert args.rounds == 0  # 0 = infinite

    def test_loop_custom_rounds(self):
        parser = build_parser()
        args = parser.parse_args(["loop", "--rounds", "5", "--symbol", "ETH"])
        assert args.rounds == 5
        assert args.symbol == "ETH"
