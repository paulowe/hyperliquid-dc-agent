"""Tests for ClaudeReasoner — AI analysis and heuristic fallback."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agents.observer.collector import SessionReport, PnLSummary
from agents.observer.reasoner import ClaudeReasoner, ExplorationAnalysis, heuristic_rank


def make_report(threshold, signal_count=10, duration=1800.0):
    """Create a minimal SessionReport for testing."""
    return SessionReport(
        session_id=f"test_{threshold}",
        symbol="SOL",
        threshold=threshold,
        sl_pct=threshold,
        tp_pct=threshold * 0.5,
        trail_pct=0.5,
        position_size_usd=100.0,
        leverage=10,
        duration_seconds=duration,
        tick_count=3000,
        signal_count=signal_count,
        dc_event_count=signal_count * 2,
        signals=[],
    )


def make_pnl(net_pnl, total_trades=5, wins=3, losses=2):
    """Create a minimal PnLSummary for testing."""
    return PnLSummary(
        total_trades=total_trades,
        gross_pnl_usd=net_pnl + 0.10,  # Assume small fee
        total_fees_usd=0.10,
        net_pnl_usd=net_pnl,
        wins=wins,
        losses=losses,
    )


class TestHeuristicRank:
    """Heuristic fallback ranking (no AI needed)."""

    def test_ranks_by_net_pnl(self):
        """Higher net PnL should rank first."""
        report_pnl_pairs = [
            (make_report(0.004), make_pnl(net_pnl=-0.50)),
            (make_report(0.01), make_pnl(net_pnl=1.20)),
            (make_report(0.015), make_pnl(net_pnl=0.80)),
        ]
        result = heuristic_rank(report_pnl_pairs)

        assert result.best_threshold == 0.01
        assert result.confidence > 0

    def test_prefers_higher_win_rate_on_tie(self):
        """When PnL is similar, higher win rate should rank higher."""
        report_pnl_pairs = [
            (make_report(0.004), make_pnl(net_pnl=1.0, wins=3, losses=2)),
            (make_report(0.01), make_pnl(net_pnl=1.0, wins=4, losses=1)),
        ]
        result = heuristic_rank(report_pnl_pairs)

        assert result.best_threshold == 0.01

    def test_handles_no_trades(self):
        """Sessions with no trades should rank lowest."""
        report_pnl_pairs = [
            (make_report(0.004), make_pnl(net_pnl=0.0, total_trades=0, wins=0, losses=0)),
            (make_report(0.01), make_pnl(net_pnl=0.50, total_trades=3, wins=2, losses=1)),
        ]
        result = heuristic_rank(report_pnl_pairs)

        assert result.best_threshold == 0.01

    def test_handles_all_negative(self):
        """Should still pick the least negative."""
        report_pnl_pairs = [
            (make_report(0.004), make_pnl(net_pnl=-2.0)),
            (make_report(0.01), make_pnl(net_pnl=-0.50)),
        ]
        result = heuristic_rank(report_pnl_pairs)

        assert result.best_threshold == 0.01

    def test_empty_input(self):
        """Empty input should return a default analysis."""
        result = heuristic_rank([])

        assert result.best_threshold is None
        assert result.confidence == 0.0


class TestExplorationAnalysis:
    """ExplorationAnalysis dataclass."""

    def test_creates_with_fields(self):
        analysis = ExplorationAnalysis(
            best_threshold=0.015,
            best_config={"threshold": 0.015, "sl_pct": 0.015},
            reasoning="Highest net PnL with good win rate",
            confidence=0.85,
            suggestions=["Try wider SL for volatile markets"],
        )
        assert analysis.best_threshold == 0.015
        assert analysis.confidence == 0.85


class TestClaudeReasonerPromptBuilding:
    """Test prompt construction (no actual API calls)."""

    def test_build_prompt_includes_all_reports(self):
        reasoner = ClaudeReasoner(model="claude-sonnet-4-20250514")
        reports = [make_report(0.004), make_report(0.01), make_report(0.015)]
        pnls = [make_pnl(0.5), make_pnl(1.2), make_pnl(0.8)]
        pairs = list(zip(reports, pnls))

        prompt = reasoner._build_analysis_prompt(pairs, {"round": 1})
        assert "0.004" in prompt
        assert "0.01" in prompt
        assert "0.015" in prompt
        assert "SOL" in prompt

    def test_build_prompt_includes_pnl_data(self):
        reasoner = ClaudeReasoner(model="claude-sonnet-4-20250514")
        pairs = [(make_report(0.01), make_pnl(1.5, wins=4, losses=1))]

        prompt = reasoner._build_analysis_prompt(pairs, {"round": 1})
        assert "1.5" in prompt or "1.50" in prompt
        assert "4" in prompt  # wins


class TestClaudeReasonerFallback:
    """Test that reasoner falls back to heuristic when SDK unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_ai(self):
        """--no-ai mode should use heuristic ranking."""
        reasoner = ClaudeReasoner(model="claude-sonnet-4-20250514", use_ai=False)
        reports = [make_report(0.004), make_report(0.01)]
        pnls = [make_pnl(-0.5), make_pnl(1.2)]
        pairs = list(zip(reports, pnls))

        result = await reasoner.analyze(pairs, {"round": 1})
        assert result.best_threshold == 0.01
        assert "heuristic" in result.reasoning.lower()
