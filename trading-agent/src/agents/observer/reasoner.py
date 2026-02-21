"""ClaudeReasoner: AI-powered analysis of exploration results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from agents.observer.collector import SessionReport, PnLSummary

logger = logging.getLogger(__name__)

# System prompt for the trading analyst agent
TRADING_ANALYST_PROMPT = """\
You are a quantitative trading analyst evaluating DC Overshoot strategy configurations.

The DC Overshoot strategy:
- Detects Directional Change (DC) events when price moves by a threshold amount
- After a DC confirmation (PDCC), enters a position expecting price to overshoot
- Uses trailing stop-loss and take-profit for risk management
- Profitability depends on choosing the right threshold for current market conditions

Your task: analyze observe-only session results and recommend the best configuration.

You will receive data from multiple concurrent sessions, each running at a different
DC threshold. Each session reports: signal count, simulated PnL, win rate, and fees.

Output your analysis as JSON with this exact structure:
{
    "best_threshold": <float>,
    "reasoning": "<2-3 sentence explanation>",
    "confidence": <0.0-1.0>,
    "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}
"""


@dataclass
class ExplorationAnalysis:
    """Result of analyzing an exploration round."""

    best_threshold: float | None
    best_config: dict | None
    reasoning: str
    confidence: float
    suggestions: list[str] = field(default_factory=list)


def heuristic_rank(
    report_pnl_pairs: list[tuple[SessionReport, PnLSummary]],
) -> ExplorationAnalysis:
    """Rank sessions by net PnL (fallback when AI is unavailable).

    Score = net_pnl_usd + 0.1 * win_rate_bonus
    Win rate bonus: extra weight for higher win rates when PnL is similar.
    """
    if not report_pnl_pairs:
        return ExplorationAnalysis(
            best_threshold=None,
            best_config=None,
            reasoning="No data to analyze (heuristic fallback).",
            confidence=0.0,
        )

    scored = []
    for report, pnl in report_pnl_pairs:
        # Primary: net PnL. Secondary: win rate bonus
        win_rate_bonus = pnl.win_rate * 0.1 if pnl.total_trades > 0 else -1.0
        score = pnl.net_pnl_usd + win_rate_bonus
        scored.append((score, report, pnl))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_report, best_pnl = scored[0]

    # Confidence based on how much better the best is vs second-best
    if len(scored) >= 2:
        gap = best_score - scored[1][0]
        confidence = min(0.95, 0.5 + gap * 0.2)
    else:
        confidence = 0.5

    return ExplorationAnalysis(
        best_threshold=best_report.threshold,
        best_config={
            "threshold": best_report.threshold,
            "sl_pct": best_report.sl_pct,
            "tp_pct": best_report.tp_pct,
            "trail_pct": best_report.trail_pct,
        },
        reasoning=(
            f"Heuristic ranking: threshold {best_report.threshold} had highest "
            f"score (net PnL ${best_pnl.net_pnl_usd:+.2f}, "
            f"{best_pnl.wins}W/{best_pnl.losses}L, "
            f"win rate {best_pnl.win_rate:.0%})."
        ),
        confidence=max(0.0, min(1.0, confidence)),
    )


class ClaudeReasoner:
    """Uses Claude Agent SDK to reason about session results."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", use_ai: bool = True):
        self._model = model
        self._use_ai = use_ai

    def _build_analysis_prompt(
        self,
        report_pnl_pairs: list[tuple[SessionReport, PnLSummary]],
        market_context: dict,
    ) -> str:
        """Build a structured prompt with session results."""
        lines = [
            f"## Exploration Round {market_context.get('round', '?')}",
            f"Symbol: {report_pnl_pairs[0][0].symbol if report_pnl_pairs else 'N/A'}",
            "",
            "## Session Results",
            "",
        ]

        for report, pnl in report_pnl_pairs:
            lines.append(f"### Threshold: {report.threshold}")
            lines.append(f"  - Signals: {report.signal_count} ({report.signal_frequency_per_min:.2f}/min)")
            lines.append(f"  - Simulated trades: {pnl.total_trades}")
            lines.append(f"  - Wins/Losses: {pnl.wins}W / {pnl.losses}L (win rate: {pnl.win_rate:.0%})")
            lines.append(f"  - Gross PnL: ${pnl.gross_pnl_usd:+.2f}")
            lines.append(f"  - Fees: ${pnl.total_fees_usd:.2f}")
            lines.append(f"  - Net PnL: ${pnl.net_pnl_usd:+.2f}")
            lines.append(f"  - Config: SL={report.sl_pct}, TP={report.tp_pct}, trail={report.trail_pct}")
            lines.append("")

        lines.append("## Task")
        lines.append("Analyze these results and recommend the best threshold configuration.")
        lines.append("Output your analysis as JSON.")

        return "\n".join(lines)

    async def analyze(
        self,
        report_pnl_pairs: list[tuple[SessionReport, PnLSummary]],
        market_context: dict,
    ) -> ExplorationAnalysis:
        """Analyze exploration results. Uses AI or falls back to heuristic."""
        if not self._use_ai:
            return heuristic_rank(report_pnl_pairs)

        try:
            return await self._analyze_with_claude(report_pnl_pairs, market_context)
        except Exception as e:
            logger.warning("Claude analysis failed, falling back to heuristic: %s", e)
            return heuristic_rank(report_pnl_pairs)

    async def _analyze_with_claude(
        self,
        report_pnl_pairs: list[tuple[SessionReport, PnLSummary]],
        market_context: dict,
    ) -> ExplorationAnalysis:
        """Call Claude Agent SDK for analysis."""
        from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage

        prompt = self._build_analysis_prompt(report_pnl_pairs, market_context)

        options = ClaudeAgentOptions(
            model=self._model,
            system_prompt=TRADING_ANALYST_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=1,
            # No tools needed — pure reasoning task
            allowed_tools=[],
        )

        # Collect response text
        response_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        return self._parse_response(response_text, report_pnl_pairs)

    def _parse_response(
        self,
        response_text: str,
        report_pnl_pairs: list[tuple[SessionReport, PnLSummary]],
    ) -> ExplorationAnalysis:
        """Parse Claude's JSON response into an ExplorationAnalysis."""
        try:
            # Extract JSON from response (may be wrapped in markdown code blocks)
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            best_threshold = data.get("best_threshold")

            # Find the matching config from reports
            best_config = None
            for report, _ in report_pnl_pairs:
                if report.threshold == best_threshold:
                    best_config = {
                        "threshold": report.threshold,
                        "sl_pct": report.sl_pct,
                        "tp_pct": report.tp_pct,
                        "trail_pct": report.trail_pct,
                    }
                    break

            return ExplorationAnalysis(
                best_threshold=best_threshold,
                best_config=best_config,
                reasoning=data.get("reasoning", "No reasoning provided"),
                confidence=float(data.get("confidence", 0.5)),
                suggestions=data.get("suggestions", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse Claude response: %s", e)
            # Fall back to heuristic
            return heuristic_rank(report_pnl_pairs)
