"""Claude reasoner for Archon trade decisions.

Calls Claude with market context and returns structured trade decisions.
Falls back to heuristic logic when AI is unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from strategies.archon.context import MarketContext

logger = logging.getLogger(__name__)

# System prompt that guides Claude's trading decisions
TRADE_DECISION_PROMPT = """\
You are Archon, an intelligent crypto trading agent managing a HYPE perpetual futures position on Hyperliquid.

Your edge comes from the Directional Change (DC) framework:
- DC events detect when price moves by a threshold amount (e.g., 2%)
- After a DC confirmation, price tends to "overshoot" in the same direction
- Your job: decide whether to trade the overshoot, and with what parameters

Key principles:
1. CAPITAL PRESERVATION is paramount — this is a $10 account, every dollar matters
2. Only trade when you have a clear edge — skipping is better than a bad trade
3. In ranging/quiet markets, overshoot tends to be small — use tighter TP
4. In trending markets, overshoot can be large — let winners run
5. After consecutive losses, be more conservative (wider confidence threshold)
6. LONG positions work better in uptrends — HYPE has been trending up since March 2026
7. Fee per round-trip is ~0.09% (0.045% taker each way) — need decent moves to profit

You will receive market context with: price action, DC events, trade history, and position state.

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "action": "enter_long" | "enter_short" | "close" | "skip",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<1-2 sentence explanation>",
    "tp_pct": <take profit as decimal, e.g. 0.008 for 0.8%>,
    "sl_pct": <stop loss as decimal, e.g. 0.015 for 1.5%>
}

Rules:
- "skip" if confidence < 0.5 or market is unclear
- "close" if you think the current position should be exited
- tp_pct and sl_pct only matter for "enter_long" or "enter_short"
- confidence should reflect how certain you are about the trade
"""


@dataclass
class TradeDecision:
    """A structured trade decision from Claude or heuristic."""

    action: str  # "enter_long", "enter_short", "close", "skip"
    confidence: float  # 0.0 - 1.0
    reasoning: str
    tp_pct: float = 0.008
    sl_pct: float = 0.015
    source: str = "heuristic"  # "claude" or "heuristic"


class ArchonReasoner:
    """Makes trade decisions using Claude or heuristic fallback.

    Calls Claude Agent SDK for intelligent analysis, with graceful
    fallback to rule-based logic when AI is unavailable.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        use_ai: bool = True,
        min_confidence: float = 0.6,
        max_calls_per_hour: int = 30,
        direction_filter: str = "long",
        system_prompt: str = "",
    ):
        self._model = model
        self._use_ai = use_ai
        self._min_confidence = min_confidence
        self._max_calls_per_hour = max_calls_per_hour
        self._direction_filter = direction_filter
        self._system_prompt = system_prompt or TRADE_DECISION_PROMPT

        # Rate limiting
        self._call_timestamps: list[float] = []

        # Stats
        self.ai_calls = 0
        self.ai_successes = 0
        self.ai_failures = 0
        self.heuristic_calls = 0

    def _check_rate_limit(self) -> bool:
        """Check if we're within the hourly call budget."""
        now = time.time()
        cutoff = now - 3600
        self._call_timestamps = [t for t in self._call_timestamps if t > cutoff]
        return len(self._call_timestamps) < self._max_calls_per_hour

    async def decide(self, context: MarketContext) -> TradeDecision:
        """Make a trade decision based on market context.

        Tries Claude first (if enabled and within rate limits),
        falls back to heuristic logic.
        """
        if self._use_ai and self._check_rate_limit():
            try:
                decision = await self._decide_with_claude(context)
                self.ai_calls += 1
                self.ai_successes += 1
                self._call_timestamps.append(time.time())
                return decision
            except Exception as e:
                self.ai_calls += 1
                self.ai_failures += 1
                logger.warning("Claude decision failed, using heuristic: %s", e)

        self.heuristic_calls += 1
        return self._decide_heuristic(context)

    async def _decide_with_claude(self, context: MarketContext) -> TradeDecision:
        """Call Claude API for a trade decision.

        Uses the anthropic Python SDK for direct API calls.
        Requires ANTHROPIC_API_KEY environment variable.
        """
        import anthropic

        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env

        prompt = context.to_prompt_text()

        message = client.messages.create(
            model=self._model,
            max_tokens=256,
            system=self._system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        return self._parse_claude_response(response_text, context)

    def _parse_claude_response(
        self, response_text: str, context: MarketContext
    ) -> TradeDecision:
        """Parse Claude's JSON response into a TradeDecision."""
        text = response_text.strip()
        # Strip markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        action = data.get("action", "skip")
        confidence = float(data.get("confidence", 0.0))

        # Apply direction filter
        if self._direction_filter == "long" and action == "enter_short":
            action = "skip"
            confidence = 0.0
        elif self._direction_filter == "short" and action == "enter_long":
            action = "skip"
            confidence = 0.0

        # Apply confidence threshold
        if action in ("enter_long", "enter_short") and confidence < self._min_confidence:
            return TradeDecision(
                action="skip",
                confidence=confidence,
                reasoning=f"Below confidence threshold ({confidence:.2f} < {self._min_confidence}). "
                f"Original: {data.get('reasoning', '')}",
                source="claude",
            )

        return TradeDecision(
            action=action,
            confidence=confidence,
            reasoning=data.get("reasoning", "No reasoning provided"),
            tp_pct=float(data.get("tp_pct", 0.008)),
            sl_pct=float(data.get("sl_pct", 0.015)),
            source="claude",
        )

    def _decide_heuristic(self, context: MarketContext) -> TradeDecision:
        """Rule-based fallback when Claude is unavailable.

        Enhanced heuristic with price-structure awareness:
        - Enter LONG on PDCC2_UP in quiet/neutral regime
        - Close on PDCC_Down when in position
        - Skip in choppy regime or after consecutive losses
        - Score entry quality based on trend alignment, range position, DC momentum
        """
        event_type = context.trigger_event.get("event_type", "")
        regime = context.regime
        has_position = context.has_position
        consecutive_losses = context.consecutive_losses

        # Skip in choppy regime
        if regime == "choppy":
            return TradeDecision(
                action="skip",
                confidence=0.3,
                reasoning=f"Choppy regime ({context.sensor_event_rate:.1f} events/min), skipping.",
            )

        # Cool down after losses (escalating threshold)
        loss_threshold = 3 if consecutive_losses < 3 else 4
        if consecutive_losses >= loss_threshold:
            return TradeDecision(
                action="skip",
                confidence=0.2,
                reasoning=f"Loss guard: {consecutive_losses} consecutive losses, cooling down.",
            )

        # Close existing position on opposing DC event
        if has_position:
            pos_side = context.position_side
            if pos_side == "LONG" and event_type == "PDCC_Down":
                return TradeDecision(
                    action="close",
                    confidence=0.7,
                    reasoning="DC Down while LONG — closing position.",
                )
            if pos_side == "SHORT" and event_type == "PDCC2_UP":
                return TradeDecision(
                    action="close",
                    confidence=0.7,
                    reasoning="DC Up while SHORT — closing position.",
                )
            # Same direction — already in position
            return TradeDecision(
                action="skip",
                confidence=0.5,
                reasoning=f"Already {pos_side}, same-direction DC event.",
            )

        # Direction filter
        if self._direction_filter == "long" and event_type == "PDCC_Down":
            return TradeDecision(
                action="skip",
                confidence=0.4,
                reasoning="Long-only mode, skipping DC Down entry.",
            )
        if self._direction_filter == "short" and event_type == "PDCC2_UP":
            return TradeDecision(
                action="skip",
                confidence=0.4,
                reasoning="Short-only mode, skipping DC Up entry.",
            )

        # === Enhanced entry scoring ===
        # Start with base confidence and adjust based on market context.
        # Base 0.58 so that without positive signals, entries need at least
        # one confirming factor to clear the 0.60 min_confidence threshold.
        # Live validation (2026-03-25): both "high in range" entries at
        # conf=0.55 were marginal — one won +0.13%, one lost -1.51%.
        confidence = 0.58
        reasons = []

        # Factor 0: Price range quality filter
        # Narrow ranges (<0.3%) correlate with losing entries
        if context.price_high > context.price_low and context.tick_count >= 10:
            range_pct = (context.price_high - context.price_low) / context.price_low * 100
            if range_pct < 0.3:
                confidence -= 0.10
                reasons.append(f"narrow range ({range_pct:.1f}%)")

        # Factor 1: Trend alignment (+/- 0.10)
        trend = context.price_trend_pct
        if event_type == "PDCC2_UP" and trend > 0.5:
            confidence += 0.10
            reasons.append(f"trend aligned ({trend:+.1f}%)")
        elif event_type == "PDCC2_UP" and trend < -1.0:
            confidence -= 0.10
            reasons.append(f"counter-trend ({trend:+.1f}%)")
        elif event_type == "PDCC_Down" and trend < -0.5:
            confidence += 0.10
            reasons.append(f"trend aligned ({trend:+.1f}%)")
        elif event_type == "PDCC_Down" and trend > 1.0:
            confidence -= 0.10
            reasons.append(f"counter-trend ({trend:+.1f}%)")

        # Factor 2: Range position (+/- 0.05)
        # For longs, prefer buying in lower half of range
        if context.price_high > context.price_low:
            range_position = (context.current_price - context.price_low) / (context.price_high - context.price_low)
            if event_type == "PDCC2_UP" and range_position < 0.4:
                confidence += 0.05
                reasons.append("low in range")
            elif event_type == "PDCC2_UP" and range_position > 0.8:
                confidence -= 0.05
                reasons.append("high in range")

        # Factor 3: DC momentum — are recent events consistent? (+/- 0.05)
        if context.dc_up_count > 0 or context.dc_down_count > 0:
            total_dc = context.dc_up_count + context.dc_down_count
            if event_type == "PDCC2_UP" and context.dc_up_count / total_dc > 0.6:
                confidence += 0.05
                reasons.append("DC momentum up")
            elif event_type == "PDCC_Down" and context.dc_down_count / total_dc > 0.6:
                confidence += 0.05
                reasons.append("DC momentum down")

        # Factor 4: Recent trade performance (+/- 0.05)
        if context.total_trades >= 3:
            recent_wr = context.win_count / context.total_trades
            if recent_wr > 0.6:
                confidence += 0.05
                reasons.append(f"hot streak ({recent_wr:.0%} WR)")
            elif recent_wr < 0.3:
                confidence -= 0.05
                reasons.append(f"cold streak ({recent_wr:.0%} WR)")

        # Calculate adaptive TP from overshoot history
        tp_pct = 0.008  # default
        if context.avg_overshoot_pct > 0:
            tp_pct = max(0.003, context.avg_overshoot_pct / 100 * 0.4)

        # Adaptive SL: tighter when confident, wider when less sure
        sl_pct = 0.015
        if confidence >= 0.7:
            sl_pct = 0.012  # tighter SL for high-confidence entries
        elif confidence < 0.55:
            sl_pct = 0.018  # wider SL for marginal entries

        # Build reasoning string
        reason_str = ", ".join(reasons) if reasons else "neutral signals"

        if event_type == "PDCC2_UP":
            return TradeDecision(
                action="enter_long",
                confidence=confidence,
                reasoning=f"DC Up in {regime}: {reason_str}.",
                tp_pct=tp_pct,
                sl_pct=sl_pct,
            )

        if event_type == "PDCC_Down":
            return TradeDecision(
                action="enter_short",
                confidence=confidence,
                reasoning=f"DC Down in {regime}: {reason_str}.",
                tp_pct=tp_pct,
                sl_pct=sl_pct,
            )

        return TradeDecision(
            action="skip",
            confidence=0.3,
            reasoning=f"Unrecognized event type: {event_type}",
        )

    def get_stats(self) -> dict:
        """Return reasoner statistics."""
        return {
            "ai_calls": self.ai_calls,
            "ai_successes": self.ai_successes,
            "ai_failures": self.ai_failures,
            "heuristic_calls": self.heuristic_calls,
            "rate_limit_remaining": self._max_calls_per_hour - len(
                [t for t in self._call_timestamps if t > time.time() - 3600]
            ),
        }
