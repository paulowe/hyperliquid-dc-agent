"""Market context builder for Archon strategy.

Collects and summarizes market state for Claude to analyze.
Maintains rolling windows of ticks, DC events, and trades.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PriceTick:
    """A single price observation."""
    price: float
    timestamp: float


@dataclass
class DCEvent:
    """A directional change event."""
    event_type: str  # "PDCC_Down" or "PDCC2_UP"
    price: float
    start_price: float
    end_price: float
    timestamp: float
    threshold: float
    is_sensor: bool = False


@dataclass
class TradeResult:
    """A completed trade."""
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    duration_s: float
    timestamp: float


@dataclass
class MarketContext:
    """Snapshot of market state for Claude analysis.

    Contains everything Claude needs to make a trade decision.
    """
    # Current state
    current_price: float
    timestamp: float
    symbol: str

    # Price action (last N ticks summarized)
    price_high: float
    price_low: float
    price_mean: float
    price_trend_pct: float  # % change over context window
    tick_count: int

    # DC event that triggered this analysis
    trigger_event: Dict[str, Any]

    # Recent DC events
    recent_dc_events: List[Dict[str, Any]]
    dc_up_count: int
    dc_down_count: int
    avg_overshoot_pct: float

    # Regime
    regime: str  # "quiet", "neutral", "choppy", "trending"
    sensor_event_rate: float  # events per minute

    # Trade history
    recent_trades: List[Dict[str, Any]]
    total_trades: int
    win_count: int
    loss_count: int
    consecutive_losses: int
    net_pnl_pct: float

    # Position state
    has_position: bool
    position_side: Optional[str] = None
    position_entry: Optional[float] = None
    position_pnl_pct: Optional[float] = None

    def to_prompt_text(self) -> str:
        """Format context as structured text for Claude prompt."""
        lines = [
            f"## Market Context — {self.symbol} @ ${self.current_price:.2f}",
            "",
            f"### Price Action (last {self.tick_count} ticks)",
            f"- Range: ${self.price_low:.2f} — ${self.price_high:.2f}",
            f"- Trend: {self.price_trend_pct:+.3f}%",
            f"- Regime: {self.regime} (sensor rate: {self.sensor_event_rate:.1f}/min)",
            "",
            f"### Trigger Event",
            f"- Type: {self.trigger_event.get('event_type', 'N/A')}",
            f"- Price moved: ${self.trigger_event.get('start_price', 0):.2f} → "
            f"${self.trigger_event.get('end_price', 0):.2f}",
            "",
            f"### Recent DC Events ({len(self.recent_dc_events)} events)",
            f"- Up confirmations: {self.dc_up_count} | Down confirmations: {self.dc_down_count}",
            f"- Avg overshoot: {self.avg_overshoot_pct:.3f}%",
        ]

        if self.recent_dc_events:
            lines.append("- Recent:")
            for ev in self.recent_dc_events[-5:]:
                lines.append(
                    f"  {ev['event_type']} @ ${ev['price']:.2f} "
                    f"(overshoot: {ev.get('overshoot_pct', 0):.3f}%)"
                )

        lines.extend([
            "",
            f"### Trade History ({self.total_trades} total)",
            f"- Wins: {self.win_count} | Losses: {self.loss_count}",
            f"- Consecutive losses: {self.consecutive_losses}",
            f"- Net P&L: {self.net_pnl_pct:+.3f}%",
        ])

        if self.recent_trades:
            lines.append("- Recent trades:")
            for t in self.recent_trades[-5:]:
                lines.append(
                    f"  {t['side']} {t['exit_reason']} "
                    f"pnl={t['pnl_pct']:+.3f}% ({t['duration_s']:.0f}s)"
                )

        if self.has_position:
            lines.extend([
                "",
                f"### Current Position",
                f"- {self.position_side} @ ${self.position_entry:.2f}",
                f"- Unrealized P&L: {self.position_pnl_pct:+.3f}%",
            ])
        else:
            lines.extend(["", "### Current Position: FLAT"])

        return "\n".join(lines)


class ContextBuilder:
    """Maintains rolling windows and builds MarketContext snapshots.

    Call record_tick(), record_dc_event(), record_trade() as data arrives.
    Call build() when a DC event triggers analysis.
    """

    def __init__(
        self,
        symbol: str,
        max_ticks: int = 60,
        max_dc_events: int = 20,
        max_trades: int = 20,
    ):
        self.symbol = symbol
        self._ticks: deque[PriceTick] = deque(maxlen=max_ticks)
        self._dc_events: deque[DCEvent] = deque(maxlen=max_dc_events)
        self._trades: deque[TradeResult] = deque(maxlen=max_trades)
        self._consecutive_losses = 0
        self._total_trades = 0
        self._win_count = 0
        self._loss_count = 0

        # Regime tracking from sensor events
        self._sensor_events: deque = deque(maxlen=100)

    def record_tick(self, price: float, timestamp: float) -> None:
        """Record a price tick."""
        self._ticks.append(PriceTick(price=price, timestamp=timestamp))

    def record_dc_event(self, event: DCEvent) -> None:
        """Record a DC event (trade-level or sensor-level)."""
        if event.is_sensor:
            direction = 1 if event.event_type == "PDCC2_UP" else -1
            self._sensor_events.append((event.timestamp, direction))
        else:
            self._dc_events.append(event)

    def record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade."""
        self._trades.append(trade)
        self._total_trades += 1
        if trade.pnl_pct > 0:
            self._win_count += 1
            self._consecutive_losses = 0
        else:
            self._loss_count += 1
            self._consecutive_losses += 1

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def get_regime(self, now: float, lookback: float = 600.0) -> str:
        """Classify current market regime from sensor events."""
        cutoff = now - lookback
        recent = [(t, d) for t, d in self._sensor_events if t > cutoff]

        if len(recent) < 3:
            return "quiet"

        # Event rate (events per minute)
        if recent:
            span = now - recent[0][0]
            rate = len(recent) / max(span / 60, 0.01)
        else:
            rate = 0.0

        if rate > 4.0:
            # High rate — check directional consistency
            directions = [d for _, d in recent]
            net = sum(directions)
            consistency = abs(net) / len(directions)
            if consistency < 0.4:
                return "choppy"
            return "trending"

        if len(recent) >= 3:
            return "neutral"

        return "quiet"

    def get_sensor_rate(self, now: float, lookback: float = 600.0) -> float:
        """Get sensor event rate (events per minute)."""
        cutoff = now - lookback
        recent = [(t, d) for t, d in self._sensor_events if t > cutoff]
        if not recent:
            return 0.0
        span = now - recent[0][0]
        return len(recent) / max(span / 60, 0.01)

    def build(
        self,
        trigger_event: Dict[str, Any],
        position_side: Optional[str] = None,
        position_entry: Optional[float] = None,
    ) -> MarketContext:
        """Build a MarketContext snapshot for Claude analysis.

        Args:
            trigger_event: The DC event that triggered this analysis
            position_side: Current position side ("LONG"/"SHORT") or None
            position_entry: Current position entry price or None

        Returns:
            MarketContext with all available data
        """
        now = time.time()
        prices = [t.price for t in self._ticks]
        current_price = prices[-1] if prices else trigger_event.get("price", 0)

        # Price summary
        if len(prices) >= 2:
            price_high = max(prices)
            price_low = min(prices)
            price_mean = sum(prices) / len(prices)
            price_trend = (prices[-1] - prices[0]) / prices[0]
        else:
            price_high = price_low = price_mean = current_price
            price_trend = 0.0

        # DC event summary
        dc_list = list(self._dc_events)
        dc_up = sum(1 for e in dc_list if e.event_type == "PDCC2_UP")
        dc_down = sum(1 for e in dc_list if e.event_type == "PDCC_Down")

        overshoots = []
        for i in range(1, len(dc_list)):
            prev = dc_list[i - 1]
            curr = dc_list[i]
            if prev.event_type != curr.event_type:
                os_pct = abs(curr.end_price - prev.end_price) / prev.end_price
                overshoots.append(os_pct)

        avg_os = (sum(overshoots) / len(overshoots) * 100) if overshoots else 0.0

        # Format DC events for context
        recent_dc = [
            {
                "event_type": e.event_type,
                "price": e.price,
                "start_price": e.start_price,
                "end_price": e.end_price,
                "overshoot_pct": abs(e.end_price - e.start_price) / e.start_price * 100,
            }
            for e in dc_list[-10:]
        ]

        # Trade summary
        recent_trades_list = [
            {
                "side": t.side,
                "pnl_pct": t.pnl_pct * 100,
                "exit_reason": t.exit_reason,
                "duration_s": t.duration_s,
            }
            for t in list(self._trades)[-10:]
        ]

        net_pnl = sum(t.pnl_pct for t in self._trades) * 100

        # Position P&L
        pos_pnl = None
        if position_side and position_entry and current_price:
            if position_side == "LONG":
                pos_pnl = (current_price - position_entry) / position_entry * 100
            else:
                pos_pnl = (position_entry - current_price) / position_entry * 100

        regime = self.get_regime(now)
        sensor_rate = self.get_sensor_rate(now)

        return MarketContext(
            current_price=current_price,
            timestamp=now,
            symbol=self.symbol,
            price_high=price_high,
            price_low=price_low,
            price_mean=price_mean,
            price_trend_pct=price_trend * 100,
            tick_count=len(prices),
            trigger_event=trigger_event,
            recent_dc_events=recent_dc,
            dc_up_count=dc_up,
            dc_down_count=dc_down,
            avg_overshoot_pct=avg_os,
            regime=regime,
            sensor_event_rate=sensor_rate,
            recent_trades=recent_trades_list,
            total_trades=self._total_trades,
            win_count=self._win_count,
            loss_count=self._loss_count,
            consecutive_losses=self._consecutive_losses,
            net_pnl_pct=net_pnl,
            has_position=position_side is not None,
            position_side=position_side,
            position_entry=position_entry,
            position_pnl_pct=pos_pnl,
        )
