"""Basis trade strategy — delta-neutral funding rate arbitrage.

Trade Idea
----------
Category: Cash-and-carry ARBITRAGE (delta-neutral)

Mechanics:
    1. Buy spot HYPE (long exposure)
    2. Short HYPE perp (short exposure, equal notional)
    3. Net delta = 0 → immune to price movements
    4. Collect hourly funding payments (shorts receive when funding > 0)
    5. Close both legs when funding turns persistently negative

Profit Source:
    - Funding rate payments from long holders to short holders
    - On Hyperliquid: anchor rate 0.01%/8h ≈ 11.6% APR
    - Can spike to 0.1%+/h during trending markets

Risk Sources:
    - Negative funding (shorts pay longs) — mitigated by exit threshold
    - Execution slippage on entry/exit of both legs
    - Spot/perp basis divergence during high volatility
    - Exchange risk (smart contract, operational)

Target Profit (at $3 capital):
    - Anchor rate: ~$0.008/day, ~$0.24/month
    - Elevated funding (0.05%/h): ~$0.03/day, ~$0.90/month
    - Break-even: 3-5 days at anchor rate (after 0.14% round-trip fees)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from strategies.basis_trade.config import BasisTradeConfig
from strategies.basis_trade.funding_monitor import FundingMonitor, FundingSnapshot

logger = logging.getLogger(__name__)


class BasisState(str, Enum):
    """State machine for the basis trade lifecycle."""
    IDLE = "idle"              # No position, monitoring funding
    ENTERING = "entering"      # Placing entry orders (spot buy + perp short)
    ACTIVE = "active"          # Both legs open, collecting funding
    EXITING = "exiting"        # Closing both legs
    STOPPED = "stopped"        # Strategy shut down


@dataclass
class BasisPosition:
    """Tracks the active basis position (both legs)."""
    # Spot leg
    spot_entry_price: float
    spot_size: float           # In base token units (e.g., HYPE)
    spot_notional: float       # Entry notional in USD

    # Perp leg
    perp_entry_price: float
    perp_size: float           # In base token units
    perp_notional: float       # Entry notional in USD

    # Timing
    entry_time: float          # Unix timestamp

    # Fees paid
    entry_fees_usd: float = 0.0

    def notional(self) -> float:
        """Average notional of the two legs."""
        return (self.spot_notional + self.perp_notional) / 2

    def hold_hours(self, now: float) -> float:
        """Hours since position was opened."""
        return (now - self.entry_time) / 3600

    def spot_pnl(self, current_price: float) -> float:
        """Unrealized P&L on the spot leg."""
        return (current_price - self.spot_entry_price) * self.spot_size

    def perp_pnl(self, current_price: float) -> float:
        """Unrealized P&L on the perp short leg."""
        return (self.perp_entry_price - current_price) * self.perp_size

    def net_price_pnl(self, current_price: float) -> float:
        """Net P&L from price movement (should be ~0 for delta-neutral)."""
        return self.spot_pnl(current_price) + self.perp_pnl(current_price)


class BasisTradeStrategy:
    """Delta-neutral basis trade: long spot + short perp to collect funding.

    State machine:
        IDLE → (funding attractive) → ENTERING → (fills confirmed) → ACTIVE
        ACTIVE → (funding turns negative / target hit) → EXITING → IDLE
        Any state → stop() → STOPPED
    """

    def __init__(self, config: BasisTradeConfig):
        self._cfg = config
        self._state = BasisState.IDLE
        self._position: Optional[BasisPosition] = None
        self._funding = FundingMonitor(window_hours=24)

        # Counters
        self._trades_opened = 0
        self._trades_closed = 0
        self._total_pnl_usd = 0.0

    @property
    def state(self) -> BasisState:
        return self._state

    @property
    def position(self) -> Optional[BasisPosition]:
        return self._position

    @property
    def funding_monitor(self) -> FundingMonitor:
        return self._funding

    def start(self) -> None:
        """Start the strategy — begin monitoring funding rates."""
        if self._state == BasisState.STOPPED:
            self._state = BasisState.IDLE
        logger.info("Basis trade strategy started for %s", self._cfg.symbol)

    def stop(self) -> None:
        """Stop the strategy."""
        self._state = BasisState.STOPPED
        logger.info(
            "Basis trade strategy stopped: opened=%d closed=%d pnl=$%.4f",
            self._trades_opened, self._trades_closed, self._total_pnl_usd,
        )

    def update_funding(self, rate: float, mark_price: float,
                       oracle_price: float = 0.0,
                       timestamp: float = 0.0) -> None:
        """Feed a new funding rate observation into the monitor.

        Should be called every hour (matching Hyperliquid's funding interval).
        """
        ts = timestamp or time.time()
        snapshot = FundingSnapshot(
            timestamp=ts, rate=rate,
            mark_price=mark_price, oracle_price=oracle_price,
        )
        self._funding.record_observation(snapshot)
        self._funding.update_streaks(
            rate,
            above_threshold=self._cfg.min_funding_rate,
            below_threshold=self._cfg.exit_funding_rate,
        )

        # Record funding payment if we have an active position
        if self._state == BasisState.ACTIVE and self._position:
            self._funding.record_payment(
                timestamp=ts,
                rate=rate,
                notional=self._position.notional(),
            )

    def should_enter(self) -> bool:
        """Check if conditions are met to open a basis position.

        Returns True when:
        - Strategy is IDLE (no open position)
        - Funding rate has been above min_funding_rate for min_funding_hours
        """
        if self._state != BasisState.IDLE:
            return False

        return self._funding.consecutive_above >= self._cfg.min_funding_hours

    def should_exit(self) -> Optional[str]:
        """Check if conditions are met to close the basis position.

        Returns exit reason string, or None if position should stay open.
        """
        if self._state != BasisState.ACTIVE or self._position is None:
            return None

        now = time.time()

        # Check 1: Funding turned negative for too long
        if self._funding.consecutive_below >= self._cfg.exit_funding_hours:
            return "funding_negative"

        # Check 2: Max hold time exceeded
        if self._cfg.max_hold_hours > 0:
            if self._position.hold_hours(now) >= self._cfg.max_hold_hours:
                return "max_hold_time"

        # Check 3: Target profit reached
        if self._cfg.target_profit_usd > 0:
            if self._funding.cumulative_funding_usd >= self._cfg.target_profit_usd:
                return "target_profit"

        # Check 4: Maximum loss exceeded
        if self._cfg.max_loss_usd > 0:
            if self._funding.cumulative_funding_usd <= -self._cfg.max_loss_usd:
                return "max_loss"

        return None

    def on_entry_filled(self, spot_price: float, spot_size: float,
                        perp_price: float, perp_size: float,
                        total_fees: float = 0.0,
                        timestamp: float = 0.0) -> None:
        """Called when both entry legs are filled.

        Transitions: IDLE/ENTERING → ACTIVE
        """
        ts = timestamp or time.time()
        self._position = BasisPosition(
            spot_entry_price=spot_price,
            spot_size=spot_size,
            spot_notional=spot_price * spot_size,
            perp_entry_price=perp_price,
            perp_size=perp_size,
            perp_notional=perp_price * perp_size,
            entry_time=ts,
            entry_fees_usd=total_fees,
        )
        self._state = BasisState.ACTIVE
        self._trades_opened += 1

        logger.info(
            "Basis position OPENED: spot %.4f %s @ $%.4f + short perp %.4f @ $%.4f | "
            "notional=$%.2f | fees=$%.4f",
            spot_size, self._cfg.symbol, spot_price,
            perp_size, perp_price,
            self._position.notional(), total_fees,
        )

    def on_exit_filled(self, spot_price: float, perp_price: float,
                       total_fees: float = 0.0, reason: str = "",
                       timestamp: float = 0.0) -> Dict[str, Any]:
        """Called when both exit legs are filled.

        Transitions: ACTIVE/EXITING → IDLE
        Returns trade summary dict.
        """
        if self._position is None:
            return {}

        pos = self._position
        spot_pnl = pos.spot_pnl(spot_price)
        perp_pnl = pos.perp_pnl(perp_price)
        price_pnl = spot_pnl + perp_pnl
        funding_pnl = self._funding.cumulative_funding_usd
        total_fees_all = pos.entry_fees_usd + total_fees
        net_pnl = price_pnl + funding_pnl - total_fees_all

        now = timestamp or time.time()
        hold_hours = pos.hold_hours(now)

        summary = {
            "symbol": self._cfg.symbol,
            "reason": reason,
            "hold_hours": round(hold_hours, 2),
            "spot_entry": pos.spot_entry_price,
            "spot_exit": spot_price,
            "spot_pnl": round(spot_pnl, 6),
            "perp_entry": pos.perp_entry_price,
            "perp_exit": perp_price,
            "perp_pnl": round(perp_pnl, 6),
            "price_pnl": round(price_pnl, 6),
            "funding_pnl": round(funding_pnl, 6),
            "total_fees": round(total_fees_all, 6),
            "net_pnl": round(net_pnl, 6),
            "funding_payments": self._funding.total_payments,
        }

        self._total_pnl_usd += net_pnl
        self._trades_closed += 1
        self._position = None
        self._state = BasisState.IDLE

        # Reset funding monitor for next trade
        self._funding = FundingMonitor(window_hours=24)

        logger.info(
            "Basis position CLOSED: reason=%s | hold=%.1fh | "
            "price_pnl=$%.4f | funding=$%.4f | fees=$%.4f | net=$%.4f",
            reason, hold_hours, price_pnl, funding_pnl, total_fees_all, net_pnl,
        )
        return summary

    def get_status(self) -> Dict[str, Any]:
        """Return strategy state for monitoring/telemetry."""
        status: Dict[str, Any] = {
            "state": self._state.value,
            "symbol": self._cfg.symbol,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "total_pnl_usd": round(self._total_pnl_usd, 6),
            "funding": self._funding.get_status(),
        }

        if self._position:
            status["position"] = {
                "spot_entry": self._position.spot_entry_price,
                "spot_size": self._position.spot_size,
                "perp_entry": self._position.perp_entry_price,
                "perp_size": self._position.perp_size,
                "notional": round(self._position.notional(), 2),
                "hold_hours": round(
                    self._position.hold_hours(time.time()), 2
                ),
                "entry_fees": self._position.entry_fees_usd,
            }

        return status
