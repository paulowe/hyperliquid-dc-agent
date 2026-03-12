"""Configuration for the basis trade strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BasisTradeConfig:
    """Configuration for delta-neutral basis trade (long spot + short perp).

    The basis trade profits from funding rate payments. When funding is positive,
    short perp holders receive payments from longs. By holding an equal spot
    position, we're delta-neutral (immune to price moves) and collect funding.

    Category: Cash-and-carry arbitrage (delta-neutral)
    """

    # Asset to trade
    symbol: str = "HYPE"

    # Spot pair name on Hyperliquid (e.g., "@107" for HYPE/USDC)
    spot_pair: str = "@107"

    # Position sizing
    # Total capital is split: spot_fraction for spot buy, rest for perp margin
    position_size_usd: float = 3.0

    # Leverage for the perp short leg
    leverage: int = 10

    # --- Entry conditions ---

    # Minimum hourly funding rate to open the basis position (decimal)
    # 0.0001 = 0.01%/h ≈ 87.6% APR — only enter when funding is attractive
    min_funding_rate: float = 0.0001

    # Number of consecutive hours funding must exceed min to trigger entry
    # Avoids entering on a single spike
    min_funding_hours: int = 3

    # --- Exit conditions ---

    # Close if funding rate falls below this for N consecutive hours (decimal)
    # Slightly negative to avoid closing on brief dips through zero
    exit_funding_rate: float = -0.00005

    # Number of consecutive hours funding must stay below exit threshold
    exit_funding_hours: int = 6

    # Maximum position hold time in hours (0 = unlimited)
    max_hold_hours: float = 0

    # Target cumulative funding profit in USD to take profit (0 = disabled)
    target_profit_usd: float = 0.0

    # Maximum cumulative loss in USD before closing (0 = disabled)
    max_loss_usd: float = 0.0

    # --- Monitoring ---

    # How often to check funding rate and log status (seconds)
    check_interval_seconds: float = 300.0  # 5 minutes

    # Slippage tolerance for market orders (decimal, e.g., 0.001 = 0.1%)
    slippage_tolerance: float = 0.002

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BasisTradeConfig:
        """Create config from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def entry_apr(self) -> float:
        """Annualized return at the minimum entry funding rate."""
        return self.min_funding_rate * 24 * 365 * 100

    def max_spot_notional(self) -> float:
        """Maximum notional for the spot leg given capital and leverage.

        With unified account mode, USDC backs both spot purchase and perp margin.
        If we buy X spot and short X perp at L leverage:
            X (spot cost) + X/L (perp margin) = total_capital
            X = total_capital * L / (L + 1)
        """
        return self.position_size_usd * self.leverage / (self.leverage + 1)
