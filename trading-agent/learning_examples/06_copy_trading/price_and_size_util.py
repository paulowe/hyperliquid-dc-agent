"""
Utility functions for price and size calculations.
https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext

# High precision to avoid intermediate rounding artifacts
getcontext().prec = 40


@dataclass(frozen=True)
class HLFormatRules:
    """
    Formatting rules derived from Hyperliquid instrument metadata.

    Attributes:
        sz_decimals: Number of decimal places allowed for order size (szDecimals).
        is_spot: True for spot markets, False for perpetuals.
    """
    sz_decimals: int
    is_spot: bool

    @property
    def max_price_decimal_places(self) -> int:
        """
        Maximum allowed decimal places for price, derived from protocol rules.

        Hyperliquid:
          - Spot: 8 total decimals
          - Perps: 6 total decimals

        Price decimal places = MAX_DECIMALS - szDecimals
        """
        max_decimals = 8 if self.is_spot else 6
        return max(0, max_decimals - self.sz_decimals)

@dataclass
class InstrumentFormatter:
    """
    Cached formatter bound to a single trading instrument.

    Use this when formatting many orders for the same symbol
    to avoid repeatedly passing szDecimals and is_spot.
    """
    rules: HLFormatRules

    def format(
        self,
        *,
        size: float | int | str | Decimal,
        price: float | int | str | Decimal,
        is_buy: bool,
    ) -> tuple[str, str]:
        """
        Format size and price using pre-bound instrument rules.

        Args:
            size: Raw order size
            price: Raw order price
            is_buy: True for buy orders, False for sell orders

        Returns:
            Tuple of (formatted_size, formatted_price)
        """
        formatted_size = format_order_size(
            size,
            self.rules,
            rounding=ROUND_DOWN,
        )

        price_rounding = ROUND_DOWN if is_buy else ROUND_UP

        formatted_price = format_order_price(
            price,
            self.rules,
            rounding=price_rounding,
        )

        return float(formatted_size), float(formatted_price)

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def to_decimal(value: float | int | str | Decimal) -> Decimal:
    """
    Convert input to Decimal safely using string conversion
    to avoid binary float precision issues.
    """
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"Invalid numeric value: {value!r}")


def decimal_quantum(decimal_places: int) -> Decimal:
    """
    Return a Decimal quantization unit for a given number of decimal places.

    Example:
        decimal_places = 3  -> Decimal('0.001')
    """
    return Decimal("1").scaleb(-decimal_places)


def strip_trailing_zeros(value: Decimal) -> str:
    """
    Convert Decimal to string and remove trailing zeros and decimal point
    if not needed.

    Required because Hyperliquid expects normalized values when signing.
    """
    s = format(value, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def is_integer_decimal(value: Decimal) -> bool:
    """
    Check whether a Decimal represents an integer value.
    """
    return value == value.to_integral_value()


def count_significant_figures(value: Decimal) -> int:
    """
    Count significant figures in a Decimal.

    Rules:
      - Leading zeros are ignored
      - Trailing zeros are ignored (after normalization)
      - Zero has 1 significant figure by convention
    """
    value = value.copy_abs()
    if value.is_zero():
        return 1
    return len(value.normalize().as_tuple().digits)


# -------------------------------------------------------------------
# Size formatting
# -------------------------------------------------------------------

def format_order_size(
    size: float | int | str | Decimal,
    rules: HLFormatRules,
    *,
    rounding=ROUND_DOWN,
) -> str:
    """
    Format order size to satisfy Hyperliquid size rules.

    Behavior:
      - Rounds to szDecimals
      - Rejects zero or negative sizes
      - Prevents rounding to zero

    Args:
        size: Raw order size
        rules: HLFormatRules instance
        rounding: Decimal rounding mode (default ROUND_DOWN)

    Returns:
        Normalized size string
    """
    size_dec = to_decimal(size)

    if size_dec <= 0:
        raise ValueError(f"Order size must be > 0, got {size_dec}")

    size_dec = size_dec.quantize(
        decimal_quantum(rules.sz_decimals),
        rounding=rounding,
    )

    if size_dec <= 0:
        raise ValueError(
            f"Order size rounded to zero with szDecimals={rules.sz_decimals}: {size}"
        )

    return strip_trailing_zeros(size_dec)


# -------------------------------------------------------------------
# Price formatting
# -------------------------------------------------------------------

def format_order_price(
    price: float | int | str | Decimal,
    rules: HLFormatRules,
    *,
    rounding=ROUND_DOWN,
) -> str:
    """
    Format order price to satisfy Hyperliquid price rules.

    Rules enforced:
      - Integer prices are always allowed
      - Non-integer prices must satisfy:
          * decimal places ≤ max_price_decimal_places
          * significant figures ≤ 5
      - Prefer maximum precision first
      - Fallback to integer if no valid fractional form exists

    Args:
        price: Raw order price
        rules: HLFormatRules instance
        rounding: Decimal rounding mode

    Returns:
        Normalized price string
    """
    price_dec = to_decimal(price)

    if price_dec <= 0:
        raise ValueError(f"Order price must be > 0, got {price_dec}")

    for dp in range(rules.max_price_decimal_places, -1, -1):
        candidate = price_dec.quantize(
            decimal_quantum(dp),
            rounding=rounding,
        )

        # Integer prices bypass sig-fig rule
        if is_integer_decimal(candidate):
            return strip_trailing_zeros(candidate)

        if count_significant_figures(candidate) <= 5:
            return strip_trailing_zeros(candidate)

    # Last-resort fallback: integer price
    return strip_trailing_zeros(
        price_dec.to_integral_value(rounding=rounding)
    )


# -------------------------------------------------------------------
# Combined order formatter
# -------------------------------------------------------------------

def format_order(
    order_size: float | int | str | Decimal,
    order_price: float | int | str | Decimal,
    *,
    sz_decimals: int,
    is_spot: bool,
    is_buy: bool,
) -> tuple[str, str]:
    """
    Format both order size and price according to Hyperliquid rules.

    Args:
        order_size: Raw order size
        order_price: Raw order price
        sz_decimals: Instrument szDecimals
        is_spot: True if spot market, False if perp
        is_buy:
            - True: round price DOWN (safer for buyers)
            - False: round price UP (safer for sellers)

    Returns:
        Tuple of (formatted_size, formatted_price)
    """
    rules = HLFormatRules(sz_decimals=sz_decimals, is_spot=is_spot)

    # Size: always round DOWN (safe)
    formatted_size = format_order_size(
        order_size,
        rules,
        rounding=ROUND_DOWN,
    )

    # Price: directional rounding
    price_rounding = ROUND_DOWN if is_buy else ROUND_UP

    formatted_price = format_order_price(
        order_price,
        rules,
        rounding=price_rounding,
    )

    return float(formatted_size), float(formatted_price)
