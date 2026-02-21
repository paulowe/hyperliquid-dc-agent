"""Validate DC Overshoot strategy configuration for safety and profitability."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """A single validation finding."""

    level: str  # "error", "warning", "info"
    code: str  # "E1", "W1", "I1"
    message: str


@dataclass
class ValidationResult:
    """Collection of validation findings."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)

    def format(self) -> str:
        """Format all issues as a human-readable report."""
        lines = []
        errors = [i for i in self.issues if i.level == "error"]
        warnings = [i for i in self.issues if i.level == "warning"]
        infos = [i for i in self.issues if i.level == "info"]

        if errors:
            lines.append("ERRORS (must fix before trading):")
            for i in errors:
                lines.append(f"  [{i.code}] {i.message}")
            lines.append("")

        if warnings:
            lines.append("WARNINGS (suboptimal, review recommended):")
            for i in warnings:
                lines.append(f"  [{i.code}] {i.message}")
            lines.append("")

        if infos:
            lines.append("INFO:")
            for i in infos:
                lines.append(f"  [{i.code}] {i.message}")
            lines.append("")

        if not errors and not warnings:
            lines.insert(0, "Config OK — no errors or warnings.\n")

        return "\n".join(lines)


def validate_dc_config(
    threshold: float,
    sl_pct: float,
    tp_pct: float,
    backstop_sl_pct: float,
    leverage: int,
    position_size_usd: float,
    trail_pct: float = 0.5,
    min_profit_to_trail_pct: float = 0.001,
    taker_fee_pct: float = 0.00035,
) -> ValidationResult:
    """Validate DC Overshoot config for safety and profitability.

    Returns a ValidationResult with errors, warnings, and info messages.
    """
    result = ValidationResult()
    liquidation_dist = 1.0 / leverage if leverage > 0 else 0.0
    roundtrip_fee = 2 * taker_fee_pct

    # --- Errors ---

    # E1: SL beyond liquidation
    if leverage > 0 and sl_pct >= liquidation_dist:
        result.issues.append(ValidationIssue(
            level="error",
            code="E1",
            message=(
                f"SL {sl_pct*100:.2f}% >= liquidation distance "
                f"{liquidation_dist*100:.1f}% at {leverage}x leverage. "
                f"You'll be liquidated before SL fires."
            ),
        ))

    # E2: Backstop beyond liquidation
    if leverage > 0 and backstop_sl_pct >= liquidation_dist:
        result.issues.append(ValidationIssue(
            level="error",
            code="E2",
            message=(
                f"Backstop SL {backstop_sl_pct*100:.1f}% >= liquidation distance "
                f"{liquidation_dist*100:.1f}%. Exchange order won't protect you."
            ),
        ))

    # E3: SL wider than backstop
    if sl_pct >= backstop_sl_pct:
        result.issues.append(ValidationIssue(
            level="error",
            code="E3",
            message=(
                f"Software SL {sl_pct*100:.2f}% >= backstop SL {backstop_sl_pct*100:.1f}%. "
                f"Backstop will fire before software SL."
            ),
        ))

    # E4: Invalid position size
    if position_size_usd <= 0:
        result.issues.append(ValidationIssue(
            level="error",
            code="E4",
            message="Position size must be positive.",
        ))

    # E5: Leverage out of range
    if leverage < 1 or leverage > 50:
        result.issues.append(ValidationIssue(
            level="error",
            code="E5",
            message=f"Leverage {leverage}x outside Hyperliquid range (1-50x).",
        ))

    # --- Warnings ---

    # W1: TP eaten by fees
    if tp_pct < roundtrip_fee:
        result.issues.append(ValidationIssue(
            level="warning",
            code="W1",
            message=(
                f"TP {tp_pct*100:.3f}% is less than round-trip fees "
                f"({roundtrip_fee*100:.3f}%). Every TP exit will lose money."
            ),
        ))

    # W2: SL too tight for threshold
    if sl_pct < threshold:
        result.issues.append(ValidationIssue(
            level="warning",
            code="W2",
            message=(
                f"SL {sl_pct*100:.2f}% < DC threshold {threshold*100:.2f}%. "
                f"Normal price oscillations will stop you out before overshoot completes."
            ),
        ))

    # W3: Backstop close to liquidation
    if leverage > 0 and backstop_sl_pct > 0.8 * liquidation_dist:
        # Only warn if backstop is still below liquidation (otherwise E2 fires)
        if backstop_sl_pct < liquidation_dist:
            result.issues.append(ValidationIssue(
                level="warning",
                code="W3",
                message=(
                    f"Backstop SL at {backstop_sl_pct*100:.1f}% is within 20% of "
                    f"liquidation distance ({liquidation_dist*100:.1f}%). Consider tightening."
                ),
            ))

    # W4: Thin profit margin after fees
    if tp_pct < 4 * taker_fee_pct:
        # Only if not already covered by W1 (which is stricter)
        if tp_pct >= roundtrip_fee:
            result.issues.append(ValidationIssue(
                level="warning",
                code="W4",
                message=(
                    f"TP {tp_pct*100:.3f}% leaves thin margin after fees "
                    f"({roundtrip_fee*100:.3f}% round-trip)."
                ),
            ))

    # W5: High leverage with wide SL = big margin loss per stop
    if sl_pct * leverage > 0.20:
        result.issues.append(ValidationIssue(
            level="warning",
            code="W5",
            message=(
                f"SL at {sl_pct*100:.2f}% with {leverage}x leverage = "
                f"{sl_pct*leverage*100:.1f}% margin loss per stop. High risk per trade."
            ),
        ))

    # W6: min_profit_to_trail too small
    if min_profit_to_trail_pct < taker_fee_pct:
        result.issues.append(ValidationIssue(
            level="warning",
            code="W6",
            message=(
                f"Min profit to trail ({min_profit_to_trail_pct*100:.3f}%) < "
                f"single-side fee ({taker_fee_pct*100:.3f}%). "
                f"Trailing may ratchet on fee-noise."
            ),
        ))

    # --- Info ---

    # I1: Margin impact
    if leverage > 0:
        result.issues.append(ValidationIssue(
            level="info",
            code="I1",
            message=(
                f"At {leverage}x: SL = {sl_pct*leverage*100:.1f}% margin loss, "
                f"TP = {tp_pct*leverage*100:.1f}% margin gain"
            ),
        ))

    # I2: Fee impact
    fee_per_trade = position_size_usd * roundtrip_fee
    result.issues.append(ValidationIssue(
        level="info",
        code="I2",
        message=(
            f"Round-trip fee: {roundtrip_fee*100:.3f}% of notional = "
            f"${fee_per_trade:.4f} per trade"
        ),
    ))

    # I3: Protection layers
    if leverage > 0:
        result.issues.append(ValidationIssue(
            level="info",
            code="I3",
            message=(
                f"Protection: Software SL ({sl_pct*100:.2f}%) -> "
                f"Backstop ({backstop_sl_pct*100:.1f}%) -> "
                f"Liquidation (~{liquidation_dist*100:.1f}%)"
            ),
        ))

    return result
