"""Tests for DC Overshoot config validation."""

from __future__ import annotations

import pytest

from strategies.dc_overshoot.validate import (
    ValidationResult,
    validate_dc_config,
)


# --- Helpers ---

def good_config(**overrides) -> dict:
    """Return a known-good config with optional overrides."""
    defaults = {
        "threshold": 0.015,
        "sl_pct": 0.015,
        "tp_pct": 0.005,
        "backstop_sl_pct": 0.05,
        "backstop_tp_pct": 0.10,
        "leverage": 10,
        "position_size_usd": 100.0,
        "trail_pct": 0.5,
        "min_profit_to_trail_pct": 0.002,
        "taker_fee_pct": 0.00035,
    }
    defaults.update(overrides)
    return defaults


def has_code(result: ValidationResult, code: str) -> bool:
    """Check if a specific issue code is present in the result."""
    return any(i.code == code for i in result.issues)


# --- Good config baseline ---

class TestGoodConfig:
    def test_no_errors(self):
        result = validate_dc_config(**good_config())
        assert not result.has_errors

    def test_no_warnings(self):
        result = validate_dc_config(**good_config())
        assert not result.has_warnings

    def test_has_info_messages(self):
        """Good config should still have info messages."""
        result = validate_dc_config(**good_config())
        info = [i for i in result.issues if i.level == "info"]
        assert len(info) >= 4  # I1, I2, I3, I4


# --- Error rules ---

class TestErrorE1SLBeyondLiquidation:
    def test_sl_equals_liquidation(self):
        """SL at exactly 1/leverage should be an error."""
        result = validate_dc_config(**good_config(sl_pct=0.10, leverage=10))
        assert result.has_errors
        assert has_code(result, "E1")

    def test_sl_beyond_liquidation(self):
        """SL wider than liquidation distance."""
        result = validate_dc_config(**good_config(sl_pct=0.15, leverage=10))
        assert has_code(result, "E1")

    def test_sl_within_liquidation(self):
        """SL tighter than liquidation — no E1."""
        result = validate_dc_config(**good_config(sl_pct=0.05, leverage=10))
        assert not has_code(result, "E1")


class TestErrorE2BackstopBeyondLiquidation:
    def test_backstop_equals_liquidation(self):
        result = validate_dc_config(**good_config(backstop_sl_pct=0.10, leverage=10))
        assert has_code(result, "E2")

    def test_backstop_beyond_liquidation(self):
        result = validate_dc_config(**good_config(backstop_sl_pct=0.20, leverage=10))
        assert has_code(result, "E2")

    def test_backstop_within_liquidation(self):
        result = validate_dc_config(**good_config(backstop_sl_pct=0.05, leverage=10))
        assert not has_code(result, "E2")


class TestErrorE3SLWiderThanBackstop:
    def test_sl_equals_backstop(self):
        result = validate_dc_config(**good_config(sl_pct=0.05, backstop_sl_pct=0.05))
        assert has_code(result, "E3")

    def test_sl_wider_than_backstop(self):
        result = validate_dc_config(**good_config(sl_pct=0.06, backstop_sl_pct=0.05))
        assert has_code(result, "E3")

    def test_sl_tighter_than_backstop(self):
        result = validate_dc_config(**good_config(sl_pct=0.01, backstop_sl_pct=0.05))
        assert not has_code(result, "E3")


class TestErrorE4PositionSize:
    def test_zero_position(self):
        result = validate_dc_config(**good_config(position_size_usd=0))
        assert has_code(result, "E4")

    def test_negative_position(self):
        result = validate_dc_config(**good_config(position_size_usd=-50))
        assert has_code(result, "E4")

    def test_positive_position(self):
        result = validate_dc_config(**good_config(position_size_usd=100))
        assert not has_code(result, "E4")


class TestErrorE5Leverage:
    def test_zero_leverage(self):
        result = validate_dc_config(**good_config(leverage=0))
        assert has_code(result, "E5")

    def test_excessive_leverage(self):
        result = validate_dc_config(**good_config(leverage=51))
        assert has_code(result, "E5")

    def test_valid_leverage(self):
        result = validate_dc_config(**good_config(leverage=10))
        assert not has_code(result, "E5")


# --- Warning rules ---

class TestWarningW1TPEatenByFees:
    def test_tp_less_than_roundtrip_fees(self):
        # taker_fee = 0.00035, roundtrip = 0.0007
        # tp_pct = 0.0005 < 0.0007
        result = validate_dc_config(**good_config(tp_pct=0.0005))
        assert has_code(result, "W1")

    def test_tp_above_roundtrip_fees(self):
        result = validate_dc_config(**good_config(tp_pct=0.005))
        assert not has_code(result, "W1")


class TestWarningW2SLTooTightForThreshold:
    def test_sl_less_than_threshold(self):
        result = validate_dc_config(**good_config(sl_pct=0.003, threshold=0.015))
        assert has_code(result, "W2")

    def test_sl_equals_threshold(self):
        """SL == threshold is fine (not strictly less)."""
        result = validate_dc_config(**good_config(sl_pct=0.015, threshold=0.015))
        assert not has_code(result, "W2")


class TestWarningW3BackstopCloseToLiquidation:
    def test_backstop_within_20pct_of_liquidation(self):
        # leverage=10, liquidation=0.10, 80% = 0.08
        # backstop=0.09 > 0.08 → warning
        result = validate_dc_config(**good_config(backstop_sl_pct=0.09, leverage=10))
        assert has_code(result, "W3")

    def test_backstop_well_below_liquidation(self):
        result = validate_dc_config(**good_config(backstop_sl_pct=0.05, leverage=10))
        assert not has_code(result, "W3")


class TestWarningW4ThinProfitMargin:
    def test_tp_less_than_4x_fee(self):
        # taker_fee = 0.00035, 4x = 0.0014
        # tp_pct = 0.001 < 0.0014
        result = validate_dc_config(**good_config(tp_pct=0.001))
        assert has_code(result, "W4")

    def test_tp_above_4x_fee(self):
        result = validate_dc_config(**good_config(tp_pct=0.005))
        assert not has_code(result, "W4")


class TestWarningW5HighLeverageWideSL:
    def test_margin_loss_over_20pct(self):
        # sl_pct=0.03 * leverage=10 = 0.30 > 0.20
        result = validate_dc_config(**good_config(sl_pct=0.03, leverage=10))
        assert has_code(result, "W5")

    def test_margin_loss_under_20pct(self):
        # sl_pct=0.015 * leverage=10 = 0.15 < 0.20
        result = validate_dc_config(**good_config(sl_pct=0.015, leverage=10))
        assert not has_code(result, "W5")


class TestWarningW6MinProfitToTrailTooSmall:
    def test_min_trail_less_than_fee(self):
        # taker_fee = 0.00035, min_profit_to_trail = 0.0002
        result = validate_dc_config(**good_config(min_profit_to_trail_pct=0.0002))
        assert has_code(result, "W6")

    def test_min_trail_above_fee(self):
        result = validate_dc_config(**good_config(min_profit_to_trail_pct=0.002))
        assert not has_code(result, "W6")


# --- Info messages ---

class TestInfoMessages:
    def test_i1_margin_impact(self):
        result = validate_dc_config(**good_config())
        assert has_code(result, "I1")

    def test_i2_fee_impact(self):
        result = validate_dc_config(**good_config())
        assert has_code(result, "I2")

    def test_i3_protection_layers(self):
        result = validate_dc_config(**good_config())
        assert has_code(result, "I3")

    def test_i4_backstop_tp(self):
        result = validate_dc_config(**good_config())
        assert has_code(result, "I4")


# --- Backstop TP rules ---

class TestErrorE6BackstopTpNotPositive:
    def test_zero_backstop_tp(self):
        result = validate_dc_config(**good_config(backstop_tp_pct=0.0))
        assert has_code(result, "E6")

    def test_negative_backstop_tp(self):
        result = validate_dc_config(**good_config(backstop_tp_pct=-0.05))
        assert has_code(result, "E6")

    def test_positive_backstop_tp_ok(self):
        result = validate_dc_config(**good_config(backstop_tp_pct=0.10))
        assert not has_code(result, "E6")


class TestWarningW7BackstopTpTighterThanSoftwareTP:
    def test_backstop_tp_less_than_software_tp(self):
        # backstop_tp_pct=0.001 < tp_pct=0.005 → exchange closes before trailing
        result = validate_dc_config(**good_config(backstop_tp_pct=0.001, tp_pct=0.005))
        assert has_code(result, "W7")

    def test_backstop_tp_wider_than_software_tp(self):
        result = validate_dc_config(**good_config(backstop_tp_pct=0.10, tp_pct=0.005))
        assert not has_code(result, "W7")


# --- Result properties ---

class TestValidationResult:
    def test_has_errors_true_when_errors_exist(self):
        result = validate_dc_config(**good_config(sl_pct=0.15, leverage=10))
        assert result.has_errors is True

    def test_has_errors_false_when_clean(self):
        result = validate_dc_config(**good_config())
        assert result.has_errors is False

    def test_has_warnings_true(self):
        result = validate_dc_config(**good_config(sl_pct=0.003, threshold=0.015))
        assert result.has_warnings is True

    def test_format_returns_nonempty_string(self):
        result = validate_dc_config(**good_config())
        formatted = result.format()
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_shows_errors(self):
        result = validate_dc_config(**good_config(sl_pct=0.15, leverage=10))
        formatted = result.format()
        assert "ERROR" in formatted or "error" in formatted.lower()
