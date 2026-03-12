"""Tests for BasisTradeConfig."""

import pytest
from strategies.basis_trade.config import BasisTradeConfig


class TestBasisTradeConfig:
    """Test configuration parsing and computed properties."""

    def test_defaults(self):
        cfg = BasisTradeConfig()
        assert cfg.symbol == "HYPE"
        assert cfg.spot_pair == "@107"
        assert cfg.position_size_usd == 3.0
        assert cfg.leverage == 10
        assert cfg.min_funding_rate == 0.0001
        assert cfg.min_funding_hours == 3
        assert cfg.exit_funding_rate == -0.00005
        assert cfg.exit_funding_hours == 6
        assert cfg.slippage_tolerance == 0.002

    def test_from_dict(self):
        cfg = BasisTradeConfig.from_dict({
            "symbol": "BTC",
            "position_size_usd": 100.0,
            "leverage": 5,
            "min_funding_rate": 0.0005,
        })
        assert cfg.symbol == "BTC"
        assert cfg.position_size_usd == 100.0
        assert cfg.leverage == 5
        assert cfg.min_funding_rate == 0.0005
        # Defaults preserved for unset keys
        assert cfg.exit_funding_hours == 6

    def test_from_dict_ignores_unknown_keys(self):
        cfg = BasisTradeConfig.from_dict({
            "symbol": "ETH",
            "unknown_param": 42,
            "another_thing": "hello",
        })
        assert cfg.symbol == "ETH"
        assert not hasattr(cfg, "unknown_param")

    def test_entry_apr(self):
        cfg = BasisTradeConfig(min_funding_rate=0.0001)
        # 0.0001 * 24 * 365 * 100 = 87.6%
        assert abs(cfg.entry_apr() - 87.6) < 0.1

    def test_entry_apr_high_funding(self):
        cfg = BasisTradeConfig(min_funding_rate=0.001)
        # 0.001 * 24 * 365 * 100 = 876%
        assert abs(cfg.entry_apr() - 876.0) < 0.1

    def test_max_spot_notional_default(self):
        # $3 capital, 10x leverage: X = 3 * 10 / 11 ≈ $2.727
        cfg = BasisTradeConfig(position_size_usd=3.0, leverage=10)
        assert abs(cfg.max_spot_notional() - 2.727) < 0.01

    def test_max_spot_notional_higher_leverage(self):
        # $100 capital, 20x leverage: X = 100 * 20 / 21 ≈ $95.24
        cfg = BasisTradeConfig(position_size_usd=100.0, leverage=20)
        assert abs(cfg.max_spot_notional() - 95.24) < 0.01

    def test_max_spot_notional_low_leverage(self):
        # $10 capital, 2x leverage: X = 10 * 2 / 3 ≈ $6.67
        cfg = BasisTradeConfig(position_size_usd=10.0, leverage=2)
        assert abs(cfg.max_spot_notional() - 6.667) < 0.01

    def test_frozen(self):
        cfg = BasisTradeConfig()
        with pytest.raises(AttributeError):
            cfg.symbol = "BTC"
