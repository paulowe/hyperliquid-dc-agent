"""Tests for DCOvershootConfig dataclass and YAML parsing."""

import pytest

from strategies.dc_overshoot.config import DCOvershootConfig


class TestDCOvershootConfigDefaults:
    """Verify default config creates a valid instance."""

    def test_default_config_creates_valid_instance(self):
        cfg = DCOvershootConfig()
        assert cfg.symbol == "BTC"
        assert cfg.dc_thresholds == [(0.001, 0.001)]
        assert cfg.position_size_usd == 50.0
        assert cfg.max_position_size_usd == 200.0
        assert cfg.initial_stop_loss_pct == 0.003
        assert cfg.initial_take_profit_pct == 0.002
        assert cfg.trail_pct == 0.5
        assert cfg.cooldown_seconds == 10.0
        assert cfg.max_open_positions == 1
        assert cfg.log_events is True

    def test_default_thresholds_are_tuple_pairs(self):
        cfg = DCOvershootConfig()
        for t in cfg.dc_thresholds:
            assert isinstance(t, tuple)
            assert len(t) == 2
            assert all(isinstance(v, float) for v in t)


class TestDCOvershootConfigFromDict:
    """Verify parsing from YAML-like dicts."""

    def test_from_dict_with_all_fields(self):
        d = {
            "symbol": "ETH",
            "dc_thresholds": [[0.005, 0.005]],
            "position_size_usd": 100.0,
            "max_position_size_usd": 500.0,
            "initial_stop_loss_pct": 0.005,
            "initial_take_profit_pct": 0.003,
            "trail_pct": 0.6,
            "cooldown_seconds": 20.0,
            "max_open_positions": 2,
            "log_events": False,
        }
        cfg = DCOvershootConfig.from_dict(d)
        assert cfg.symbol == "ETH"
        assert cfg.dc_thresholds == [(0.005, 0.005)]
        assert cfg.position_size_usd == 100.0
        assert cfg.initial_stop_loss_pct == 0.005
        assert cfg.initial_take_profit_pct == 0.003
        assert cfg.trail_pct == 0.6
        assert cfg.cooldown_seconds == 20.0
        assert cfg.max_open_positions == 2
        assert cfg.log_events is False

    def test_from_dict_with_defaults(self):
        """Empty dict should produce valid defaults."""
        cfg = DCOvershootConfig.from_dict({})
        assert cfg.symbol == "BTC"
        assert cfg.initial_stop_loss_pct == 0.003

    def test_from_dict_multiple_thresholds(self):
        d = {"dc_thresholds": [[0.001, 0.001], [0.005, 0.005]]}
        cfg = DCOvershootConfig.from_dict(d)
        assert len(cfg.dc_thresholds) == 2
        assert cfg.dc_thresholds[0] == (0.001, 0.001)
        assert cfg.dc_thresholds[1] == (0.005, 0.005)

    def test_from_dict_scalar_threshold_becomes_symmetric(self):
        """A single float should be treated as symmetric (down=up)."""
        d = {"dc_thresholds": [0.002]}
        cfg = DCOvershootConfig.from_dict(d)
        assert cfg.dc_thresholds == [(0.002, 0.002)]


class TestDCOvershootConfigValidation:
    """Verify invalid configs raise errors."""

    def test_negative_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss"):
            DCOvershootConfig.from_dict({"initial_stop_loss_pct": -0.01})

    def test_negative_take_profit_raises(self):
        with pytest.raises(ValueError, match="take_profit"):
            DCOvershootConfig.from_dict({"initial_take_profit_pct": -0.01})

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            DCOvershootConfig.from_dict({"dc_thresholds": [[0.0, 0.0]]})

    def test_trail_pct_out_of_range_raises(self):
        """trail_pct must be in (0, 1]."""
        with pytest.raises(ValueError, match="trail_pct"):
            DCOvershootConfig.from_dict({"trail_pct": 0.0})
        with pytest.raises(ValueError, match="trail_pct"):
            DCOvershootConfig.from_dict({"trail_pct": 1.5})

    def test_invalid_threshold_format_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            DCOvershootConfig.from_dict({"dc_thresholds": [[0.001, 0.001, 0.001]]})

    def test_negative_position_size_raises(self):
        with pytest.raises(ValueError, match="position_size"):
            DCOvershootConfig.from_dict({"position_size_usd": -10.0})

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown"):
            DCOvershootConfig.from_dict({"cooldown_seconds": -1.0})
