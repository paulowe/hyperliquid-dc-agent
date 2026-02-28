"""Tests for MultiScaleConfig."""

from __future__ import annotations

import pytest

from strategies.dc_overshoot.multi_scale_config import MultiScaleConfig


class TestDefaults:
    def test_from_dict_defaults(self):
        cfg = MultiScaleConfig.from_dict({})
        assert cfg.symbol == "BTC"
        assert len(cfg.sensor_thresholds) == 3
        assert cfg.trade_threshold == (0.015, 0.015)
        assert cfg.momentum_alpha == 0.3
        assert cfg.min_momentum_score == 0.3
        assert cfg.trail_pct == 0.3

    def test_from_dict_custom(self):
        cfg = MultiScaleConfig.from_dict({
            "symbol": "SOL",
            "sensor_thresholds": [(0.001, 0.001), (0.005, 0.005)],
            "trade_threshold": (0.02, 0.02),
            "momentum_alpha": 0.5,
            "min_momentum_score": 0.1,
            "position_size_usd": 200.0,
            "initial_stop_loss_pct": 0.02,
            "initial_take_profit_pct": 0.01,
            "trail_pct": 0.5,
        })
        assert cfg.symbol == "SOL"
        assert len(cfg.sensor_thresholds) == 2
        assert cfg.trade_threshold == (0.02, 0.02)
        assert cfg.momentum_alpha == 0.5
        assert cfg.min_momentum_score == 0.1
        assert cfg.position_size_usd == 200.0


class TestThresholdKeys:
    def test_all_thresholds_combines_sensor_and_trade(self):
        cfg = MultiScaleConfig.from_dict({
            "sensor_thresholds": [(0.002, 0.002), (0.004, 0.004)],
            "trade_threshold": (0.015, 0.015),
        })
        all_t = cfg.all_thresholds()
        assert len(all_t) == 3
        assert (0.002, 0.002) in all_t
        assert (0.004, 0.004) in all_t
        assert (0.015, 0.015) in all_t

    def test_sensor_threshold_keys(self):
        cfg = MultiScaleConfig.from_dict({
            "sensor_thresholds": [(0.002, 0.002), (0.004, 0.004)],
        })
        keys = cfg.sensor_threshold_keys()
        assert "0.002:0.002" in keys
        assert "0.004:0.004" in keys
        assert len(keys) == 2

    def test_trade_threshold_key(self):
        cfg = MultiScaleConfig.from_dict({
            "trade_threshold": (0.015, 0.015),
        })
        assert cfg.trade_threshold_key() == "0.015:0.015"

    def test_sensor_and_trade_keys_disjoint(self):
        cfg = MultiScaleConfig.from_dict({
            "sensor_thresholds": [(0.002, 0.002), (0.004, 0.004)],
            "trade_threshold": (0.015, 0.015),
        })
        sensor_keys = cfg.sensor_threshold_keys()
        trade_key = cfg.trade_threshold_key()
        assert trade_key not in sensor_keys


class TestValidation:
    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            MultiScaleConfig.from_dict({
                "trade_threshold": (-0.01, 0.01),
            })

    def test_negative_sensor_threshold_raises(self):
        with pytest.raises(ValueError):
            MultiScaleConfig.from_dict({
                "sensor_thresholds": [(-0.002, 0.002)],
            })

    def test_invalid_trail_pct_raises(self):
        with pytest.raises(ValueError):
            MultiScaleConfig.from_dict({"trail_pct": 1.5})

    def test_zero_position_size_raises(self):
        with pytest.raises(ValueError):
            MultiScaleConfig.from_dict({"position_size_usd": 0.0})

    def test_single_float_sensor_threshold(self):
        """A single float should be treated as (float, float) pair."""
        cfg = MultiScaleConfig.from_dict({
            "sensor_thresholds": [0.003, 0.006],
        })
        assert cfg.sensor_thresholds == [(0.003, 0.003), (0.006, 0.006)]

    def test_single_float_trade_threshold(self):
        """A single float trade threshold should become (float, float) pair."""
        cfg = MultiScaleConfig.from_dict({
            "trade_threshold": 0.02,
        })
        assert cfg.trade_threshold == (0.02, 0.02)
