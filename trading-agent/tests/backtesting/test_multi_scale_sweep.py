"""Tests for MultiScaleParameterSweep."""

from __future__ import annotations

import pytest

from backtesting.sweep import MultiScaleSweepConfig, MultiScaleParameterSweep


class TestSweepConfig:
    def test_default_grid_size(self):
        cfg = MultiScaleSweepConfig()
        combos = cfg.combinations()
        # 4 trade_thresholds * 3 alphas * 4 min_scores * 4 sl * 4 tp * 3 trail * 2 mpt
        assert len(combos) == 4 * 3 * 4 * 4 * 4 * 3 * 2  # 4608

    def test_custom_grid(self):
        cfg = MultiScaleSweepConfig(
            trade_thresholds=[0.01, 0.02],
            momentum_alphas=[0.3],
            min_momentum_scores=[0.2],
            sl_pcts=[0.01],
            tp_pcts=[0.005],
            trail_pcts=[0.3],
            min_profit_to_trail_pcts=[0.001],
        )
        combos = cfg.combinations()
        assert len(combos) == 2  # 2 * 1 * 1 * 1 * 1 * 1 * 1

    def test_combinations_return_correct_types(self):
        cfg = MultiScaleSweepConfig(
            trade_thresholds=[0.01],
            momentum_alphas=[0.3],
            min_momentum_scores=[0.2],
            sl_pcts=[0.01],
            tp_pcts=[0.005],
            trail_pcts=[0.3],
            min_profit_to_trail_pcts=[0.001],
        )
        combos = cfg.combinations()
        assert len(combos) == 1
        c = combos[0]
        assert c.trade_threshold == 0.01
        assert c.momentum_alpha == 0.3
        assert c.min_momentum_score == 0.2

    def test_sensor_thresholds_propagated(self):
        """Sensor thresholds from sweep config should be in each combo."""
        sensors = [(0.001, 0.001), (0.003, 0.003)]
        cfg = MultiScaleSweepConfig(
            sensor_thresholds=sensors,
            trade_thresholds=[0.01],
            momentum_alphas=[0.3],
            min_momentum_scores=[0.2],
            sl_pcts=[0.01],
            tp_pcts=[0.005],
            trail_pcts=[0.3],
            min_profit_to_trail_pcts=[0.001],
        )
        combos = cfg.combinations()
        assert combos[0].sensor_thresholds == sensors
