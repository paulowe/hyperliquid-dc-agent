"""Tests for model loading integration in DCForecastStrategy.

Tests that the strategy correctly loads models and scalers from config URIs,
and falls back gracefully when model files are unavailable.
"""

import json
import os
import tempfile
import pytest
import numpy as np

from strategies.dc_forecast.dc_forecast_strategy import DCForecastStrategy
from strategies.dc_forecast.config import DEFAULT_FEATURE_NAMES
from strategies.dc_forecast.model_loader import ModelLoader, ScalerParams
from interfaces.strategy import MarketData


def _write_features_metadata(path: str, feature_order=None):
    """Write a valid features_metadata.json for testing."""
    feature_order = feature_order or [
        "PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"
    ]
    # Continuous cols: those not in indicators
    indicators = {"PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"}
    continuous = [f for f in feature_order if f not in indicators]

    meta = {
        "feature_order": feature_order,
        "mean_": [50000.0] + [0.0] * (len(continuous) - 1),
        "scale_": [5000.0] + [1.0] * (len(continuous) - 1),
        "std_feature_order": [
            f"{f}_std" if f not in indicators else f for f in feature_order
        ],
        "scale_indicators": False,
    }

    with open(path, "w") as f:
        json.dump(meta, f)
    return path


class TestModelLoaderFromFile:
    """Test ModelLoader can load scaler from a real JSON file."""

    def test_load_scaler_from_temp_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            meta_path = f.name
            _write_features_metadata(meta_path)

        try:
            loader = ModelLoader(model_uri="", scaler_uri=meta_path)
            scaler = loader.load_scaler_params()

            assert isinstance(scaler, ScalerParams)
            assert scaler.feature_order == [
                "PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"
            ]
            assert len(scaler.continuous_cols) == 3  # PRICE, OSV_Down, OSV_Up
            assert len(scaler.indicator_cols) == 4
            assert scaler.mean[0] == 50000.0
            assert scaler.scale[0] == 5000.0
        finally:
            os.unlink(meta_path)

    def test_scaler_caching(self):
        """Second call should return cached scaler."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            meta_path = f.name
            _write_features_metadata(meta_path)

        try:
            loader = ModelLoader(model_uri="", scaler_uri=meta_path)
            scaler1 = loader.load_scaler_params()
            scaler2 = loader.load_scaler_params()
            assert scaler1 is scaler2  # Same object (cached)
        finally:
            os.unlink(meta_path)

    def test_apply_scaling_with_real_metadata(self):
        """End-to-end: load metadata → apply scaling → verify output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            meta_path = f.name
            _write_features_metadata(meta_path)

        try:
            loader = ModelLoader(model_uri="", scaler_uri=meta_path)
            scaler = loader.load_scaler_params()

            raw_features = {
                "PRICE": 55000.0,
                "PDCC_Down": 1,
                "OSV_Down": 0.5,
                "PDCC2_UP": 0,
                "OSV_Up": 0.0,
                "regime_up": 0,
                "regime_down": 1,
            }
            scaled = scaler.apply_scaling(raw_features)

            # PRICE: (55000 - 50000) / 5000 = 1.0
            assert abs(scaled["PRICE_std"] - 1.0) < 0.001
            # Indicators pass through
            assert scaled["PDCC_Down"] == 1
            assert scaled["regime_down"] == 1
            assert scaled["PDCC2_UP"] == 0
        finally:
            os.unlink(meta_path)


class TestStrategyAutoLoadScaler:
    """Test that DCForecastStrategy auto-loads scaler at init when URI is provided."""

    def test_strategy_loads_scaler_on_init(self):
        """Strategy should auto-load scaler when scaler_uri is provided."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            meta_path = f.name
            _write_features_metadata(meta_path)

        try:
            config = {
                "symbol": "BTC",
                "dc_thresholds": [(0.01, 0.01)],
                "scaler_uri": meta_path,
                "model_uri": "",  # No model
                "log_dc_events": False,
            }
            strategy = DCForecastStrategy(config)

            # Scaler should be loaded
            assert strategy._scaler_params is not None
            assert strategy._scaler_params.feature_order[0] == "PRICE"
            # Model is not loaded (empty URI)
            assert strategy._model is None
            assert strategy.has_model is False
        finally:
            os.unlink(meta_path)

    def test_strategy_graceful_without_scaler(self):
        """Strategy should work fine without scaler (Phase 1 mode)."""
        config = {
            "symbol": "BTC",
            "dc_thresholds": [(0.01, 0.01)],
            "scaler_uri": "",
            "model_uri": "",
            "log_dc_events": False,
        }
        strategy = DCForecastStrategy(config)

        # No scaler, no model → Phase 1
        assert strategy._scaler_params is None
        assert strategy._model is None
        assert strategy.has_model is False

        # Should still process ticks (Phase 1 mode)
        md = MarketData(asset="BTC", price=100.0, volume_24h=0.0, timestamp=1000.0)
        signals = strategy.generate_signals(md, [], 10000.0)
        assert signals == []

    def test_strategy_graceful_with_bad_scaler_path(self):
        """Strategy should log error but not crash on bad scaler path."""
        config = {
            "symbol": "BTC",
            "dc_thresholds": [(0.01, 0.01)],
            "scaler_uri": "/nonexistent/path/features_metadata.json",
            "model_uri": "",
            "log_dc_events": False,
        }
        # Should not raise
        strategy = DCForecastStrategy(config)

        # Scaler failed to load → Phase 1 fallback
        assert strategy._scaler_params is None

    def test_strategy_uses_loaded_scaler_for_features(self):
        """When scaler is loaded, features should be scaled properly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            meta_path = f.name
            _write_features_metadata(meta_path)

        try:
            config = {
                "symbol": "BTC",
                "dc_thresholds": [(0.01, 0.01)],
                "scaler_uri": meta_path,
                "model_uri": "",
                "log_dc_events": False,
                "window_size": 3,
            }
            strategy = DCForecastStrategy(config)

            # Feed a few ticks
            for i in range(3):
                md = MarketData(asset="BTC", price=50000.0 + i, volume_24h=0.0, timestamp=1000.0 + i)
                strategy.generate_signals(md, [], 10000.0)

            # Buffer should have scaled values
            assert strategy._buffer.is_ready()
            window = strategy._buffer.get_window()

            # First column (PRICE_std) should be scaled: (50000 - 50000) / 5000 = 0.0
            assert abs(window[0, 0] - 0.0) < 0.01
        finally:
            os.unlink(meta_path)
