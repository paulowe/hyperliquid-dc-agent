"""Tests for ModelLoader and ScalerParams."""

import json
import os
import tempfile

import numpy as np
import pytest
from strategies.dc_forecast.model_loader import ModelLoader, ScalerParams


class TestScalerParams:
    """Test ScalerParams dataclass and scaling logic."""

    def _make_scaler_params(self) -> ScalerParams:
        """Create a test ScalerParams matching the pipeline format."""
        return ScalerParams(
            feature_order=["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
            mean=np.array([50000.0, 0.0, 0.0]),  # Only continuous cols
            scale=np.array([5000.0, 1.0, 1.0]),
            continuous_cols=["PRICE", "OSV_Down", "OSV_Up"],
            indicator_cols=["PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"],
            std_feature_order=["PRICE_std", "PDCC_Down", "OSV_Down_std", "PDCC2_UP", "OSV_Up_std", "regime_up", "regime_down"],
        )

    def test_scale_continuous_feature(self):
        """Continuous features should be standardized: (x - mean) / scale."""
        sp = self._make_scaler_params()
        features = {
            "PRICE": 55000.0,
            "PDCC_Down": 0,
            "OSV_Down": 2.0,
            "PDCC2_UP": 1,
            "OSV_Up": 0.0,
            "regime_up": 1,
            "regime_down": 0,
        }
        scaled = sp.apply_scaling(features)
        # PRICE: (55000 - 50000) / 5000 = 1.0
        assert abs(scaled["PRICE_std"] - 1.0) < 0.001
        # OSV_Down: (2.0 - 0.0) / 1.0 = 2.0
        assert abs(scaled["OSV_Down_std"] - 2.0) < 0.001

    def test_indicators_pass_through(self):
        """Indicator columns should not be scaled."""
        sp = self._make_scaler_params()
        features = {
            "PRICE": 50000.0,
            "PDCC_Down": 1,
            "OSV_Down": 0.0,
            "PDCC2_UP": 0,
            "OSV_Up": 0.0,
            "regime_up": 0,
            "regime_down": 1,
        }
        scaled = sp.apply_scaling(features)
        assert scaled["PDCC_Down"] == 1
        assert scaled["PDCC2_UP"] == 0
        assert scaled["regime_up"] == 0
        assert scaled["regime_down"] == 1

    def test_inverse_scale_recovers_original(self):
        """Inverse transform should recover the original value."""
        sp = self._make_scaler_params()
        original_price = 55000.0
        scaled_price = (original_price - 50000.0) / 5000.0  # 1.0
        recovered = sp.inverse_scale_feature("PRICE", scaled_price)
        assert abs(recovered - original_price) < 0.001

    def test_output_keys_are_std_names(self):
        """Scaled output should use std feature names."""
        sp = self._make_scaler_params()
        features = {
            "PRICE": 50000.0,
            "PDCC_Down": 0,
            "OSV_Down": 0.0,
            "PDCC2_UP": 0,
            "OSV_Up": 0.0,
            "regime_up": 0,
            "regime_down": 0,
        }
        scaled = sp.apply_scaling(features)
        assert "PRICE_std" in scaled
        assert "OSV_Down_std" in scaled
        assert "OSV_Up_std" in scaled
        # Indicators keep their names
        assert "PDCC_Down" in scaled
        assert "regime_up" in scaled


class TestModelLoader:
    """Test ModelLoader metadata loading and parsing."""

    def _write_test_metadata(self, path: str) -> dict:
        """Write a test features_metadata.json and return its contents."""
        meta = {
            "feature_order": ["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up", "regime_up", "regime_down"],
            "mean_": [50000.0, 0.5, 0.3],
            "scale_": [5000.0, 1.2, 0.8],
            "std_feature_order": ["PRICE_std", "PDCC_Down", "OSV_Down_std", "PDCC2_UP", "OSV_Up_std", "regime_up", "regime_down"],
            "scale_indicators": False,
        }
        with open(path, "w") as f:
            json.dump(meta, f)
        return meta

    def test_load_scaler_from_local_json(self):
        """Load scaler params from a local features_metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "features_metadata.json")
            self._write_test_metadata(meta_path)

            loader = ModelLoader(model_uri="", scaler_uri=meta_path)
            scaler = loader.load_scaler_params()

            assert isinstance(scaler, ScalerParams)
            assert len(scaler.feature_order) == 7
            np.testing.assert_array_almost_equal(scaler.mean, [50000.0, 0.5, 0.3])
            np.testing.assert_array_almost_equal(scaler.scale, [5000.0, 1.2, 0.8])

    def test_scaler_continuous_vs_indicator_cols(self):
        """Verify correct separation of continuous and indicator columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "features_metadata.json")
            self._write_test_metadata(meta_path)

            loader = ModelLoader(model_uri="", scaler_uri=meta_path)
            scaler = loader.load_scaler_params()

            # Indicators should be: PDCC_Down, PDCC2_UP, regime_up, regime_down
            assert "PDCC_Down" in scaler.indicator_cols
            assert "regime_up" in scaler.indicator_cols
            # Continuous should be: PRICE, OSV_Down, OSV_Up
            assert "PRICE" in scaler.continuous_cols
            assert "OSV_Down" in scaler.continuous_cols

    def test_model_loaded_flag(self):
        """ModelLoader should track whether model and scaler are loaded."""
        loader = ModelLoader(model_uri="", scaler_uri="")
        assert loader.is_model_loaded is False
        assert loader.is_scaler_loaded is False
