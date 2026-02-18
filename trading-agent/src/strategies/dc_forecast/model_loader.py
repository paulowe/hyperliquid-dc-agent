"""Model loader for TensorFlow models and scaler parameters.

Loads trained models and StandardScaler parameters from local paths or GCS,
as saved by the Vertex AI training pipeline (tf_data_splitter.py).

The scaler metadata format (features_metadata.json):
    {
        "feature_order": ["PRICE", "PDCC_Down", "OSV_Down", ...],
        "mean_": [50000.0, ...],   # means for continuous cols only
        "scale_": [5000.0, ...],   # stds for continuous cols only
        "std_feature_order": ["PRICE_std", "PDCC_Down", "OSV_Down_std", ...],
        "scale_indicators": false
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Columns that are binary indicators (not scaled by StandardScaler)
INDICATOR_COLUMNS = {"PDCC_Down", "PDCC2_UP", "regime_up", "regime_down"}


@dataclass
class ScalerParams:
    """StandardScaler parameters for feature normalization.

    Continuous features are standardized: (x - mean) / scale.
    Indicator features (binary 0/1) are passed through unchanged.
    """

    feature_order: List[str]
    mean: np.ndarray
    scale: np.ndarray
    continuous_cols: List[str]
    indicator_cols: List[str]
    std_feature_order: List[str]

    def apply_scaling(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale a feature dict and return with standardized names.

        Args:
            features: Raw features keyed by original names (e.g., "PRICE").

        Returns:
            Scaled features keyed by std names (e.g., "PRICE_std").
            Indicator cols keep their original names.
        """
        scaled = {}

        # Build mapping from continuous col name to index in mean/scale arrays
        cont_to_idx = {col: i for i, col in enumerate(self.continuous_cols)}

        for raw_name, std_name in zip(self.feature_order, self.std_feature_order):
            if raw_name in cont_to_idx:
                idx = cont_to_idx[raw_name]
                val = float(features.get(raw_name, 0.0))
                scaled[std_name] = (val - float(self.mean[idx])) / float(self.scale[idx])
            else:
                # Indicator: pass through with original name
                scaled[std_name] = float(features.get(raw_name, 0.0))

        return scaled

    def inverse_scale_feature(self, raw_name: str, scaled_value: float) -> float:
        """Inverse-transform a single scaled feature back to original scale.

        Args:
            raw_name: Original feature name (e.g., "PRICE").
            scaled_value: Standardized value.

        Returns:
            Original-scale value.
        """
        cont_to_idx = {col: i for i, col in enumerate(self.continuous_cols)}
        if raw_name in cont_to_idx:
            idx = cont_to_idx[raw_name]
            return scaled_value * float(self.scale[idx]) + float(self.mean[idx])
        return scaled_value


class ModelLoader:
    """Loads TF models and scaler params from local or GCS paths.

    Usage:
        loader = ModelLoader(
            model_uri="gs://bucket/model.keras",
            scaler_uri="gs://bucket/features_metadata.json",
        )
        scaler = loader.load_scaler_params()
        model = loader.load_model()  # Returns keras.Model
    """

    def __init__(self, model_uri: str, scaler_uri: str) -> None:
        self._model_uri = model_uri
        self._scaler_uri = scaler_uri
        self._model: Any = None
        self._scaler_params: Optional[ScalerParams] = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_scaler_loaded(self) -> bool:
        return self._scaler_params is not None

    def load_scaler_params(self) -> ScalerParams:
        """Load and parse scaler parameters from features_metadata.json.

        Returns:
            ScalerParams instance.
        """
        if self._scaler_params is not None:
            return self._scaler_params

        meta = self._read_json(self._scaler_uri)

        feature_order = meta["feature_order"]
        mean = np.array(meta["mean_"], dtype=np.float64)
        scale = np.array(meta["scale_"], dtype=np.float64)
        std_feature_order = meta.get("std_feature_order", [])
        scale_indicators = meta.get("scale_indicators", False)

        # Separate continuous and indicator columns
        indicator_cols = [c for c in feature_order if c in INDICATOR_COLUMNS]
        continuous_cols = [c for c in feature_order if c not in INDICATOR_COLUMNS]

        # If std_feature_order not provided, generate it
        if not std_feature_order:
            std_feature_order = [
                f"{c}_std" if c in continuous_cols else c for c in feature_order
            ]

        self._scaler_params = ScalerParams(
            feature_order=feature_order,
            mean=mean,
            scale=scale,
            continuous_cols=continuous_cols,
            indicator_cols=indicator_cols,
            std_feature_order=std_feature_order,
        )

        logger.info(
            "[ModelLoader] Scaler loaded: %d features (%d continuous, %d indicator)",
            len(feature_order),
            len(continuous_cols),
            len(indicator_cols),
        )
        return self._scaler_params

    def load_model(self) -> Any:
        """Load TensorFlow model from model_uri.

        Returns:
            keras.Model instance, or None if model_uri is empty.
        """
        if self._model is not None:
            return self._model

        if not self._model_uri:
            logger.warning("[ModelLoader] No model_uri configured; inference disabled.")
            return None

        try:
            import tensorflow as tf

            # Handle GCS paths
            if self._model_uri.startswith("gs://"):
                # TF can load directly from GCS
                self._model = tf.keras.models.load_model(self._model_uri)
            else:
                self._model = tf.keras.models.load_model(self._model_uri)

            logger.info("[ModelLoader] Model loaded from %s", self._model_uri)
            return self._model
        except ImportError:
            logger.error("[ModelLoader] TensorFlow not installed. Cannot load model.")
            return None
        except Exception as e:
            logger.error("[ModelLoader] Failed to load model from %s: %s", self._model_uri, e)
            return None

    @staticmethod
    def _read_json(uri: str) -> dict:
        """Read a JSON file from local path or GCS."""
        if uri.startswith("gs://"):
            try:
                import tensorflow as tf
                with tf.io.gfile.GFile(uri, "r") as f:
                    return json.loads(f.read())
            except ImportError:
                from google.cloud import storage
                # Parse gs://bucket/path
                parts = uri[5:].split("/", 1)
                bucket_name, blob_path = parts[0], parts[1]
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                return json.loads(blob.download_as_text())
        else:
            with open(uri, "r") as f:
                return json.load(f)
