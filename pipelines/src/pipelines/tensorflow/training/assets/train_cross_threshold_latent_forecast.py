import argparse
import io
import os
import json
import logging
import google.cloud.logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense

# from tensorflow.data import Dataset  # unused in current tf.data API usage
from typing import List, Tuple, Optional, Dict

# This training script consumes the latent TFRecord bundles produced by
# concat_latent_datasets() and trains a Keras MLP to forecast the future price using
# the latent vectors from multiple directional-change thresholds at once.

# Set up logging
client = google.cloud.logging.Client()
client.setup_logging()

text = "Cross-Threshold Latent Forecast Training Script"
logging.info(text)

# -----------------------------------------------------------------------------
# Constants / Defaults
# -----------------------------------------------------------------------------
TRAINING_DATASET_INFO = "training_dataset.json"
SCALER_FILENAME = "scaler.joblib"
SCALER_METADATA_FILENAME = "scaler_metadata.json"

# Default hyperparameters remain close to the single-threshold variant but add extra
# knobs (`label_source_threshold`, `label_source_index`) so the caller can pick which
# threshold's labels to train against after concatenation.
DEFAULT_HPARAMS = dict(
    batch_size=256,
    epochs=5,
    loss_fn="MeanSquaredError",
    optimizer="Adam",
    learning_rate=0.001,
    latent_dim=3,
    seq_length=50,
    n_features=6,
    forecast_steps=10,  # Number of steps to forecast into the future
    patience=5,
    metrics=[
        "RootMeanSquaredError",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredLogarithmicError",
    ],
    hidden_units=[(64, "relu"), (32, "relu")],
    distribute_strategy="single",
    early_stopping_epochs=3,
    label_name="PRICE_std",
    use_mixed_precision=True,
    use_xla_jit=True,
    # --- NEW: latent TFRecord settings (keys & compression) ---
    latent_key="z",
    label_key="y",
    z_mean_key="z_mean",  # optional present in files; not used by model
    z_log_var_key="z_log_var",  # optional present in files; not used by model
    compression="GZIP",  # "GZIP" or "" (uncompressed)
    label_source_index=2,  # default to 0.010 threshold (3rd entry)
    label_source_threshold="0.010000",
)


def _canonical_threshold(value) -> Optional[str]:
    """Normalize threshold identifiers to a consistent string for comparisons.

    The concatenation step may emit threshold identifiers as floats or strings with
    varying precision. We collapse everything into a fixed-width string so equality
    checks stay stable no matter how the metadata was written.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        numeric = float(value)
        return f"{numeric:.6f}"
    except (TypeError, ValueError):
        return str(value)


def _smape(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Symmetric mean absolute percentage error with zero-safe denominator."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = tf.abs(y_pred - y_true)
    denominator = tf.maximum(tf.abs(y_true) + tf.abs(y_pred), tf.constant(1e-6))
    return tf.reduce_mean(2.0 * numerator / denominator, axis=-1)


_smape.__name__ = "smape"  # Helps Keras report metrics under an intuitive name.


def _resolve_optimizer(hparams: Dict) -> keras.optimizers.Optimizer:
    """Instantiate an optimizer from hparams, validating the requested type."""
    optimizer_name = hparams.get("optimizer", "Adam")
    learning_rate = float(hparams.get("learning_rate", 0.001))
    optimizer_cls = getattr(optimizers, optimizer_name, None)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}' requested")
    return optimizer_cls(learning_rate=learning_rate)

# -----------------------------------------------------------------------------
# Helper: distribution strategy (unchanged)
# -----------------------------------------------------------------------------


def get_distribution_strategy(distribute_strategy: str) -> tf.distribute.Strategy:
    """Set distribute strategy based on input string.
    Args:
        distribute_strategy (str): single, mirror or multi
    Returns:
        strategy (tf.distribute.Strategy): distribution strategy
    """
    logging.info(f"Distribution strategy: {distribute_strategy}")

    # Single machine, single compute device
    if distribute_strategy == "single":
        if len(tf.config.list_physical_devices("GPU")):
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # Single machine, multiple compute device
    elif distribute_strategy == "mirror":
        strategy = tf.distribute.MirroredStrategy()
    # Multiple machine, multiple compute device
    elif distribute_strategy == "multi":
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise RuntimeError(f"Distribute strategy: {distribute_strategy} not supported")
    return strategy


def _is_chief(strategy: tf.distribute.Strategy) -> bool:
    """Determine whether current worker is the chief (master). See more info:
    - https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    - https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/ClusterResolver # noqa: E501
    Args:
        strategy (tf.distribute.Strategy): strategy
    Returns:
        is_chief (bool): True if worker is chief, otherwise False
    """
    cr = strategy.cluster_resolver
    return (cr is None) or (cr.task_type == "chief" and cr.task_id == 0)


def _to_local_path(maybe_gcs_path: str) -> Path:
    """Convert a gs:// URI to the local /gcs mount if necessary.

    Vertex AI mounts buckets at /gcs/<bucket>/..., so we rewrite cloud URIs into their
    on-disk counterpart before interacting with the filesystem.
    """
    if maybe_gcs_path.startswith("gs://"):
        return Path("/gcs/" + maybe_gcs_path[5:])
    return Path(maybe_gcs_path)


def _load_scaler(scaler_uri: str):
    """Load a persisted StandardScaler artifact, returning (scaler, feature_order).

    The scaler is optional for training but highly useful for reporting metrics back
    in the original price space, so we reuse the artifact emitted by tf_data_splitter.
    """
    if not scaler_uri:
        return None, None

    scaler_dir = _to_local_path(scaler_uri.rstrip("/"))
    scaler_path = scaler_dir / SCALER_FILENAME
    if not scaler_path.exists():
        logging.warning(
            "Scaler artifact path %s missing %s; inverse transform skipped.",
            scaler_dir,
            SCALER_FILENAME,
        )
        return None, None

    try:
        import joblib  # type: ignore
    except ImportError:
        joblib = None

    scaler_obj = None
    if joblib is not None:
        try:
            scaler_obj = joblib.load(scaler_path)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("joblib.load failed (%s); falling back to pickle.", exc)
    if scaler_obj is None:
        import pickle

        with open(scaler_path, "rb") as fh:
            scaler_obj = pickle.load(fh)

    feature_order = None
    meta_path = scaler_dir / SCALER_METADATA_FILENAME
    if meta_path.exists():
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
            feature_order = meta.get("feature_order")
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to read scaler metadata (%s); continuing.", exc)

    return scaler_obj, feature_order


def _save_prediction_plot(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, model_dir: Path):
    """Plot a slice of predictions vs. ground truth and persist to local/GCS paths."""
    sample_len = min(200, y_true_raw.shape[0])
    if sample_len == 0:
        logging.warning("Prediction plot skipped: no samples available.")
        return None, None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("matplotlib unavailable; skipping prediction plot (%s).", exc)
        return None, None

    x_axis = np.arange(sample_len)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_axis, y_true_raw[:sample_len], label="Actual", linewidth=1.5)
    ax.plot(x_axis, y_pred_raw[:sample_len], label="Forecast", linewidth=1.2)
    ax.set_title("Latent Forecast: Predictions vs Actual (Raw Scale)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    rel_path = Path("figures") / "latent_forecast_pred_vs_truth.png"
    local_path = model_dir / rel_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    with open(local_path, "wb") as fp:
        fp.write(buf.getvalue())

    gcs_uri = None
    model_dir_str = str(model_dir)
    if model_dir_str.startswith("/gcs/"):
        gcs_uri = "gs://" + model_dir_str[5:]
        if not gcs_uri.endswith("/"):
            gcs_uri += "/"
        gcs_uri += rel_path.as_posix()
        buf.seek(0)
        with tf.io.gfile.GFile(gcs_uri, "wb") as gcs_fp:
            gcs_fp.write(buf.getvalue())
        logging.info("Saved prediction plot to %s", gcs_uri)
    else:
        logging.info("Saved prediction plot locally to %s", local_path)

    plt.close(fig)
    return local_path.as_posix(), gcs_uri


def inspect_dataset_v2(ds: tf.data.Dataset, name="dataset", max_batches=1):
    try:
        for i, (x, y) in enumerate(ds.take(max_batches)):
            tf.debugging.assert_all_finite(
                x, f"{name}: found NaN/Inf in features (batch {i})"
            )
            tf.debugging.assert_all_finite(
                y, f"{name}: found NaN/Inf in labels (batch {i})"
            )
        logging.info(f"{name}: basic NaN/Inf checks passed on {max_batches} batch(es).")
    except Exception as e:
        logging.exception(f"Inspection error on {name}: {e}")


# -----------------------------------------------------------------------------
# (Deprecated): CSV helpers
# -----------------------------------------------------------------------------
def read_csv_any(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Robust CSV reader that works for local files or GCS URIs via tf.io.gfile."""
    if path.startswith("gs://"):
        with tf.io.gfile.GFile(path, "rb") as f:
            return pd.read_csv(f, usecols=usecols)
    return pd.read_csv(path, usecols=usecols)


# -----------------------------------------------------------------------------
# (Deprecated): Latent TFRecord input pipeline
# -----------------------------------------------------------------------------
def _latent_feature_spec(hp: Dict) -> Dict[str, tf.io.FixedLenFeature]:
    """Build a parsing spec for latent TFRecords."""
    spec = {
        hp["latent_key"]: tf.io.FixedLenFeature([hp["latent_dim"]], tf.float32),
        hp["label_key"]: tf.io.FixedLenFeature([hp["forecast_steps"]], tf.float32),
    }
    # Optionally parse z_mean / z_log_var if present (ignored by the model)
    # Using default_value=[] lets parse succeed even if absent; we’ll guard later.
    if hp.get("z_mean_key"):
        spec[hp["z_mean_key"]] = tf.io.FixedLenFeature(
            [hp["latent_dim"]], tf.float32, default_value=[0.0] * hp["latent_dim"]
        )
    if hp.get("z_log_var_key"):
        spec[hp["z_log_var_key"]] = tf.io.FixedLenFeature(
            [hp["latent_dim"]], tf.float32, default_value=[0.0] * hp["latent_dim"]
        )
    return spec


def _parse_latent_example(
    serialized: tf.Tensor, spec: Dict, latent_key: str, label_key: str
):
    """Parse a single Example -> (z, y)."""
    ex = tf.io.parse_single_example(serialized, features=spec)
    z = ex[latent_key]  # [latent_dim]
    y = ex[label_key]  # [forecast_steps]
    return z, y


# Historical note: earlier iterations kept fully commented helpers for reading
# TFRecords (custom ``_make_latent_dataset_from_dir`` variants). Those snippets
# were removed here to keep linting happy; refer to git history if the legacy
# versions are needed again.


def _read_manifest_if_any(data_dir: str):
    """Return the manifest saved alongside the TFRecord shards (if present).

    The concat component writes shape, compression, and source threshold metadata.
    Reading it here lets us parse the multi-threshold records without relying solely
    on global hparams defaults.
    """
    mpath = os.path.join(data_dir, "_manifest.json")
    if tf.io.gfile.exists(mpath):
        with tf.io.gfile.GFile(mpath, "r") as f:
            return json.load(f)
    return None


# v3 list_tfrecord_files
def _list_tfrecord_files(data_dir: str, compression: str) -> tf.data.Dataset:
    pattern = "*.tfrecord.gz" if (compression or "").upper() == "GZIP" else "*.tfrecord"
    files_glob = os.path.join(data_dir, pattern)
    return tf.data.Dataset.list_files(files_glob, shuffle=False)


# v2 make_latent_dataset_from_dir
#     """Build (z, y) dataset from TFRecords written by the VAE step."""


# Legacy parsing fallbacks for variable-length label layouts have been omitted
# for brevity. If unusual TFRecord schemas resurface, restore the original
# parsing logic from version control.

# v3
def _filter_files_with_keys(
    filepaths: list[str], required_keys: set[str], compression: str
) -> list[str]:
    good = []
    for fp in filepaths:
        try:
            for rec in tf.data.TFRecordDataset([fp], compression_type=compression).take(
                1
            ):
                ex = tf.train.Example()
                ex.ParseFromString(bytes(rec.numpy()))
                feats = ex.features.feature
                if all(k in feats for k in required_keys):
                    good.append(fp)
                else:
                    missing = [k for k in required_keys if k not in feats]
                    logging.warning(f"Skipping {fp}: missing keys {missing}")
                break
        except Exception as e:
            logging.warning(f"Skipping {fp}: cannot read first record ({e})")
    return good


def _make_latent_dataset_from_dir(
    data_dir: str, hparams: dict, shuffle: bool
) -> tf.data.Dataset:
    """
    Load latent TFRecord dataset (z, y) from a run-scoped split directory.

    Args:
        data_dir: Directory containing TFRecord shards for one split
            (train/valid/test).
        hparams: Dict of hyperparameters (latent_dim, forecast_steps,
            batch_size, etc.).
        shuffle: Whether to shuffle records (True for training, False for eval).
    Returns:
        tf.data.Dataset of (z, y) pairs.
    """

    # --------------------------
    # Step 1. Resolve config values
    # --------------------------
    # Default compression is GZIP (since Train Script 1 writes .tfrecord.gz).
    # The upstream pipeline defaults to GZIP-compressed shards; callers can override
    # the flag in hparams but we fall back to that expectation.
    compression = (hparams.get("compression") or "GZIP").upper()
    compression_type = "GZIP" if compression == "GZIP" else ""

    # Shapes from hparams
    latent_dim = int(hparams["latent_dim"])
    forecast_steps = int(hparams["forecast_steps"])

    # Feature keys (allow overrides in hparams, fallback to defaults)
    latent_key = str(hparams.get("latent_key", "z"))
    mean_key = str(hparams.get("mean_key", "z_mean"))
    logvar_key = str(hparams.get("logvar_key", "z_log_var"))
    label_key = str(hparams.get("label_key", "y"))

    per_threshold_label_dim = forecast_steps
    num_thresholds = int(hparams.get("num_thresholds", 1))
    # We store the manifest-provided thresholds as canonical strings so we can map
    # a human readable threshold (like "0.010000") back to the correct row later.
    thresholds_resolved: List[Optional[str]] = []

    # --------------------------
    # Step 2. Override with manifest if present
    # --------------------------
    manifest = _read_manifest_if_any(data_dir)
    if manifest:
        feats = manifest.get("features", {})

        required_manifest_keys = [latent_key, label_key]
        if mean_key:
            required_manifest_keys.append(mean_key)
        if logvar_key:
            required_manifest_keys.append(logvar_key)

        for k in required_manifest_keys:
            if k not in feats:
                raise ValueError(f"Manifest missing required key '{k}' in {data_dir}")

        latent_dim_manifest = int(feats[latent_key]["shape"][0])
        total_label_dim_manifest = int(feats[label_key]["shape"][0])

        # Each concatenated record keeps track of the original latent dims. This lets
        # us deduce how many thresholds were combined even if the hparams were left at
        # their defaults.
        source_latent_dims = manifest.get("sourceLatentDims") or []
        if source_latent_dims:
            num_thresholds = len(source_latent_dims)

        if manifest.get("numThresholds"):
            try:
                num_thresholds = max(1, int(manifest["numThresholds"]))
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid numThresholds in manifest (%s); keeping %s",
                    manifest.get("numThresholds"),
                    num_thresholds,
                )

        src_label_dim = manifest.get("sourceLabelDim")
        if src_label_dim is not None:
            try:
                per_threshold_label_dim = max(1, int(src_label_dim))
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid sourceLabelDim in manifest (%s); keeping %s",
                    src_label_dim,
                    per_threshold_label_dim,
                )

        thresholds_meta = manifest.get("sourceThresholds") or []
        thresholds_resolved = [
            _canonical_threshold(th) for th in thresholds_meta if th is not None
        ]
        if thresholds_resolved:
            num_thresholds = len(thresholds_resolved)

        total_label_dim = per_threshold_label_dim * max(1, num_thresholds)
        if total_label_dim_manifest != total_label_dim:
            if (
                num_thresholds > 0
                and total_label_dim_manifest % num_thresholds == 0
            ):
                logging.warning(
                    "Label dimension mismatch (manifest=%s vs computed=%s). "
                    "Adjusting per-threshold label dim.",
                    total_label_dim_manifest,
                    total_label_dim,
                )
                # When the mismatch is divisible we assume the manifest is the source of
                # truth and recalibrate the per-threshold dimension accordingly.
                per_threshold_label_dim = max(
                    1, total_label_dim_manifest // num_thresholds
                )
                total_label_dim = total_label_dim_manifest
            else:
                raise ValueError(
                    "Unable to reconcile label dimensions: manifest=%s, "
                    "num_thresholds=%s."
                    % (total_label_dim_manifest, num_thresholds)
                )
        else:
            total_label_dim = total_label_dim_manifest

        if latent_dim_manifest != latent_dim:
            logging.warning(
                "Override latent dim from hparams %s -> manifest %s",
                latent_dim,
                latent_dim_manifest,
            )
            latent_dim = latent_dim_manifest

        # Update hparams so downstream consumers see resolved values
        hparams["latent_dim"] = latent_dim
        hparams["forecast_steps"] = per_threshold_label_dim
        hparams["num_thresholds"] = num_thresholds

        # Override compression if manifest disagrees
        comp_m = (manifest.get("compression") or compression).upper()
        if comp_m != compression:
            logging.warning(
                "Override compression from hparams -> manifest: %s->%s",
                compression,
                comp_m,
            )
            compression = comp_m
            compression_type = "GZIP" if compression == "GZIP" else ""
    else:
        total_label_dim = per_threshold_label_dim * max(1, num_thresholds)

    num_thresholds = max(1, num_thresholds)
    per_threshold_label_dim = max(1, per_threshold_label_dim)
    total_label_dim = max(1, total_label_dim)

    # Determine which threshold's labels to target
    requested_threshold = _canonical_threshold(hparams.get("label_source_threshold"))
    label_source_index = int(hparams.get("label_source_index", 0))

    # Primary path: try to match the human-readable threshold first so pipeline users
    # can request "0.01" without worrying about the underlying ordering.
    if requested_threshold and thresholds_resolved:
        if requested_threshold in thresholds_resolved:
            label_source_index = thresholds_resolved.index(requested_threshold)
        else:
            logging.warning(
                "Requested threshold %s not found in %s. Falling back to index %s.",
                requested_threshold,
                thresholds_resolved,
                label_source_index,
            )

    if label_source_index < 0 or label_source_index >= num_thresholds:
        logging.warning(
            "label_source_index %s out of range [0, %s). Clamping.",
            label_source_index,
            num_thresholds,
        )
        # Guardrail: prevent out-of-bounds slicing when the caller provides an invalid
        # index (perhaps because fewer thresholds were produced downstream).
        label_source_index = min(max(label_source_index, 0), num_thresholds - 1)

    if not thresholds_resolved:
        thresholds_resolved = [None] * num_thresholds
    elif len(thresholds_resolved) != num_thresholds:
        logging.warning(
            "Manifest threshold count mismatch (threshold list len=%s, num_thresholds=%s).",
            len(thresholds_resolved),
            num_thresholds,
        )
        thresholds_resolved = (thresholds_resolved + [None] * num_thresholds)[
            :num_thresholds
        ]

    selected_threshold_display = thresholds_resolved[label_source_index]

    hparams["label_source_index"] = label_source_index
    hparams["label_source_threshold_resolved"] = selected_threshold_display
    hparams["num_thresholds"] = num_thresholds
    hparams["forecast_steps_per_threshold"] = per_threshold_label_dim
    hparams["label_thresholds"] = thresholds_resolved

    logging.info(
        "[reader] %s -> latent_dim=%s, total_label_dim=%s, per_threshold_label_dim=%s, "
        "num_thresholds=%s, label_source_index=%s, label_source_threshold=%s",
        data_dir,
        latent_dim,
        total_label_dim,
        per_threshold_label_dim,
        num_thresholds,
        label_source_index,
        selected_threshold_display,
    )

    # --------------------------
    # Step 3. Collect candidate TFRecord shards
    # --------------------------
    # Pattern matches .tfrecord.gz if compression=GZIP else .tfrecord
    pattern = "*.tfrecord.gz" if compression == "GZIP" else "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(data_dir, pattern))
    logging.info(f"[reader] {data_dir}: found {len(files)} shards matching {pattern}")
    if not files:
        raise ValueError(f"No TFRecord files found in {data_dir} (pattern={pattern})")

    # --------------------------
    # Step 4. Filter shards for required keys
    # --------------------------
    required_keys = {latent_key, label_key}
    if mean_key:
        required_keys.add(mean_key)
    if logvar_key:
        required_keys.add(logvar_key)
    files = _filter_files_with_keys(files, required_keys, compression_type)
    logging.info(
        "[reader] %s: %s shards after key filter %s",
        data_dir,
        len(files),
        sorted(required_keys),
    )
    if not files:
        raise ValueError(
            "No valid TFRecord files found in %s with required keys %s"
            % (data_dir, sorted(required_keys))
        )

    # --------------------------
    # Step 5. Define parse spec
    # --------------------------
    # We parse the stacked label vector as a single block of length
    # num_thresholds * per_threshold_label_dim, then split it later when we choose the
    # target threshold.
    feature_spec = {
        latent_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
        label_key: tf.io.FixedLenFeature([total_label_dim], tf.float32),
    }
    if mean_key:
        feature_spec[mean_key] = tf.io.FixedLenFeature([latent_dim], tf.float32)
    if logvar_key:
        feature_spec[logvar_key] = tf.io.FixedLenFeature([latent_dim], tf.float32)

    def _parse(serialized):
        """Parse a single Example -> (z, y)"""
        ex = tf.io.parse_single_example(serialized, feature_spec)
        # Ensure shapes are exactly as expected
        z = tf.ensure_shape(ex[latent_key], [latent_dim])
        y_flat = tf.ensure_shape(ex[label_key], [total_label_dim])
        y_flat = tf.cast(y_flat, tf.float32)
        y_matrix = tf.reshape(y_flat, [num_thresholds, per_threshold_label_dim])
        y_selected = tf.gather(y_matrix, label_source_index, axis=0)
        # Cast to float32 (guard against float64 sneaking in)
        return tf.cast(z, tf.float32), tf.cast(y_selected, tf.float32)

    # --------------------------
    # Step 6. Build the tf.data pipeline
    # --------------------------
    # Read shards with parallelism
    ds = tf.data.TFRecordDataset(
        files,
        compression_type=compression_type,
        num_parallel_reads=tf.data.AUTOTUNE,
    )

    # Parse into (z, y)
    # Map each serialized Example into (concatenated latent vector, selected label).
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle if requested (buffer/seed from hparams or defaults)
    if shuffle:
        buf = int(hparams.get("shuffle_buffer", 10000))
        seed = int(hparams.get("shuffle_seed", 1234))
        ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=True)

    # Batch and prefetch
    ds = ds.batch(int(hparams["batch_size"]), drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# -----------------------------------------------------------------------------
# Build forecasting model that takes latent vectors as input (unchanged)
# -----------------------------------------------------------------------------
def build_latent_forecast_model(
    latent_dim: int,
    forecast_steps: int,
    hidden_units: List[Tuple[int, str]],
    optimizer: keras.optimizers.Optimizer,
    loss_fn: str,
    metric_fns: List,
    num_labels: int = 1,
) -> keras.Model:
    """Build a forecasting model that takes latent vectors as input.

    Args:
        latent_dim (int): Dimension of latent vectors
        forecast_steps (int): Number of steps to forecast
        hidden_units (List[Tuple[int, str]]): List of (units, activation) tuples
        num_labels (int): Number of label columns to forecast

    Returns:
        model (keras.Model): Compiled forecasting model
    """
    inputs = keras.Input(shape=(latent_dim,))

    x = inputs
    for units, activation in hidden_units:
        x = Dense(units, activation=activation)(x)
        x = layers.Dropout(0.2)(x)

    # Output layer for forecasting multiple labels
    # Shape: (batch_size, forecast_steps * num_labels)
    outputs = Dense(forecast_steps * num_labels, name="forecast", activation="linear")(
        x
    )

    model = keras.Model(inputs=inputs, outputs=outputs, name="latent_forecast_model")

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metric_fns)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument(
        "--model_dir", type=str, default=os.getenv("AIP_MODEL_DIR", "model")
    )
    # Kept for backward compatibility; not needed when consuming precomputed latents.
    parser.add_argument("--parent_model_uri", type=str, required=False)
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    parser.add_argument("--scaler_artifact_uri", type=str, default="")
    args = parser.parse_args()

    # ============================================================
    # Convert GCS paths if present
    # Resolve model_dir for Vertex (gs:// -> /gcs/...)
    # ============================================================
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])

    # Reload the StandardScaler produced upstream so we can optionally report metrics
    # in the original price space after training.
    scaler, scaler_feature_order = _load_scaler(args.scaler_artifact_uri)
    price_index = None
    if scaler is not None:
        logging.info("Loaded scaler artifact from %s", args.scaler_artifact_uri)
        if scaler_feature_order and "PRICE" in scaler_feature_order:
            price_index = scaler_feature_order.index("PRICE")
        else:
            default_order = [
                "start_price",
                "PRICE",
                "vol_quote",
                "cvd_quote",
                "PDCC_Down",
                "OSV_Down",
                "PDCC2_UP",
                "OSV_Up",
            ]
            price_index = default_order.index("PRICE")
    else:
        logging.info("No scaler artifact supplied; metrics remain in standardized scale.")

    # ============================================================
    # Merge hyperparameters
    # ============================================================
    hparams = {**DEFAULT_HPARAMS, **args.hparams}

    def _normalise_hidden_units(units_spec):
        normalised = []
        for item in units_spec or []:
            if isinstance(item, dict):
                units = int(item.get("units"))
                activation = item.get("activation")
                if activation is None:
                    raise ValueError(
                        f"Missing activation in hidden_units entry: {item}"
                    )
                normalised.append((units, activation))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                normalised.append((int(item[0]), item[1]))
            else:
                raise ValueError(f"Unsupported hidden_units entry: {item}")
        return normalised

    hparams["hidden_units"] = _normalise_hidden_units(hparams.get("hidden_units"))
    logging.info(f"Using model hyper-parameters: {hparams}")

    # ============================================================
    # Enable JIT + mixed precision (GPU only)
    # ============================================================
    gpu_devices = tf.config.list_physical_devices("GPU")
    if hparams.get("use_xla_jit", True) and gpu_devices:
        try:
            tf.config.optimizer.set_jit(True)
            logging.info("XLA JIT enabled")
        except Exception as e:
            logging.warning(f"Could not enable XLA JIT: {e}")
    elif hparams.get("use_xla_jit", True):
        logging.info("Skipping XLA JIT: no GPU devices detected")

    if hparams.get("use_mixed_precision", True) and tf.config.list_physical_devices(
        "GPU"
    ):
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            logging.info("Mixed precision enabled")
        except Exception as e:
            logging.warning(f"Could not enable mixed precision: {e}")

    # ============================================================
    # Set distribution strategy
    # ============================================================
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # ============================================================
    # Load latent datasets (local or GCS) with error logging
    # ============================================================
    logging.info(f"[reader] args.train_data={args.train_data}")
    files = tf.io.gfile.glob(os.path.join(args.train_data, "*.tfrecord.gz"))
    logging.info(f"[reader] found {len(files)} gz shards")
    train_ds = _make_latent_dataset_from_dir(args.train_data, hparams, shuffle=True)
    valid_ds = _make_latent_dataset_from_dir(args.valid_data, hparams, shuffle=False)
    test_ds = _make_latent_dataset_from_dir(args.test_data, hparams, shuffle=False)

    # Quick integrity check (first batch)
    # Run small-sample integrity checks before we hand the datasets to Keras. This
    # surfaces NaNs or shape mismatches early instead of failing deep in model.fit.
    inspect_dataset_v2(train_ds, "train_ds", 1)
    inspect_dataset_v2(valid_ds, "val_ds", 1)
    inspect_dataset_v2(test_ds, "test_ds", 1)

    # ============================================================
    # Build model
    # ============================================================
    loss_fn = hparams.get("loss_fn", "mse")
    metric_fns = [
        keras.metrics.MeanAbsoluteError(name="mae"),
        _smape,
        keras.metrics.RootMeanSquaredError(name="rmse"),
    ]

    with strategy.scope():
        # num_labels is 1 because y is already [forecast_steps] for the single label.
        model = build_latent_forecast_model(
            latent_dim=hparams["latent_dim"],
            forecast_steps=hparams["forecast_steps"],
            hidden_units=hparams["hidden_units"],
            optimizer=_resolve_optimizer(hparams),
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            num_labels=1,
        )

    model.summary()

    # Set up callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=hparams["early_stopping_epochs"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    # ============================================================
    # Train the model
    # ============================================================
    logging.info("Starting training...")
    model.fit(
        train_ds,  # (z, y), y shape = [forecast_steps]
        epochs=hparams["epochs"],
        validation_data=valid_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # Only persist output if this worker is chief
    if not _is_chief(strategy):
        logging.info("Non-chief worker: exiting without saving model/metrics.")
        sys.exit(0)

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = (
        f"{args.model_dir}/latent_forecast_{hparams.get('threshold', 'default')}.keras"
    )
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # Evaluate on test set
    eval_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    logging.info(f"Evaluation metrics: {eval_metrics}")

    # Compute forecast vs actual means on the test set

    # Ground truth mean. Stay in TF runtime until final scalar
    # y_true/y_pred remain on standardized scale until an optional inverse transform.
    y_true = tf.concat([y for _, y in test_ds], axis=0)
    y_true_np = y_true.numpy()
    mean_Y = float(np.mean(y_true_np))

    # Predictions: model.predict from Keras always returns NumPy
    y_pred = model.predict(test_ds, verbose=0)
    mean_Y_pred = float(np.mean(y_pred))

    logging.info(f"mean_Y={mean_Y}, mean_Y_pred={mean_Y_pred}")

    # Save metrics
    metrics = {
        # Model evaluation metrics
        "rootMeanSquaredError": eval_metrics.get("rmse"),
        "meanAbsoluteError": eval_metrics.get("mae"),
        "symmetricMeanAbsolutePercentageError": eval_metrics.get("smape"),
        # Custom metrics
        "mean_Y": mean_Y,
        "mean_Y_pred": mean_Y_pred,
        "rSquared": None,
        "rootMeanSquaredLogError": None,
        # Contextual info
        "problemType": "regression",
        "forecastSteps": hparams["forecast_steps"],
        "latentDim": hparams["latent_dim"],
        "numThresholds": hparams.get("num_thresholds"),
        "forecastStepsPerThreshold": hparams.get("forecast_steps_per_threshold"),
        "labelSourceIndex": hparams.get("label_source_index"),
        "labelSourceThreshold": hparams.get("label_source_threshold_resolved"),
    }

    if scaler is not None and price_index is not None:
        mean_arr = getattr(scaler, "mean_", None)
        scale_arr = getattr(scaler, "scale_", None)
        if mean_arr is None or scale_arr is None:
            logging.warning(
                "Scaler object missing mean_ or scale_; cannot inverse transform."
            )
        else:
            mean_price = float(mean_arr[price_index])
            scale_price = float(scale_arr[price_index])
            if np.isclose(scale_price, 0.0):
                logging.warning(
                    "Scaler scale for PRICE is zero; skipping inverse transform."
                )
            else:
                y_true_raw = y_true_np * scale_price + mean_price
                y_pred_raw = y_pred * scale_price + mean_price
                mean_Y_raw = float(np.mean(y_true_raw))
                mean_Y_pred_raw = float(np.mean(y_pred_raw))
                diff_raw = y_pred_raw - y_true_raw
                rmse_raw = float(np.sqrt(np.mean(np.square(diff_raw))))
                mae_raw = float(np.mean(np.abs(diff_raw)))
                smape_raw = float(
                    np.mean(
                        2.0
                        * np.abs(diff_raw)
                        / np.maximum(
                            np.abs(y_true_raw) + np.abs(y_pred_raw), np.full_like(y_true_raw, 1e-6)
                        )
                    )
                )
                metrics.update(
                    {
                        "mean_Y_raw": mean_Y_raw,
                        "mean_Y_pred_raw": mean_Y_pred_raw,
                        "rootMeanSquaredErrorRaw": rmse_raw,
                        "mae_raw": mae_raw,
                        "smape_raw": smape_raw,
                    }
                )
                local_plot, gcs_plot = _save_prediction_plot(
                    y_true_raw, y_pred_raw, Path(args.model_dir)
                )
                # Persist the plot location so downstream steps (or model registry)
                # can surface the visualization alongside scalar metrics.
                # if gcs_plot:
                #     metrics["predictionsPlotUri"] = gcs_plot
                # elif local_plot:
                #     metrics["predictionsPlotPath"] = local_plot
                logging.info(
                    "Inverse-transformed metrics -> mean_Y_raw=%s, mean_Y_pred_raw=%s, rmse_raw=%s, mae_raw=%s, smape_raw=%s",
                    mean_Y_raw,
                    mean_Y_pred_raw,
                    rmse_raw,
                    mae_raw,
                    smape_raw,
                )

    # Save metrics to JSON file
    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp)
    logging.info(f"Metrics saved to {args.metrics}")

    # Save training dataset info for model monitoring
    path = Path(args.model_dir) / TRAINING_DATASET_INFO
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "tfrecord",
        "targetColumn": hparams.get(
            "label_key", "y"
        ),  # y is the label tensor in TFRecords
    }
    with open(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)
    logging.info(f"Training dataset info saved to {path}")



if __name__ == "__main__":
    main()
