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
from google.cloud import bigquery

# from tensorflow.data import Dataset  # unused in current tf.data API usage
from typing import List, Tuple, Optional, Dict

# Set up cloud profiler
from google.cloud.aiplatform.training_utils import cloud_profiler

# Initialize the profiler.
# https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler#enable
cloud_profiler.init()

# Set up logging
client = google.cloud.logging.Client()
client.setup_logging()

text = "Latent Forecast Training Script"
logging.info(text)

# -----------------------------------------------------------------------------
# Constants / Defaults
# -----------------------------------------------------------------------------
TRAINING_DATASET_INFO = "training_dataset.json"
SCALER_FILENAME = "scaler.joblib"
SCALER_METADATA_FILENAME = "scaler_metadata.json"

DEFAULT_HPARAMS = dict(
    batch_size=256,
    epochs=5,
    loss_fn="MeanSquaredError",
    optimizer="Adam",
    learning_rate=0.001,
    latent_dim=3,
    seq_length=50,
    n_features=6,
    forecast_steps=1,  # Number of steps to forecast into the future
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
)


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
    """Convert a gs:// URI to the local /gcs mount if necessary."""
    if maybe_gcs_path.startswith("gs://"):
        return Path("/gcs/" + maybe_gcs_path[5:])
    return Path(maybe_gcs_path)


def _load_scaler(scaler_uri: str):
    """Load a persisted StandardScaler artifact, returning (scaler, feature_order)."""
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


def _save_prediction_plot(
    y_true_raw: np.ndarray, y_pred_raw: np.ndarray, model_dir: Path
):
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

    # --------------------------
    # Step 2. Override with manifest if present
    # --------------------------
    manifest = _read_manifest_if_any(data_dir)
    if manifest:
        feats = manifest.get("features", {})

        # Sanity check: all required keys must be present
        for k in (latent_key, mean_key, logvar_key, label_key):
            if k not in feats:
                raise ValueError(f"Manifest missing required key '{k}' in {data_dir}")

        # Override latent_dim and forecast_steps from manifest
        ld_m = int(feats[latent_key]["shape"][0])
        fs_m = int(feats[label_key]["shape"][0])
        if (ld_m, fs_m) != (latent_dim, forecast_steps):
            logging.warning(
                "Override shapes from hparams -> manifest: latent_dim %s->%s, "
                "forecast_steps %s->%s",
                latent_dim,
                ld_m,
                forecast_steps,
                fs_m,
            )
            latent_dim, forecast_steps = ld_m, fs_m

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
    required_keys = {latent_key, mean_key, logvar_key, label_key}
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
    feature_spec = {
        latent_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
        mean_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
        logvar_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
        label_key: tf.io.FixedLenFeature([forecast_steps], tf.float32),
    }

    def _parse(serialized):
        """Parse a single Example -> (z, y)"""
        ex = tf.io.parse_single_example(serialized, feature_spec)
        # Ensure shapes are exactly as expected
        z = tf.ensure_shape(ex[latent_key], [latent_dim])
        y = tf.ensure_shape(ex[label_key], [forecast_steps])
        # Cast to float32 (guard against float64 sneaking in)
        return tf.cast(z, tf.float32), tf.cast(y, tf.float32)

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

def _load_latent_dataset(saved_dir: str, hparams: Dict, shuffle: bool) -> tf.data.Dataset:
    """
    Load (z, y) from a Dataset.save(...) snapshot written by the VAE script.

    The saved elements are dicts like {"z": [B, latent_dim], "y": [B, forecast_steps]}
    because the VAE pipeline saved *batched* elements. We unbatch to per-example,
    then batch to our desired batch_size.
    """
    compression = (hparams.get("compression") or "GZIP")
    ds = tf.data.Dataset.load(saved_dir, compression=compression)

    # Map dict -> (z, y)
    latent_key = hparams.get("latent_key", "z")
    label_key  = hparams.get("label_key", "y")
    ds = ds.map(lambda d: (d[latent_key], d[label_key]), num_parallel_calls=tf.data.AUTOTUNE)

    # The saved elements were batched at write time; make them per-example again
    ds = ds.unbatch()

    if shuffle:
        buf = int(hparams.get("shuffle_buffer", 10000))
        seed = int(hparams.get("shuffle_seed", 1234))
        ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=True)

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

def _write_predictions_to_bq(
    predictions_df: pd.DataFrame,
    bq_dataset: str,
    bq_table: str,
    bq_project: str,
    bq_location: str,
    bq_write_disposition: str,
):
    """
    Write predictions to BigQuery as rows (sample_index, horizon, y_true_raw, y_pred_raw, ...).
    - Auto-creates the dataset if needed.
    - Uses autodetect schema.
    """
    if not bq_project or not bq_dataset or not bq_table:
        raise ValueError("BQ project/dataset/table must be provided to load predictions.")

    # Build fully-qualified ids
    dataset_fqid = f"{bq_project}.{bq_dataset}"
    table_fqid = f"{bq_project}.{bq_dataset}.{bq_table}"

    client = bigquery.Client(project=bq_project, location=bq_location)

    # Ensure dataset exists
    try:
        client.get_dataset(dataset_fqid)
    except Exception:
        ds = bigquery.Dataset(dataset_fqid)
        client.create_dataset(ds, exists_ok=True)
        logging.info(f"Ensured BigQuery dataset exists: {dataset_fqid}")

    # Configure load
    job_config = bigquery.LoadJobConfig(
        write_disposition=bq_write_disposition,
        autodetect=True,
    )

    # Load
    load_job = client.load_table_from_dataframe(
        predictions_df, table_fqid, job_config=job_config
    )
    result = load_job.result()  # wait

    # Log outcome
    table = client.get_table(table_fqid)
    logging.info(
        "Loaded %d rows to %s (total rows now: %d, job_id=%s)",
        len(predictions_df), table_fqid, table.num_rows, load_job.job_id
    )


def _build_predictions_dataframe(
    y_true_values: np.ndarray,
    y_pred_values: np.ndarray,
    value_kind: str = "raw",
) -> pd.DataFrame:
    """Normalize prediction arrays into a flat DataFrame for downstream sinks."""
    y_true_arr = np.asarray(y_true_values)
    y_pred_arr = np.asarray(y_pred_values)
    true_col = "y_true_raw" if value_kind == "raw" else "y_true"
    pred_col = "y_pred_raw" if value_kind == "raw" else "y_pred"

    if y_true_arr.ndim == 2 and y_true_arr.shape[1] == 1:
        # Single-step: flatten to 1-D columns
        y_true_col = y_true_arr.squeeze(-1).astype(np.float64)
        y_pred_col = y_pred_arr.squeeze(-1).astype(np.float64)
        predictions_df = pd.DataFrame(
            {
                "sample_index": np.arange(y_true_col.shape[0], dtype=np.int64),
                true_col: y_true_col,
                pred_col: y_pred_col,
            }
        )
    elif y_true_arr.ndim == 1:
        # Already 1-D
        predictions_df = pd.DataFrame(
            {
                "sample_index": np.arange(y_true_arr.shape[0], dtype=np.int64),
                true_col: y_true_arr.astype(np.float64),
                pred_col: y_pred_arr.astype(np.float64),
            }
        )
    else:
        # Multi-step: long format with horizon
        N, H = y_true_arr.shape
        frames = []
        for h in range(H):
            frames.append(
                pd.DataFrame(
                    {
                        "sample_index": np.arange(N, dtype=np.int64),
                        "horizon": (h + 1),
                        true_col: y_true_arr[:, h].astype(np.float64),
                        pred_col: y_pred_arr[:, h].astype(np.float64),
                    }
                )
            )
        predictions_df = pd.concat(frames, ignore_index=True)

    return predictions_df


# def _save_predictions_dataset(predictions_df: pd.DataFrame, output_uri: str) -> bool:
#     """Persist the predictions dataframe as CSV at the requested URI (GCS/local)."""
#     if not output_uri:
#         return False

#     directory = os.path.dirname(output_uri)
#     if directory and directory not in (".", ""):
#         tf.io.gfile.makedirs(directory)

#     buffer = io.StringIO()
#     predictions_df.to_csv(buffer, index=False)
#     with tf.io.gfile.GFile(output_uri, "w") as fh:
#         fh.write(buffer.getvalue())

#     logging.info(
#         "Saved predictions DataFrame with %d rows to %s", len(predictions_df), output_uri
#     )
#     return True
def _save_predictions_dataset(predictions_df: pd.DataFrame, output_uri: str) -> str:
    """
    Writes predictions to a directory-like Output[Dataset].uri location.
    Ensures a file inside it, plus metadata + _SUCCESS.
    Returns final file URI.
    """
    if not output_uri:
        raise ValueError("output_uri is required")

    # Treat output_uri as a directory/prefix (which it is for KFP Output[Dataset])
    output_dir = output_uri.rstrip("/")
    file_uri = f"{output_dir}/predictions.csv"

    # Make directory (works for GCS via tf.io.gfile)
    tf.io.gfile.makedirs(output_dir)

    # Write CSV
    buf = io.StringIO()
    predictions_df.to_csv(buf, index=False)
    with tf.io.gfile.GFile(file_uri, "w") as fh:
        fh.write(buf.getvalue())

    # Optional: metadata + success marker (helps downstream components reason)
    meta_uri = f"{output_dir}/metadata.json"
    success_uri = f"{output_dir}/_SUCCESS"

    meta = {
        "rows": int(len(predictions_df)),
        "format": "csv",
        "filename": "predictions.csv",
    }

    with tf.io.gfile.GFile(meta_uri, "w") as mfh:
        mfh.write(json.dumps(meta))

    with tf.io.gfile.GFile(success_uri, "w") as sfh:
        sfh.write("")

    logging.info("Saved %d rows → %s", len(predictions_df), file_uri)
    return file_uri


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
    parser.add_argument("--bq_project", type=str, default="")
    parser.add_argument("--bq_dataset", type=str, default="")
    parser.add_argument("--bq_table", type=str, default="")
    parser.add_argument("--bq_location", type=str, default="")
    parser.add_argument("--load_to_bq", type=bool, default=False)
    # parser.add_argument("--load_to_bq", action="store_true", help="If set, write predictions to BigQuery.")
    parser.add_argument("--bq_write_disposition", type=str, default="WRITE_TRUNCATE")
    parser.add_argument("--predictions_output_uri", type=str, default="")
    args = parser.parse_args()

    # ============================================================
    # Convert GCS paths if present
    # Resolve model_dir for Vertex (gs:// -> /gcs/...)
    # ============================================================
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])

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
        logging.info(
            "No scaler artifact supplied; metrics remain in standardized scale."
        )

    # ============================================================
    # Merge hyperparameters
    # ============================================================
    hparams = {**DEFAULT_HPARAMS, **args.hparams}

    def _normalise_hidden_units(units_spec):
        """Convert list items to (units, activation) tuples."""
        normalised = []
        for item in units_spec or []:
            if isinstance(item, dict):
                units = int(item.get("units"))
                activation = item.get("activation")
                if activation is None:
                    raise ValueError(f"Missing activation in hidden_units entry: {item}")
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
    # logging.info(f"[reader] args.train_data={args.train_data}")
    # files = tf.io.gfile.glob(os.path.join(args.train_data, "*.tfrecord.gz"))
    # logging.info(f"[reader] found {len(files)} gz shards")
    # train_ds = _make_latent_dataset_from_dir(args.train_data, hparams, shuffle=True)
    # valid_ds = _make_latent_dataset_from_dir(args.valid_data, hparams, shuffle=False)
    # test_ds = _make_latent_dataset_from_dir(args.test_data, hparams, shuffle=False)

    # ============================================================
    # Load latent datasets saved via Dataset.save(...) (dicts: {"z","y"})
    # ============================================================
    logging.info(f"[reader] loading latents via Dataset.load: {args.train_data}")
    train_ds = _load_latent_dataset(args.train_data, hparams, shuffle=False)
    valid_ds = _load_latent_dataset(args.valid_data, hparams, shuffle=False)
    test_ds  = _load_latent_dataset(args.test_data,  hparams, shuffle=False)
    # Quick integrity check (first batch)
    # inspect_dataset_v2(train_ds, "train_ds", 1)
    # inspect_dataset_v2(valid_ds, "val_ds", 1)
    # inspect_dataset_v2(test_ds, "test_ds", 1)

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
    # https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements#tensorboard
    # Profile from batches 5 to 10
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'],
        histogram_freq=1,
        profile_batch='5, 10'
    )
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams["early_stopping_epochs"],
        restore_best_weights=True,
    )
    reduce_lr_on_plateau_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    #  TO DO: Model Checkpoint callback
    # https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements#resilience
    callbacks = [
        tensorboard_callback,
        early_stopping_callback,
        reduce_lr_on_plateau_callback,
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
    }

    predictions_df = None
    predictions_df_kind = None

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
                            np.abs(y_true_raw) + np.abs(y_pred_raw),
                            np.full_like(y_true_raw, 1e-6),
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
                predictions_df = _build_predictions_dataframe(
                    y_true_raw, y_pred_raw, value_kind="raw"
                )
                predictions_df_kind = "raw"

                if args.load_to_bq:
                    _write_predictions_to_bq(
                        predictions_df=predictions_df,
                        bq_dataset=args.bq_dataset,
                        bq_table=args.bq_table,
                        bq_project=args.bq_project,
                        bq_location=args.bq_location,
                        bq_write_disposition=args.bq_write_disposition,
                    )
                # save prediction plot
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
                    "Inverse-transformed metrics -> mean_Y_raw=%s, "
                    "mean_Y_pred_raw=%s, rmse_raw=%s, mae_raw=%s, smape_raw=%s",
                    mean_Y_raw,
                    mean_Y_pred_raw,
                    rmse_raw,
                    mae_raw,
                    smape_raw,
                )
    if predictions_df is None:
        predictions_df = _build_predictions_dataframe(
            y_true_np, y_pred, value_kind="scaled"
        )
        predictions_df_kind = "scaled"

    if predictions_df is not None and args.predictions_output_uri:
        _save_predictions_dataset(predictions_df, args.predictions_output_uri)
        logging.info(
            "Persisted %s predictions to %s",
            predictions_df_kind,
            args.predictions_output_uri,
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
