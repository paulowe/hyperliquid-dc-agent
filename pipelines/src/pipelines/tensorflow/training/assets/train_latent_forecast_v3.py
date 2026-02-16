import argparse
import os
import json
import logging
import google.cloud.logging
import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense
from typing import List, Tuple, Optional, Dict

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
client = google.cloud.logging.Client()
client.setup_logging()
logging.info("Latent Forecast Training Script — last-12 timeline plot")

# -----------------------------------------------------------------------------
# Matplotlib for headless environments
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants / Defaults
# -----------------------------------------------------------------------------
TRAINING_DATASET_INFO = "training_dataset.json"

DEFAULT_HPARAMS = dict(
    batch_size=256,
    epochs=5,
    loss_fn="MeanSquaredError",
    optimizer="Adam",
    learning_rate=0.001,
    latent_dim=3,
    seq_length=50,
    n_features=6,
    forecast_steps=10,  # steps to forecast
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
    # --- Latent TFRecord settings (keys & compression) ---
    latent_key="z",
    label_key="y",
    z_mean_key="z_mean",          # optional; ignored by model
    z_log_var_key="z_log_var",    # optional; ignored by model
    compression="GZIP",           # "GZIP" or "" (uncompressed)

    # --- Plot controls ---
    enable_plots=True,            # create the last-12 timeline plot
    plot_examples=12,             # how many last records to include
)

# -----------------------------------------------------------------------------
# Strategy helpers
# -----------------------------------------------------------------------------
def get_distribution_strategy(distribute_strategy: str) -> tf.distribute.Strategy:
    logging.info(f"Distribution strategy: {distribute_strategy}")
    if distribute_strategy == "single":
        if len(tf.config.list_physical_devices("GPU")):
            return tf.distribute.OneDeviceStrategy("/gpu:0")
        return tf.distribute.OneDeviceStrategy("/cpu:0")
    if distribute_strategy == "mirror":
        return tf.distribute.MirroredStrategy()
    if distribute_strategy == "multi":
        return tf.distribute.MultiWorkerMirroredStrategy()
    raise RuntimeError(f"Distribute strategy not supported: {distribute_strategy}")

def _is_chief(strategy: tf.distribute.Strategy) -> bool:
    cr = strategy.cluster_resolver
    return (cr is None) or (cr.task_type == "chief" and cr.task_id == 0)

def inspect_dataset_v2(ds: tf.data.Dataset, name="dataset", max_batches=1):
    try:
        for i, (x, y) in enumerate(ds.take(max_batches)):
            tf.debugging.assert_all_finite(x, f"{name}: found NaN/Inf in features (batch {i})")
            tf.debugging.assert_all_finite(y, f"{name}: found NaN/Inf in labels (batch {i})")
        logging.info(f"{name}: NaN/Inf checks passed on {max_batches} batch(es).")
    except Exception as e:
        logging.exception(f"Inspection error on {name}: {e}")

# -----------------------------------------------------------------------------
# (Legacy) CSV reader (unused here)
# -----------------------------------------------------------------------------
def read_csv_any(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if path.startswith("gs://"):
        with tf.io.gfile.GFile(path, "rb") as f:
            return pd.read_csv(f, usecols=usecols)
    return pd.read_csv(path, usecols=usecols)

# -----------------------------------------------------------------------------
# Manifest reader
# -----------------------------------------------------------------------------
def _read_manifest_if_any(data_dir: str):
    mpath = os.path.join(data_dir, "_manifest.json")
    if tf.io.gfile.exists(mpath):
        with tf.io.gfile.GFile(mpath, "r") as f:
            return json.load(f)
    return None

# -----------------------------------------------------------------------------
# TFRecord pipeline for latent (z, y)
# -----------------------------------------------------------------------------
def _make_latent_dataset_from_dir(data_dir: str, hparams: dict, shuffle: bool) -> tf.data.Dataset:
    """
    Load latent TFRecord dataset (z, y). Only z & y are required.
    z_mean/z_log_var are optional and ignored by the model.
    """
    compression = (hparams.get("compression") or "GZIP").upper()
    compression_type = "GZIP" if compression == "GZIP" else ""

    latent_dim     = int(hparams["latent_dim"])
    forecast_steps = int(hparams["forecast_steps"])

    latent_key  = str(hparams.get("latent_key", "z"))
    label_key   = str(hparams.get("label_key", "y"))
    z_mean_key  = str(hparams.get("z_mean_key", "z_mean"))
    z_logv_key  = str(hparams.get("z_log_var_key", "z_log_var"))

    # Manifest overrides if present
    manifest = _read_manifest_if_any(data_dir)
    if manifest:
        feats = manifest.get("features", {})
        for k in (latent_key, label_key):  # only require z & y
            if k not in feats:
                raise ValueError(f"Manifest missing required key '{k}' in {data_dir}")

        ld_m = int(feats[latent_key]["shape"][0])
        fs_m = int(feats[label_key]["shape"][0])
        if (ld_m, fs_m) != (latent_dim, forecast_steps):
            logging.warning(
                f"Override shapes from hparams -> manifest: "
                f"latent_dim {latent_dim}->{ld_m}, forecast_steps {forecast_steps}->{fs_m}"
            )
            latent_dim, forecast_steps = ld_m, fs_m

        comp_m = (manifest.get("compression") or compression).upper()
        if comp_m != compression:
            logging.warning(f"Override compression: {compression}->{comp_m}")
            compression = comp_m
            compression_type = "GZIP" if compression == "GZIP" else ""

    pattern = "*.tfrecord.gz" if compression == "GZIP" else "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(data_dir, pattern))
    logging.info(f"[reader] {data_dir}: found {len(files)} shards ({pattern})")
    if not files:
        raise ValueError(f"No TFRecord files found in {data_dir} (pattern={pattern})")

    # Spec: require z & y; optional z_mean/z_log_var with defaults
    feature_spec = {
        latent_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
        label_key:  tf.io.FixedLenFeature([forecast_steps], tf.float32),
        z_mean_key: tf.io.FixedLenFeature([latent_dim], tf.float32, default_value=[0.0]*latent_dim),
        z_logv_key: tf.io.FixedLenFeature([latent_dim], tf.float32, default_value=[0.0]*latent_dim),
    }

    def _parse(serialized):
        ex = tf.io.parse_single_example(serialized, feature_spec)
        z = tf.ensure_shape(ex[latent_key], [latent_dim])
        y = tf.ensure_shape(ex[label_key], [forecast_steps])
        return tf.cast(z, tf.float32), tf.cast(y, tf.float32)

    ds = tf.data.TFRecordDataset(
        files,
        compression_type=compression_type,
        num_parallel_reads=tf.data.AUTOTUNE,
    ).map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(int(hparams.get("shuffle_buffer", 10000)),
                        seed=int(hparams.get("shuffle_seed", 1234)),
                        reshuffle_each_iteration=True)

    ds = ds.batch(int(hparams["batch_size"]), drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_latent_forecast_model(
    latent_dim: int,
    forecast_steps: int,
    hidden_units: List[Tuple[int, str]],
    num_labels: int = 1,
) -> keras.Model:
    inputs = keras.Input(shape=(latent_dim,))
    x = inputs
    for units, activation in hidden_units:
        x = Dense(units, activation=activation)(x)
        x = layers.Dropout(0.2)(x)
    outputs = Dense(forecast_steps * num_labels, name="forecast", activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="latent_forecast_model")
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            "mae",
            "mape",
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]
    )
    return model

# -----------------------------------------------------------------------------
# Plot: single timeline overlay for the LAST K test records
# -----------------------------------------------------------------------------
def plot_last_k_timeline(model: keras.Model,
                         test_ds: tf.data.Dataset,
                         forecast_steps: int,
                         label_name: str,
                         out_png: Path,
                         k: int = 12):
    """
    Collect the LAST k examples from test_ds, predict with model,
    and plot a single timeline overlay (actual solid, predicted dashed).
    X-axis = forecast step index (1..T).
    """
    # Keep only the last k (z, y) pairs without materializing the whole test set
    buf_x = deque(maxlen=k)
    buf_y = deque(maxlen=k)
    for x_b, y_b in test_ds:
        x_np = x_b.numpy()
        y_np = y_b.numpy()
        for i in range(x_np.shape[0]):
            buf_x.append(x_np[i])
            buf_y.append(y_np[i])

    n = len(buf_x)
    if n == 0:
        logging.warning("No test records available to plot.")
        return

    X_last = np.stack(list(buf_x), axis=0).astype(np.float32)   # [n, latent_dim]
    Y_true = np.stack(list(buf_y), axis=0).astype(np.float32)   # [n, steps]
    steps = forecast_steps

    # Predict and reshape
    Y_pred = model.predict(X_last, verbose=0).reshape(n, steps).astype(np.float32)

    # Horizon (1..T). Replace with timestamps later if you add them to TFRecords.
    horizon = np.arange(1, steps + 1)

    plt.figure(figsize=(10, 6))

    # Light overlays for each sample
    for i in range(n):
        plt.plot(horizon, Y_true[i], linewidth=1.0, alpha=0.35)
        plt.plot(horizon, Y_pred[i], linestyle="--", linewidth=1.0, alpha=0.35)

    # Bold averages to make it readable at a glance
    mean_true = Y_true.mean(axis=0)
    mean_pred = Y_pred.mean(axis=0)
    plt.plot(horizon, mean_true, linewidth=2.5, label="Actual (avg)")
    plt.plot(horizon, mean_pred, linestyle="--", linewidth=2.5, label="Predicted (avg)")

    plt.title(f"Last {n} Records — Forecast Timeline (Steps 1..{steps})")
    plt.xlabel("Forecast step")
    plt.ylabel(label_name)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    logging.info(f"Saved combined last-{n} timeline plot: {out_png}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=os.getenv("AIP_MODEL_DIR", "model"))
    parser.add_argument("--parent_model_uri", type=str, required=False)  # unused; kept for compat
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    args = parser.parse_args()

    # Resolve model_dir for Vertex
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])
    else:
        args.model_dir = Path(args.model_dir)

    # Merge hparams
    hparams = {**DEFAULT_HPARAMS, **args.hparams}
    logging.info(f"Using model hyper-parameters: {hparams}")

    # XLA + mixed precision
    if hparams.get("use_xla_jit", True):
        try:
            tf.config.optimizer.set_jit(True)
            logging.info("XLA JIT enabled")
        except Exception as e:
            logging.warning(f"Could not enable XLA JIT: {e}")

    if hparams.get("use_mixed_precision", True) and tf.config.list_physical_devices("GPU"):
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            logging.info("Mixed precision enabled")
        except Exception as e:
            logging.warning(f"Could not enable mixed precision: {e}")

    # Strategy
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # Datasets
    logging.info(f"[reader] train_data={args.train_data}")
    train_ds = _make_latent_dataset_from_dir(args.train_data, hparams, shuffle=True)
    valid_ds = _make_latent_dataset_from_dir(args.valid_data, hparams, shuffle=False)
    test_ds  = _make_latent_dataset_from_dir(args.test_data,  hparams, shuffle=False)

    inspect_dataset_v2(train_ds, "train_ds", 1)
    inspect_dataset_v2(valid_ds, "val_ds", 1)
    inspect_dataset_v2(test_ds, "test_ds", 1)

    # Model
    with strategy.scope():
        model = build_latent_forecast_model(
            latent_dim=hparams["latent_dim"],
            forecast_steps=hparams["forecast_steps"],
            hidden_units=hparams["hidden_units"],
            num_labels=1,
        )

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=hparams["early_stopping_epochs"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
        ),
    ]

    # Train
    logging.info("Starting training…")
    history = model.fit(
        train_ds,
        epochs=hparams["epochs"],
        validation_data=valid_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # Chief-only outputs
    if not _is_chief(strategy):
        logging.info("Non-chief worker: exiting without saving artifacts.")
        sys.exit(0)

    # Save model
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"latent_forecast_{hparams.get('threshold', 'default')}.keras"
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # Evaluate on test set
    eval_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    logging.info(f"Evaluation metrics (test): {eval_metrics}")

    # Save metrics (Vertex-style)
    metrics = {
        "problemType": "regression",
        "rootMeanSquaredError": eval_metrics.get("rmse"),
        "meanAbsoluteError": eval_metrics.get("mae"),
        "meanAbsolutePercentageError": eval_metrics.get("mape"),
        "rSquared": None,
        "rootMeanSquaredLogError": None,
        "forecastSteps": hparams["forecast_steps"],
        "latentDim": hparams["latent_dim"],
    }
    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp)
    logging.info(f"Metrics saved to {args.metrics}")

    # Training dataset info for monitoring
    info_path = args.model_dir / TRAINING_DATASET_INFO
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "tfrecord",
        "targetColumn": hparams.get("label_key", "y"),
    }
    with open(info_path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)
    logging.info(f"Training dataset info saved to {info_path}")

    # ---------------------------
    # Single plot: last-12 timeline
    # ---------------------------
    try:
        if bool(hparams.get("enable_plots", True)):
            plots_dir = args.model_dir / "plots"
            out_png = plots_dir / "last12_timeline.png"
            plot_last_k_timeline(
                model=model,
                test_ds=test_ds,  # batched dataset
                forecast_steps=int(hparams["forecast_steps"]),
                label_name=str(hparams.get("label_name", "label")),
                out_png=out_png,
                k=int(hparams.get("plot_examples", 12)),
            )
    except Exception as e:
        logging.exception(f"Failed to produce last-12 timeline plot: {e}")

if __name__ == "__main__":
    main()
