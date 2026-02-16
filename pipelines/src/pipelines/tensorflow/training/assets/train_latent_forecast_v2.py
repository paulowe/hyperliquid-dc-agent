import argparse
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
from tensorflow.data import Dataset
from typing import List, Tuple, Optional, Dict

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
client = google.cloud.logging.Client()
client.setup_logging()

text = "Latent Forecast Training Script"
logging.info(text)

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
    # --- Latent TFRecord settings (keys & compression) ---
    latent_key="z",
    label_key="y",
    z_mean_key="z_mean",          # optional present in files; not used by model
    z_log_var_key="z_log_var",    # optional present in files; not used by model
    compression="GZIP",           # "GZIP" or "" (uncompressed)

    # --- Plotting controls (OFF by default for production speed) ---
    enable_plots=True,
    plot_backend="matplotlib",    # "matplotlib" or "plotly"
    plot_examples=12,             # small sample to keep overhead tiny
)

# -----------------------------------------------------------------------------
# Helper: distribution strategy
# -----------------------------------------------------------------------------
def get_distribution_strategy(distribute_strategy: str) -> tf.distribute.Strategy:
    """Set distribute strategy based on input string."""
    logging.info(f"Distribution strategy: {distribute_strategy}")

    if distribute_strategy == "single":
        if len(tf.config.list_physical_devices("GPU")):
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif distribute_strategy == "mirror":
        strategy = tf.distribute.MirroredStrategy()
    elif distribute_strategy == "multi":
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise RuntimeError(f"Distribute strategy: {distribute_strategy} not supported")
    return strategy

def _is_chief(strategy: tf.distribute.Strategy) -> bool:
    """Determine whether current worker is the chief (master)."""
    cr = strategy.cluster_resolver
    return (cr is None) or (cr.task_type == "chief" and cr.task_id == 0)

def inspect_dataset_v2(ds: tf.data.Dataset, name="dataset", max_batches=1):
    try:
        for i, (x, y) in enumerate(ds.take(max_batches)):
            tf.debugging.assert_all_finite(x, f"{name}: found NaN/Inf in features (batch {i})")
            tf.debugging.assert_all_finite(y, f"{name}: found NaN/Inf in labels (batch {i})")
        logging.info(f"{name}: basic NaN/Inf checks passed on {max_batches} batch(es).")
    except Exception as e:
        logging.exception(f"Inspection error on {name}: {e}")

# -----------------------------------------------------------------------------
# (Deprecated) CSV helper – kept for compatibility but unused now
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
# Shard filtering (validate keys exist)
# -----------------------------------------------------------------------------
def _filter_files_with_keys(filepaths: list[str], required_keys: set[str], compression: str) -> list[str]:
    good = []
    for fp in filepaths:
        try:
            for rec in tf.data.TFRecordDataset([fp], compression_type=compression).take(1):
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

# -----------------------------------------------------------------------------
# Build tf.data pipeline for latent TFRecords
# -----------------------------------------------------------------------------
def _make_latent_dataset_from_dir(data_dir: str, hparams: dict, shuffle: bool) -> tf.data.Dataset:
    """
    Load latent TFRecord dataset (z, y) from a run-scoped split directory.
    """
    # --- Resolve compression
    compression = (hparams.get("compression") or "GZIP").upper()
    compression_type = "GZIP" if compression == "GZIP" else ""

    # --- Shapes & keys
    latent_dim     = int(hparams["latent_dim"])
    forecast_steps = int(hparams["forecast_steps"])

    latent_key  = str(hparams.get("latent_key", "z"))
    mean_key    = str(hparams.get("mean_key", "z_mean"))
    logvar_key  = str(hparams.get("logvar_key", "z_log_var"))
    label_key   = str(hparams.get("label_key", "y"))

    # --- Manifest overrides
    manifest = _read_manifest_if_any(data_dir)
    if manifest:
        feats = manifest.get("features", {})
        for k in (latent_key, mean_key, logvar_key, label_key):
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
            logging.warning(f"Override compression from hparams -> manifest: {compression}->{comp_m}")
            compression = comp_m
            compression_type = "GZIP" if compression == "GZIP" else ""

    # --- Collect shards
    pattern = "*.tfrecord.gz" if compression == "GZIP" else "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(data_dir, pattern))
    logging.info(f"[reader] {data_dir}: found {len(files)} shards matching {pattern}")
    if not files:
        raise ValueError(f"No TFRecord files found in {data_dir} (pattern={pattern})")

    # --- Filter shards by keys
    required_keys = {latent_key, mean_key, logvar_key, label_key}
    files = _filter_files_with_keys(files, required_keys, compression_type)
    logging.info(f"[reader] {data_dir}: {len(files)} shards after key filter {sorted(required_keys)}")
    if not files:
        raise ValueError(f"No valid TFRecord files found in {data_dir} with required keys {sorted(required_keys)}")

    # --- Parse spec
    feature_spec = {
        latent_key:  tf.io.FixedLenFeature([latent_dim], tf.float32),
        mean_key:    tf.io.FixedLenFeature([latent_dim], tf.float32),
        logvar_key:  tf.io.FixedLenFeature([latent_dim], tf.float32),
        label_key:   tf.io.FixedLenFeature([forecast_steps], tf.float32),
    }

    def _parse(serialized):
        ex = tf.io.parse_single_example(serialized, feature_spec)
        z = tf.ensure_shape(ex[latent_key], [latent_dim])
        y = tf.ensure_shape(ex[label_key], [forecast_steps])
        return tf.cast(z, tf.float32), tf.cast(y, tf.float32)

    # --- Build dataset
    ds = tf.data.TFRecordDataset(
        files,
        compression_type=compression_type,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        buf = int(hparams.get("shuffle_buffer", 10000))
        seed = int(hparams.get("shuffle_seed", 1234))
        ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=True)

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
    """Build a forecasting model that takes latent vectors as input."""
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
# Plot export (optional; off by default)
# -----------------------------------------------------------------------------
def _export_plots(X, Y_true, Y_pred, hparams, out_dir: Path):
    """Export per-example forecast vs actual plots using matplotlib or plotly.
       PNG (or SVG fallback) only; no heavy HTML. Keep sample size tiny."""
    backend = str(hparams.get("plot_backend", "matplotlib")).lower()
    max_examples = int(hparams.get("plot_examples", 12))
    label_name = hparams.get("label_name", "label")
    steps = int(hparams["forecast_steps"])
    horizon = np.arange(1, steps + 1)

    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "plotly":
        try:
            import plotly.graph_objects as go  # requires plotly
        except Exception as e:
            logging.warning(f"Plotly not available, falling back to matplotlib: {e}")
            backend = "matplotlib"

    for i in range(min(len(X), max_examples)):
        if backend == "plotly":
            fig = go.Figure()
            fig.add_scatter(x=horizon, y=Y_true[i], mode="lines+markers", name="Actual")
            fig.add_scatter(x=horizon, y=Y_pred[i], mode="lines+markers", name="Forecast")
            fig.update_layout(
                title=f"Example {i} — {steps}-step ahead",
                xaxis_title="Forecast step",
                yaxis_title=label_name,
                margin=dict(l=40, r=20, t=50, b=40),
                width=640, height=420,
            )
            out_path = out_dir / f"forecast_vs_actual_{i:02d}.png"
            try:
                # Static export via kaleido; requires kaleido to be installed
                fig.write_image(str(out_path))
            except Exception as e:
                logging.warning(f"Kaleido export failed ({e}); trying SVG")
                fig.write_image(str(out_dir / f"forecast_vs_actual_{i:02d}.svg"))
            # help GC
            try:
                fig.__dict__.clear()
            except Exception:
                pass
        else:
            # Matplotlib (Agg)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(horizon, Y_true[i], label="Actual")
            plt.plot(horizon, Y_pred[i], label="Forecast")
            plt.title(f"Example {i} — {steps}-step ahead")
            plt.xlabel("Forecast step")
            plt.ylabel(label_name)
            plt.legend()
            plt.tight_layout()
            out_path = out_dir / f"forecast_vs_actual_{i:02d}.png"
            plt.savefig(out_path)
            plt.close()
        logging.info(f"Saved plot: {out_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=os.getenv("AIP_MODEL_DIR", "model"))
    # Kept for backward compatibility; not needed when consuming precomputed latents.
    parser.add_argument("--parent_model_uri", type=str, required=False)
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    args = parser.parse_args()

    # --- Resolve model_dir for Vertex (gs:// -> /gcs/...)
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])

    # --- Merge hyperparameters
    hparams = {**DEFAULT_HPARAMS, **args.hparams}
    logging.info(f"Using model hyper-parameters: {hparams}")

    # --- Enable JIT + mixed precision (GPU only)
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

    # --- Strategy
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # --- Latent datasets
    logging.info(f"[reader] args.train_data={args.train_data}")
    files = tf.io.gfile.glob(os.path.join(args.train_data, "*.tfrecord.gz"))
    logging.info(f"[reader] found {len(files)} gz shards")
    train_ds = _make_latent_dataset_from_dir(args.train_data, hparams, shuffle=True)
    valid_ds = _make_latent_dataset_from_dir(args.valid_data, hparams, shuffle=False)
    test_ds  = _make_latent_dataset_from_dir(args.test_data,  hparams, shuffle=False)

    inspect_dataset_v2(train_ds, "train_ds", 1)
    inspect_dataset_v2(valid_ds, "val_ds", 1)
    inspect_dataset_v2(test_ds, "test_ds", 1)

    # --- Build model
    with strategy.scope():
        model = build_latent_forecast_model(
            latent_dim=hparams["latent_dim"],
            forecast_steps=hparams["forecast_steps"],
            hidden_units=hparams["hidden_units"],
            num_labels=1,
        )

    model.summary()

    # --- Callbacks
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

    # --- Train
    logging.info("Starting training...")
    model.fit(
        train_ds,
        epochs=hparams["epochs"],
        validation_data=valid_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Chief-only persistence
    if not _is_chief(strategy):
        logging.info("Non-chief worker: exiting without saving model/metrics.")
        sys.exit(0)

    # --- Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = f"{args.model_dir}/latent_forecast_{hparams.get('threshold', 'default')}.keras"
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # --- Evaluate
    eval_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    logging.info(f"Evaluation metrics: {eval_metrics}")

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

    # --- Training dataset info for model monitoring
    path = Path(args.model_dir) / TRAINING_DATASET_INFO
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "tfrecord",
        "targetColumn": hparams.get("label_key", "y"),
    }
    with open(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)
    logging.info(f"Training dataset info saved to {path}")

    # --- Optional: lightweight plots (disabled by default)
    try:
        if bool(hparams.get("enable_plots", False)):
            xs, ys = [], []
            for x_i, y_i in test_ds.unbatch().take(int(hparams.get("plot_examples", 12))):
                xs.append(x_i.numpy())
                ys.append(y_i.numpy())
            if xs:
                X = np.stack(xs, axis=0)  # [N, latent_dim]
                Y_true = np.stack(ys, axis=0)  # [N, forecast_steps]
                Y_pred = model.predict(X, verbose=0).reshape(-1, hparams["forecast_steps"])

                plots_dir = Path(args.model_dir) / "plots"
                _export_plots(X, Y_true, Y_pred, hparams, plots_dir)

                # small CSV for quick checks
                rows = [
                    (i, t + 1, float(Y_true[i, t]), float(Y_pred[i, t]))
                    for i in range(len(X)) for t in range(hparams["forecast_steps"])
                ]
                pd.DataFrame(rows, columns=["example_id", "step", "y_true", "y_pred"])\
                  .to_csv(plots_dir / "forecast_vs_actual_sample.csv", index=False)
            else:
                logging.info("No test examples available to plot.")
    except Exception as e:
        logging.exception(f"Plot export skipped due to error: {e}")

if __name__ == "__main__":
    main()
