import argparse
import os
import json
import logging
import google.cloud.logging
import sys
from pathlib import Path

import numpy as np

# import pandas as pd  # unused legacy helpers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization, StringLookup

# from tensorflow.keras import Input, Model, optimizers  # unused
# from tensorflow.keras.layers import Dense, Concatenate  # unused
from tensorflow.data import Dataset

# from matplotlib import pyplot as plt
from typing import Dict, List, Optional

# from typing import Tuple  # unused
from datetime import datetime

# Set up logging
client = google.cloud.logging.Client()
client.setup_logging()

text = "VAE Training Script"
logging.warning(text)

# used for monitoring during prediction time
TRAINING_DATASET_INFO = "training_dataset.json"

# Default hyperparameters
DEFAULT_HPARAMS = dict(
    batch_size=128,
    epochs=1,
    learning_rate=0.001,
    latent_dim=3,
    seq_length=50,
    # n_features=6,
    n_features=8,
    kl_init_weight=1e-4,
    patience=10,
    metrics=[
        "RootMeanSquaredError",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredLogarithmicError",
    ],
    hidden_units=[(10, "relu")],
    distribute_strategy="single",
    early_stopping_epochs=5,
    label_name="PRICE_std",
    use_mixed_precision=True,  # only takes effect if a GPU is present
    use_xla_jit=True,  # try XLA; keep if step time drops
    cache_to_disk="/tmp/dcvae_data.cache",  # set path to cache-to-file
    # falsy/None caches in RAM instead of disk
)

# ---------------------------------------------------------------------
# (Unchanged) helpers kept for compatibility or future use
# ---------------------------------------------------------------------


def get_distribution_strategy(distribute_strategy: str) -> tf.distribute.Strategy:
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


def normalization(name: str, dataset: Dataset) -> Normalization:
    logging.info(f"Normalizing numerical input '{name}'...")
    x = Normalization(axis=None, name=f"normalize_{name}")
    x.adapt(dataset.map(lambda y, _: y[name]))
    return x


def str_lookup(name: str, dataset: Dataset, output_mode: str) -> StringLookup:
    logging.info(f"Encoding categorical input '{name}' ({output_mode})...")
    x = StringLookup(output_mode=output_mode, name=f"str_lookup_{output_mode}_{name}")
    x.adapt(dataset.map(lambda y, _: y[name]))
    logging.info(f"Vocabulary: {x.get_vocabulary()}")
    return x


def _is_chief(strategy: tf.distribute.Strategy) -> bool:
    cr = strategy.cluster_resolver
    return (cr is None) or (cr.task_type == "chief" and cr.task_id == 0)


def _get_temp_dir(dirpath, task_id):
    base_dirpath = "workertemp_" + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def inspect_dataset(ds, name="dataset", max_batches=3):
    logging.info(f"Inspecting {name} for shape/NaN (first {max_batches} batches)...")
    try:
        for i, batch in enumerate(ds.take(max_batches)):
            x = batch if not isinstance(batch, (tuple, list)) else batch[0]
            x_np = x.numpy() if tf.is_tensor(x) else np.array(x)
            if np.isnan(x_np).any():
                logging.error(f"Found NaN in {name} at batch {i}")
            logging.info(f"{name} batch {i} shape: {x_np.shape}")
    except Exception as e:
        logging.exception(f"Error while inspecting {name}: {e}")


# ---------------------------------------------------------------------
# NEW: TFRecord input pipeline (replaces CSV/pandas path)
# ---------------------------------------------------------------------


def _tfrecord_feature_spec(
    float_feature_names: List[str],
) -> Dict[str, tf.io.FixedLenFeature]:
    # We parse only what we need for training; string timestamps are ignored.
    spec = {name: tf.io.FixedLenFeature([], tf.float32) for name in float_feature_names}
    # If you ever need timestamps, you can add:
    # spec["start_time"] = tf.io.FixedLenFeature([], tf.string)
    # spec["load_time_toronto"] = tf.io.FixedLenFeature([], tf.string)
    return spec


def _parse_example(serialized, feature_spec, feature_order: List[str]):
    parsed = tf.io.parse_single_example(serialized, feature_spec)
    # Assemble features in a fixed order -> vector [n_features]
    x = tf.stack([parsed[name] for name in feature_order], axis=-1)
    return x  # shape: [n_features], dtype float32


def _make_sequence_dataset_from_tfrecord_dir(
    data_dir: str,
    seq_length: int,
    batch_size: int,
    feature_names: List[str],
    shuffle_sequences: bool,
    compression_type: str = "GZIP",
) -> tf.data.Dataset:
    """
    Build a dataset of shape [batch, seq_length, n_features] from TFRecord shards.
    Assumes TFRecords are the row-wise standardized features emitted by the splitter.
    """
    pattern = os.path.join(
        data_dir, "*.tfrecord.gz" if compression_type == "GZIP" else "*.tfrecord"
    )
    logging.info(f"Reading TFRecords from pattern: {pattern}")

    # Important for time series: keep file order deterministic (no shuffling at the
    # file level).
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#list_files
    files = tf.data.Dataset.list_files(pattern, shuffle=False)

    # Interleave record datasets; keep ordering across shards deterministic.
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave
    ds = files.interleave(
        lambda fp: tf.data.TFRecordDataset(fp, compression_type=compression_type),
        cycle_length=1,  # keep in-order; bump if cross-shard mixing is OK
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    feature_spec = _tfrecord_feature_spec(feature_names)
    ds = ds.map(
        lambda s: _parse_example(s, feature_spec, feature_names),
        num_parallel_calls=tf.data.AUTOTUNE,
    )  # each element: [n_features]

    # Sliding windows of consecutive rows -> [seq_length, n_features]
    # window() keeps order; drop_remainder enforces fixed length
    windowed = ds.window(size=seq_length, shift=1, drop_remainder=True)
    windowed = windowed.flat_map(lambda w: w.batch(seq_length, drop_remainder=True))

    # Optionally shuffle windows for training
    if shuffle_sequences:
        windowed = windowed.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    # Batch and prefetch
    windowed = windowed.batch(batch_size, drop_remainder=True)
    cache_path = DEFAULT_HPARAMS.get("cache_to_disk")
    if cache_path:
        tf.io.gfile.makedirs(os.path.dirname(cache_path))
        windowed = windowed.cache(cache_path)
    else:
        windowed = windowed.cache()
    windowed = windowed.prefetch(tf.data.AUTOTUNE)
    return windowed  # each element: [batch, seq_length, n_features]


# ---------------------------------------------------------------------
# NEW: Latent TFRecord writer (adds y; training logic unchanged)
# ---------------------------------------------------------------------

# def _float_feature_list(arr: np.ndarray) -> tf.train.Feature:
#     return tf.train.Feature(
#         float_list=tf.train.FloatList(value=arr.astype(np.float32).ravel())
#     )


def _float_feature_list(arr):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    return tf.train.Feature(float_list=tf.train.FloatList(value=arr.tolist()))


# def _latent_example(
#     z: np.ndarray,
#     z_mean: np.ndarray,
#     z_log_var: np.ndarray,
#     y: Optional[np.ndarray] = None,
#     latent_key="z",
#     mean_key="z_mean",
#     logvar_key="z_log_var",
#     label_key="y",
# ) -> tf.train.Example:
#     feats = {
#         latent_key: _float_feature_list(z),
#         mean_key:   _float_feature_list(z_mean),
#         logvar_key: _float_feature_list(z_log_var),
#     }
#     if y is not None:
#         feats[label_key] = _float_feature_list(y)
#     return tf.train.Example(features=tf.train.Features(feature=feats))


def _latent_example(
    z,
    z_mean,
    z_log_var,
    y,
    latent_key="z",
    mean_key="z_mean",
    logvar_key="z_log_var",
    label_key="y",
):
    feats = {
        latent_key: _float_feature_list(z),
        mean_key: _float_feature_list(z_mean),
        logvar_key: _float_feature_list(z_log_var),
    }
    if y is not None:
        feats[label_key] = _float_feature_list(y)
    return tf.train.Example(features=tf.train.Features(feature=feats))


def _write_manifest(
    dirpath,
    *,
    latent_dim,
    forecast_steps,
    compression,
    latent_key="z",
    mean_key="z_mean",
    logvar_key="z_log_var",
    label_key="y",
    threshold=None,
):
    thr_float = None
    thr_str = None
    if threshold is not None:
        try:
            thr_float = float(threshold)
            thr_str = f"{thr_float:.6f}"
        except (TypeError, ValueError):
            thr_float = None
            thr_str = str(threshold)

    manifest = {
        "version": 1,
        "compression": compression,  # "GZIP" or ""
        "features": {
            latent_key: {"dtype": "float32", "shape": [latent_dim]},
            mean_key: {"dtype": "float32", "shape": [latent_dim]},
            logvar_key: {"dtype": "float32", "shape": [latent_dim]},
            label_key: {"dtype": "float32", "shape": [forecast_steps]},
        },
    }
    if thr_float is not None or thr_str is not None:
        manifest["threshold"] = thr_float if thr_float is not None else threshold
        manifest["threshold_str"] = thr_str

    with tf.io.gfile.GFile(os.path.join(dirpath, "_manifest.json"), "w") as f:
        json.dump(manifest, f)


def _write_latent_tfrecords(
    windows_ds: tf.data.Dataset,
    encoder: tf.keras.Model,
    out_dir: str,
    num_shards: int = 16,
    compression: str = "GZIP",
    latent_key: str = "z",
    mean_key: str = "z_mean",
    logvar_key: str = "z_log_var",
    label_key: str = "y",
    label_builder: Optional[callable] = None,  # function(batch) -> Tensor [B, H]
):
    """
    Encode window batches with the VAE encoder and write sharded TFRecords
    that contain z, z_mean, z_log_var and (optionally) y per example.
    This section does not alter any training/evaluation logic above.
    """
    tf.io.gfile.makedirs(out_dir)
    opts = tf.io.TFRecordOptions(compression)
    writers = [
        tf.io.TFRecordWriter(
            os.path.join(
                out_dir,
                f"part-{i:05d}.tfrecord" + (".gz" if compression == "GZIP" else ""),
            ),
            options=opts,
        )
        for i in range(num_shards)
    ]
    shard = 0
    ex_count = 0

    for batch in windows_ds:  # batch: [B, seq_len, n_features]
        z_mean, z_log_var, z = encoder(batch, training=False)
        z_np = z.numpy()
        zm_np = z_mean.numpy()
        zl_np = z_log_var.numpy()

        if label_builder is not None:
            y = label_builder(batch)  # Tensor [B, H]
            y_np = y.numpy()
        else:
            y_np = None

        B = z_np.shape[0]
        for i in range(B):
            ex = _latent_example(
                z=z_np[i],
                z_mean=zm_np[i],
                z_log_var=zl_np[i],
                y=None if y_np is None else y_np[i],
                latent_key=latent_key,
                mean_key=mean_key,
                logvar_key=logvar_key,
                label_key=label_key,
            )
            writers[shard].write(ex.SerializeToString())
            shard = (shard + 1) % num_shards
            ex_count += 1

    for w in writers:
        w.close()
    logging.info(
        "Wrote %s latent examples to %s (shards=%s, compression=%s)",
        ex_count,
        out_dir,
        num_shards,
        compression,
    )


def _write_single_field(
    windows_ds: tf.data.Dataset,
    encoder: tf.keras.Model,
    out_dir: str,
    field: str,  # "z_mean" or "z_log_var" or "z"
    num_shards: int = 8,
    compression: str = "GZIP",
):
    """
    Convenience: write a dataset containing ONLY one field vector per example.
    Useful if a downstream job wants means/logvars separately.
    """
    tf.io.gfile.makedirs(out_dir)
    opts = tf.io.TFRecordOptions(compression)
    writers = [
        tf.io.TFRecordWriter(
            os.path.join(
                out_dir,
                f"part-{i:05d}.tfrecord" + (".gz" if compression == "GZIP" else ""),
            ),
            options=opts,
        )
        for i in range(num_shards)
    ]
    shard = 0
    ex_count = 0

    for batch in windows_ds:
        z_mean, z_log_var, z = encoder(batch, training=False)
        pick = {"z": z, "z_mean": z_mean, "z_log_var": z_log_var}[field].numpy()
        for vec in pick:
            ex = tf.train.Example(
                features=tf.train.Features(feature={field: _float_feature_list(vec)})
            )
            writers[shard].write(ex.SerializeToString())
            shard = (shard + 1) % num_shards
            ex_count += 1

    for w in writers:
        w.close()
    logging.info(
        "Wrote %s '%s' examples to %s (shards=%s, compression=%s)",
        ex_count,
        field,
        out_dir,
        num_shards,
        compression,
    )


# ---------------------------------------------------------------------
# Model parts (unchanged)
# ---------------------------------------------------------------------


class Sampling(layers.Layer):
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.generator = tf.random.Generator.from_seed(seed)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = self.generator.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(seq_length, n_features, latent_dim):
    inputs = tf.keras.Input(shape=(seq_length, n_features))
    x = layers.LSTM(32, return_sequences=True, dropout=0.1)(inputs)
    x = layers.LSTM(16)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -10, 10))(z_log_var)
    sampling = Sampling()
    z = sampling([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(seq_length, n_features, latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.RepeatVector(seq_length)(latent_inputs)
    x = layers.LSTM(16, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def compute_losses(self, data, training):
        z_mean, z_log_var, z = self.encoder(data, training=training)
        reconstruction = self.decoder(z, training=training)
        recon_per_example = tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2])
        reconstruction_loss = tf.reduce_mean(recon_per_example)
        kl_per_example = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.kl_weight * tf.reduce_mean(tf.reduce_sum(kl_per_example, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss, _ = self.compute_losses(
                data, training=True
            )
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        total_loss, reconstruction_loss, kl_loss, _ = self.compute_losses(
            data, training=False
        )
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction


def build_vae(latent_dim, seq_length, n_features, kl_init_weight):
    encoder = build_encoder(seq_length, n_features, latent_dim)
    decoder = build_decoder(seq_length, n_features, latent_dim)
    return VAE(encoder, decoder, kl_init_weight)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument(
        "--model_dir", type=str, default=os.getenv("AIP_MODEL_DIR", "model")
    )
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    args = parser.parse_args()

    # Convert GCS model path if present
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])

    # Hparams
    hparams = {**DEFAULT_HPARAMS, **args.hparams}
    logging.info(f"Using model hyper-parameters: {hparams}")

    # Perf knobs
    if hparams.get("use_xla_jit", True):
        try:
            tf.config.optimizer.set_jit(True)
            logging.info("XLA JIT enabled")
        except Exception as e:
            logging.warning(f"Could not enable XLA JIT: {e}")
    if hparams.get("use_mixed_precision", True) and tf.config.list_physical_devices(
        "GPU"
    ):
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            logging.info("Mixed precision enabled")
        except Exception as e:
            logging.warning(f"Could not enable mixed precision: {e}")

    # Strategy
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # ===========================
    # Datasets from TFRecords
    # ===========================
    # We consume the standardized features written by the splitter.
    # features = [
    #     "start_price_std",
    #     "PRICE_std",
    #     "PDCC_Down_std",
    #     "OSV_Down_std",
    #     "PDCC2_UP_std",
    #     "OSV_Up_std",
    # ]
    features = [
        "PRICE_std",
        "vol_quote_std",
        "cvd_quote_std",
        "PDCC_Down",
        "OSV_Down_std",
        "PDCC2_UP",
        "regime_up",
        "regime_down",
    ]
    
    assert hparams["n_features"] == len(
        features
    ), f"n_features={hparams['n_features']} does not match {len(features)}"

    # Build sequence datasets [batch, seq_length, n_features]
    train_ds = _make_sequence_dataset_from_tfrecord_dir(
        data_dir=args.train_data,
        seq_length=hparams["seq_length"],
        batch_size=hparams["batch_size"],
        feature_names=features,
        shuffle_sequences=True,  # shuffle windows for training
        compression_type="GZIP",
    )
    val_ds = _make_sequence_dataset_from_tfrecord_dir(
        data_dir=args.valid_data,
        seq_length=hparams["seq_length"],
        batch_size=hparams["batch_size"],
        feature_names=features,
        shuffle_sequences=False,
        compression_type="GZIP",
    )
    test_ds = _make_sequence_dataset_from_tfrecord_dir(
        data_dir=args.test_data,
        seq_length=hparams["seq_length"],
        batch_size=hparams["batch_size"],
        feature_names=features,
        shuffle_sequences=False,
        compression_type="GZIP",
    )

    # Build & compile
    with strategy.scope():
        vae = build_vae(
            latent_dim=hparams["latent_dim"],
            seq_length=hparams["seq_length"],
            n_features=hparams["n_features"],
            kl_init_weight=hparams["kl_init_weight"],
        )
        optimizer = keras.optimizers.AdamW(learning_rate=hparams["learning_rate"])
        vae.compile(optimizer=optimizer, loss=None)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=hparams["patience"], restore_best_weights=True
        )
    ]

    # Quick sanity
    inspect_dataset(train_ds, name="train_ds", max_batches=1)
    inspect_dataset(val_ds, name="val_ds", max_batches=1)
    inspect_dataset(test_ds, name="test_ds", max_batches=1)

    # Train (kept identical semantics; remove .take(1) if you want full dataset)
    logging.info("Starting training...")
    vae.fit(
        train_ds.take(1),
        epochs=hparams["epochs"],
        validation_data=val_ds.take(1),
        callbacks=callbacks,
    )

    # Chief-only save
    if not _is_chief(strategy):
        logging.info("Non-chief worker: exiting without saving model/metrics.")
        sys.exit(0)

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    vae.save(f"{args.model_dir}/vae_{hparams['threshold']}.keras")
    logging.info(f"Model saved to {args.model_dir}")

    # Evaluate
    eval_metrics_list = vae.evaluate(test_ds)
    eval_names = vae.metrics_names
    eval_metrics = dict(
        zip(
            eval_names,
            eval_metrics_list
            if isinstance(eval_metrics_list, (list, tuple))
            else [eval_metrics_list],
        )
    )
    metrics = {
        "problemType": "regression",
        "rootMeanSquaredError": eval_metrics.get("root_mean_squared_error"),
        "meanAbsoluteError": eval_metrics.get("mean_absolute_error"),
        "meanAbsolutePercentageError": eval_metrics.get(
            "mean_absolute_percentage_error"
        ),
        "rSquared": None,
        "rootMeanSquaredLogError": eval_metrics.get("mean_squared_logarithmic_error"),
        "klLoss": float(eval_metrics.get("kl_loss", 0.0))
        if eval_metrics.get("kl_loss", None) is not None
        else None,
        "reconstructionLoss": float(eval_metrics.get("reconstruction_loss", 0.0))
        if eval_metrics.get("reconstruction_loss", None) is not None
        else None,
        "totalLoss": float(eval_metrics.get("total_loss", 0.0))
        if eval_metrics.get("total_loss", None) is not None
        else None,
    }
    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp)
    logging.info(f"Metrics saved to {args.metrics}")

    # -----------------------------------------------------------------
    # NEW: Write latents + y as TFRecords for train/valid/test (append-only)
    # -----------------------------------------------------------------
    try:
        # Label builder: take the last timestep value of `label_name`.
        # To support multi-step output without changing hparams, you can set
        # environment variable FORECAST_STEPS (default=1). If H>1, we repeat
        # the last value H times to form y of shape [H].
        label_name = hparams["label_name"]
        assert (
            label_name in features
        ), f"label_name '{label_name}' not found in features list"
        label_idx = features.index(label_name)
        # make forecast_steps a single source of truth
        forecast_steps = int(hparams["forecast_steps"])
        env_fs = int(os.getenv("FORECAST_STEPS", str(forecast_steps)))
        if env_fs != forecast_steps:
            raise ValueError(
                f"forecast_steps mismatch: hparams={forecast_steps} env={env_fs}"
            )
        # FORECAST_STEPS = int(os.getenv("FORECAST_STEPS", "1"))

        # def _label_builder(batch_windows: tf.Tensor) -> tf.Tensor:
        #     # batch_windows: [B, T, F]
        #     last = batch_windows[:, -1, label_idx]         # [B]
        #     if FORECAST_STEPS <= 1:
        #         return tf.expand_dims(last, axis=-1)       # [B, 1]
        #     # Repeat last value H times -> [B, H]
        #     return tf.repeat(
        #         tf.expand_dims(last, -1),
        #         repeats=FORECAST_STEPS,
        #         axis=-1,
        #     )
        def _label_builder(batch_windows: tf.Tensor) -> tf.Tensor:
            # batch_windows: [B, T, F]
            last = batch_windows[:, -1, label_idx]  # [B]
            return tf.repeat(
                tf.expand_dims(last, -1), repeats=forecast_steps, axis=-1
            )  # [B, H]

        # Run-scoped base dir
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        base_lat_dir = os.path.join(args.model_dir, "latents", run_id)
        tf.io.gfile.makedirs(base_lat_dir)

        def _write_split(split_name, ds):
            out_dir = os.path.join(base_lat_dir, split_name)
            _write_latent_tfrecords(
                ds,
                vae.encoder,
                out_dir,
                num_shards=16 if split_name == "train" else 8,
                compression="GZIP",
                latent_key="z",
                mean_key="z_mean",
                logvar_key="z_log_var",
                label_key="y",
                label_builder=_label_builder,
            )
            _write_manifest(
                out_dir,
                latent_dim=int(hparams["latent_dim"]),
                forecast_steps=forecast_steps,
                compression="GZIP",
                latent_key="z",
                mean_key="z_mean",
                logvar_key="z_log_var",
                label_key="y",
                threshold=hparams.get("threshold"),
            )

        _write_split("train", train_ds)
        _write_split("valid", val_ds)
        _write_split("test", test_ds)

        # 1) Emit a per-run index with absolute split URIs
        index = {
            "version": 1,
            "run_id": run_id,
            "forecast_steps": forecast_steps,
            "splits": {
                "train": os.path.join(base_lat_dir, "train"),
                "valid": os.path.join(base_lat_dir, "valid"),
                "test": os.path.join(base_lat_dir, "test"),
            },
        }

        # Write the index inside this run's folder so the run is self-contained
        with tf.io.gfile.GFile(os.path.join(base_lat_dir, "_index.json"), "w") as f:
            json.dump(index, f)

        # 2) (Optional but handy) Update a pointer to the latest run for this model_dir
        # This makes discovery easy if your wrapper wants the most recent run.
        latents_root = os.path.join(args.model_dir, "latents")
        with tf.io.gfile.GFile(os.path.join(latents_root, "LATEST"), "w") as f:
            f.write(run_id + "\n")

        logging.info(f"Wrote latent index: {os.path.join(base_lat_dir, '_index.json')}")

        # Optional convenience datasets (also run-scoped; distinct subtree)
        base_means_dir = os.path.join(args.model_dir, "latents_means", run_id)
        base_logs_dir = os.path.join(args.model_dir, "latents_logvars", run_id)
        _write_single_field(
            train_ds,
            vae.encoder,
            os.path.join(base_means_dir, "train"),
            field="z_mean",
            num_shards=8,
            compression="GZIP",
        )
        _write_single_field(
            val_ds,
            vae.encoder,
            os.path.join(base_means_dir, "valid"),
            field="z_mean",
            num_shards=4,
            compression="GZIP",
        )
        _write_single_field(
            test_ds,
            vae.encoder,
            os.path.join(base_means_dir, "test"),
            field="z_mean",
            num_shards=4,
            compression="GZIP",
        )

        _write_single_field(
            train_ds,
            vae.encoder,
            os.path.join(base_logs_dir, "train"),
            field="z_log_var",
            num_shards=8,
            compression="GZIP",
        )
        _write_single_field(
            val_ds,
            vae.encoder,
            os.path.join(base_logs_dir, "valid"),
            field="z_log_var",
            num_shards=4,
            compression="GZIP",
        )
        _write_single_field(
            test_ds,
            vae.encoder,
            os.path.join(base_logs_dir, "test"),
            field="z_log_var",
            num_shards=4,
            compression="GZIP",
        )

        logging.info(f"Latent TFRecords (with y) written under: {base_lat_dir}")

        # Historical reference: use `_write_latent_tfrecords(...)` for
        # train/val/test plus optional `_write_single_field(..., field="z_mean")`
        # and `_write_single_field(..., field="z_log_var")` exports.
    except Exception as e:
        logging.exception(f"Failed to write latent TFRecords: {e}")
        raise

    # Training dataset info (format updated to tfrecord)
    path = Path(args.model_dir) / TRAINING_DATASET_INFO
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "tfrecord",
        "threshold": hparams["threshold"],
        "targetField": "UNSUPERVISED",
    }
    logging.info(f"Saving training dataset info for model monitoring: {path}")
    logging.info(f"Training dataset: {training_dataset_for_monitoring}")
    with open(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)


if __name__ == "__main__":
    main()
