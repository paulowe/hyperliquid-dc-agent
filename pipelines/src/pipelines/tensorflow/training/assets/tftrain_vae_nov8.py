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

# Set up cloud profiler
from google.cloud.aiplatform.training_utils import cloud_profiler

# Initialize the profiler.
# https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler#enable
cloud_profiler.init()

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
    kl_init_weight=1e-4,
    patience=10,
    metrics=[
        "MeanAbsoluteError",
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
# Load windows from TFRecord files
# ---------------------------------------------------------------------
def load_windows(saved_dir: str, compression: Optional[str] = "GZIP"):
    """
    Load a tf.data.Dataset of (x, y) pairs created with
    tf.data.Dataset.save(..., compression=...).

    Returns:
      ds_xy: dataset yielding (x, y)
      x_spec, y_spec: TensorSpecs for x and y
    """
    ds_xy = tf.data.Dataset.load(saved_dir, compression=compression)
    ds_xy = ds_xy.prefetch(tf.data.AUTOTUNE)

    element_spec = ds_xy.element_spec
    # element_spec expected to be (TensorSpec([B, L, F], ...), TensorSpec([B, H], ...)) per batch
    if not isinstance(element_spec, (tuple, list)) or len(element_spec) != 2:
        raise ValueError(f"Expected (x,y) dataset; got element_spec={element_spec}")
    x_spec, y_spec = element_spec
    return ds_xy, x_spec, y_spec

# ---------------------------------------------------------------------
# Model parts
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
    # z_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -10, 10))(z_log_var)
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


def build_vae(latent_dim, input_width, n_features, kl_init_weight):
    encoder = build_encoder(input_width, n_features, latent_dim)
    decoder = build_decoder(input_width, n_features, latent_dim)
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
    parser.add_argument("--z_train", type=str, required=True)
    parser.add_argument("--z_valid", type=str, required=True)
    parser.add_argument("--z_test", type=str, required=True)
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

    # Load windows
    train_xy, x_spec_tr, y_spec_tr = load_windows(args.train_data, compression="GZIP")
    val_xy,   _, _ = load_windows(args.valid_data, compression="GZIP")
    test_xy,  _, _ = load_windows(args.test_data, compression="GZIP")

    input_width = x_spec_tr.shape[-2]  # input_width
    n_features = x_spec_tr.shape[-1]  # number of features
    label_width = y_spec_tr.shape[-1]  # label_width (forecast horizon)

    # For VAE reconstruction training, we only need X (drop Y)
    train_ds = (train_xy
        # .shuffle(10000, reshuffle_each_iteration=True)
        .batch(hparams["batch_size"], drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .map(lambda x, y: x)
    )

    val_ds = (val_xy
        .batch(hparams["batch_size"], drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .map(lambda x, y: x)
    )

    test_ds = (test_xy
        .batch(hparams["batch_size"], drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .map(lambda x, y: x)
    )


    # Build & compile
    with strategy.scope():
        vae = build_vae(
            latent_dim=hparams["latent_dim"],
            input_width=input_width,
            n_features=n_features,
            kl_init_weight=hparams["kl_init_weight"],
        )
        optimizer = keras.optimizers.AdamW(learning_rate=hparams["learning_rate"])
        vae.compile(optimizer=optimizer, loss=None)

    # Callbacks
    # https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements#tensorboard
    # Profile from batches 5 to 10
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'],
        histogram_freq=1,
        profile_batch='5, 10',
    )
    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=hparams["patience"], restore_best_weights=True
    )

    #  TO DO: Model Checkpoint callback
    # https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements#resilience

    callbacks = [
        tensorboard_callback,
        early_stopping_callback,    
    ]

    # Quick sanity
    inspect_dataset(train_ds, name="train_ds", max_batches=1)
    inspect_dataset(val_ds, name="val_ds", max_batches=1)
    inspect_dataset(test_ds, name="test_ds", max_batches=1)
    logging.info("Inspecting (x,y) pairs...")
    try:
        for x_b, y_b in train_xy.take(1):
            logging.info(f"x batch shape: {x_b.shape}")  # [B, L, F]
            logging.info(f"y batch shape: {y_b.shape}")  # [B, H]
    except Exception as e:
        logging.warning(f"Could not inspect (x,y): {e}")

    # Train (kept identical semantics; remove .take(1) if you want full dataset)
    logging.info("Starting training...")
    vae.fit(
        # train_ds.take(1),
        train_ds,
        epochs=hparams["epochs"],
        # validation_data=val_ds.take(1),
        validation_data=val_ds,
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
        "problemType": "dimensionality_reduction",
        "klLoss": float(eval_metrics.get("kl_loss", 0.0)),
        "reconstructionLoss": float(eval_metrics.get("reconstruction_loss", 0.0)),
        "totalLoss": float(eval_metrics.get("total_loss", 0.0)),
    }
    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp)
    logging.info(f"Metrics saved to {args.metrics}")

    # -----------------------------------------------------------------
    # Write latents TFRecords for train/valid/test
    # -----------------------------------------------------------------
    try:
        # Build: (x,y) -> map encoder -> {z, y}
        def encode_xy(x, y):
            z_mean, z_log_var, z = vae.encoder(x, training=False)
            return {"z": z, "y": y}   # <-- only store what you need

        train_latents_ds = (train_xy
            .batch(hparams["batch_size"], drop_remainder=False)
            .map(encode_xy, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

        val_latents_ds = (val_xy
            .batch(hparams["batch_size"], drop_remainder=False)
            .map(encode_xy, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

        test_latents_ds = (test_xy
            .batch(hparams["batch_size"], drop_remainder=False)
            .map(encode_xy, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

        train_latents_ds.save(args.z_train, compression="GZIP")
        val_latents_ds.save(args.z_valid, compression="GZIP")
        test_latents_ds.save(args.z_test, compression="GZIP")

    except Exception as e:
        logging.exception(f"Failed to write latent snapshots: {e}")
        raise


if __name__ == "__main__":
    main()
