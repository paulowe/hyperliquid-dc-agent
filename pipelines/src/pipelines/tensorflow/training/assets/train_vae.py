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
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization, StringLookup

# from tensorflow.keras import Input, Model, optimizers  # unused
# from tensorflow.keras.layers import Dense, Concatenate  # unused
from tensorflow.data import Dataset
from matplotlib import pyplot as plt
from typing import Optional, List, Dict, Tuple

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
    n_features=6,
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
    cache_to_disk="/tmp/dcvae_data.cache",  # set to a path (e.g. "/tmp/ts.cache")
    # to enable cache-to-file instead of RAM
)

# Helper function to create dataset


def create_dataset(
    input_data: Path,
    label_name: str,
    model_params: dict,
    selected_columns: list[str],
    shuffle: bool = True,
) -> Dataset:
    """Create a TF Dataset from input csv files.
    Args:
        input_data (Input[Dataset]): Train/Valid data in CSV format
        label_name (str): Name of column containing the labels
        model_params (dict): model parameters
        file_pattern (str): Read data from one or more files. If empty, then
            training and validation data is read from single file respectively.
            For multiple files, use a pattern e.g. "files-*.csv".
    Returns:
        dataset (TF Dataset): TF dataset where each element is a (features, labels)
            tuple that corresponds to a batch of CSV rows
    """

    # shuffle & shuffle_buffer_size added to rearrange input data
    # passed into model training
    # num_rows_for_inference is for auto detection of datatypes
    # while creating the dataset.
    # If a float column has a high proportion of integer values (0/1 etc),
    # the method wrongly detects it as a tf.int32 which fails during training time,
    # hence the high hardcoded value (default is 100)

    # Apply data sharding: Sharded elements are produced by the dataset
    # Each worker will process the whole dataset and discard the portion that is
    # not for itself. Note that for this mode to correctly partitions the dataset
    # elements, the dataset needs to produce elements in a deterministic order.
    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    logging.info(f"Creating dataset from CSV file(s) at {input_data}...")
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=str(input_data),
        batch_size=model_params["batch_size"],
        label_name=label_name,
        # num_epochs=model_params["epochs"],
        num_epochs=1,  # IMPORTANT: do not repeat here; `fit(epochs=...)` will
        shuffle=shuffle,
        shuffle_buffer_size=10000 if shuffle else 1,
        num_rows_for_inference=20000,
        num_parallel_reads=tf.data.AUTOTUNE,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        select_columns=selected_columns,  # read only what we actually use
    )
    # Cache small/medium datasets to RAM (or a file path if RAM is tight)
    ds = ds.cache()
    ds = ds.with_options(data_options)
    return ds.prefetch(tf.data.AUTOTUNE)


# Helper function to get distribution strategy


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


# Helper function to normalize numerical input


def normalization(name: str, dataset: Dataset) -> Normalization:
    """Create a Normalization layer for a feature.
    Args:
        name (str): name of feature to be normalized
        dataset (Dataset): dataset to adapt layer
    Returns:
        normalization layer (Normalization): adapted normalization layer
            of shape (?,1)
    """
    logging.info(f"Normalizing numerical input '{name}'...")
    x = Normalization(axis=None, name=f"normalize_{name}")
    x.adapt(dataset.map(lambda y, _: y[name]))
    return x


# Helper function to encode categorical input


def str_lookup(name: str, dataset: Dataset, output_mode: str) -> StringLookup:
    """Create a StringLookup layer for a feature.
    Args:
        name (str): name of feature to be encoded
        dataset (Dataset): dataset to adapt layer
        output_mode (str): argument for StringLookup layer (e.g. 'one_hot', 'int')
    Returns:
        StringLookup layer (StringLookup): adapted StringLookup layer of shape (?,X)
    """
    logging.info(f"Encoding categorical input '{name}' ({output_mode})...")
    x = StringLookup(output_mode=output_mode, name=f"str_lookup_{output_mode}_{name}")
    x.adapt(dataset.map(lambda y, _: y[name]))
    logging.info(f"Vocabulary: {x.get_vocabulary()}")
    return x


# Helper function to load numpy dataset


def load_numpy_dataset(path, batch_size, shuffle=True):
    data = np.load(path)["arr_0"]
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data))
    return ds.batch(batch_size)


# Helper function to check if current worker is chief


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


# Helper function to get temp directory


def _get_temp_dir(dirpath, task_id):
    base_dirpath = "workertemp_" + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


# Helper function to inspect dataset for None values


def inspect_dataset(ds, name="dataset", max_batches=3):
    logging.info(
        f"Inspecting {name} for None values (checking first {max_batches} batches)..."
    )
    try:
        for i, batch in enumerate(ds.take(max_batches)):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
            else:
                x, y = batch, None

            # Convert to numpy for easier inspection
            x_np = x.numpy() if tf.is_tensor(x) else np.array(x)
            y_np = (
                y.numpy()
                if (y is not None and tf.is_tensor(y))
                else np.array(y)
                if y is not None
                else None
            )

            # Check features for missing values
            if x_np is None or np.any(pd.isna(x_np)):
                logging.error(f"Found None or NaN in features at batch {i}")

            # Check labels if they exist
            if y_np is not None and np.any(pd.isna(y_np)):
                logging.error(f"Found None or NaN in labels at batch {i}")

            logging.info(f"Batch {i} of {name} passed basic None/NaN checks.")
    except Exception as e:
        logging.exception(f"Error while inspecting {name}: {e}")


# -----------------------------------------------------------------------------
# Read CSV using tf.io.gfile (local or GCS)
# -----------------------------------------------------------------------------


def read_csv_any(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Robust CSV reader that works for local files or GCS URIs via tf.io.gfile."""
    if path.startswith("gs://"):
        with tf.io.gfile.GFile(path, "rb") as f:
            return pd.read_csv(f, usecols=usecols)
    return pd.read_csv(path, usecols=usecols)


# Window Generator


class WindowGenerator:
    """
    Utility to generate input/label windows from time-series DataFrames.

    Splits each of train, validation, and test DataFrames into overlapping
    windows of fixed length, returning slices of inputs and labels. Optionally
    supports selecting only certain label columns.

    Attributes:
        train_df (pd.DataFrame): Full training data.
        val_df (pd.DataFrame): Full validation data.
        test_df (pd.DataFrame): Full test data.
        input_width (int): Number of time steps per input window.
        label_width (int): Number of time steps to predict.
        shift (int): Offset between the end of the input window and label start.
        total_window_size (int): ``input_width + shift``.
        input_slice (slice): Input indices ``[0:input_width]``.
        labels_slice (slice): Label indices ``[label_start:]``.
        input_indices (np.ndarray): Input indices within the total window.
        label_indices (np.ndarray): Label indices within the total window.
        column_indices (dict): Maps each column name to its index.
        label_columns (list[str] | None): Columns to use as labels.
        label_columns_indices (dict): Maps each label column to its index.
    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        label_columns: Optional[List[str]] = ["PRICE_std"],
    ):
        # Raw splits
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift  # Includes label width

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def _split_window(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Given a batch of windows, split into (inputs, labels) tensors.
        """
        inputs = batch[:, self.input_slice, :]
        labels = batch[:, self.labels_slice, :]
        if self.label_columns:
            labels = tf.stack(
                [labels[:, :, self.column_indices[c]] for c in self.label_columns],
                axis=-1,
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def _split(
        self, features: tf.Tensor, latents: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Given a list of consecutive inputs, convert them into input and label
        windows.

        Split a full window into (inputs, labels), and optionally append latents.

        Parameters:
            features: (batch_size, total_window_size, num_features)
            latents: (batch_size, latent_dim)

        Returns:
            inputs: (batch_size, input_width, num_features) OR
            (batch_size, input_width, num_features + latent_dim)
            labels: (batch_size, label_width, num_labels)

        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # select only label columns
        if self.label_columns:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # attach latents if provided
        if latents is not None:
            # expand and tile latents to match time dimension
            lat_tiled = tf.repeat(
                latents[:, tf.newaxis, :],  # (batch_size, 1, latent_dim)
                repeats=self.input_width,  # (batch_size, input_width, latent_dim)
                axis=1,
            )
            # (batch_size, input_width, num_features + latent_dim)
            inputs = tf.concat([inputs, lat_tiled], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def _make_base_dataset(
        self, data: pd.DataFrame, batch_size: int, shuffle: bool
    ) -> tf.data.Dataset:
        """
        Build base tf.data.Dataset of raw windows from a DataFrame.
        """
        arr = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=arr,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return ds

    def make_dataset(
        self,
        split: pd.DataFrame,
        encoder: Optional[tf.keras.Model] = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Construct a tf.data.Dataset, optionally mapping through a VAE encoder.

        Returns a tf.data.Dataset of (inputs, labels) or (latent, labels)
        for a given split.

        Args:
            split:    One of 'train', 'val', 'test'.
            encoder:  If provided, the VAE encoder to map raw windows to latents.
            batch_size: Batch size.
            shuffle:  Whether to shuffle (only applies to 'train' by default).
        """
        # Get the DataFrame for the split
        df = getattr(self, f"{split}_df")
        if df is None:
            raise ValueError(f"No DataFrame for split '{split}'")

        ds = self._make_base_dataset(df, batch_size, shuffle)

        # If an encoder is provided, map to attach latent
        if encoder:
            # map to attach latent
            def map_fn(batch):
                # encode the batch to get the latent vector z
                # z shape: (batch_size, latent_dim)
                _, _, z = encoder(batch, training=False)

                # # tile z over time axis
                # z_tiled = tf.repeat(
                #     z[:, tf.newaxis, :], # (batch_size, 1, latent_dim)
                #     repeats=self.input_width, # (batch_size, input_width, latent_dim)
                #     axis=1)

                # get the corresponding label y for the forecasting task
                x, y = self._split(batch)
                return z, y

            ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(
                lambda batch: self._split(batch), num_parallel_calls=tf.data.AUTOTUNE
            )

        return ds.prefetch(tf.data.AUTOTUNE)

    def get_latent_dict(
        self,
        encoders: Dict[float, tf.keras.Model],
        thresholds: List[float],
        batch_size: int = 32,
    ) -> Dict[float, np.ndarray]:
        """
        Returns a dict of raw latent arrays (n_windows, latent_dim) per threshold,
        using the 'train' split of each WindowGenerator in self.
        """
        z_dict: Dict[float, np.ndarray] = {}
        for thr in thresholds:
            # Build a dataset that yields (latent_windows, labels) batches
            ds = self.make_dataset(
                split="train",
                encoder=encoders[thr],
                batch_size=batch_size,
                shuffle=False,
            )
            # Collect all batches into a list of tuples
            batches = list(ds)  # [(z_batch, y_batch), ...]
            # Transpose to two tuples: one of all z_batches, one of all y_batches
            z_batches, _ = zip(*batches)
            # Concatenate along the batch dimension and store
            z_dict[thr] = np.concatenate([zb.numpy() for zb in z_batches], axis=0)
        return z_dict

    # v2 plot
    def plot(self, model=None, plot_col="PRICE_std", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            # Plot labels safely
            label_vals = labels[n, :, label_col_index]
            if len(self.label_indices) != label_vals.shape[0]:
                plt.scatter(
                    [self.label_indices[-1]],
                    [label_vals[-1]],
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )
            else:
                plt.scatter(
                    self.label_indices,
                    label_vals,
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )

            # Plot predictions safely
            if model is not None:
                predictions = model(inputs)
                pred_vals = predictions[n, :, label_col_index]
                if len(self.label_indices) != pred_vals.shape[0]:
                    plt.scatter(
                        [self.label_indices[-1]],
                        [pred_vals[-1]],
                        marker="X",
                        edgecolors="k",
                        label="Predictions",
                        c="#ff7f0e",
                        s=64,
                    )
                else:
                    plt.scatter(
                        self.label_indices,
                        pred_vals,
                        marker="X",
                        edgecolors="k",
                        label="Predictions",
                        c="#ff7f0e",
                        s=64,
                    )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [ticks]")

    @property
    def train(self) -> tf.data.Dataset:
        return self.make_dataset("train")

    @property
    def val(self) -> tf.data.Dataset:
        return self.make_dataset("val") if self.val_df is not None else None

    @property
    def test(self) -> tf.data.Dataset:
        return self.make_dataset("test") if self.test_df is not None else None

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        """Human-readable summary of the window configuration."""
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Shift: {self.shift}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )


# Sampling Layer


class Sampling(layers.Layer):
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        # Create a random generator with a fixed seed
        self.generator = tf.random.Generator.from_seed(seed)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Use the generator to draw reproducible normals
        epsilon = self.generator.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# class Sampling(layers.Layer):
#     def __init__(self, seed: int = 42, **kwargs):
#         super().__init__(**kwargs)
#         # Create a random generator with a fixed seed
#         self.generator = tf.random.Generator.from_seed(seed)

#     def call(self, inputs, training=True):
#         z_mean, z_log_var = inputs
#         if training:
#             # Use the generator to draw reproducible normals
#             epsilon = self.generator.normal(shape=tf.shape(z_mean))
#         else:
#             logging.info("Sampling layer is deterministic at serve/export")
#             epsilon = 0.0   # deterministic at serve/export

#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# class Sampling(layers.Layer):
#     def __init__(self, seed: int = 42, **kwargs):
#         super().__init__(**kwargs)
#         self.seed = int(seed)
#         # Prefer Keras trackable RNG; fall back to stateless if unavailable
#         try:
#             from keras import random as krandom  # Keras 3
#             self._krandom = krandom
#             self.rng = krandom.SeedGenerator(self.seed)  # <-- trackable
#             self._use_trackable_rng = True
#         except Exception:
#             self._krandom = None
#             self._seed_tensor = tf.constant([self.seed, self.seed + 1], tf.int32)
#             self._use_trackable_rng = False

#     def call(self, inputs, training=None):
#         z_mean, z_log_var = inputs
#         if training:
#             if self._use_trackable_rng:
#                 eps = self._krandom.normal(tf.shape(z_mean), rng=self.rng)
#             else:
#                 # Stateless fallback (also exports cleanly)
#                 eps = tf.random.stateless_normal(
#                     tf.shape(z_mean), seed=self._seed_tensor
#                 )
#         else:
#             # Deterministic at serve/export
#             logging.info("Sampling layer is deterministic at serve/export")
#             eps = tf.zeros_like(z_mean)

#         return z_mean + tf.exp(0.5 * z_log_var) * eps

#     def get_config(self):
#         cfg = super().get_config()
#         cfg.update({"seed": self.seed})
#         return cfg

# Encoder


def build_encoder(seq_length, n_features, latent_dim):
    inputs = tf.keras.Input(shape=(seq_length, n_features))
    x = layers.LSTM(32, return_sequences=True, dropout=0.1)(inputs)
    x = layers.LSTM(16)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -10, 10))(z_log_var)
    # z_log_var = keras.ops.clip(z_log_var, -10.0, 10.0)

    # Training flag is magically passed to the Sampling layer by Keras
    sampling = Sampling()
    z = sampling([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


# Decoder


def build_decoder(seq_length, n_features, latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.RepeatVector(seq_length)(latent_inputs)
    x = layers.LSTM(16, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


# VAE Class


class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

        # Trackers for loss componenets
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
        """
        Shared logic for computing reconstruction, KL, and total loss.
        Returns: total_loss, reconstruction_loss, kl_loss, reconstruction
        """
        z_mean, z_log_var, z = self.encoder(data, training=training)
        reconstruction = self.decoder(z, training=training)

        # Reconstruction loss: sum over time+features per example, then mean over batch
        recon_per_example = tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2])
        reconstruction_loss = tf.reduce_mean(recon_per_example)

        # KL divergence per example, mean over batch, then scale
        kl_per_example = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.kl_weight * tf.reduce_mean(tf.reduce_sum(kl_per_example, axis=1))

        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss, reconstruction

    def train_step(self, data):
        # Forward pass
        with tf.GradientTape() as tape:
            (
                total_loss,
                reconstruction_loss,
                kl_loss,
                reconstruction,
            ) = self.compute_losses(data, training=True)

        # Backprop and update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)  # optional clipping
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update KL weight dynamically
        # self.kl_weight = min(1.0, self.kl_weight * 1.10)
        # Gradually increase KL contribution

        # Update metrics trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        total_loss, reconstruction_loss, kl_loss, reconstruction = self.compute_losses(
            data, training=False
        )

        # Update trackers (no gradient)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs, training=False):
        """
        This method defines the forward pass for inference.
        Typically, you'd want to return the reconstruction (or z if you prefer).
        """
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction


# Build VAE


def build_vae(latent_dim, seq_length, n_features, kl_init_weight):
    encoder = build_encoder(seq_length, n_features, latent_dim)
    decoder = build_decoder(seq_length, n_features, latent_dim)
    return VAE(encoder, decoder, kl_init_weight)


def main():
    # ============================================================
    # Parse arguments (now debug args are applied if DEBUG_MODE=1)
    # ============================================================
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

    # ============================================================
    # Convert GCS paths if present
    # ============================================================
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])

    # ============================================================
    # Merge hyperparameters
    # ============================================================
    hparams = {**DEFAULT_HPARAMS, **args.hparams}
    logging.info(f"Using model hyper-parameters: {hparams}")

    # ============================================================
    # Enable JIT + mixed precision (GPU only)
    # ============================================================
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

    # ============================================================
    # Setup distributed training strategy
    # ============================================================
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # ============================================================
    # Load datasets with error logging
    # ============================================================
    features = [
        "start_price_std",
        "PRICE_std",
        "PDCC_Down_std",
        "OSV_Down_std",
        "PDCC2_UP_std",
        "OSV_Up_std",
    ]
    label_columns = [hparams["label_name"]]  # typically "PRICE_std"
    selected_cols = list({*features})  # (already includes label)

    def _resolve_file(path_like: str) -> str:
        # Accept "dir" or "dir/file.csv"; default to "data.csv" inside the dir.
        if path_like.endswith(".csv"):
            return path_like
        return os.path.join(path_like, "data.csv")

    train_path = _resolve_file(args.train_data)
    val_path = _resolve_file(args.valid_data)
    test_path = _resolve_file(args.test_data)

    try:
        train_df = read_csv_any(train_path, usecols=selected_cols)
        val_df = read_csv_any(val_path, usecols=selected_cols)
        test_df = read_csv_any(test_path, usecols=selected_cols)
    except Exception:
        logging.exception(
            "Failed reading CSVs (train=%s, val=%s, test=%s)",
            train_path,
            val_path,
            test_path,
        )
        raise

    # Ensure float32 (saves memory; matches TF default)
    train_df = train_df.astype(np.float32)
    val_df = val_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    # ============================================================
    # Create WindowGenerator
    # ============================================================
    window = WindowGenerator(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_width=hparams["seq_length"],
        shift=1,
        label_width=1,
        label_columns=label_columns,
    )
    logging.info(f"Generating train, val, test datasets...")

    train_ds = window.train.map(lambda x, y: x)
    val_ds = window.val.map(lambda x, y: x)
    test_ds = window.test.map(lambda x, y: x)

    # ============================================================
    # Build and compile VAE under distribution strategy
    # ============================================================
    with strategy.scope():
        vae = build_vae(
            latent_dim=hparams["latent_dim"],
            seq_length=hparams["seq_length"],
            n_features=hparams["n_features"],
            kl_init_weight=hparams["kl_init_weight"],
        )
        optimizer = keras.optimizers.AdamW(learning_rate=hparams["learning_rate"])
        vae.compile(
            optimizer=optimizer,
            loss=None,
        )

    # ============================================================
    # Setup callbacks
    # ============================================================
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=hparams["patience"], restore_best_weights=True
        )
    ]

    # ============================================================
    # Inspect datasets for None/NaN before training
    # ============================================================
    inspect_dataset(train_ds, name="train_ds", max_batches=1)
    inspect_dataset(val_ds, name="val_ds", max_batches=1)
    inspect_dataset(test_ds, name="test_ds", max_batches=1)

    # ============================================================
    # Train the model
    # ============================================================
    logging.info("Starting training...")
    vae.fit(
        train_ds.take(1),
        epochs=hparams["epochs"],
        validation_data=val_ds.take(1),
        callbacks=callbacks,
    )

    # ============================================================
    # Only persist output if this worker is chief
    # ============================================================
    if not _is_chief(strategy):
        logging.info("Non-chief worker: exiting without saving model/metrics.")
        sys.exit(0)

    # ============================================================
    # Save model
    # ============================================================
    os.makedirs(args.model_dir, exist_ok=True)
    vae.save(f"{args.model_dir}/vae_{hparams['threshold']}.keras")
    logging.info(f"Model saved to {args.model_dir}")

    # model_dir = os.environ.get("AIP_MODEL_DIR")  # <- magic path for auto-upload
    # if not model_dir:
    #     raise RuntimeError(
    #         "AIP_MODEL_DIR not set; ensure you're running under Vertex "
    #         "CustomTrainingJob with a serving container."
    #     )

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(
    #             [None, hparams["seq_length"], hparams["n_features"]], tf.float32
    #         )
    #     ]
    # )
    # def serving_fn_encoder(x):
    #     z_mean, z_log_var, z = vae.encoder(x, training=False)
    #     return {"z_mean": z_mean, "z_log_var": z_log_var, "z": z}
    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec([None, hparams["latent_dim"]], tf.float32),
    #         tf.TensorSpec([2], tf.int32)  # seed
    #     ]
    # )
    # def sampling_fn(z_mean, seed):
    #     eps = tf.random.stateless_normal(tf.shape(z_mean), seed=seed)
    #     return {"outputs": vae.decoder(z_mean + eps, training=False)}
    # # Save as SavedModel
    # # tf.saved_model.save(vae, model_dir)
    # tf.saved_model.save(
    #     vae,
    #     model_dir,
    #     signatures={
    #         "serving_default": serving_fn_encoder,
    #         # "sampling": sampling_fn # optional stochastic API
    #     }
    # )
    # logging.info(f"SavedModel exported to: {model_dir}")

    # ============================================================
    # Save evaluation metrics on test set
    # ============================================================
    eval_metrics = dict(zip(vae.metrics_names, vae.evaluate(test_ds)))
    metrics = {
        "problemType": "regression",
        "rootMeanSquaredError": eval_metrics.get("root_mean_squared_error"),
        "meanAbsoluteError": eval_metrics.get("mean_absolute_error"),
        "meanAbsolutePercentageError": eval_metrics.get(
            "mean_absolute_percentage_error"
        ),
        "rSquared": None,
        "rootMeanSquaredLogError": eval_metrics.get("mean_squared_logarithmic_error"),
        "klLoss": float(eval_metrics.get("kl_loss", 0.0)),
        "reconstructionLoss": float(eval_metrics.get("reconstruction_loss", 0.0)),
        "totalLoss": float(eval_metrics.get("total_loss", 0.0)),
    }

    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp)
    logging.info(f"Metrics saved to {args.metrics}")

    # ============================================================
    # Save training dataset info for model monitoring
    # ============================================================
    path = Path(args.model_dir) / TRAINING_DATASET_INFO
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "csv",
        "threshold": hparams["threshold"],
        "targetField": "UNSUPERVISED",
    }
    logging.info(f"Saving training dataset info for model monitoring: {path}")
    logging.info(f"Training dataset: {training_dataset_for_monitoring}")

    with open(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)


if __name__ == "__main__":
    main()
