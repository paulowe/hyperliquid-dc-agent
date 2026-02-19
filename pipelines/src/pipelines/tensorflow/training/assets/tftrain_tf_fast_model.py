import argparse
import os
import json
import logging
import google.cloud.logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # kept for compatibility with helper functions; not used in TFRecord path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, layers, optimizers
from tensorflow.keras.layers import Dense, Normalization, StringLookup, Concatenate
from tensorflow.data import Dataset
from matplotlib import pyplot as plt
from typing import Optional, List, Dict, Tuple

# Set up cloud profiler
from google.cloud.aiplatform.training_utils import cloud_profiler

# Initialize the profiler.
# https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler#enable
cloud_profiler.init()

# Set up logging
client = google.cloud.logging.Client()
client.setup_logging()

text = "DC Forecast Training Script"
logging.info(text)

# -----------------------------------------------------------------------------
# Constants / Defaults
# -----------------------------------------------------------------------------
TRAINING_DATASET_INFO = "training_dataset.json"

DEFAULT_HPARAMS = dict(
    batch_size=256,                    # larger batch = fewer steps, better throughput
    epochs=5,
    loss_fn="MeanSquaredError",
    optimizer="Adam",
    learning_rate=0.001,
    # Windowing parameters - Nov 5, 2025
    # -------------------
    label_width=1,       # 1 = single-step, >1 = multi-step horizon
    shift=1,             # gap between last input step and first label step
    sample_stride=1,     # stride between successive training examples
    # -------------------
    # n_features=6,
    n_features=8,
    kl_init_weight=1e-4,
    patience=5, # early stopping patience
    bottleneck_dim=0,  # 0 = no bottleneck (backward compat); >0 = project flattened input to this dim
    dropout_rate=0.0,  # 0.0 = no dropout; >0 = dropout after each hidden Dense layer
    l2_reg=0.0,        # 0.0 = no L2; >0 = kernel_regularizer=l2(l2_reg) on hidden Dense layers
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
    use_mixed_precision=True,          # only takes effect if a GPU is present
    use_xla_jit=True,                 # try XLA; keep if step time drops
    cache_to_disk="/tmp/dcvae_data.cache",  # set to a path (e.g., "/tmp/ts.cache") to cache-to-file
)

# Helper function to create dataset
# def create_dataset(
#     input_data: Path,
#     label_name: str,
#     model_params: dict,
#     selected_columns: list[str],
#     shuffle: bool = True,
# ) -> Dataset:
#     """Create a TF Dataset from input files.

#     NOTE: This function previously created a dataset from CSV using
#     `make_csv_dataset`. To improve I/O throughput and avoid text parsing,
#     it now reads **TFRecords** (as written by the upstream data splitter)
#     and builds a *finite* (one pass) tf.data pipeline with parallel reads +
#     prefetch. Keras `fit(epochs=...)` controls total passes.

#     Args:
#         input_data (Input[Dataset]): directory containing TFRecord shards
#         label_name (str): Name of column containing the labels (not used here;
#                           we window sequences and split later if needed)
#         model_params (dict): model parameters
#         selected_columns (list[str]): names of float features to parse
#         shuffle (bool): shuffle the resulting windows if True

#     Returns:
#         dataset (TF Dataset): dataset of windows shaped
#             (batch, seq_length, n_features). Labels are produced later in the
#             pipeline (model is next-step forecaster with label_width=1).
#     """
#     # Apply data sharding: Sharded elements are produced by the dataset
#     # Each worker will process the whole dataset and discard the portion that is
#     # not for itself. Note that for this mode to correctly partitions the dataset
#     # elements, the dataset needs to produce elements in a deterministic order.
#     # Make a *finite* (one pass) tf.data pipeline with parallel reads + prefetch.
#     # Keras `fit(epochs=...)` controls total passes.

#     data_options = tf.data.Options()
#     data_options.experimental_distribute.auto_shard_policy = (
#         tf.data.experimental.AutoShardPolicy.DATA
#     )

#     # NEW: Build a TFRecord -> parsed floats -> sliding windows pipeline.
#     # Expect files named part-*.tfrecord[.gz] written by the splitter.
#     compression = "GZIP"  # the splitter writes .tfrecord.gz by default
#     pattern = os.path.join(str(input_data), "*.tfrecord.gz")

#     logging.info(f"Creating dataset from TFRecord file(s) at {pattern}...")
#     files = tf.data.Dataset.list_files(pattern, shuffle=False)

#     ds = files.interleave(
#         lambda fp: tf.data.TFRecordDataset(fp, compression_type=compression),
#         cycle_length=1,                      # keep order across shards for time series
#         num_parallel_calls=tf.data.AUTOTUNE,
#         deterministic=True,
#     )

#     # Parse exactly the float features we will use for training
#     feature_spec = {name: tf.io.FixedLenFeature([], tf.float32) for name in selected_columns}

#     def _parse(serialized):
#         ex = tf.io.parse_single_example(serialized, feature_spec)
#         x = tf.stack([ex[name] for name in selected_columns], axis=-1)  # [n_features]
#         return x

#     ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

#     # Sliding windows of consecutive rows -> [seq_length, n_features]
#     seq_len = int(model_params.get("seq_length", 50))
#     windowed = ds.window(size=seq_len, shift=1, drop_remainder=True)
#     windowed = windowed.flat_map(lambda w: w.batch(seq_len, drop_remainder=True))

#     # Optionally shuffle windows for training
#     if shuffle:
#         windowed = windowed.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

#     # Batch and prefetch
#     windowed = windowed.batch(model_params["batch_size"], drop_remainder=True)
#     cache_path = model_params.get("cache_to_disk")
#     if cache_path:
#         tf.io.gfile.makedirs(os.path.dirname(cache_path))
#         windowed = windowed.cache(cache_path)
#     else:
#         windowed = windowed.cache()

#     windowed = windowed.with_options(data_options)
#     return windowed.prefetch(tf.data.AUTOTUNE)

# def build_context_label_windows(
#     input_data_dir: Path,
#     selected_columns: list[str],
#     label_columns: list[str],
#     input_width: int,
#     label_width: int,
#     shift: int,
#     sample_stride: int,
#     batch_size: int,
#     shuffle: bool,
#     cache_to_disk: str | None = None,
# ) -> tf.data.Dataset:
#     """Stream TFRecords -> float tensors -> sliding windows -> (x,y) pairs.

#     Shapes:
#       x: [B, input_width, n_features]
#       y: [B, label_width, n_labels]
#     """
#     # 1) Stream & parse
#     compression = "GZIP"
#     pattern = os.path.join(str(input_data_dir), "*.tfrecord.gz")
#     files = tf.data.Dataset.list_files(pattern, shuffle=False)

#     ds = files.interleave(
#         lambda fp: tf.data.TFRecordDataset(fp, compression_type=compression),
#         cycle_length=1,
#         num_parallel_calls=tf.data.AUTOTUNE,
#         deterministic=True,
#     )

#     feature_spec = {name: tf.io.FixedLenFeature([], tf.float32) for name in selected_columns}

#     def _parse(serialized):
#         ex = tf.io.parse_single_example(serialized, feature_spec)
#         row = tf.stack([ex[name] for name in selected_columns], axis=-1)  # [n_features]
#         return row

#     ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

#     # 2) Form sliding windows (total = inputs + gap + labels)
#     total_window = input_width + shift + label_width

#     # Use sample_stride for how far we slide between examples
#     windows = ds.window(size=total_window, shift=sample_stride, drop_remainder=True)
#     windows = windows.flat_map(lambda w: w.batch(total_window, drop_remainder=True))  # [T, F]

#     # 3) Split into (x, y) with zero leakage
#     label_indices = [selected_columns.index(c) for c in (label_columns or [selected_columns[0]])]

#     def _split(batch_tf: tf.Tensor):
#         # batch_tf: [T, F] -> add a batch dim by batching later
#         x = batch_tf[:input_width, :]                                 # [input_width, F]
#         y_slice = batch_tf[input_width + shift : input_width + shift + label_width, :]  # [label_width, F]
#         y = tf.gather(y_slice, indices=label_indices, axis=-1)        # [label_width, L]
#         return x, y

#     windows = windows.map(_split, num_parallel_calls=tf.data.AUTOTUNE)

#     # 4) (Optional) shuffle examples, then batch & cache
#     if shuffle:
#         windows = windows.shuffle(10000, reshuffle_each_iteration=True)

#     windows = windows.batch(batch_size, drop_remainder=True)

#     if cache_to_disk:
#         tf.io.gfile.makedirs(os.path.dirname(cache_to_disk))
#         windows = windows.cache(cache_to_disk)
#     else:
#         windows = windows.cache()

#     # 5) Perf options
#     opts = tf.data.Options()
#     opts.experimental_deterministic = False
#     opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#     return windows.with_options(opts).prefetch(tf.data.AUTOTUNE)

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
    x.adapt(dataset.map(lambda y, _: y[name], num_parallel_calls=tf.data.AUTOTUNE))
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

            # Check features for None / NaN
            if np.any(x_np == None) or np.isnan(x_np).any():
                logging.error(f"Found None or NaN in features at batch {i}")

            # Check labels if they exist
            if y_np is not None:
                if np.any(y_np == None) or np.isnan(y_np).any():
                    logging.error(f"Found None or NaN in labels at batch {i}")

            logging.info(f"Batch {i} of {name} passed basic None/NaN checks.")
    except Exception as e:
        logging.exception(f"Error while inspecting {name}: {e}")

# -----------------------------------------------------------------------------
# Debug helper (kept lightweight)
# -----------------------------------------------------------------------------
def inspect_dataset_v2(ds: tf.data.Dataset, name="dataset", max_batches=1):
    try:
        for i, (x, y) in enumerate(ds.take(max_batches)):
            tf.debugging.assert_all_finite(x, f"{name}: found NaN/Inf in features (batch {i})")
            tf.debugging.assert_all_finite(y, f"{name}: found NaN/Inf in labels (batch {i})")
        logging.info(f"{name}: basic NaN/Inf checks passed on {max_batches} batch(es).")
    except Exception as e:
        logging.exception(f"Inspection error on {name}: {e}")

# -----------------------------------------------------------------------------
# Read CSV using tf.io.gfile (local or GCS)
# -----------------------------------------------------------------------------
def read_csv_any(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Robust CSV reader that works for local files or GCS URIs via tf.io.gfile.

    NOTE: left in place for backwards compatibility and debugging, but the
    training pipeline now reads TFRecords to maximize throughput.
    """
    if path.startswith("gs://"):
        with tf.io.gfile.GFile(path, "rb") as f:
            return pd.read_csv(f, usecols=usecols)
    return pd.read_csv(path, usecols=usecols)

# -----------------------------------------------------------------------------
# tf.data performance options (shared)
# -----------------------------------------------------------------------------
def dataset_with_fast_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    # Let fast elements overtake slow ones → higher throughput
    opts.experimental_deterministic = False
    # If you scale out, allow data sharding by element
    opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return ds.with_options(opts)

# -----------------------------------------------------------------------------
# Window Generator (retained for reference / plotting; not used in TFRecord path)
# -----------------------------------------------------------------------------
# class WindowGenerator:
#     """
#     Utility to generate input/label windows from time-series DataFrames for model training.
#     (Kept for compatibility; the training path below uses TFRecords directly.)
#     """
#     def __init__(
#         self,
#         input_width: int,
#         label_width: int,
#         shift: int,
#         train_df: pd.DataFrame,
#         val_df: Optional[pd.DataFrame] = None,
#         test_df: Optional[pd.DataFrame] = None,
#         label_columns: Optional[List[str]] = ["PRICE_std"],
#         cache_to_disk: Optional[str] = True,
#     ):
#         self.train_df = train_df
#         self.val_df = val_df
#         self.test_df = test_df
#         self.label_columns = label_columns or []
#         if label_columns is not None:
#             self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
#         else:
#             self.label_columns_indices = {}
#         self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift  # Includes label width
#         self.total_window_size = input_width + shift
#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]
#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
#         self.cache_to_disk = cache_to_disk

#     def _split_window(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
#         inputs = batch[:, self.input_slice, :]
#         labels = batch[:, self.labels_slice, :]
#         if self.label_columns:
#             labels = tf.stack(
#                 [labels[:, :, self.column_indices[c]] for c in self.label_columns],
#                 axis=-1,
#             )
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#         return inputs, labels

#     def _split(
#         self, features: tf.Tensor, latents: Optional[tf.Tensor] = None
#     ) -> Tuple[tf.Tensor, tf.Tensor]:
#         inputs = features[:, self.input_slice, :]
#         labels = features[:, self.labels_slice, :]
#         if self.label_columns:
#             labels = tf.stack(
#                 [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#                 axis=-1,
#             )
#         if latents is not None:
#             lat_tiled = tf.repeat(latents[:, tf.newaxis, :], repeats=self.input_width, axis=1)
#             inputs = tf.concat([inputs, lat_tiled], axis=-1)
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#         return inputs, labels

#     def _make_base_dataset(self, data: pd.DataFrame, batch_size: int, shuffle: bool) -> tf.data.Dataset:
#         arr = np.asarray(data, dtype=np.float32)
#         ds = tf.keras.utils.timeseries_dataset_from_array(
#             data=arr,
#             targets=None,
#             sequence_length=self.total_window_size,
#             sequence_stride=1,
#             sampling_rate=1,
#             shuffle=shuffle,
#             batch_size=batch_size,
#         )
#         return ds

#     def make_dataset(
#         self,
#         split: str,
#         encoder: Optional[tf.keras.Model] = None,
#         batch_size: int = 32,
#         shuffle: bool = False,
#     ) -> tf.data.Dataset:
#         df = getattr(self, f"{split}_df")
#         if df is None:
#             raise ValueError(f"No DataFrame for split '{split}'")
#         ds = self._make_base_dataset(df, batch_size, shuffle)
#         if encoder:
#             def map_fn(batch):
#                 _, _, z = encoder(batch, training=False)
#                 x, y = self._split(batch)
#                 return z, y
#             ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
#         else:
#             ds = ds.map(lambda batch: self._split(batch), num_parallel_calls=tf.data.AUTOTUNE)
#         if self.cache_to_disk:
#             ds = ds.cache(self.cache_to_disk + f".{split}")
#         else:
#             ds = ds.cache()
#         opts = tf.data.Options()
#         opts.experimental_deterministic = True # because we are doing stateful RNN training
#         opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#         ds = ds.with_options(opts)
#         return ds.prefetch(tf.data.AUTOTUNE)

#     # Properties kept for plotting/debugging; not used in TFRecord training path below.
#     @property
#     def train(self) -> tf.data.Dataset:
#         return self.make_dataset("train", batch_size=self._batch_size, shuffle=False)
#     @property
#     def val(self) -> Optional[tf.data.Dataset]:
#         if self.val_df is None:
#             return None
#         return self.make_dataset("val", batch_size=self._batch_size, shuffle=False)
#     @property
#     def test(self) -> Optional[tf.data.Dataset]:
#         if self.test_df is None:
#             return None
#         return self.make_dataset("test", batch_size=self._batch_size, shuffle=False)
#     @property
#     def example(self):
#         result = getattr(self, "_example", None)
#         if result is None:
#             result = next(iter(self.train))
#             self._example = result
#         return result
#     @property
#     def _batch_size(self) -> int:
#         return getattr(self, "__batch_size", 256)
#     @_batch_size.setter
#     def _batch_size(self, value: int):
#         setattr(self, "__batch_size", int(value))

# -----------------------------------------------------------------------------
# Build multi-step dense model
# -----------------------------------------------------------------------------
def build_multi_step_dense(n_outputs: int, label_width: int, n_labels: int, hparams: dict = None) -> keras.Model:
    """
    Dense forecaster with optional bottleneck projection, dropout, and L2 regularization.

    When bottleneck_dim > 0, a projection layer is inserted after Flatten to map all
    input dimensionalities to the same size. This equalizes effective model capacity
    regardless of the number of input features.

    Args:
        label_width: Number of steps to predict
        n_labels: Number of labels to predict
        n_outputs: Number of outputs to predict
        hparams: Hyperparameters dict; reads bottleneck_dim, dropout_rate, l2_reg

    Returns:
        keras.Model: The compiled model
    """
    hparams = hparams or {}
    bottleneck_dim = int(hparams.get("bottleneck_dim", 0))
    dropout_rate = float(hparams.get("dropout_rate", 0.0))
    l2_reg = float(hparams.get("l2_reg", 0.0))

    # Build regularizer if requested
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model_layers = [
        # Shape: (time, features) => (time*features)
        keras.layers.Flatten(),
    ]

    # Optional bottleneck: project flattened input to a fixed dimensionality
    if bottleneck_dim > 0:
        model_layers.append(keras.layers.Dense(bottleneck_dim, activation="relu", name="bottleneck"))
        if dropout_rate > 0:
            model_layers.append(keras.layers.Dropout(dropout_rate))

    # Hidden layers with optional L2 and dropout
    model_layers.append(keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer))
    if dropout_rate > 0:
        model_layers.append(keras.layers.Dropout(dropout_rate))

    model_layers.append(keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer))
    if dropout_rate > 0:
        model_layers.append(keras.layers.Dropout(dropout_rate))

    # Output layer (no regularization, no dropout)
    model_layers.append(keras.layers.Dense(label_width))

    return keras.Sequential(model_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)   # dir or file
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument(
        "--model_dir", type=str, default=os.getenv("AIP_MODEL_DIR", "model")
    )
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    parser.add_argument(
        "--predictions", type=str, default="",
        help="Optional path to write predictions CSV (sample_index, y_true_scaled, y_pred_scaled)",
    )
    args = parser.parse_args()

    # ============================================================
    # Convert GCS paths if present
    # Resolve model_dir for Vertex (gs:// -> /gcs/...)
    # ============================================================
    if args.model_dir.startswith("gs://"):
        args.model_dir = Path("/gcs/" + args.model_dir[5:])

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

    # ============================================================
    # Load datasets (local or GCS) with error logging
    # Now reads TFRecords written by the upstream splitter; no CSV parsing.
    # ============================================================
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
    label_columns = [hparams["label_name"]]  # typically "PRICE_std"
    selected_cols = list({*features})        # training uses standardized features

    # Build sequence datasets directly from TFRecords:
    # - Each element is a window of shape (seq_length, n_features)
    # - The forecasting target (label) is the *next step* of the same feature(s)
    #   and is created on-the-fly below to preserve the original meaning
    #   (label_width=1 next-step prediction).
    try:
        # train_windows = create_dataset(
        #     input_data=Path(args.train_data),
        #     label_name=hparams["label_name"],
        #     model_params=hparams,
        #     selected_columns=selected_cols,
        #     shuffle=True,
        # )
        # val_windows = create_dataset(
        #     input_data=Path(args.valid_data),
        #     label_name=hparams["label_name"],
        #     model_params=hparams,
        #     selected_columns=selected_cols,
        #     shuffle=False,
        # )
        # test_windows = create_dataset(
        #     input_data=Path(args.test_data),
        #     label_name=hparams["label_name"],
        #     model_params=hparams,
        #     selected_columns=selected_cols,
        #     shuffle=False,
        # )
        # train_ds = build_context_label_windows(
        #     input_data_dir=Path(args.train_data),
        #     selected_columns=selected_cols,
        #     label_columns=label_columns,
        #     input_width=int(hparams["input_width"]),
        #     label_width=int(hparams["label_width"]),
        #     shift=int(hparams["shift"]),
        #     sample_stride=int(hparams.get("sample_stride", 1)),
        #     batch_size=int(hparams["batch_size"]),
        #     shuffle=True,
        #     cache_to_disk=(hparams.get("cache_to_disk") or None) and (hparams["cache_to_disk"] + ".train"),
        # )

        # val_ds = build_context_label_windows(
        #     input_data_dir=Path(args.valid_data),
        #     selected_columns=selected_cols,
        #     label_columns=label_columns,
        #     input_width=int(hparams["input_width"]),
        #     label_width=int(hparams["label_width"]),
        #     shift=int(hparams["shift"]),
        #     sample_stride=int(hparams.get("sample_stride", 1)),
        #     batch_size=int(hparams["batch_size"]),
        #     shuffle=False,
        #     cache_to_disk=(hparams.get("cache_to_disk") or None) and (hparams["cache_to_disk"] + ".val"),
        # )

        # test_ds = build_context_label_windows(
        #     input_data_dir=Path(args.test_data),
        #     selected_columns=selected_cols,
        #     label_columns=label_columns,
        #     input_width=int(hparams["input_width"]),
        #     label_width=int(hparams["label_width"]),
        #     shift=int(hparams["shift"]),
        #     sample_stride=int(hparams.get("sample_stride", 1)),
        #     batch_size=int(hparams["batch_size"]),
        #     shuffle=False,
        #     cache_to_disk=(hparams.get("cache_to_disk") or None) and (hparams["cache_to_disk"] + ".test"),
        # )
        train_xy, x_spec_tr, y_spec_tr = load_windows(args.train_data, compression="GZIP")
        val_xy,   _, _ = load_windows(args.valid_data, compression="GZIP")
        test_xy,  _, _ = load_windows(args.test_data, compression="GZIP")

        input_width = x_spec_tr.shape[-2]  # input_width
        n_features = x_spec_tr.shape[-1]  # number of features
        label_width = y_spec_tr.shape[-1]  # label_width (forecast horizon)
        logging.info(f"input_width: {input_width}, n_features: {n_features}, label_width: {label_width}")

        # Determine output dimensionality (label_width * n_labels) for the final Dense
        # Update the model's output size to match multi-step/multi-label:
        n_labels = len(label_columns)
        n_outputs   = n_labels * label_width   # total scalars

        # For VAE reconstruction training, we only need X (drop Y)
        train_ds = (train_xy
            .shuffle(10000, reshuffle_each_iteration=True)
            .batch(hparams["batch_size"], drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

        val_ds = (val_xy
            .batch(hparams["batch_size"], drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

        test_ds = (test_xy
            .batch(hparams["batch_size"], drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

    except Exception:
        logging.exception(
            f"Failed reading TFRecords (train={args.train_data}, val={args.valid_data}, test={args.test_data})"
        )
        raise

    # ============================================================
    # Create (features, labels) pairs from window tensors
    # Original logic: label_width=1 forecasting of label_columns (e.g., PRICE_std)
    # Keep the same semantics by splitting the last time step as the label.
    # ============================================================
    # seq_len = int(hparams["seq_length"])
    # label_idx = [selected_cols.index(c) for c in label_columns] if label_columns else [selected_cols.index("PRICE_std")]

    # def split_x_y(window_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    #     """
    #     Given a batch of windows shaped (batch, seq_length, n_features),
    #     return (inputs, labels) where labels correspond to the **next-step**
    #     (last position in window) for the requested label columns.
    #     """
    #     # inputs: all timesteps except the last one if you want strict next-step,
    #     # but original dense forecaster expects (seq_len, features) -> predict 1 step.
    #     # We keep inputs as the full window (seq_len, n_features) as before.
    #     x = window_batch  # [B, T, F]
    #     last = window_batch[:, -1:, :]  # [B, 1, F]
    #     y = tf.gather(last, indices=label_idx, axis=-1)  # [B, 1, L]
    #     return x, y

    # train_ds = train_windows.map(split_x_y, num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds   = val_windows.map(split_x_y,   num_parallel_calls=tf.data.AUTOTUNE) if val_windows is not None else None
    # test_ds  = test_windows.map(split_x_y,  num_parallel_calls=tf.data.AUTOTUNE) if test_windows is not None else None

    # Quick integrity check (first batch)
    inspect_dataset_v2(train_ds, "train_ds", 1)
    if val_ds is not None:
        inspect_dataset_v2(val_ds, "val_ds", 1)
    if test_ds is not None:
        inspect_dataset_v2(test_ds, "test_ds", 1)

    

    # ============================================================
    # Build and compile model under distribution strategy
    # ============================================================
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    with strategy.scope():
        model = build_multi_step_dense(n_outputs=n_outputs, label_width=label_width, n_labels=n_labels, hparams=hparams)
        # AdamW is a good default; if mixed precision is enabled, Keras handles loss scaling
        optimizer = keras.optimizers.AdamW(learning_rate=hparams["learning_rate"])

        # Convert metric strings to actual Keras metric objects
        metrics_objects = []
        for metric_name in hparams["metrics"]:
            if metric_name == "RootMeanSquaredError":
                metrics_objects.append(keras.metrics.RootMeanSquaredError())
            elif metric_name == "MeanAbsoluteError":
                metrics_objects.append(keras.metrics.MeanAbsoluteError())
            elif metric_name == "MeanAbsolutePercentageError":
                metrics_objects.append(keras.metrics.MeanAbsolutePercentageError())
            elif metric_name == "MeanSquaredLogarithmicError":
                metrics_objects.append(keras.metrics.MeanSquaredLogarithmicError())
            else:
                logging.warning(f"Unknown metric: {metric_name}")

        model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            optimizer=optimizer,
            metrics=metrics_objects,
            # jit_compile=True,  # Alternative per-model JIT; measure before/after
        )

    # ============================================================
    # Setup callbacks
    # ============================================================
    # Callbacks
    # https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements#tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'],
        histogram_freq=1
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=hparams["patience"],
        restore_best_weights=True,
    )
    #  TO DO: Model Checkpoint callback
    # https://docs.cloud.google.com/vertex-ai/docs/training/code-requirements#resilience
    callbacks = [tensorboard_callback, early_stopping]

    # ============================================================
    # Train the model
    # ============================================================
    logging.info("Starting training...")
    model.fit(
        train_ds,                      # already batched dataset
        epochs=hparams["epochs"],      # dataset is finite; epochs control total passes
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ============================================================
    # Chief-only save/eval (for multi-worker)
    # Only persist output if this worker is chief
    # ============================================================
    if not _is_chief(strategy):
        logging.info("Non-chief worker: exiting without saving model/metrics.")
        sys.exit(0)

    # ============================================================
    # Save model
    # ============================================================
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = f"{args.model_dir}/forecast_engine_{hparams.get('threshold','default')}.keras"
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # ============================================================
    # Save evaluation metrics on test set (KFP artifact + JSON file)
    # ============================================================
    import math

    # Evaluate on test set
    eval_metrics = model.evaluate(test_ds, verbose=0, return_dict=True) if test_ds is not None else {}
    logging.info(f"Evaluation (return_dict): {eval_metrics}")

    def safe_float(v):
        """Convert to float if possible and finite, else None."""
        if v is None:
            return None
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    # Human-friendly full JSON
    pretty_metrics = {
        "problemType": "regression",
        "rootMeanSquaredError":        safe_float(eval_metrics.get("root_mean_squared_error")) or safe_float(eval_metrics.get("rmse")) or safe_float(eval_metrics.get("RootMeanSquaredError")),
        "meanAbsoluteError":           safe_float(eval_metrics.get("mean_absolute_error")) or safe_float(eval_metrics.get("mae")) or safe_float(eval_metrics.get("MeanAbsoluteError")),
        "meanAbsolutePercentageError": safe_float(eval_metrics.get("mean_absolute_percentage_error")) or safe_float(eval_metrics.get("mape")) or safe_float(eval_metrics.get("MeanAbsolutePercentageError")),
        "rootMeanSquaredLogError":     safe_float(eval_metrics.get("mean_squared_logarithmic_error")) or safe_float(eval_metrics.get("msle")) or safe_float(eval_metrics.get("MeanSquaredLogarithmicError")),
        "rSquared":                    safe_float(eval_metrics.get("r2")) or safe_float(eval_metrics.get("r_squared")) or safe_float(eval_metrics.get("R2")),
        "klLoss":                      safe_float(eval_metrics.get("kl_loss")),
        "reconstructionLoss":          safe_float(eval_metrics.get("reconstruction_loss")) or safe_float(eval_metrics.get("loss")),
        "totalLoss":                   safe_float(eval_metrics.get("total_loss")) or safe_float(eval_metrics.get("loss")),
    }
    # Save metrics to JSON file
    with open(args.metrics, "w") as fp:
        json.dump(pretty_metrics, fp)
    logging.info(f"Metrics saved to {args.metrics}")


    # Log which metrics were found
    logging.info(f"Extracted metrics: {pretty_metrics}")

    # ============================================================
    # Generate predictions on test set and save to CSV (optional)
    # ============================================================
    if args.predictions:
        logging.info("Generating predictions on test set...")
        y_true_list = []
        y_pred_list = []
        ref_price_list = []
        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch, verbose=0)
            y_true_list.append(y_batch.numpy().reshape(-1))
            y_pred_list.append(preds.reshape(-1))
            # Extract last input PRICE_std as reference for directional accuracy.
            # PRICE_std is always feature index 0 (first in feature_names list).
            ref_price_list.append(x_batch[:, -1, 0].numpy().reshape(-1))
        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        ref_price_all = np.concatenate(ref_price_list)

        pred_df = pd.DataFrame({
            "sample_index": np.arange(len(y_true_all)),
            "y_true_scaled": y_true_all,
            "y_pred_scaled": y_pred_all,
            "reference_price": ref_price_all,
        })
        predictions_path = args.predictions
        os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
        pred_df.to_csv(predictions_path, index=False)
        logging.info(f"Predictions saved to {predictions_path} ({len(pred_df)} samples)")

        # Add metadata to metrics JSON
        pretty_metrics["total_params"] = int(model.count_params())
        pretty_metrics["n_features"] = int(n_features)
        pretty_metrics["input_width"] = int(input_width)
        pretty_metrics["model_label"] = hparams.get("model_label", "unknown")
        with open(args.metrics, "w") as fp:
            json.dump(pretty_metrics, fp)
        logging.info(f"Updated metrics with prediction metadata: {args.metrics}")

    # ============================================================
    # Save training dataset info for model monitoring
    # ============================================================
    path = os.path.join(args.model_dir, TRAINING_DATASET_INFO)
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "tfrecord",   # UPDATED: reflect actual format
        "threshold": hparams.get("threshold", "default"),
        "targetField": hparams["label_name"],
    }
    logging.info(f"Saving training dataset info for model monitoring: {path}")
    logging.info(f"Training dataset: {training_dataset_for_monitoring}")
    with tf.io.gfile.GFile(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)
    logging.info(f"Training dataset info saved to {path}")


if __name__ == "__main__":
    main()
