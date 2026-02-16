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
from tensorflow.keras import Input, Model, layers, optimizers
from tensorflow.keras.layers import Dense, Normalization, StringLookup, Concatenate
from tensorflow.data import Dataset
from matplotlib import pyplot as plt
from typing import Optional, List, Dict, Tuple

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
    latent_dim=3,
    seq_length=50,
    n_features=6,
    kl_init_weight=1e-4,
    patience=5, # early stopping patience
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
    # Make a *finite* (one pass) tf.data pipeline with parallel reads + prefetch.
    # Keras `fit(epochs=...)` controls total passes.
    
    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )


    logging.info(f"Creating dataset from CSV file(s) at {input_data}...")
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=str(input_data),
        batch_size=model_params["batch_size"],
        label_name=label_name,
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
    """Robust CSV reader that works for local files or GCS URIs via tf.io.gfile."""
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


# Window Generator


class WindowGenerator:
    """
    Utility to generate input/label windows from time-series DataFrames for model training.

    Splits each of train, validation, and test DataFrames into overlapping
    windows of fixed length, returning slices of inputs and corresponding labels.

    Optionally supports selecting only certain label columns.

    Attributes:
        train_df (pd.DataFrame): Full training data.
        val_df   (pd.DataFrame): Full validation data.
        test_df  (pd.DataFrame): Full test data.
        input_width      (int): Number of time steps in each input window.
        label_width      (int): Number of time steps to predict.
        shift            (int): Offset between end of input window and start of label window.
        total_window_size(int): input_width + shift.
        input_slice      (slice): Python slice for input indices [0:input_width].
        labels_slice     (slice): Python slice for label indices [label_start:].
        input_indices    (np.ndarray): Array of indices for inputs within the total window.
        label_indices    (np.ndarray): Array of indices for labels within the total window.
        column_indices   (dict): Maps each column name to its integer index.
        label_columns    (list[str] or None): Names of columns to be used as labels.
        label_columns_indices (dict): Maps each label column name to its index.
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
        cache_to_disk: Optional[str] = True,
    ):
        # Raw splits
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Label column indices
        self.label_columns = label_columns or []
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(self.label_columns)
            }
        else:
            self.label_columns_indices = {}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift  # Includes label width

        # Total window size
        self.total_window_size = input_width + shift

        # Input slice
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Cache to disk
        self.cache_to_disk = cache_to_disk

    # Split a batch of windows into (inputs, labels) tensors
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
        Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels.
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
        # Convert to float32 avoid copies when possible.
        arr = np.asarray(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=arr,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return ds

    def make_dataset(
        self,
        split: str,
        encoder: Optional[tf.keras.Model] = None,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """
        Construct a tf.data.Dataset, optionally mapping through a VAE encoder.

        Returns a tf.data.Dataset of (inputs, labels) or (latent, labels) for a given split.

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

        # Cache small/medium datasets; time-series windows for 30k rows are typically safe to cache.
        if self.cache_to_disk:
            ds = ds.cache(self.cache_to_disk + f".{split}")
        else:
            ds = ds.cache()

        # tf.data performance options (shared) for all datasets
        opts = tf.data.Options()
        # Let fast elements overtake slow ones to improve throughput
        opts.experimental_deterministic = True # because we are doing stateful RNN training
        # If you scale out, allow data sharding by element. Use "FILE"/"OFF" on multi-worker
        opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(opts)

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
        return self.make_dataset("train", batch_size=self._batch_size, shuffle=False)

    @property
    def val(self) -> Optional[tf.data.Dataset]:
        if self.val_df is None:
            return None
        return self.make_dataset("val", batch_size=self._batch_size, shuffle=False)

    @property
    def test(self) -> Optional[tf.data.Dataset]:
        if self.test_df is None:
            return None
        return self.make_dataset("test", batch_size=self._batch_size, shuffle=False)    
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
    # Allow external control of the batch size after construction
    @property
    def _batch_size(self) -> int:
        return getattr(self, "__batch_size", 256)

    @_batch_size.setter
    def _batch_size(self, value: int):
        setattr(self, "__batch_size", int(value))

# -----------------------------------------------------------------------------
# Build multi-step dense model
# -----------------------------------------------------------------------------
def build_multi_step_dense(n_outputs: int) -> keras.Model:
    """
    Simple dense forecaster;

    Args:
        n_outputs: Number of outputs to predict

    Returns:
        keras.Model: The compiled model
    """
    return keras.Sequential(
        [
            # Shape: (time, features) => (time*features)
            keras.layers.Flatten(), # (time, features) -> (time*features)
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(n_outputs), # predict label_width * n_labels
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            keras.layers.Reshape([1, -1]),
        ]
    )


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
    selected_cols = list({*features})        # (already includes label)

    def _resolve_file(path_like: str) -> str:
        # Accept "dir" or "dir/file.csv"; default to "data.csv" inside the dir.
        if path_like.endswith(".csv"):
            return path_like
        return os.path.join(path_like, "data.csv")

    train_path = _resolve_file(args.train_data)
    val_path   = _resolve_file(args.valid_data)
    test_path  = _resolve_file(args.test_data)

    try:
        train_df = read_csv_any(train_path, usecols=selected_cols)
        val_df   = read_csv_any(val_path,   usecols=selected_cols)
        test_df  = read_csv_any(test_path,  usecols=selected_cols)
    except Exception:
        logging.exception(f"Failed reading CSVs (train={train_path}, val={val_path}, test={test_path})")
        raise

    # Ensure float32 (saves memory; matches TF default)
    train_df = train_df.astype(np.float32)
    val_df   = val_df.astype(np.float32)
    test_df  = test_df.astype(np.float32)

    # ============================================================
    # Create windowed datasets
    # ============================================================
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    window = WindowGenerator(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_width=hparams["seq_length"],
        shift=1,
        label_width=1,
        label_columns=label_columns,
        cache_to_disk=(hparams["cache_to_disk"] if isinstance(hparams["cache_to_disk"], str) else None),
    )
    window._batch_size = hparams["batch_size"]

    train_ds = window.train
    val_ds   = window.val
    test_ds  = window.test

    # ============================================================
    # Quick integrity check (first batch)
    # Inspect datasets for None/NaN before training
    # ============================================================
    inspect_dataset_v2(train_ds, "train_ds", 1)
    if val_ds is not None:
        inspect_dataset_v2(val_ds, "val_ds", 1)
    if test_ds is not None:
        inspect_dataset_v2(test_ds, "test_ds", 1)

    # Determine output dimensionality (label_width * n_labels) for the final Dense
    n_labels = len(label_columns) if label_columns else 1
    n_outputs = n_labels  # label_width=1; expand if you forecast multiple steps per sample

    # ============================================================
    # Build and compile VAE under distribution strategy
    # ============================================================
    with strategy.scope():
        model = build_multi_step_dense(n_outputs=n_outputs)
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
            loss=keras.losses.MeanSquaredError(),
            optimizer=optimizer,
            metrics=metrics_objects,
            # jit_compile=True,  # Alternative per-model JIT; measure before/after
        )

    # ============================================================
    # Setup callbacks
    # ============================================================
    early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=hparams["patience"], 
            restore_best_weights=True,
        )
    callbacks = [
        early_stopping
    ]
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
    model_path = f"{args.model_dir}/forecast_engine_{hparams['threshold']}.keras"
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
    
    # Log which metrics were found
    logging.info(f"Extracted metrics: {pretty_metrics}")

    # Write the metrics to a JSON file (debug-friendly)
    metrics_path = os.path.join(args.model_dir, "forecast_eval_metrics_test_set.json")
    with tf.io.gfile.GFile(metrics_path, "w") as f:
        json.dump(pretty_metrics, f, indent=2, allow_nan=False)
    logging.info(f"Raw test set eval metrics written to {metrics_path}")

    # KFP Metrics artifact (only numeric values the UI will show)
    kfp_pairs = {
        "rmse": pretty_metrics["rootMeanSquaredError"],
        "mae":  pretty_metrics["meanAbsoluteError"],
        "mape": pretty_metrics["meanAbsolutePercentageError"],
        "msle": pretty_metrics["rootMeanSquaredLogError"],
        "loss": pretty_metrics["totalLoss"],
        # "r2":   pretty_metrics["rSquared"],   # uncomment if you compute it
        # "loss": pretty_metrics["totalLoss"],
    }

    kfp_payload = {
        "metrics": [
            {"name": name, "numberValue": val, "format": "RAW"}
            for name, val in kfp_pairs.items() if val is not None
        ]
    }

    with tf.io.gfile.GFile(args.metrics, "w") as f:
        json.dump(kfp_payload, f, allow_nan=False)
    logging.info(f"KFP metrics written to {args.metrics}")


    # ============================================================
    # Save training dataset info for model monitoring
    # ============================================================
    path = os.path.join(args.model_dir, TRAINING_DATASET_INFO)
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "csv",
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

