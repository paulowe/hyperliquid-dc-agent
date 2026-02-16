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
from tensorflow.keras.layers import Normalization, StringLookup

# from tensorflow.keras import Input, Model, layers, optimizers  # unused
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
)

# Helper function to create dataset


def create_dataset(input_data: Path, label_name: str, model_params: dict) -> Dataset:
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
    created_dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=str(input_data),
        batch_size=model_params["batch_size"],
        label_name=label_name,
        num_epochs=model_params["epochs"],
        shuffle=True,
        shuffle_buffer_size=1000,
        num_rows_for_inference=20000,
    )
    return created_dataset.with_options(data_options)


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


# Build multi-step dense model


def build_multi_step_dense():
    return tf.keras.Sequential(
        [
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ]
    )


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
    # Setup distributed training strategy
    # ============================================================
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # ============================================================
    # Load datasets with error logging
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
    label_columns = ["PRICE_std"]

    try:
        train_df = pd.read_csv(f"{args.train_data}/data.csv")[features]
        val_df = pd.read_csv(f"{args.valid_data}/data.csv")[features]
        test_df = pd.read_csv(f"{args.test_data}/data.csv")[features]
    except Exception as e:
        logging.exception(
            "Failed to load one of the datasets: train=%s, valid=%s, test=%s",
            args.train_data,
            args.valid_data,
            args.test_data,
        )
        raise

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

    train_ds = window.train.map(lambda x, y: x)
    val_ds = window.val.map(lambda x, y: x)
    test_ds = window.test.map(lambda x, y: x)

    # ============================================================
    # Build and compile VAE under distribution strategy
    # ============================================================
    with strategy.scope():
        model = build_multi_step_dense()
        optimizer = keras.optimizers.AdamW(learning_rate=hparams["learning_rate"])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

    # ============================================================
    # Setup callbacks
    # ============================================================
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=hparams["patience"],
        restore_best_weights=True,
    )
    callbacks = [early_stopping]

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
    model.fit(
        window.train,
        epochs=hparams["epochs"],
        validation_data=window.val,
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
    model.save(f"{args.model_dir}/forecast_engine_{hparams['threshold']}.keras")
    logging.info(f"Model saved to {args.model_dir}")

    # ============================================================
    # Save evaluation metrics on test set
    # ============================================================
    eval_metrics = dict(zip(model.metrics_names, model.evaluate(window.test)))
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
        "targetField": "PRICE_std",
    }
    logging.info(f"Saving training dataset info for model monitoring: {path}")
    logging.info(f"Training dataset: {training_dataset_for_monitoring}")

    with open(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)


if __name__ == "__main__":
    main()
