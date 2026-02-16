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

text = "VAE Training Script"
logging.warning(text)
# used for monitoring during prediction time
TRAINING_DATASET_INFO = "training_dataset.json"

# numeric/categorical features in Chicago trips dataset to be preprocessed
NUM_COLS = ["dayofweek", "hourofday", "trip_distance", "trip_miles", "trip_seconds"]
ORD_COLS = ["company"]
OHE_COLS = ["payment_type"]

# Default hyperparameters
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
    patience=10,
    metrics=[
        "RootMeanSquaredError",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredLogarithmicError",
    ],
    hidden_units=[(64, "relu"), (32, "relu")],
    distribute_strategy="single",
    early_stopping_epochs=3,
    label="PRICE_std",
    use_mixed_precision=True,          # only takes effect if a GPU is present
    use_xla_jit=True,                  # try XLA; disable if it hurts a given model
)

logging.getLogger().setLevel(logging.INFO)


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


def create_adapt_dataset(
    input_data: Path,
    label_name: str,
    model_params: dict,
    selected_columns: list[str],
) -> Dataset:
    """
    Clean, single-pass dataset for .adapt() to avoid re-scanning the shuffled train DS.
    """
    logging.info("Building dataset for preprocessing .adapt() passes...")
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=str(input_data),
        batch_size=max(model_params["batch_size"], 1024),  # big batches reduce passes
        label_name=label_name,
        num_epochs=1,
        shuffle=False,
        num_rows_for_inference=20000,
        num_parallel_reads=tf.data.AUTOTUNE,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        select_columns=selected_columns,
    )
    return ds.cache().prefetch(tf.data.AUTOTUNE)


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
    x.adapt(dataset.map(lambda y, _: y[name], num_parallel_calls=tf.data.AUTOTUNE))
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
    # If vocab is huge, comment the next line to avoid slow logs:
    # logging.debug(f"Vocabulary({name}): {x.get_vocabulary()}")
    return x


def build_and_compile_model(dataset: Dataset, model_params: dict) -> Model:
    """Build and compile model.
    Args:
        dataset (Dataset): training dataset
        model_params (dict): model parameters
    Returns:
        model (Model): built and compiled model
    """

    # create inputs (scalars with shape `()`)
    num_ins = {name: Input(shape=(), name=name, dtype=tf.float32) for name in NUM_COLS}
    ord_ins = {name: Input(shape=(), name=name, dtype=tf.string) for name in ORD_COLS}
    cat_ins = {name: Input(shape=(), name=name, dtype=tf.string) for name in OHE_COLS}

    # join all inputs and expand by 1 dimension. NOTE: this is useful when passing
    # in scalar inputs to a model in Vertex AI batch predictions or endpoints e.g.
    # `{"instances": {"input1": 1.0, "input2": "str"}}` instead of
    # `{"instances": {"input1": [1.0], "input2": ["str"]}`
    all_ins = {**num_ins, **ord_ins, **cat_ins}
    exp_ins = {n: tf.expand_dims(i, axis=-1) for n, i in all_ins.items()}

    # preprocess expanded inputs
    num_encoded = [normalization(n, dataset)(exp_ins[n]) for n in NUM_COLS]
    ord_encoded = [str_lookup(n, dataset, "int")(exp_ins[n]) for n in ORD_COLS]
    ohe_encoded = [str_lookup(n, dataset, "one_hot")(exp_ins[n]) for n in OHE_COLS]

    # ensure ordinal encoded layers is of type float32 (like the other layers)
    ord_encoded = [tf.cast(x, tf.float32) for x in ord_encoded]

    # concat encoded inputs and add dense layers including output layer
    x = num_encoded + ord_encoded + ohe_encoded
    x = Concatenate()(x)
    for units, activation in model_params["hidden_units"]:
        x = Dense(units, activation=activation)(x)
    x = Dense(1, name="output", activation="linear")(x)

    model = Model(inputs=all_ins, outputs=x, name="nn_model")
    model.summary()

    logging.info(f"Use optimizer {model_params['optimizer']}")
    optimizer = optimizers.get(model_params["optimizer"])
    optimizer.learning_rate = model_params["learning_rate"]

    # Optional accelerations
    if model_params.get("use_xla_jit", True):
        try:
            tf.config.optimizer.set_jit(True)
            logging.info("XLA JIT enabled")
        except Exception as e:
            logging.warning(f"Could not enable XLA JIT: {e}")

    if model_params.get("use_mixed_precision", True) and tf.config.list_physical_devices("GPU"):
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            logging.info("Mixed precision enabled (GPU)")
        except Exception as e:
            logging.warning(f"Could not enable mixed precision: {e}")

    model.compile(
        loss=model_params["loss_fn"],
        optimizer=optimizer,
        metrics=model_params["metrics"],
        # If using TF>=2.11, you can also try jit_compile=True per-model:
        # jit_compile=True,
    )
    return model

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
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
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
    parser.add_argument("--model", type=str, default=os.getenv("AIP_MODEL_DIR"), help="model")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    args = parser.parse_args()

    # Vertex AI model dir: write to /gcs/ mount if given as gs://
    if args.model and args.model.startswith("gs://"):
        args.model = Path("/gcs/" + args.model[5:])
    else:
        args.model = Path(args.model)

    # Merge user hparams by overwriting default_model_params if provided in model_params
    hparams = {**DEFAULT_HPARAMS, **args.hparams}
    logging.info(f"Using model hyper-parameters: {hparams}")
    label = hparams["label"]

    # Use only the columns we need (+ label)
    SELECTED_COLUMNS = list({label, *NUM_COLS, *ORD_COLS, *OHE_COLS})
    features = [
        'start_price_std',
        'PRICE_std',
        'PDCC_Down_std',
        'OSV_Down_std',
        'PDCC2_UP_std',
        'OSV_Up_std'
    ]
    label_columns = ["PRICE_std"]

    # Set distribute strategy before any TF operations
    strategy = get_distribution_strategy(hparams["distribute_strategy"])
    logging.info(f"Using strategy: {type(strategy).__name__}")

    # Build datasets
    adapt_ds = create_adapt_dataset(Path(args.train_data), label, hparams, SELECTED_COLUMNS) # for preprocessing .adapt() passes
    train_ds = create_dataset(Path(args.train_data), label, hparams, SELECTED_COLUMNS, shuffle=True) # for training
    valid_ds = create_dataset(Path(args.valid_data), label, hparams, SELECTED_COLUMNS, shuffle=False) # for validation
    test_ds = create_dataset(Path(args.test_data),  label, hparams, SELECTED_COLUMNS, shuffle=False) # for evaluation

    # Sanity log
    train_features = list(train_ds.element_spec[0].keys())
    valid_features = list(valid_ds.element_spec[0].keys())
    logging.info(f"Training feature names: {train_features}")
    logging.info(f"Validation feature names: {valid_features}")

    if len(train_features) != len(valid_features):
        raise RuntimeError("No. of training features != # validation features")

    # Build model (adapt on clean dataset)
    with strategy.scope():
        tf_model = build_and_compile_model(adapt_ds, hparams)

    # Early stopping on validation for real stopping power
    logging.info("Use early stopping (on val_loss)")
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=hparams["early_stopping_epochs"],
        restore_best_weights=True
    )

    logging.info("Fit model...")
    history = tf_model.fit(
        train_ds,
        epochs=hparams["epochs"],
        validation_data=valid_ds,
        callbacks=[callback],
        verbose=1,
    )

    # only persist output files if current worker is chief
    if not _is_chief(strategy):
        logging.info("Not chief node, exiting now")
        sys.exit()

    # Save model
    logging.info(f"Save model to: {args.model}")
    args.model.mkdir(parents=True, exist_ok=True)
    tf_model.save(str(args.model), save_format="tf")

    # Evaluate & write metrics
    logging.info(f"Save metrics to: {args.metrics}")
    eval_names = tf_model.metrics_names
    eval_values = tf_model.evaluate(test_ds, verbose=0)
    eval_metrics = dict(zip(eval_names, eval_values))

    metrics = {
        "problemType": "regression",
        "rootMeanSquaredError": float(eval_metrics.get("root_mean_squared_error", 0.0)),
        "meanAbsoluteError": float(eval_metrics.get("mean_absolute_error", 0.0)),
        "meanAbsolutePercentageError": float(eval_metrics.get("mean_absolute_percentage_error", 0.0)),
        "rSquared": None,
        "rootMeanSquaredLogError": float(eval_metrics.get("mean_squared_logarithmic_error", 0.0)),
    }

    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp)

    # Persist URIs of training file(s) for model monitoring in batch predictions
    # See https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.TrainingDataset  # noqa: E501
    # for the expected schema.
    path = args.model / TRAINING_DATASET_INFO
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "csv",
        "targetField": label,
    }
    logging.info(f"Save training dataset info for model monitoring: {path}")
    logging.info(f"Training dataset: {training_dataset_for_monitoring}")

    with open(path, "w") as fp:
        json.dump(training_dataset_for_monitoring, fp)

if __name__ == "__main__":
    main()

