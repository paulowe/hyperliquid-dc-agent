from kfp.v2.dsl import component, Input, Output, Dataset, Model
from typing import NamedTuple
from typing import List
import os, datetime


@component(
    base_image="python:3.11",
    packages_to_install=[
        # Upgrade pip/setuptools so we get the latest PEP 517 hooks
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        # Force Cython < 3 so build‐time hooks don’t look for the removed API
        "cython<3.0.0",
        # Install PyYAML without isolation so it sees our Cython
        "--no-build-isolation",
        "pyyaml==5.4.1",
        # Other runtime deps
        "IPython",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "tensorflow",
    ],
    # base_image="us-central1-docker.pkg.dev/derivatives-417104/dc-vae-kfp-base-repo/dc-vae-base:2025-06-25",
    # install_kfp_package=False,
    # packages_to_install=[]
)
def window_generator(
    dataset: Input[Dataset],
    input_width: int,
    label_width: int,
    shift: int,
    label_columns: List[str],
    windowed_dataset: Output[Dataset],
):
    # Define WindowGenerator class and use it to create windowed dataset splits.
    import os, datetime
    import IPython, IPython.display
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf

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
            train_df,
            val_df,
            test_df,
            label_columns: List[str] | None = None,
        ):
            # Store the raw dataframes
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            # If the user specifies label_columns, map their names to indices
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {
                    name: i for i, name in enumerate(label_columns)
                }
            # Map all column names to indices for convenient slicing
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

            # Window dimensions
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            # Total size of each window (inputs + gap to labels)
            self.total_window_size = input_width + shift

            # Define the slice object and index arrays for inputs
            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            # Define the slice object and index arrays for labels
            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        def split_window(self, features):
            """
            Split a full window of features into (inputs, labels) pairs.

            Args:
              features: 3D array of shape (batch, total_window_size, num_features),
                        typically from tf.data.Dataset or a NumPy stack.

            Returns:
              inputs: 3D tensor of shape (batch, input_width, num_features)
              labels: 3D tensor of shape (batch, label_width, num_label_columns)
            """
            # Extract the input slice from the full window
            inputs = features[:, self.input_slice, :]

            # Extract the label slice from the full window
            labels = features[:, self.labels_slice, :]

            # If specific label columns are set, select only those features
            if self.label_columns is not None:
                labels = tf.stack(
                    [
                        labels[:, :, self.column_indices[name]]
                        for name in self.label_columns
                    ],
                    axis=-1,
                )

            # After slicing, TensorFlow loses static shape info, so we set it manually:
            # - batch size is unknown (None)
            # - time steps for inputs and labels as defined
            # - feature dimension remains unspecified (None)
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])

            return inputs, labels

        def plot(self, model=None, plot_col="T (degC)", max_subplots=3):
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

                plt.scatter(
                    self.label_indices,
                    labels[n, :, label_col_index],
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(
                        self.label_indices,
                        predictions[n, :, label_col_index],
                        marker="X",
                        edgecolors="k",
                        label="Predictions",
                        c="#ff7f0e",
                        s=64,
                    )

                if n == 0:
                    plt.legend()
            plt.xlabel("Time [h]")

        def make_dataset(self, data):
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,
            )

            ds = ds.map(self.split_window)

            return ds

        @property
        def train(self):
            return self.make_dataset(self.train_df)

        @property
        def val(self):
            return self.make_dataset(self.val_df)

        @property
        def test(self):
            return self.make_dataset(self.test_df)

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
                    f"Label indices: {self.label_indices}",
                    f"Label column name(s): {self.label_columns}",
                ]
            )

    # Start here

    # dataset: Input[Dataset],
    # input_width: int,
    # label_width: int,
    # shift: int,
    # label_columns: List[str],
    # windowed_dataset: Output[Dataset],
    features = ["PRICE", "PDCC_Down", "OSV_Down", "PDCC2_UP", "OSV_Up"]
    train_df = pd.read_csv(dataset.path)
    val_df = None
    test_df = None

    w = WindowGenerator(
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        train_df=train_df[features],
        val_df=val_df,
        test_df=test_df,
        label_columns=label_columns,
    )

    print(w)

    # Split window w1, w2, w3, ...
    # example_inputs, example_labels = w2.split_window(example_window)

    # Print the structure, data types, and shapes of the dataset elements
    print(w.train.element_spec)

    for example_inputs, example_labels in w.train.take(1):
        print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
        print(f"Labels shape (batch, time, features): {example_labels.shape}")

    inputs_list, labels_list = [], []
    for x_batch, y_batch in w.train:
        inputs_list.append(x_batch.numpy())
        labels_list.append(y_batch.numpy())
    X = np.concatenate(inputs_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)

    # Save to the component’s output path
    # .npz preserves dtype & shape, and compresses automatically
    np.savez(windowed_dataset.path, inputs=X, labels=Y)
    # now windowed_dataset.path points at a .npz that can consume downstream!
