from kfp.v2.dsl import Input, component, Metrics, Output, HTML, Dataset
from typing import List

@component(base_image="python:3.10",
           packages_to_install=["pandas", "numpy", "tensorflow", "google-cloud-logging", "matplotlib"])
def window_dataset(
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    train_windowed: Output[Dataset],
    valid_windowed: Output[Dataset],
    test_windowed: Output[Dataset],
    plots_output: Output[HTML],
    metrics: Output[Metrics],
    feature_names: List[str] = [
        "PRICE_std",
        "vol_quote_std",
        "cvd_quote_std",
        "PDCC_Down",
        "OSV_Down_std",
        "PDCC2_UP",
        "regime_up",
        "regime_down",
    ],
    label_columns: List[str] = ["PRICE_std"],
    input_width: int = 50,
    label_width: int = 1,
    shift: int = 50,
    batch_size: int = 32,
):
    from typing import Dict, List, Optional
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import os
    import logging
    import google.cloud.logging

    import io
    import base64

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Set up logging
    client = google.cloud.logging.Client()
    client.setup_logging()

    def _tfrecord_feature_spec(
        float_feature_names: List[str],
    ) -> Dict[str, tf.io.FixedLenFeature]:
        """
        Build a TFRecord feature spec for parsing float features.

        Parameters
        ----------
        float_feature_names : list of str
            Names of scalar float features stored in each Example. Each will be
            parsed as a `tf.float32` FixedLenFeature of shape [] (scalar).

        Returns
        -------
        dict[str, tf.io.FixedLenFeature]
            A mapping `{feature_name: tf.io.FixedLenFeature([], tf.float32)}` that
            can be passed to `tf.io.parse_single_example`.

        Notes
        -----
        - Only float features are included. If you later need string timestamps
        (e.g., `"start_time"`), add:
        `spec["start_time"] = tf.io.FixedLenFeature([], tf.string)`.

        Examples
        --------
        >>> spec = _tfrecord_feature_spec(["PRICE_std", "vol_quote_std"])
        >>> isinstance(spec["PRICE_std"], tf.io.FixedLenFeature)
        True
        >>> spec["PRICE_std"].dtype == tf.float32
        True
        """
        # We parse only what we need for training; string timestamps are ignored.
        spec = {name: tf.io.FixedLenFeature([], tf.float32) for name in float_feature_names}
        # If you ever need timestamps, you can add:
        # spec["start_time"] = tf.io.FixedLenFeature([], tf.string)
        # spec["load_time_toronto"] = tf.io.FixedLenFeature([], tf.string)
        return spec

    def _parse_example(serialized, feature_spec, feature_order: List[str]):
        """
        Parse a single serialized TF Example into a fixed-order feature vector.

        Parameters
        ----------
        serialized : tf.Tensor (scalar string)
            A single serialized `tf.train.Example` proto.
        feature_spec : dict
            Mapping of feature names to `tf.io.FixedLenFeature` (e.g. from
            `_tfrecord_feature_spec`).
        feature_order : list of str
            The exact order of feature names to stack into the output vector.

        Returns
        -------
        tf.Tensor
            Tensor of shape `[n_features]` (`tf.float32`), where `n_features = len(feature_order)`.

        Notes
        -----
        - Ensures deterministic column ordering by stacking features in `feature_order`.
        - Suitable for feeding into subsequent windowing ops.

        Examples
        --------
        >>> spec = _tfrecord_feature_spec(["PRICE_std", "vol_quote_std"])
        >>> # ds_raw yields serialized Examples (strings)
        >>> # x = ds_raw.map(lambda s: _parse_example(s, spec, ["PRICE_std", "vol_quote_std"]))
        >>> # `x` elements will each have shape [2] (PRICE_std, vol_quote_std)
        """
        parsed = tf.io.parse_single_example(serialized, feature_spec)
        # Assemble features in a fixed order -> vector [n_features]
        x = tf.stack([parsed[name] for name in feature_order], axis=-1)
        return x  # shape: [n_features], dtype float32


    def _make_forecasting_dataset_from_tfrecord_dir(
        data_dir: str,
        *,
        input_width: int,          # L
        label_width: int,          # H (multi-step length); use 1 for single-point
        feature_names: List[str],
        label_name: str,           # e.g. hparams["label_name"] == "PRICE_std"
        shift: int = 0,            # gap between input end and label start (0 => labels start immediately after inputs)
        batch_size: int = 128,
        shuffle_sequences: bool = True,
        compression_type: str = "GZIP",
    ) -> tf.data.Dataset:
        """
        Create a windowed forecasting dataset `(x, y)` from TFRecord shards.

        Parameters
        ----------
        data_dir : str
            Directory containing `*.tfrecord[.gz]` shards.
        input_width : int
            Number of timesteps in the input window `L`.
        label_width : int
            Number of future timesteps to predict `H` (multi-step horizon).
            This **is** the `H` you see in `y` shape `[batch, H]`.
        feature_names : list of str
            All float feature names to parse. Order defines column order of `x`.
        label_name : str
            Name of the **single** target feature to extract for `y`.
        shift : int, default 0
            Number of timesteps between end of the input window and the start of labels.
            `shift=0` makes labels start immediately after inputs.
        batch_size : int, default 128
            Batch size after windowing.
        shuffle_sequences : bool, default True
            If True, shuffle windowed sequences (use False for deterministic inspection).
        compression_type : {"GZIP", None}, default "GZIP"
            Compression used when writing TFRecords.

        Returns
        -------
        tf.data.Dataset
            Dataset yielding tuples `(x, y)` where:
            - `x` has shape `[batch, input_width, n_features]`
            - `y` has shape `[batch, label_width]` (only the single target feature)

        Windowing Semantics
        -------------------
        Let `L = input_width`, `H = label_width` and `S = shift`.
        Each contiguous **chunk** has length `L + S + H`.
        We split it as:
        inputs: indices `[0 .. L-1]`
        gap:    indices `[L .. L+S-1]` (ignored)
        labels: indices `[L+S .. L+S+H-1]`  (from `label_name` only)
        """
        total_window = input_width + shift + label_width
        label_idx = feature_names.index(label_name)

        pattern = os.path.join(
            data_dir, "*.tfrecord.gz" if compression_type == "GZIP" else "*.tfrecord"
        )
        logging.info(f"Reading TFRecords from pattern: {pattern}")

        # Important for time series: keep file order deterministic (no shuffling at the
        # file level).
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#list_files
        files = tf.data.Dataset.list_files(pattern, shuffle=False)

        # Stream the shards deterministically
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave
        ds = files.interleave(
            lambda fp: tf.data.TFRecordDataset(fp, compression_type=compression_type),
            cycle_length=1, # keep in-order; bump if cross-shard mixing is OK
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        feature_spec = _tfrecord_feature_spec(feature_names)
        ds = ds.map(
            lambda s: _parse_example(s, feature_spec, feature_names),  # -> [n_features]
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Sliding windows of consecutive rows -> [seq_length, n_features]
        # window() keeps order; drop_remainder enforces fixed length
        ds = ds.window(size=total_window, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(total_window, drop_remainder=True))  # -> [T_total, F]

        # Split into (inputs, labels)
        def split_window(chunk: tf.Tensor):
            # chunk: [total_window, n_features]
            x = chunk[:input_width, :]  # [L, F]
            # labels start after input_width + shift
            y_start = input_width + shift
            y = chunk[y_start : y_start + label_width, label_idx]  # [H]
            return x, y

        ds = ds.map(split_window, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle windows for training
        if shuffle_sequences:
            ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        
        # Batch and prefetch
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # -------------------------------------------------------------------------
    # Write windowed dataset to TFRecords under an output artifact directory
    # -------------------------------------------------------------------------
    def _write_windowed_dataset(
        ds: tf.data.Dataset,
        out_dir: str,
        *,
        compression_type: str = "GZIP",
        shard_prefix: str = "data",
        max_examples: int = None,  # None = all
    ) -> int:
        """Materialize (x, y) windowed dataset as TFRecords. Returns num_examples."""
        os.makedirs(out_dir, exist_ok=True)
        suffix = ".tfrecord.gz" if compression_type == "GZIP" else ".tfrecord"
        shard_path = os.path.join(out_dir, f"{shard_prefix}-00000-of-00001{suffix}")

        options = tf.io.TFRecordOptions(compression_type=compression_type) if compression_type else None
        writer = tf.io.TFRecordWriter(shard_path, options=options)

        def _float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=values))

        def _int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

        n = 0
        for x_batch, y_batch in ds:
            x_np = x_batch.numpy()  # [B, L, F]
            y_np = y_batch.numpy()  # [B, H]
            B, L, F = x_np.shape
            H = y_np.shape[1]

            for i in range(B):
                x_flat = x_np[i].reshape(-1).astype(np.float32)  # L*F
                y_flat = y_np[i].astype(np.float32)              # H
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "x_flat": _float_feature(x_flat.tolist()),
                            "x_shape": _int_feature([L, F]),
                            "y": _float_feature(y_flat.tolist()),
                            "y_shape": _int_feature([H]),
                        }
                    )
                )
                writer.write(example.SerializeToString())
                n += 1
                if max_examples is not None and n >= max_examples:
                    break
            if max_examples is not None and n >= max_examples:
                break

        writer.close()
        logging.info(f"Wrote {n} examples to {shard_path}")
        return n
    
    train_xy = _make_forecasting_dataset_from_tfrecord_dir(
        data_dir=train_data.path,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        batch_size=batch_size,
        feature_names=feature_names,
        label_name=label_columns[0],
        shuffle_sequences=False,
        compression_type="GZIP",
    )
    val_xy = _make_forecasting_dataset_from_tfrecord_dir(
        data_dir=valid_data.path,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        batch_size=batch_size,
        feature_names=feature_names,
        label_name=label_columns[0],
        shuffle_sequences=False,
        compression_type="GZIP",
    )
    test_xy = _make_forecasting_dataset_from_tfrecord_dir(
        data_dir=test_data.path,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        batch_size=batch_size,
        feature_names=feature_names,
        label_name=label_columns[0],
        shuffle_sequences=False,
        compression_type="GZIP",
    )

    # -------------------------------------------------------------------------
    # Metrics: number of records across all batches
    # -------------------------------------------------------------------------
    logging.info("Computing metrics...")
    # ------------- Count records (streaming, no materialization) -------------
    def count_records(ds: tf.data.Dataset) -> int:
        # ds is batched: each element is (x_b, y_b) with x_b.shape[0] = batch size
        return int(
            ds.map(lambda x, y: tf.shape(x)[0])
              .reduce(tf.constant(0, tf.int64), lambda acc, b: acc + tf.cast(b, tf.int64))
              .numpy()
        )

    train_n = count_records(train_xy)
    val_n = count_records(val_xy)
    test_n  = count_records(test_xy)

    # Log metrics
    metrics.log_metric("train_num_records", float(train_n))
    metrics.log_metric("valid_num_records", float(val_n))
    metrics.log_metric("test_num_records", float(test_n))
    metrics.log_metric("total_window_size", float(input_width + shift + label_width))
    metrics.log_metric("batch_size", float(batch_size))
    metrics.log_metric("input_width", float(input_width))
    metrics.log_metric("num_features", float(len(feature_names)))
    metrics.log_metric("label_width", float(label_width))
    metrics.log_metric("shift", float(shift))
    
    
    # -------------------------------------------------------------------------
    # Write output datasets (materialize to TFRecords)
    # -------------------------------------------------------------------------
    # _ = _write_windowed_dataset(train_xy, train_windowed.path, compression_type="GZIP")
    # _ = _write_windowed_dataset(val_xy,   valid_windowed.path, compression_type="GZIP")
    # _ = _write_windowed_dataset(test_xy,  test_windowed.path,  compression_type="GZIP")

    # ------------- Save datasets (most scalable & simple) -------------
    # Save as tuple (x, y). The loader must pass the same element_spec.
    # ------------- element_spec for save/load -------------
    # Each element is (x: [L,F], y: [H])
    # element_spec = (
    #     tf.TensorSpec(shape=(input_width, len(feature_names)), dtype=tf.float32),
    #     tf.TensorSpec(shape=(label_width,), dtype=tf.float32),
    # )
    # tf.data.experimental.save(train_xy.unbatch(), train_windowed.path, compression="GZIP")  # unbatch -> per-example
    # tf.data.experimental.save(val_xy.unbatch(), valid_windowed.path, compression="GZIP")
    # tf.data.experimental.save(test_xy.unbatch(),  test_windowed.path,  compression="GZIP")
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#save
    train_xy.unbatch().save(train_windowed.path, compression="GZIP")
    val_xy.unbatch().save(valid_windowed.path, compression="GZIP")
    test_xy.unbatch().save(test_windowed.path, compression="GZIP")
    
    # Plot examples
    # ------------- Plot 3 windows from first train batch -> HTML -------------
    first_xb, first_yb = next(iter(train_xy.take(1)))
    B, L, F = first_xb.shape
    H = first_yb.shape[1]
    metrics.log_metric("observed_batch_size", float(B))

    label_idx = feature_names.index(label_columns[0])

    def plot_png(x_seq, y_seq, target, L, S, H):
        xs = np.arange(L)
        ys = np.arange(L + S, L + S + H)
        fig = plt.figure(figsize=(10, 4))
        plt.plot(xs, x_seq[:, label_idx], marker="o")
        plt.axvline(x=L - 0.5, linestyle="--")
        if S > 0:
            plt.axvspan(L - 0.5, L + S - 0.5, alpha=0.15)
        plt.scatter(ys, y_seq, zorder=3)
        for xi, yi, v in zip(ys, y_seq, y_seq):
            plt.text(xi, yi, f"{float(v):.4f}", fontsize=8, rotation=90, va="bottom")
        plt.title(f"Window L={L}, shift={S}, H={H}")
        plt.xlabel("Relative timestep"); plt.ylabel(target); plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
        buf.seek(0); return base64.b64encode(buf.read()).decode("ascii")

    imgs = []
    for i in range(min(3, B)):
        imgs.append(plot_png(first_xb[i].numpy(), first_yb[i].numpy(),
                             label_columns[0], input_width, shift, label_width))

    html = f"""
    <html><body style="font-family:Inter,system-ui,Arial,sans-serif;padding:16px;">
      <h2>Windowing Preview (Train)</h2>
      <p><b>x shape:</b> [{B}, {L}, {F}] &nbsp; <b>y shape:</b> [{B}, {H}]</p>
      <table style="border-collapse:collapse;margin:10px 0;">
        <tr><th style="border:1px solid #ccc;padding:6px 10px;">Split</th>
            <th style="border:1px solid #ccc;padding:6px 10px;">Records</th></tr>
        <tr><td style="border:1px solid #ccc;padding:6px 10px;">train</td><td style="border:1px solid #ccc;padding:6px 10px;">{train_n}</td></tr>
        <tr><td style="border:1px solid #ccc;padding:6px 10px;">valid</td><td style="border:1px solid #ccc;padding:6px 10px;">{val_n}</td></tr>
        <tr><td style="border:1px solid #ccc;padding:6px 10px;">test</td><td style="border:1px solid #ccc;padding:6px 10px;">{test_n}</td></tr>
      </table>
      <h3>Three Example Windows</h3>
      {''.join(f'<img style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;margin:6px 0;" src="data:image/png;base64,{b64}" />' for b64 in imgs)}
    </body></html>
    """
    os.makedirs(os.path.dirname(plots_output.path), exist_ok=True)
    with open(plots_output.path, "w", encoding="utf-8") as f:
        f.write(html)
