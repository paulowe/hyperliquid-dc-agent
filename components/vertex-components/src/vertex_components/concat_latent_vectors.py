from typing import List
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics

@component(
    base_image="python:3.10",
    packages_to_install=[
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        "cython<3.0.0",
        "--no-build-isolation",
        "pyyaml==5.4.1",
        "numpy>=1.21.0",
        "tensorflow==2.17.0",
    ],
)
def concat_latent_vectors(
    train_latents: Input[List[Dataset]],
    valid_latents: Input[List[Dataset]],
    test_latents: Input[List[Dataset]],
    joint_train: Output[Dataset],
    joint_valid: Output[Dataset],
    joint_test: Output[Dataset],
    metrics: Output[Metrics],
):
    """
    Step 3 (full): Combine latent vectors (z, z_mean, z_log_var) across thresholds
    into a single joint representation, retaining one target vector y.

    Input:
        - Multiple TFRecord latent datasets per split (train/valid/test)
          Each record contains keys: {"z", "z_mean", "z_log_var", "y"}

    Output:
        - TFRecord per split with combined latent representation:
          {"z_joint", "z_mean_joint", "z_log_var_joint", "y"}
    """
    import os
    import json
    import numpy as np
    import tensorflow as tf

    COMPRESSION = "GZIP"

    def _load_per_example(ds_dir: str) -> tf.data.Dataset:
        """
        Load a saved tf.data snapshot and return per-example dicts {"z","y"}.
        If the snapshot was saved with batched elements, unbatch to regain examples.
        """
        ds = tf.data.Dataset.load(ds_dir, compression=COMPRESSION)

        # Keep only the keys we need.
        ds = ds.map(
            lambda d: {"z": d["z"], "y": d["y"]},
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Since TFTrain VAE writer saved batched elements we gotta unbatch to per-example.
        ds = ds.unbatch()

        return ds

    def _concat_split(datasets: List[Dataset], out_dir: str):
        if not datasets:
            raise ValueError("No latent datasets provided for concatenation.")

        # 1) Load each threshold as per-example dict dataset
        per_thr = [_load_per_example(d.path) for d in datasets]

        # 2) zip across thresholds so each element is a tuple of dicts
        zipped = tf.data.Dataset.zip(tuple(per_thr))

        # 3) concat z along last dimension; keep y from the first threshold
        def _concat_tuple(*elems):
            zs = [e["z"] for e in elems]
            y0 = elems[0]["y"]
            return {"z": tf.concat(zs, axis=-1), "y": y0}

        joint_ds = zipped.map(_concat_tuple, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch
        joint_ds = joint_ds.prefetch(tf.data.AUTOTUNE)

        writer_batch_size = 1
        joint_batched = joint_ds.batch(writer_batch_size, drop_remainder=False)
        joint_batched = joint_batched.prefetch(tf.data.AUTOTUNE)

        # 4) save snapshot (directory) using Dataset.save
        tf.io.gfile.makedirs(out_dir)
        joint_batched.save(out_dir, compression=COMPRESSION)

        print(f"Wrote joint latent snapshot to {out_dir}")

    _concat_split(train_latents, joint_train.path)
    _concat_split(valid_latents, joint_valid.path)
    _concat_split(test_latents, joint_test.path)
