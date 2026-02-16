from typing import List
from kfp.v2.dsl import component, Input, Output, Dataset

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

    def _concat_split(datasets: List[Dataset], output_path: str):
        if not datasets:
            raise ValueError("No latent datasets provided for concatenation.")

        parsed_splits = []
        latent_dims = []

        # Load datasets per threshold
        for d in datasets:
            pattern = tf.io.gfile.glob(os.path.join(d.path, "*.tfrecord*"))
            if not pattern:
                raise ValueError(f"No TFRecord files found under {d.path}")

            comp = "GZIP" if any(p.endswith(".gz") for p in pattern) else ""
            raw_dataset = tf.data.TFRecordDataset(pattern, compression_type=comp)

            feature_spec = {
                "z": tf.io.VarLenFeature(tf.float32),
                "z_mean": tf.io.VarLenFeature(tf.float32),
                "z_log_var": tf.io.VarLenFeature(tf.float32),
                "y": tf.io.VarLenFeature(tf.float32),
            }

            def _parse_fn(ex):
                parsed = tf.io.parse_single_example(ex, feature_spec)
                return {
                    "z": tf.sparse.to_dense(parsed["z"]),
                    "z_mean": tf.sparse.to_dense(parsed["z_mean"]),
                    "z_log_var": tf.sparse.to_dense(parsed["z_log_var"]),
                    "y": tf.sparse.to_dense(parsed["y"]),
                }

            parsed_ds = list(raw_dataset.map(_parse_fn).as_numpy_iterator())
            parsed_splits.append(parsed_ds)
            latent_dims.append(len(parsed_ds[0]["z"]))

        num_examples = len(parsed_splits[0])
        for ds in parsed_splits:
            if len(ds) != num_examples:
                raise ValueError("Latent datasets have mismatched example counts")

        # Retain y from the first dataset
        ys = [parsed_splits[0][i]["y"] for i in range(num_examples)]

        # Combine z, z_mean, z_log_var
        concatenated = []
        for i in range(num_examples):
            zs = [parsed_splits[t][i]["z"] for t in range(len(parsed_splits))]
            zm = [parsed_splits[t][i]["z_mean"] for t in range(len(parsed_splits))]
            zv = [parsed_splits[t][i]["z_log_var"] for t in range(len(parsed_splits))]

            concatenated.append(
                {
                    "z_joint": np.concatenate(zs, axis=-1),
                    "z_mean_joint": np.concatenate(zm, axis=-1),
                    "z_log_var_joint": np.concatenate(zv, axis=-1),
                    "y": ys[i],
                }
            )

        # Write TFRecord
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, "joint_latents.tfrecord.gz")
        with tf.io.TFRecordWriter(out_file, options="GZIP") as writer:
            for ex in concatenated:
                def _float_feature(arr):
                    arr = np.asarray(arr, dtype=np.float32).ravel()
                    return tf.train.Feature(float_list=tf.train.FloatList(value=arr))

                # tf_ex = tf.train.Example(
                #     features=tf.train.Features(
                #         feature={
                #             "z_joint": _float_feature(ex["z_joint"]),
                #             "z_mean_joint": _float_feature(ex["z_mean_joint"]),
                #             "z_log_var_joint": _float_feature(ex["z_log_var_joint"]),
                #             "y": _float_feature(ex["y"]),
                #         }
                #     )
                # )
                tf_ex = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "z": _float_feature(ex["z_joint"]),
                            "z_mean": _float_feature(ex["z_mean_joint"]),
                            "z_log_var": _float_feature(ex["z_log_var_joint"]),
                            "y": _float_feature(ex["y"]),
                        }
                    )
                )
                writer.write(tf_ex.SerializeToString())

        # Write manifest
        # manifest = {
        #     "num_thresholds": len(datasets),
        #     "latent_dims": latent_dims,
        #     "combined_dim": int(sum(latent_dims)),
        #     "num_examples": num_examples,
        #     "has_target": True,
        #     "fields": ["z_joint", "z_mean_joint", "z_log_var_joint", "y"],
        # }

        combined_dim = int(sum(latent_dims))
        manifest = {
            # "features" schema for compatibility with training readers
            "features": {
                "z": {"shape": [combined_dim]},
                "z_mean": {"shape": [combined_dim]},
                "z_log_var": {"shape": [combined_dim]},
                "y": {"shape": [1]},
            },
            "compression": "GZIP",  # since TFRecords are written as .gz
            # Preserve existing metadata
            "num_thresholds": len(datasets),
            "latent_dims": latent_dims,
            "combined_dim": combined_dim,
            "num_examples": num_examples,
            "has_target": True,
            # keep fields list for convenience
            "fields": ["z", "z_mean", "z_log_var", "y"],
        }
        with tf.io.gfile.GFile(os.path.join(output_path, "_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"Wrote {num_examples} examples to {out_file} "
            f"(combined_dim={manifest['combined_dim']}, has_target=True)"
        )

    _concat_split(train_latents, joint_train.path)
    _concat_split(valid_latents, joint_valid.path)
    _concat_split(test_latents, joint_test.path)
