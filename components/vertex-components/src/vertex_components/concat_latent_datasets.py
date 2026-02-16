from typing import List, Optional, Union

from kfp.v2.dsl import Artifact, Dataset, Input, Output, component


def _normalize_threshold(threshold) -> Optional[str]:
    if threshold is None:
        return None
    if isinstance(threshold, (list, tuple)):
        if not threshold:
            return None
        threshold = threshold[0]
    try:
        value = float(threshold)
        return f"{value:.6f}"
    except (TypeError, ValueError):
        return str(threshold)


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
        "tensorflow==2.17.0",
        "numpy>=1.21.0",
    ],
)
def concat_latent_datasets(
    train_latents: Input[List[Dataset]],
    valid_latents: Input[List[Dataset]],
    test_latents: Input[List[Dataset]],
    scalers: Input[List[Artifact]],
    joint_train: Output[Dataset],
    joint_valid: Output[Dataset],
    joint_test: Output[Dataset],
    reference_scaler: Output[Artifact],
    latent_key: str = "z",
    mean_key: str = "z_mean",
    logvar_key: str = "z_log_var",
    label_key: str = "y",
    compression: str = "GZIP",
):
    """Concatenate latent TFRecord splits from multiple thresholds into one dataset.

    Args:
        train_latents: Per-threshold latent datasets for the train split.
        valid_latents: Per-threshold latent datasets for the validation split.
        test_latents: Per-threshold latent datasets for the test split.
        scalers: Scaler artifacts produced by the upstream splitters (first one
            is propagated so downstream consumers can still inverse-transform).
        joint_train: Concatenated latent dataset (train split).
        joint_valid: Concatenated latent dataset (validation split).
        joint_test: Concatenated latent dataset (test split).
        reference_scaler: Reference scaler artifact forwarded downstream.
        latent_key: Key used for the latent vector within the TFRecord.
        mean_key: Key used for the latent mean vector.
        logvar_key: Key used for the latent log-variance vector.
        label_key: Key used for labels/targets.
        compression: Compression to use for output TFRecords ("GZIP" or "").
    """
    import json
    import logging
    import os
    from typing import Dict, Iterable

    import numpy as np
    import tensorflow as tf

    def _artifact_uri(artifact: Union[Dataset, Artifact]) -> str:
        uri = getattr(artifact, "uri", None) or getattr(artifact, "path", None)
        if not uri:
            raise ValueError("Artifact is missing uri/path")
        return uri.rstrip("/")

    def _read_manifest(dir_path: str) -> Dict:
        manifest_path = os.path.join(dir_path, "_manifest.json")
        if tf.io.gfile.exists(manifest_path):
            with tf.io.gfile.GFile(manifest_path, "r") as src:
                return json.load(src)
        return {}

    def _build_parser(
        dir_path: str,
        manifest: Dict,
    ):
        comp = (manifest.get("compression") or compression or "").upper()
        if comp not in ("", "GZIP"):
            raise ValueError(f"Unsupported compression '{comp}' for {dir_path}")

        features = manifest.get("features", {})
        latent_dim = int(features.get(latent_key, {}).get("shape", [0])[0])
        label_dim = int(features.get(label_key, {}).get("shape", [0])[0])

        if latent_dim <= 0 or label_dim <= 0:
            raise ValueError(
                f"Manifest at {dir_path} missing latent/label dims for keys "
                f"'{latent_key}' / '{label_key}'"
            )

        feature_spec = {
            latent_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
            mean_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
            logvar_key: tf.io.FixedLenFeature([latent_dim], tf.float32),
            label_key: tf.io.FixedLenFeature([label_dim], tf.float32),
        }

        def _parse(serialized):
            parsed = tf.io.parse_single_example(serialized, feature_spec)
            return {
                latent_key: tf.ensure_shape(parsed[latent_key], [latent_dim]),
                mean_key: tf.ensure_shape(parsed[mean_key], [latent_dim]),
                logvar_key: tf.ensure_shape(parsed[logvar_key], [latent_dim]),
                label_key: tf.ensure_shape(parsed[label_key], [label_dim]),
            }

        return _parse, comp, latent_dim, label_dim

    def _load_split(datasets: Iterable[Dataset], split_name: str):
        datasets = list(datasets)
        if not datasets:
            raise ValueError(f"No latent datasets supplied for split '{split_name}'")

        parsers: List = []
        latent_dims: List[int] = []
        label_dims: List[int] = []
        data_iters = []
        thresholds: List[str] = []
        comp = None

        for idx, artifact in enumerate(datasets):
            dir_path = _artifact_uri(artifact)
            manifest = _read_manifest(dir_path)
            parser, this_comp, latent_dim, label_dim = _build_parser(dir_path, manifest)

            if comp is None:
                comp = this_comp
            elif this_comp != comp:
                raise ValueError(
                    f"Compression mismatch: '{this_comp}' vs '{comp}' for {dir_path}"
                )

            pattern = "*.tfrecord.gz" if comp == "GZIP" else "*.tfrecord"
            shard_paths = sorted(tf.io.gfile.glob(os.path.join(dir_path, pattern)))
            if not shard_paths:
                raise ValueError(f"No TFRecord shards found under {dir_path}")

            ds = tf.data.TFRecordDataset(
                shard_paths,
                compression_type="GZIP" if comp == "GZIP" else "",
            )
            ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
            data_iters.append(iter(ds.as_numpy_iterator()))
            parsers.append(parser)
            latent_dims.append(latent_dim)
            label_dims.append(label_dim)

            thr_raw = manifest.get("threshold_str") if manifest else None
            if thr_raw is None and manifest:
                thr_raw = manifest.get("threshold")
            thresholds.append(_normalize_threshold(thr_raw) or f"index_{idx}")
            logging.info(
                "[concat] %s split input #%d -> latent_dim=%d label_dim=%d shards=%d",
                split_name,
                idx,
                latent_dim,
                label_dim,
                len(shard_paths),
            )

        if len(set(label_dims)) != 1:
            raise ValueError(
                f"Label dimensions differ across thresholds for split '{split_name}': "
                f"{label_dims}"
            )

        return data_iters, latent_dims, label_dims[0], thresholds, comp

    def _float_feature(values: np.ndarray) -> tf.train.Feature:
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(v) for v in values.ravel()])
        )

    def _write_split(
        split_name: str,
        inputs: Iterable[Dataset],
        output: Output[Dataset],
    ):
        data_iters, latent_dims, label_dim, thresholds, comp = _load_split(
            inputs, split_name
        )

        out_dir = output.path
        tf.io.gfile.makedirs(out_dir)
        suffix = ".tfrecord.gz" if comp == "GZIP" else ".tfrecord"
        num_shards = 16 if split_name == "train" else 8
        options = (
            tf.io.TFRecordOptions(compression_type="GZIP")
            if comp == "GZIP"
            else tf.io.TFRecordOptions()
        )
        writers = [
            tf.io.TFRecordWriter(
                os.path.join(out_dir, f"part-{shard:05d}{suffix}"),
                options=options,
            )
            for shard in range(num_shards)
        ]

        shard_idx = 0
        example_count = 0
        num_thresholds = len(data_iters)
        total_label_dim = label_dim * num_thresholds

        for records in zip(*data_iters):
            latents = [rec[latent_key] for rec in records]
            means = [rec[mean_key] for rec in records]
            logvars = [rec[logvar_key] for rec in records]
            labels = [rec[label_key] for rec in records]

            combined_latent = np.concatenate(latents, axis=-1)
            combined_mean = np.concatenate(means, axis=-1)
            combined_logvar = np.concatenate(logvars, axis=-1)
            stacked_labels = np.stack(labels, axis=0).reshape(total_label_dim)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        latent_key: _float_feature(combined_latent),
                        mean_key: _float_feature(combined_mean),
                        logvar_key: _float_feature(combined_logvar),
                        label_key: _float_feature(stacked_labels),
                    }
                )
            )

            writers[shard_idx].write(example.SerializeToString())
            shard_idx = (shard_idx + 1) % num_shards
            example_count += 1

        # Ensure no iterator has leftover elements (length mismatch guard)
        for idx, iterator in enumerate(data_iters):
            try:
                next(iterator)
                raise ValueError(
                    f"Latent datasets have mismatched lengths for split '{split_name}'"
                )
            except StopIteration:
                continue

        for writer in writers:
            writer.close()

        combined_latent_dim = int(sum(latent_dims))
        manifest = {
            "version": 1,
            "compression": comp,
            "sourceLatentDims": latent_dims,
            "sourceLabelDim": label_dim,
            "numThresholds": num_thresholds,
            "sourceThresholds": thresholds,
            "labelDimTotal": total_label_dim,
            "features": {
                latent_key: {"dtype": "float32", "shape": [combined_latent_dim]},
                mean_key: {"dtype": "float32", "shape": [combined_latent_dim]},
                logvar_key: {"dtype": "float32", "shape": [combined_latent_dim]},
                label_key: {"dtype": "float32", "shape": [total_label_dim]},
            },
            "exampleCount": example_count,
        }
        with tf.io.gfile.GFile(os.path.join(out_dir, "_manifest.json"), "w") as dst:
            json.dump(manifest, dst, indent=2)

        output.uri = output.path
        output.metadata["latentDim"] = combined_latent_dim
        output.metadata["labelDim"] = label_dim
        output.metadata["labelDimTotal"] = total_label_dim
        output.metadata["compression"] = comp
        output.metadata["sourceLatentDims"] = latent_dims
        output.metadata["exampleCount"] = example_count
        output.metadata["numThresholds"] = num_thresholds
        output.metadata["sourceThresholds"] = thresholds

        logging.info(
            "[concat] Wrote %s examples to %s (latent_dim=%d label_dim=%d total_label_dim=%d)",
            example_count,
            out_dir,
            combined_latent_dim,
            label_dim,
            total_label_dim,
        )

    _write_split("train", train_latents, joint_train)
    _write_split("valid", valid_latents, joint_valid)
    _write_split("test", test_latents, joint_test)

    scaler_list = list(scalers)
    if not scaler_list:
        raise ValueError("At least one scaler artifact is required")

    first_scaler = scaler_list[0]
    reference_uri = _artifact_uri(first_scaler)
    reference_scaler.uri = reference_uri
    first_meta = getattr(first_scaler, "metadata", {}) or {}
    for key, value in first_meta.items():
        reference_scaler.metadata[key] = value
