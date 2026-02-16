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
def concat_threshold_features(
    # Per-split lists of snapshots saved by window_dataset (elements are (x:[L,F], y:[H]))
    train_datasets: Input[List[Dataset]],
    valid_datasets: Input[List[Dataset]],
    test_datasets: Input[List[Dataset]],

    # Outputs
    joint_train: Output[Dataset],
    joint_valid: Output[Dataset],
    joint_test: Output[Dataset],
    metrics: Output[Metrics],

    # Feature config
    feature_names: List[str] = [
        "PRICE_std","vol_quote_std","cvd_quote_std",
        "PDCC_Down","OSV_Down_std","PDCC2_UP","regime_up","regime_down",
    ],
    common_feature_names: List[str] = ["PRICE_std","vol_quote_std","cvd_quote_std"],
    per_threshold_feature_names: List[str] = ["PDCC_Down","OSV_Down_std","PDCC2_UP","regime_up","regime_down"],

    # Save / checks
    compression: str = "GZIP",
    validate_y_equal: bool = True,
    y_rtol: float = 1e-6,
    y_atol: float = 1e-6,
):
    """
    Build (x_joint, y) per example by:
      • Keeping a single copy of `common_feature_names` from threshold 0 (first dataset)
      • Concatenating `per_threshold_feature_names` from *every* threshold in order
      • Preserving y from threshold 0; optionally assert all thresholds carry the same y

    Output layout of x_joint:
      [ common_feature_names(th0) , per_threshold_feature_names(th0) , ... , per_threshold_feature_names(th{T-1}) ]
    """
    import tensorflow as tf

    # --- index helpers
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    try:
        common_idx = tf.constant([name_to_idx[n] for n in common_feature_names], dtype=tf.int32)
        per_thr_idx = tf.constant([name_to_idx[n] for n in per_threshold_feature_names], dtype=tf.int32)
    except KeyError as e:
        raise ValueError(f"Feature name not found in feature_names: {e}")

    def _load_snapshot(dir_path: str) -> tf.data.Dataset:
        # Each element is (x:[L,F], y:[H]); already unbatched in your window_dataset
        return tf.data.Dataset.load(dir_path, compression=compression)

    def _concat_one_split(dsets: List[Dataset], out_dir: str):
        if not dsets:
            raise ValueError("Expected at least one dataset per split.")
        # Load datasets in threshold order
        per_thr_ds = [_load_snapshot(d.path) for d in dsets]

        # Zip across thresholds -> ((x1,y1),(x2,y2),...,(xT,yT))
        zipped = tf.data.Dataset.zip(tuple(per_thr_ds))

        def _concat_examples(*pairs):
            xs = [p[0] for p in pairs]  # list of [L,F]
            ys = [p[1] for p in pairs]  # list of [H]

            # Keep y from first threshold
            y0 = ys[0]

            # Optionally assert all y are equal (within tol)
            if validate_y_equal and len(ys) > 1:
                for yk in ys[1:]:
                    tf.debugging.assert_near(yk, y0, rtol=y_rtol, atol=y_atol,
                                             message="y differs across thresholds")

            # Common features from threshold 0 only
            x_common = tf.gather(xs[0], common_idx, axis=-1)           # [L, n_common]

            # Per-threshold feature blocks concatenated across thresholds
            per_blocks = [tf.gather(x, per_thr_idx, axis=-1) for x in xs]  # T * [L, k]
            x_per_joint = tf.concat(per_blocks, axis=-1)                    # [L, k*T]

            x_joint = tf.concat([x_common, x_per_joint], axis=-1)           # [L, n_common + k*T]
            return (x_joint, y0)

        joint = zipped.map(_concat_examples, num_parallel_calls=tf.data.AUTOTUNE)

        # Inspect 1 example for shape metrics
        first = next(iter(joint.take(1)))
        x0, y0 = first
        L = int(x0.shape[0])
        F_out = int(x0.shape[1])
        H = int(y0.shape[0]) if y0.shape.rank == 1 else int(y0.shape[-1])

        # Count examples efficiently
        count = int(
            joint.reduce(
                tf.constant(0, tf.int64),
                lambda c, _: c + tf.constant(1, tf.int64)
            ).numpy()
        )

        # Save snapshot
        tf.io.gfile.makedirs(out_dir)
        joint.save(out_dir, compression=compression)

        return count, L, F_out, H

    tr_count, tr_L, tr_F, tr_H = _concat_one_split(train_datasets, joint_train.path)
    va_count, va_L, va_F, va_H = _concat_one_split(valid_datasets, joint_valid.path)
    te_count, te_L, te_F, te_H = _concat_one_split(test_datasets, joint_test.path)

    # Metrics
    metrics.log_metric("num_thresholds", float(len(train_datasets)))
    metrics.log_metric("n_common_features", float(len(common_feature_names)))
    metrics.log_metric("n_per_threshold_features", float(len(per_threshold_feature_names)))
    metrics.log_metric("output_features", float(tr_F))
    metrics.log_metric("sequence_length_L", float(tr_L))
    metrics.log_metric("label_length_H", float(tr_H))
    metrics.log_metric("train_examples", float(tr_count))
    metrics.log_metric("valid_examples", float(va_count))
    metrics.log_metric("test_examples", float(te_count))
