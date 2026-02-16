# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import component, Input, Output, Dataset, Artifact


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
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "gcsfs>=2023.0.0",
        "pyarrow>=12.0.0",
        "fastavro>=1.7.0",
        "joblib>=1.3.0",
        "tensorflow>=2.12.0",
    ],
)
def tf_data_splitter(
    dataset: Input[Dataset],
    train_data: Output[Dataset],
    valid_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler_artifact: Output[Artifact],
    num_shards: int = 16,
    compress: bool = True,
    columns_json: str = "",
    scale_indicators: bool = False,
):
    """
    Split extracted data into train/validation/test with feature scaling.
    Saves splits to Output[Dataset]s as sharded TFRecords (.tfrecord[.gz])
    and persists a StandardScaler as an artifact.

    Args:
        dataset: Input dataset dir (expects files under dataset.uri)
        train_data: Output training dataset (70%)
        valid_data: Output validation dataset (20%)
        test_data: Output test dataset (10%)
        scaler_artifact: Output artifact directory to store scaler.joblib + metadata
        num_shards: Number of TFRecord shards per split (default 16)
        compress: If True, uses GZIP compression for TFRecords
        columns_json: Optional JSON mapping of column names to adapt to other schemas.
    """
    import logging
    from pathlib import Path
    import json

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import joblib

    import gcsfs
    import pyarrow.parquet as pq
    from fastavro import reader as avro_reader

    import tensorflow as tf
    import numpy as np

    # ---------- Schema mapping ----------
    default_cols = {
        "time_col": "trade_ts",     # main ordering/timestamp column
        "feature_price": "PRICE",
        "feature_vol_quote": "vol_quote",
        "feature_cvd_quote": "cvd_quote",
        "feature_pdcc_down": "PDCC_Down",
        "feature_osv_down": "OSV_Down",
        "feature_pdcc_up": "PDCC2_UP",
        "feature_osv_up": "OSV_Up",
        "feature_regime_up": "regime_up",
        "feature_regime_down": "regime_down",

    }
    overrides = json.loads(columns_json) if columns_json else {}
    C = {**default_cols, **overrides}

    # ---------- Timestamp normalization helper ----------
    def _to_utc_naive(series: pd.Series) -> pd.Series:
        """Normalize timestamp-like column to UTC naive."""
        try:
            series = series.astype(str).str.replace(" UTC", "", regex=False)
            s = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception as e:
            raise ValueError(f"[tf_data_splitter] Timestamp conversion failed: {e}")

    fs = gcsfs.GCSFileSystem()
    logging.info(f"Reading from: {dataset.uri}")

    # ---------------------------------------------------------
    # Read input files (CSV / Parquet / Avro / extension-less)
    # ---------------------------------------------------------
    all_files = fs.ls(dataset.uri)
    logging.info(f"Files in dataset: {all_files}")

    csv_files = [f for f in all_files if f.endswith(".csv")]
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    avro_files = [f for f in all_files if f.endswith(".avro")]
    untyped_files = [
        f
        for f in all_files
        if not any(f.endswith(ext) for ext in [".csv", ".parquet", ".avro"])
    ]

    df = None
    if csv_files:
        logging.info("Reading CSV file(s)")
        df = pd.concat(
            [pd.read_csv(fs.open(f, "rb")) for f in csv_files], ignore_index=True
        )
    elif parquet_files:
        logging.info("Reading Parquet file(s)")
        df = pd.concat(
            [pq.read_table(fs.open(f, "rb")).to_pandas() for f in parquet_files],
            ignore_index=True,
        )
    elif avro_files:
        logging.info("Reading Avro file(s)")
        with fs.open(avro_files[0], "rb") as f:
            records = list(avro_reader(f))
            df = pd.DataFrame.from_records(records)
    elif untyped_files:
        logging.warning("No extension detected — attempting to read first file as CSV")
        with fs.open(untyped_files[0], "rb") as f:
            head = f.read(2048).decode("utf-8", errors="ignore")
            if "," in head and "\n" in head:
                f.seek(0)
                df = pd.read_csv(f)
            else:
                raise ValueError("Untyped file does not look like CSV.")
    else:
        raise ValueError(f"No supported data files found in {dataset.uri}")

    if df is None or df.empty:
        raise ValueError("Loaded dataframe is empty.")

    # -----------------------------------------
    # Required columns & basic ordering
    # -----------------------------------------
    required_cols = list(C.values())
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Normalize timestamp column to UTC naive
    df[C["time_col"]] = _to_utc_naive(df[C["time_col"]])
    df = df.sort_values(C["time_col"]).reset_index(drop=True)

    # -----------------------------------------
    # Time-based split 70/20/10
    # -----------------------------------------
    n = len(df)
    if n < 100:
        raise ValueError(f"Not enough rows ({n}) to split into train/val/test.")

    i_train = int(n * 0.7)
    i_val = int(n * 0.9)

    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()

    logging.info(
        "Split sizes -> Train: %s, Val: %s, Test: %s",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # -----------------------------------------
    # Feature scaling (fit on TRAIN only)
    # -----------------------------------------
    feature_cols = [
        C["feature_price"],
        C["feature_vol_quote"],
        C["feature_cvd_quote"],
        C["feature_pdcc_down"],
        C["feature_osv_down"],
        C["feature_pdcc_up"],
        C["feature_osv_up"],
        C["feature_regime_up"],
        C["feature_regime_down"],
    ]

    # Separate indicator-type columns (binary flags)
    indicator_cols = [
        C["feature_regime_up"],
        C["feature_regime_down"],
        C["feature_pdcc_up"],
        C["feature_pdcc_down"],
    ]
    # Continuous columns = everything else
    continuous_cols = [c for c in feature_cols if c not in indicator_cols]

    std_cols = [f"{c}_std" for c in feature_cols]

    # Coerce to numeric & simple imputation (ffill/bfill) to avoid NaNs in scaler
    for split in (train_df, val_df, test_df):
        split[feature_cols] = split[feature_cols].apply(pd.to_numeric, errors="coerce")
        split[feature_cols] = split[feature_cols].ffill().bfill()

    scaler = StandardScaler().fit(train_df[continuous_cols].to_numpy())

    def _apply_scaling(split_df):
        split_df = split_df.copy()
        if scale_indicators:
            # Apply same scaler to continuous + indicator cols
            scaled = scaler.transform(split_df[continuous_cols])
            scaled_df = split_df.copy()
            scaled_df[continuous_cols] = scaled
            # Indicators are linearly shifted/scaled by same scaler instance
            scaled_df[indicator_cols] = (split_df[indicator_cols] - 0.5) * 2.0  # Optional symmetric scaling
        else:
            # Scale continuous only; leave indicators as raw 0/1
            scaled_cont = scaler.transform(split_df[continuous_cols])
            scaled_df = split_df.copy()
            scaled_df[continuous_cols] = scaled_cont
        return scaled_df

    # Columns to write (timestamps + raw + standardized)
    out_cols = [C["time_col"]] + feature_cols + std_cols

    # -----------------------------------------
    # Helpers: TFExample serialization
    # -----------------------------------------
    def _bytes_feature(v: bytes) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _float_feature_list(values) -> tf.train.Feature:
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(x) for x in values])
        )

    def _float_feature(v: float) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    def _to_example(row_dict: dict) -> bytes:
        # Ensure timestamp-like fields are strings (preserve your original meaning)
        time_str = str(row_dict[C["time_col"]])
        feat = {C["time_col"]: _bytes_feature(time_str.encode("utf-8"))}
        for c in feature_cols:
            feat[c] = _float_feature(row_dict[c])

        # Standardized features
        for c in std_cols:
            feat[c] = _float_feature(row_dict[c])

        example = tf.train.Example(features=tf.train.Features(feature=feat))
        return example.SerializeToString()

    def _scale_and_write_tfrecords(split_df: pd.DataFrame, out_dir: str, name: str):
        # transform (standardize) using fitted scaler
        # scaled = scaler.transform(split_df[feature_cols].to_numpy())
        # split_df = split_df.copy()  # avoid SettingWithCopy on caller
        # split_df.loc[:, std_cols] = scaled
        split_df = _apply_scaling(split_df)
        split_df.loc[:, std_cols] = split_df[feature_cols].to_numpy()

        # Only keep expected columns (same as CSV version)
        cols_present = [c for c in out_cols if c in split_df.columns]
        split_df = split_df.loc[:, cols_present]

        # Prepare output directory and shard writers
        tf.io.gfile.makedirs(out_dir)
        options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
        suffix = ".tfrecord.gz" if compress else ".tfrecord"

        writers = []
        for i in range(max(1, num_shards)):
            shard_path = f"{out_dir}/part-{i:05d}{suffix}"
            w = tf.io.TFRecordWriter(shard_path, options=options)
            writers.append(w)

        # Write rows to shards
        total = len(split_df)
        for idx, (_, row) in enumerate(split_df.iterrows()):
            ex = _to_example(row.to_dict())
            writers[idx % len(writers)].write(ex)

        for w in writers:
            w.close()

        logging.info(
            "Saved %s TFRecords to %s (rows=%s, shards=%s, gzip=%s)",
            name,
            out_dir,
            total,
            len(writers),
            compress,
        )

        if name == "train":
            means = split_df[std_cols].mean(numeric_only=True)
            stds = split_df[std_cols].std(numeric_only=True, ddof=0)
            for c in std_cols:
                logging.info(f"  {c}: mean={means[c]:.4f}, std={stds[c]:.4f}")

        # Emit a small schema/metadata JSON to help
        # downstream parsing (optional but useful)
        meta = {
            # --- Core TFRecord schema ---
            "string_features": [C["time_col"]],
            "float_features": feature_cols + std_cols,
            "compression": "GZIP" if compress else "NONE",
            "num_shards": int(len(writers)),
            "rows": int(total),
            "standardized_cols": std_cols,
            "raw_feature_cols": feature_cols,

            # --- Scaler / experiment info ---
            "framework": "scikit-learn",
            "version": ">=1.0.0",
            "type": "StandardScaler",
            "feature_order": feature_cols,
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "fitted_on_rows": int(len(train_df)),
            "std_feature_order": std_cols,
            "scale_indicators": bool(scale_indicators),
        }




        with tf.io.gfile.GFile(f"{out_dir}/features_metadata.json", "w") as f:
            f.write(json.dumps(meta, indent=2))
        
        return meta

    # -----------------------------------------
    # Save splits as TFRecords (replaces CSV)
    # -----------------------------------------
    meta_train = _scale_and_write_tfrecords(train_df, train_data.path, "train")
    _scale_and_write_tfrecords(val_df, valid_data.path, "val")
    _scale_and_write_tfrecords(test_df, test_data.path, "test")

    # -----------------------------------------
    # Persist scaler as an artifact (joblib + metadata)
    # -----------------------------------------
    scaler_dir = Path(scaler_artifact.path)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    scaler_bin_path = scaler_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_bin_path)

    # meta = {
    #     "framework": "scikit-learn",
    #     "version": ">=1.0.0",
    #     "type": "StandardScaler",
    #     "feature_order": feature_cols,
    #     "mean_": scaler.mean_.tolist(),
    #     "scale_": scaler.scale_.tolist(),
    #     "fitted_on_rows": int(len(train_df)),
    #     # Helpful for the trainer to know shapes/order:
    #     "std_feature_order": std_cols,
    #     "string_features": [C["time_col"]],
    #     "compression": "GZIP" if compress else "NONE",
    #     "num_shards_per_split": int(max(1, num_shards)),
    # }
    with open(scaler_dir / "scaler_metadata.json", "w") as f:
        json.dump(meta_train, f, indent=2)

    # Optional: surface metadata in UI
    scaler_artifact.metadata["type"] = "StandardScaler"
    scaler_artifact.metadata["n_features"] = len(feature_cols)
    scaler_artifact.metadata["feature_order"] = feature_cols
    scaler_artifact.uri = scaler_dir.as_posix()
