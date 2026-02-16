# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
        "--upgrade", "pip", "setuptools", "wheel",
        "cython<3.0.0",
        "--no-build-isolation", "pyyaml==5.4.1",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "gcsfs>=2023.0.0",
        "pyarrow>=12.0.0",
        "fastavro>=1.7.0",
        "joblib>=1.3.0",
    ],
)
def data_splitter(
    dataset: Input[Dataset],
    train_data: Output[Dataset],
    valid_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler_artifact: Output[Artifact],   
):
    """
    Split extracted data into train/validation/test with feature scaling.
    Saves splits to Output[Dataset]s and persists a StandardScaler as an artifact.

    Args:
        dataset: Input dataset dir (expects files under dataset.uri)
        train_data: Output training dataset (70%)
        valid_data: Output validation dataset (20%)
        test_data: Output test dataset (10%)
        scaler_artifact: Output artifact directory to store scaler.joblib + metadata
    """
    import logging
    from pathlib import Path
    import json

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import joblib

    import gcsfs
    import pyarrow.parquet as pq
    from fastavro import reader as avro_reader

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
    untyped_files = [f for f in all_files if not any(f.endswith(ext) for ext in [".csv", ".parquet", ".avro"])]

    df = None
    if csv_files:
        logging.info("Reading CSV file(s)")
        df = pd.concat([pd.read_csv(fs.open(f, "rb")) for f in csv_files], ignore_index=True)
    elif parquet_files:
        logging.info("Reading Parquet file(s)")
        df = pd.concat([pq.read_table(fs.open(f, "rb")).to_pandas() for f in parquet_files], ignore_index=True)
    elif avro_files:
        logging.info("Reading Avro file(s)")
        with fs.open(avro_files[0], "rb") as f:
            records = list(avro_reader(f))
            df = pd.DataFrame.from_records(records)
    elif untyped_files:
        logging.warning("No extension detected — attempting to read first file as CSV")
        with fs.open(untyped_files[0], "rb") as f:
            head = f.read(2048).decode("utf-8", errors="ignore")
            if "," in head and "\n" in head and "start_time" in head:
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
    required_cols = [
        "start_time",
        "load_time_toronto",
        "start_price",
        "PRICE",
        "vol_quote",
        "cvd_quote",
        "PDCC_Down",
        "OSV_Down",
        "PDCC2_UP",
        "OSV_Up",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values("load_time_toronto").reset_index(drop=True)

    # -----------------------------------------
    # Time-based split 70/20/10
    # -----------------------------------------
    n = len(df)
    if n < 100:
        raise ValueError(f"Not enough rows ({n}) to split into train/val/test.")

    i_train = int(n * 0.7)
    i_val   = int(n * 0.9)

    train_df = df.iloc[:i_train].copy()
    val_df   = df.iloc[i_train:i_val].copy()
    test_df  = df.iloc[i_val:].copy()

    logging.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # -----------------------------------------
    # Feature scaling (fit on TRAIN only)
    # -----------------------------------------
    feature_cols = [
        "start_price",
        "PRICE",
        "vol_quote",
        "cvd_quote",
        "PDCC_Down",
        "OSV_Down",
        "PDCC2_UP",
        "OSV_Up",
    ]
    std_cols = [f"{c}_std" for c in feature_cols]

    # Coerce to numeric & simple imputation (ffill/bfill) to avoid NaNs in scaler
    for split in (train_df, val_df, test_df):
        split[feature_cols] = split[feature_cols].apply(pd.to_numeric, errors="coerce")
        split[feature_cols] = split[feature_cols].ffill().bfill()

    scaler = StandardScaler().fit(train_df[feature_cols].to_numpy())

    # Columns to write (timestamps + raw + standardized)
    out_cols = ["start_time", "load_time_toronto"] + feature_cols + std_cols

    def scale_and_save(split_df: pd.DataFrame, out_dir: str, name: str):
        # transform
        scaled = scaler.transform(split_df[feature_cols].to_numpy())
        split_df.loc[:, std_cols] = scaled

        # write
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "data.csv"

        cols_present = [c for c in out_cols if c in split_df.columns]
        split_df.to_csv(out_file, index=False, columns=cols_present)
        logging.info(f"Saved {name} data to: {out_file}")

        if name == "train":
            means = split_df[std_cols].mean(numeric_only=True)
            stds  = split_df[std_cols].std(numeric_only=True, ddof=0)
            for c in std_cols:
                logging.info(f"  {c}: mean={means[c]:.4f}, std={stds[c]:.4f}")

    # Save splits
    scale_and_save(train_df, train_data.path, "train")
    scale_and_save(val_df,   valid_data.path, "val")
    scale_and_save(test_df,  test_data.path, "test")

    # -----------------------------------------
    # Persist scaler as an artifact (joblib + metadata)
    # -----------------------------------------
    scaler_dir = Path(scaler_artifact.path)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    scaler_bin_path = scaler_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_bin_path)

    meta = {
        "framework": "scikit-learn",
        "version": ">=1.0.0",
        "type": "StandardScaler",
        "feature_order": feature_cols,
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "fitted_on_rows": int(len(train_df)),
    }
    with open(scaler_dir / "scaler_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: surface some metadata in UI
    scaler_artifact.metadata["type"] = "StandardScaler"
    scaler_artifact.metadata["n_features"] = len(feature_cols)
    scaler_artifact.metadata["feature_order"] = feature_cols
