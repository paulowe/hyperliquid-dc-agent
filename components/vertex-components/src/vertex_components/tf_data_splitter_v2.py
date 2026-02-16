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
        "tensorflow>=2.12.0",
        "google-cloud-bigquery>=3.11.0",
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
    # -------- BigQuery (optional) ----------
    bq_project: str = "",
    bq_dataset: str = "",
    bq_table_base: str = "",  # final tables: <base>_{train,val,test}
    bq_location: str = "US",
    bq_write_disposition: str = "WRITE_TRUNCATE",  # or WRITE_APPEND / WRITE_EMPTY
    bq_time_partition_field: str = "load_time_toronto",  # TIMESTAMP/DATE for partitioning
):
    """
    Split data, scale features, save TFRecords and (optionally) write splits to BigQuery.

    To enable BigQuery writes, provide: bq_project, bq_dataset, bq_table_base.
    Creates/overwrites tables:
      {bq_project}.{bq_dataset}.{bq_table_base}_train
      {bq_project}.{bq_dataset}.{bq_table_base}_val
      {bq_project}.{bq_dataset}.{bq_table_base}_test
    """
    import logging
    from pathlib import Path
    import json
    from typing import List

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import joblib

    import gcsfs
    import pyarrow.parquet as pq
    from fastavro import reader as avro_reader

    import tensorflow as tf

    # BigQuery
    from google.cloud import bigquery

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

    # Ensure timestamp dtypes are proper for BQ
    # (tz-aware -> TIMESTAMP in BigQuery)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["load_time_toronto"] = pd.to_datetime(df["load_time_toronto"], errors="coerce", utc=True)

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
    feature_cols: List[str] = [
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
    out_cols = ["start_time", "load_time_toronto"] + feature_cols + std_cols

    for split in (train_df, val_df, test_df):
        split[feature_cols] = split[feature_cols].apply(pd.to_numeric, errors="coerce")
        split[feature_cols] = split[feature_cols].ffill().bfill()

    scaler = StandardScaler().fit(train_df[feature_cols].to_numpy())

    def _apply_scaling_keep_cols(split_df: pd.DataFrame) -> pd.DataFrame:
        scaled = scaler.transform(split_df[feature_cols].to_numpy())
        out = split_df.copy()
        out.loc[:, std_cols] = scaled
        cols_present = [c for c in out_cols if c in out.columns]
        return out.loc[:, cols_present]

    # Prepare scaled DataFrames for BOTH TFRecords & BQ
    train_out = _apply_scaling_keep_cols(train_df)
    val_out   = _apply_scaling_keep_cols(val_df)
    test_out  = _apply_scaling_keep_cols(test_df)

    # -----------------------------------------
    # TFExample helpers
    # -----------------------------------------
    def _bytes_feature(v: bytes) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _float_feature(v: float) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    def _to_example(row: pd.Series) -> bytes:
        # Encode timestamps as ISO-8601 strings in TFRecords
        st = pd.to_datetime(row["start_time"], utc=True).isoformat() if pd.notna(row["start_time"]) else ""
        lt = pd.to_datetime(row["load_time_toronto"], utc=True).isoformat() if pd.notna(row["load_time_toronto"]) else ""

        feat = {
            "start_time":        _bytes_feature(st.encode("utf-8")),
            "load_time_toronto": _bytes_feature(lt.encode("utf-8")),
        }
        for c in feature_cols + std_cols:
            feat[c] = _float_feature(row[c])

        example = tf.train.Example(features=tf.train.Features(feature=feat))
        return example.SerializeToString()

    def _write_tfrecords(split_df: pd.DataFrame, out_dir: str, name: str):
        tf.io.gfile.makedirs(out_dir)
        options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
        suffix = ".tfrecord.gz" if compress else ".tfrecord"

        writers = []
        for i in range(max(1, num_shards)):
            shard_path = f"{out_dir}/part-{i:05d}{suffix}"
            w = tf.io.TFRecordWriter(shard_path, options=options)
            writers.append(w)

        total = len(split_df)
        for idx, (_, row) in enumerate(split_df.iterrows()):
            writers[idx % len(writers)].write(_to_example(row))

        for w in writers:
            w.close()

        logging.info(f"Saved {name} TFRecords to: {out_dir} (rows={total}, shards={len(writers)}, gzip={compress})")

        if name == "train":
            means = split_df[std_cols].mean(numeric_only=True)
            stds  = split_df[std_cols].std(numeric_only=True, ddof=0)
            for c in std_cols:
                logging.info(f"  {c}: mean={means[c]:.4f}, std={stds[c]:.4f}")

        meta = {
            "string_features": ["start_time", "load_time_toronto"],
            "float_features": feature_cols + std_cols,
            "compression": "GZIP" if compress else "NONE",
            "num_shards": int(len(writers)),
            "rows": int(total),
            "standardized_cols": std_cols,
            "raw_feature_cols": feature_cols,
        }
        with tf.io.gfile.GFile(f"{out_dir}/features_metadata.json", "w") as f:
            f.write(json.dumps(meta, indent=2))

    # -----------------------------------------
    # BigQuery writer (optional)
    # -----------------------------------------
    def _bq_enabled() -> bool:
        return bool(bq_project and bq_dataset and bq_table_base)

    def _write_bq(split_df: pd.DataFrame, suffix: str):
        """
        Writes DataFrame to BigQuery as {bq_table_base}_{suffix}.
        - Partitions by bq_time_partition_field (TIMESTAMP/DATE).
        - Clusters by standardized columns for selective queries.
        """
        client = bigquery.Client(project=bq_project, location=bq_location)
        table_id = f"{bq_project}.{bq_dataset}.{bq_table_base}_{suffix}"

        # Job config
        job_config = bigquery.LoadJobConfig(
            write_disposition=getattr(bigquery.WriteDisposition, bq_write_disposition, bigquery.WriteDisposition.WRITE_TRUNCATE),
            time_partitioning=bigquery.TimePartitioning(field=bq_time_partition_field),
            clustering_fields=[c for c in std_cols if c in split_df.columns][:4],  # cap clustering fields
        )

        # Ensure types are suitable
        # (pandas datetime64[ns, UTC] -> TIMESTAMP)
        # Floats are fine; NaNs will be loaded as NULL.
        load_job = client.load_table_from_dataframe(
            split_df, table_id, job_config=job_config
        )
        result = load_job.result()
        table = client.get_table(table_id)
        logging.info(
            f"Loaded to BigQuery: {table_id} "
            f"(rows={table.num_rows}, location={table.location}, partitioned_by={bq_time_partition_field})"
        )

    # -----------------------------------------
    # Save TFRecords
    # -----------------------------------------
    _write_tfrecords(train_out, train_data.path, "train")
    _write_tfrecords(val_out,   valid_data.path, "val")
    _write_tfrecords(test_out,  test_data.path,  "test")

    # -----------------------------------------
    # Save to BigQuery (if enabled)
    # -----------------------------------------
    if _bq_enabled():
        try:
            _write_bq(train_out, "train")
            _write_bq(val_out,   "val")
            _write_bq(test_out,  "test")
        except Exception as e:
            # Fail fast or downgrade to warning depending on your preference:
            # raise
            logging.exception(f"BigQuery load failed: {e}")
            raise

    # -----------------------------------------
    # Persist scaler as an artifact
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
        "std_feature_order": std_cols,
        "string_features": ["start_time", "load_time_toronto"],
        "compression": "GZIP" if compress else "NONE",
        "num_shards_per_split": int(max(1, num_shards)),
        "bq_project": bq_project,
        "bq_dataset": bq_dataset,
        "bq_table_base": bq_table_base,
        "bq_enabled": _bq_enabled(),
    }
    with open(scaler_dir / "scaler_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    scaler_artifact.metadata["type"] = "StandardScaler"
    scaler_artifact.metadata["n_features"] = len(feature_cols)
    scaler_artifact.metadata["feature_order"] = feature_cols
