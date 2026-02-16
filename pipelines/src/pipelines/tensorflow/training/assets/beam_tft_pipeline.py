# beam_tft_pipeline.py
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, GoogleCloudOptions, WorkerOptions

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2

import tensorflow_data_validation as tfdv

# ------------- Config (adjust to your domain) ----------------
STRING_COLS = ["start_time", "load_time_toronto"]
NUMERIC_COLS = [
    "start_price",
    "PRICE",
    "vol_quote",
    "cvd_quote",
    "PDCC_Down",
    "OSV_Down",
    "PDCC2_UP",
    "OSV_Up",
]
REQUIRED_COLS = STRING_COLS + NUMERIC_COLS
TIME_COL = "load_time_toronto"  # split by this event/load timestamp

def parse_iso(ts: str) -> datetime:
    # Accepts 'YYYY-mm-ddTHH:MM:SS' or pandas-ish strings; adjust as needed
    return datetime.fromisoformat(str(ts).replace("Z","").replace(" ", "T").split(".")[0])

# ------------- TFT preprocessing_fn --------------------------
def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    - Pass through timestamps (string)
    - Scale numeric columns with z-score (mean/std learned on ANALYZE data)
    - Also emit raw numeric (optional) for inspection
    """
    outputs = {}
    for c in STRING_COLS:
        # keep as-is
        outputs[c] = inputs[c]

    for c in NUMERIC_COLS:
        # Ensure floats
        x = tf.cast(inputs[c], tf.float32)
        outputs[c] = x  # raw
        outputs[f"{c}_z"] = tft.scale_to_z_score(x)

    return outputs

# ------------- Helpers ---------------------------------------
class EnsureRequired(beam.DoFn):
    def process(self, row: Dict):
        missing = [c for c in REQUIRED_COLS if c not in row or row[c] is None]
        if missing:
            return  # drop bad rows; alternatively send to deadletter
        yield row

class KeepSplit(beam.DoFn):
    def __init__(self, split_name: str, end_ts: str, start_ts: str = None):
        self.split_name = split_name
        self.end_ts = end_ts
        self.start_ts = start_ts

    def setup(self):
        self.end_dt = parse_iso(self.end_ts) if self.end_ts else None
        self.start_dt = parse_iso(self.start_ts) if self.start_ts else None

    def process(self, row: Dict):
        ts = parse_iso(row[TIME_COL])
        if self.start_dt and ts < self.start_dt:
            return
        if self.end_dt and ts > self.end_dt:
            return
        yield row

def _make_raw_schema() -> schema_pb2.Schema:
    # Build a TFMD schema for your raw input
    schema = schema_pb2.Schema()
    for c in STRING_COLS:
        f = schema.feature.add()
        f.name = c
        f.type = schema_pb2.FeatureType.BYTES
    for c in NUMERIC_COLS:
        f = schema.feature.add()
        f.name = c
        f.type = schema_pb2.FeatureType.FLOAT
    return schema

def to_tfexample(features: Dict[str, tf.train.Feature]) -> bytes:
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex.SerializeToString()

def dict_to_sequence_example(d: Dict[str, tf.Tensor]) -> bytes:
    """
    SequenceExample is recommended for sequences; if you have 1-row = 1-example
    (no variable-length lists), a normal Example also works. For generality, we
    demonstrate SequenceExample with all features as context.
    """
    context_features = {}
    for k, v in d.items():
        if isinstance(v, tf.Tensor):
            v = v.numpy()  # in TF graph this won’t run; we are in Beam w/ numpy values
        if isinstance(v, (bytes, bytearray, str)):
            val = v if isinstance(v, (bytes, bytearray)) else str(v).encode("utf-8")
            context_features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))
        else:
            # assume scalar float or int convertible
            context_features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    seq_ex = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features),
        feature_lists=tf.train.FeatureLists(feature_list={})
    )
    return seq_ex.SerializeToString()

class DictToExampleDoFn(beam.DoFn):
    def __init__(self, use_sequence_example: bool = True):
        self.use_sequence_example = use_sequence_example

    def process(self, row: Dict):
        if self.use_sequence_example:
            yield dict_to_sequence_example(row)
        else:
            # Regular Example
            feat = {}
            for k, v in row.items():
                if isinstance(v, (bytes, bytearray, str)):
                    b = v if isinstance(v, (bytes, bytearray)) else str(v).encode("utf-8")
                    feat[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
                else:
                    feat[k] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))
            yield to_tfexample(feat)

# ------------- Read sources ----------------------------------
def read_input(p: beam.Pipeline, input_format: str, input_path: str) -> beam.PCollection:
    if input_format.lower() == "parquet":
        return (p
                | "ReadParquet" >> beam.io.ReadFromParquet(input_path))
    elif input_format.lower() == "csv":
        return (p
                | "ReadCSV" >> beam.io.ReadFromText(input_path, skip_header_lines=1)
                | "CSV->Dict" >> beam.Map(lambda line: line.split(","))
                # Replace with a proper CSV parser & header inference in production
               )
    else:
        raise ValueError("Unsupported input_format. Use 'parquet' or 'csv'.")

# ------------- Main ------------------------------------------
def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--temp_location", required=True)
    parser.add_argument("--staging_location", required=True)
    parser.add_argument("--runner", default="DataflowRunner")

    parser.add_argument("--input_format", choices=["parquet","csv"], default="parquet")
    parser.add_argument("--input_path", required=True)  # e.g., gs://bucket/ds/*.parquet
    parser.add_argument("--output_base", required=True) # gs://bucket/processed/exp1
    # Explicit, reproducible cutoffs:
    parser.add_argument("--train_end_ts", required=True)  # e.g., 2025-08-31T23:59:59
    parser.add_argument("--val_end_ts",   required=True)  # e.g., 2025-09-07T23:59:59
    parser.add_argument("--val_start_ts", required=False) # optional, defaults to >train_end_ts
    parser.add_argument("--test_start_ts", required=True) # e.g., 2025-09-08T00:00:00

    parser.add_argument("--shards_per_split", type=int, default=32)
    parser.add_argument("--use_sequence_example", action="store_true")
    args, beam_argv = parser.parse_known_args(argv)

    # Beam pipeline options
    options = PipelineOptions(beam_argv, save_main_session=True, streaming=False)
    gcp_opts = options.view_as(GoogleCloudOptions)
    gcp_opts.project = args.project
    gcp_opts.region = args.region
    gcp_opts.temp_location = args.temp_location
    gcp_opts.staging_location = args.staging_location
    options.view_as(SetupOptions).save_main_session = True
    worker_opts = options.view_as(WorkerOptions)
    # (optional) tune workers:
    # worker_opts.max_num_workers = 100
    # worker_opts.worker_machine_type = "n2-standard-4"

    train_out = os.path.join(args.output_base, "train")
    val_out   = os.path.join(args.output_base, "val")
    test_out  = os.path.join(args.output_base, "test")
    stats_out = os.path.join(args.output_base, "tfdv")
    tft_out   = os.path.join(args.output_base, "tft")  # TransformGraph export
    schema_out= os.path.join(args.output_base, "schema")

    raw_schema = _make_raw_schema()
    raw_feature_spec = schema_utils.schema_as_feature_spec(raw_schema).feature_spec

    with beam.Pipeline(options=options) as p:
        # 1) Read
        raw_rows = (read_input(p, args.input_format, args.input_path)
                    | "EnsureRequired" >> beam.ParDo(EnsureRequired()))

        # 2) Split by explicit timestamps (reproducible)
        # train: <= train_end_ts
        train_rows = (raw_rows
                      | "SplitTrain" >> beam.ParDo(KeepSplit("train", end_ts=args.train_end_ts)))

        # val: (val_start_ts or >train_end_ts) .. <= val_end_ts
        val_start = args.val_start_ts or args.train_end_ts
        val_rows = (raw_rows
                    | "SplitVal" >> beam.ParDo(KeepSplit("val", end_ts=args.val_end_ts, start_ts=val_start)))

        # test: >= test_start_ts (open-ended)
        test_rows = (raw_rows
                     | "SplitTest" >> beam.ParDo(KeepSplit("test", end_ts=None, start_ts=args.test_start_ts)))

        # 3) TFDV Stats & Schema
        for split_name, pcoll in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            split_stats = (pcoll
                           | f"GenStats-{split_name}" >> tfdv.GenerateStatistics())
            _ = (split_stats
                 | f"WriteStats-{split_name}" >> tfdv.WriteStatisticsToText(os.path.join(stats_out, split_name)))

        # Infer schema from TRAIN stats (or use a curated schema)
        train_stats = (train_rows | "GenStats-TrainSchema" >> tfdv.GenerateStatistics())
        inferred_schema = tfdv.infer_schema(statistics=train_stats)
        _ = (beam.pvalue.AsSingleton(train_stats), p | "WriteSchema" >> beam.Create([0])  # trigger write once
             | "SaveSchema" >> beam.Map(lambda _, s=inferred_schema: tf.io.gfile.GFile(os.path.join(schema_out,"schema.pbtxt"), "wb").write(s.SerializeToString())))

        # 4) TFT Analyze on TRAIN; Transform TRAIN/VAL/TEST
        with tft_beam.Context(temp_dir=os.path.join(args.temp_location, "tft_tmp")):
            # Convert dict->(dict, schema) for TFT
            train_ds = (train_rows, raw_schema)
            transformed_train_ds, transform_fn = (
                train_ds
                | "AnalyzeAndTransform" >> tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
            )
            # Apply same transform to val/test
            val_ds = (val_rows, raw_schema)
            test_ds = (test_rows, raw_schema)
            transformed_val_ds  = (val_ds  | "TransformVal"  >> tft_beam.TransformDataset(transform_fn))
            transformed_test_ds = (test_ds | "TransformTest" >> tft_beam.TransformDataset(transform_fn))

            transformed_train, transformed_metadata = transformed_train_ds
            transformed_val,  _ = transformed_val_ds
            transformed_test, _ = transformed_test_ds

            # Export the TransformGraph (to be used by training/serving)
            _ = (transform_fn
                 | "WriteTransformGraph" >> tft_beam.WriteTransformGraph(tft_out))

        # 5) Serialize to TFRecords (SequenceExample or Example)
        for split_name, pcoll in [("train", transformed_train),
                                  ("val",   transformed_val),
                                  ("test",  transformed_test)]:
            _ = (pcoll
                 | f"DictToTFExample-{split_name}" >> beam.ParDo(DictToExampleDoFn(use_sequence_example=args.use_sequence_example))
                 | f"WriteTFRecord-{split_name}" >> beam.io.tfrecordio.WriteToTFRecord(
                        file_path_prefix=os.path.join(args.output_base, split_name, "part"),
                        file_name_suffix=".tfrecord.gz",
                        num_shards=args.shards_per_split,
                        compression_type=beam.io.filesystem.CompressionTypes.GZIP
                    ))

if __name__ == "__main__":
    run()
