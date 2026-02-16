"""Kubeflow pipeline definitions for streaming directional change detection."""

from __future__ import annotations

from typing import List

from google_cloud_pipeline_components.v1.dataflow import DataflowPythonJobOp
from kfp import dsl

PROJECT = "latent-forecast-446118"
REGION = "us-central1"
BUCKET = "latent-forecast-446118-bucket"
PIPELINE_ROOT = f"gs://{BUCKET}/pipelines/directional-change"


def _threshold_args(thresholds: List[str]) -> List[str]:
    args: List[str] = []
    for threshold in thresholds:
        cleaned = threshold.strip()
        if not cleaned:
            continue
        args.extend(["--threshold", cleaned])
    if not args:
        raise ValueError("At least one threshold must be supplied to the pipeline.")
    return args


@dsl.pipeline(
    name="directional-change-streaming",
    pipeline_root=PIPELINE_ROOT,
)
def directional_change_streaming_pipeline(
    input_subscription: str,
    output_topic: str,
    thresholds: str = "0.01",
    dead_letter_topic: str = "",
    price_field: str = "price",
    timestamp_field: str = "timestamp",
    symbol_field: str = "symbol",
    max_num_workers: int = 5,
    machine_type: str = "n1-standard-4",
):
    """Launch a streaming Dataflow job that emits directional change events."""

    threshold_values = [t for t in thresholds.split(",") if t.strip()]
    threshold_cli_args = _threshold_args(threshold_values)

    job_args = [
        "--runner=DataflowRunner",
        f"--project={PROJECT}",
        f"--region={REGION}",
        f"--temp_location=gs://{BUCKET}/tmp",
        f"--staging_location=gs://{BUCKET}/staging",
        f"--input_subscription={input_subscription}",
        f"--output_topic={output_topic}",
        f"--price_field={price_field}",
        f"--timestamp_field={timestamp_field}",
        f"--symbol_field={symbol_field}",
        "--experiments=use_runner_v2",
        "--experiments=enable_streaming_engine",
        "--autoscaling_algorithm=THROUGHPUT_BASED",
        f"--worker_machine_type={machine_type}",
        f"--max_num_workers={max_num_workers}",
    ] + threshold_cli_args

    if dead_letter_topic:
        job_args.append(f"--dead_letter_topic={dead_letter_topic}")

    DataflowPythonJobOp(
        project=PROJECT,
        location=REGION,
        display_name="directional-change-streaming-job",
        python_module_path="dataflow_components.directional_change_streaming",
        temp_location=f"gs://{BUCKET}/tmp",
        staging_location=f"gs://{BUCKET}/staging",
        args=job_args,
    )
