# Copyright 2022 Google LLC
from typing import List, Dict, Optional, Sequence

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import Input, component, Metrics, Output, Artifact, Dataset


@component(
    base_image="python:3.10",
    packages_to_install=[
        # Upgrade pip/setuptools so we get the latest PEP 517 hooks
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        # Force Cython < 3 so build‐time hooks don’t look for the removed API
        "cython<3.0.0",
        # Install PyYAML without isolation so it sees our Cython
        "--no-build-isolation",
        "pyyaml==5.4.1",
        # Other runtime deps
        "google-cloud-aiplatform==1.24.1",
        "python-json-logger",
        "google-cloud-logging",
        "tensorflow==2.17.0",
        "datetime",
    ],
)
def custom_vae_train_job(
    train_script_uri: str,
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    z_train: Output[Dataset],
    z_valid: Output[Dataset],
    z_test: Output[Dataset],
    project_id: str,
    project_location: str,
    # model_display_name: List[str],
    train_container_uri: str,
    serving_container_uri: str,
    model: Output[Artifact],
    metrics: Output[Metrics],
    staging_bucket: str,
    threshold: float,
    parent_model: str = None,
    model_display_name: Optional[str] = None,
    requirements: Optional[List[str]] = None,
    job_name: Optional[str] = None,
    hparams: Optional[Dict[str, str]] = None,
    # threshold: Optional[List[float]] = None,
    replica_count: int = 1,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
    accelerator_count: int = 0,
):
    """Run a custom training job using a training script.

    The provided script will be invoked by passing the following command-line arguments:

    ```
    train.py \
        --train_data <train_data.path> \
        --valid_data <valid_data.path> \
        --test_data <test_data.path> \
        --metrics <metrics.path> \
        --hparams json.dumps(<hparams>)
    ```

    Ensure that your train script can read these arguments and outputs metrics
    to the provided path and the model to the correct path based on:
    https://cloud.google.com/vertex-ai/docs/training/code-requirements.

    Args:
        train_script_uri (str): gs:// uri to python train script. See:
            https://cloud.google.com/vertex-ai/docs/training/code-requirements.
        train_data (Dataset): Training data (passed as an argument to train script)
        valid_data (Dataset): Validation data (passed as an argument to train script)
        test_data (Dataset): Test data (passed as an argument to train script).
        staging_bucket (str): Staging bucket for CustomTrainingJob.
        project_location (str): location of the Google Cloud project.
        project_id (str): project id of the Google Cloud project.
        model_display_name (str): Name of the new trained model version.
        train_container_uri (str): Container URI for running train script.
        serving_container_uri (str): Container URI for deploying the output model.
        model (Model): Trained model output.
        metrics (Metrics): Output metrics of trained model.
        requirements (List[str]): Additional python dependencies for training script.
        job_name (str): Name of training job.
        hparams (Dict[str, str]): Hyperparameters (passed as a JSON serialised argument
            to train script)
        replica_count (int): Number of replicas (increase for distributed training).
        machine_type (str): Machine type of compute.
        accelerator_type (str): Accelerator type (change for GPU support).
        accelerator_count (str): Accelerator count (increase for GPU cores).
        parent_model (str): Resource name of existing parent model (optional).
            If `None`, a new model will be uploaded. Otherwise, a new model version
            for the parent model will be uploaded.
    Returns:
        parent_model (str): Resource URI of the parent model (empty string if the
            trained model is the first model version of its kind).
    """
    import json
    import logging
    import os.path
    import time
    import tensorflow as tf
    import google.cloud.aiplatform as aip
    from datetime import datetime

    logging.info(f"Using train script: {train_script_uri}")
    script_path = "/gcs/" + train_script_uri[5:]
    if not os.path.exists(script_path):
        raise ValueError(
            "Train script was not found. "
            f"Check if the path is correct: {train_script_uri}"
        )

    # display_name = _coerce_single_value(model_display_name)
    thr_str = str(threshold).replace('.', 'p')
    if model_display_name is None:
        model_display_name = f"vae_model_{thr_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        model_display_name = model_display_name
    job_name = f"vae_job_{thr_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    job = aip.CustomTrainingJob(
        project=project_id,
        location=project_location,
        staging_bucket=staging_bucket,
        display_name=job_name if job_name else f"Custom job {int(time.time())}",
        script_path=script_path,
        container_uri=train_container_uri,
        requirements=requirements,
        model_serving_container_image_uri=serving_container_uri,
    )
    logging.info(f"train_data path: {train_data.path}")

    # Add threshold to hparams
    hp = dict(hparams or {})
    if threshold is not None:
        hp["threshold"] = float(threshold)
    hp_json = json.dumps(hp)

    cmd_args = [
        f"--train_data={train_data.path}",
        f"--valid_data={valid_data.path}",
        f"--test_data={test_data.path}",
        f"--z_train={z_train.path}",
        f"--z_valid={z_valid.path}",
        f"--z_test={z_test.path}",
        f"--metrics={metrics.path}",
        f"--hparams={hp_json}",
    ]
    # https://cloud.google.com/python/docs/reference/aiplatform/1.18.3/google.cloud.aiplatform.CustomTrainingJob#google_cloud_aiplatform_CustomTrainingJob_run
    uploaded_model = job.run(
        model_display_name=model_display_name,
        parent_model=parent_model,
        is_default_version=(parent_model is None),
        args=cmd_args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )

    # resource_name = f"{uploaded_model.resource_name}@{uploaded_model.version_id}"
    resource_name = uploaded_model.resource_name
    if "@" not in resource_name and getattr(uploaded_model, "version_id", None):
        resource_name = f"{resource_name}@{uploaded_model.version_id}"

    model.metadata["resourceName"] = resource_name
    model.metadata["displayName"] = uploaded_model.display_name
    model.metadata["containerSpec"] = {"imageUri": serving_container_uri}
    model.uri = uploaded_model.uri
    model.TYPE_NAME = "google.VertexModel"


    model_root = uploaded_model.uri.rstrip("/")
    latents_root = f"{model_root}/latents"

    # Prefer LATEST
    index_path = None
    latest_ptr = f"{latents_root}/LATEST"
    if tf.io.gfile.exists(latest_ptr):
        run_id = tf.io.gfile.GFile(latest_ptr, "r").read().strip()
        index_path = f"{latents_root}/{run_id}/_index.json"

    # Fallback: newest _index.json
    if index_path is None or not tf.io.gfile.exists(index_path):
        candidates = tf.io.gfile.glob(f"{latents_root}/*/_index.json")
        if not candidates:
            raise FileNotFoundError(
                f"No _index.json found under {latents_root}. "
                "Ensure Train Script 1 writes run-scoped latents + index."
            )
        index_path = max(candidates, key=lambda p: tf.io.gfile.stat(p).mtime_nsec)

    with tf.io.gfile.GFile(index_path, "r") as f:
        idx = json.load(f)

    z_train.uri = idx["splits"]["train"].rstrip("/")
    z_valid.uri = idx["splits"]["valid"].rstrip("/")
    z_test.uri = idx["splits"]["test"].rstrip("/")

    logging.info(f"[VAE wrapper] z_train.uri = {z_train.uri}")
    logging.info(f"[VAE wrapper] z_valid.uri = {z_valid.uri}")
    logging.info(f"[VAE wrapper] z_test.uri  = {z_test.uri}")

    # 4) Optional: surface helpful metadata (read from split manifest if present)
    def _apply_manifest_meta(split_dir, out_ds):
        mpath = f"{split_dir}/_manifest.json"
        if tf.io.gfile.exists(mpath):
            with tf.io.gfile.GFile(mpath, "r") as f:
                man = json.load(f)
            out_ds.metadata["format"] = "tfrecord"
            out_ds.metadata["compression"] = man.get("compression", "")
            feats = man.get("features", {})
            if "z" in feats and "shape" in feats["z"]:
                out_ds.metadata["latentDim"] = feats["z"]["shape"][0]
            if "y" in feats and "shape" in feats["y"]:
                out_ds.metadata["forecastSteps"] = feats["y"]["shape"][0]
        else:
            # Sensible defaults if no manifest
            out_ds.metadata["format"] = "tfrecord"

    _apply_manifest_meta(z_train.uri, z_train)
    _apply_manifest_meta(z_valid.uri, z_valid)
    _apply_manifest_meta(z_test.uri, z_test)

    with open(metrics.path, "r") as fp:
        parsed_metrics = json.load(fp)

    logging.info(parsed_metrics)
    for k, v in parsed_metrics.items():
        if type(v) is float:
            metrics.log_metric(k, v)
