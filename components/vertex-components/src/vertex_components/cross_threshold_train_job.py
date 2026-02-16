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
        "datetime",
    ],
)
def custom_train_job(
    train_script_uri: str,
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    project_id: str,
    project_location: str,
    # model_display_name: List[str],
    
    train_container_uri: str,
    serving_container_uri: str,
    model: Output[Artifact],
    metrics: Output[Metrics],
    staging_bucket: str,
    # scaler_artifact: Input[Artifact],
    threshold: float,
    parent_model: str = None,
    requirements: Optional[List[str]] = None,
    job_name: Optional[str] = None,
    hparams: Optional[Dict[str, str]] = None,
    # threshold: Optional[List[float]] = None,
    model_display_name: Optional[str] = None,
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
        scaler_artifact (Artifact): Optional fitted scaler artifact to reuse downstream.
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
    import google.cloud.aiplatform as aip
    from datetime import datetime

    logging.info(f"Using train script: {train_script_uri}")
    script_path = "/gcs/" + train_script_uri[5:]
    if not os.path.exists(script_path):
        raise ValueError(
            "Train script was not found. "
            f"Check if the path is correct: {train_script_uri}"
        )


    thr_str = str(threshold).replace('.', 'p')
    if model_display_name is None:
        model_display_name = f"latent_forecast_model_{thr_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        model_display_name = model_display_name
    job_name = f"latent_forecast_job_{thr_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

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
        f"--train_data={train_data.uri}",
        f"--valid_data={valid_data.uri}",
        f"--test_data={test_data.uri}",
        f"--metrics={metrics.path}",
        f"--hparams={hp_json}",
    ]
    if scaler_artifact is not None:
        scaler_uri = getattr(scaler_artifact, "uri", "") or getattr(
            scaler_artifact, "path", ""
        )
        if scaler_uri.startswith("/gcs/"):
            scaler_uri = "gs://" + scaler_uri[5:]
        if scaler_uri:
            cmd_args.append(f"--scaler_artifact_uri={scaler_uri}")
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

    with open(metrics.path, "r") as fp:
        parsed_metrics = json.load(fp)

    logging.info(parsed_metrics)
    # for k, v in parsed_metrics.items():
    #     if type(v) is float:
    #         metrics.log_metric(k, v)
    for k, v in parsed_metrics.items():
        if isinstance(v, (float, int)) and v is not None:
            metrics.log_metric(k, float(v))
        else:
            # Non-numeric values are logged to GCP logs for debugging,
            # but won’t appear in Vertex UI
            logging.info(f"Non-numeric metric {k}: {v}")
