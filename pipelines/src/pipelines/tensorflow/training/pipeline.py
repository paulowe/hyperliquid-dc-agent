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

import os
import pathlib
from kfp.v2 import compiler, dsl
from pipelines import generate_query
from bigquery_components import bq_query_to_table, extract_bq_to_dataset, load_dataset_to_bigquery
from vertex_components import (
    lookup_model,
    # sliding_window_transform,
    custom_train_job,
    custom_vae_train_job,
    custom_forecast_train_job,
    # import_model_evaluation,
    # update_best_model,
    # data_splitter,
    directional_change_detector,
    feature_engineering,
    # get_dc_configs,
    # unpack_dc_config,
    # get_model_resource_name,
    tf_data_splitter,
    concat_threshold_features,
    concat_latent_vectors,
    # cross_threshold_latent_forecast_train_job,
    pick_reference_scaler,
    window_dataset,
)

# Dataflow components
# from google_cloud_pipeline_components.v1.dataflow import DataflowPythonJobOp
# from google_cloud_pipeline_components.types.artifact_types import Artifact
# from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp


@dsl.pipeline(name="dc-vae-train-pipeline")
def tensorflow_pipeline(
    project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    project_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    model_name: str = "dcvae",
    dataset_id: str = "coindesk",
    dataset_location: str = os.environ.get("DATASET_LOCATION"),
    ingestion_dataset_id: str = "coindesk",
    timestamp: str = "2025-06-21 00:00:00",
    staging_bucket: str = os.environ.get("VERTEX_PIPELINE_ROOT"),
    pipeline_files_gcs_path: str = os.environ.get("PIPELINE_FILES_GCS_PATH"),
    test_dataset_uri: str = "",
    # New parameters for data extraction
    instrument: str = "BTC-USD",
    start_time: str = "2025-08-01 00:00:00",
    # end_time: str = "2025-08-31 23:59:59",
    end_time: str = "2025-08-08 00:00:00",
    dc_thresholds: list = [0.001, 0.005, 0.010, 0.015],
):
    """
    Tensorflow Keras training pipeline which:
     1. Extracts raw data from BigQuery using extract.sql
     2. Runs directional change detection for 4 different thresholds
     3. Splits and extracts datasets from BQ to GCS
     4. Trains models via Vertex AI CustomTrainingJob
     5. Evaluates the model against the current champion model
     6. If better the model becomes the new default model

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        pipeline_files_gcs_path (str): GCS path where the pipeline files are located
        ingestion_project_id (str): project id containing the source bigquery data
            for ingestion. This can be the same as `project_id` if the source data is
            in the same project where the ML pipeline is executed.
        model_name (str): name of model
        model_label (str): label of model
        dataset_id (str): id of BQ dataset used to store all staging data & predictions
        dataset_location (str): location of dataset
        ingestion_dataset_id (str): dataset id of ingestion data
        timestamp (str): Optional. Empty or a specific timestamp in ISO 8601 format
            (YYYY-MM-DDThh:mm:ss.sss±hh:mm or YYYY-MM-DDThh:mm:ss).
            If any time part is missing, it will be regarded as zero.
        staging_bucket (str): Staging bucket for pipeline artifacts.
        pipeline_files_gcs_path (str): GCS path where the pipeline files are located
        test_dataset_uri (str): Optional. GCS URI of statis held-out test dataset.
        instrument (str): Trading instrument (e.g., "BTC-USD")
        start_time (str): Start time for data extraction
        end_time (str): End time for data extraction
    """

    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline
    label_column_names = ["PRICE"]
    time_column = "load_time_toronto"
    raw_ticks_table = "coindesk_latest_tick"
    ingestion_table = "training"
    table_suffix = "_tf_training"  # suffix to table names
    ingested_table = "ingested_data" + table_suffix
    preprocessed_table = "preprocessed_data" + table_suffix
    train_table = "train_data" + table_suffix
    valid_table = "valid_data" + table_suffix
    test_table = "test_data" + table_suffix
    x1_table = "x1" + table_suffix
    x2_table = "x2" + table_suffix
    x3_table = "x3" + table_suffix
    x4_table = "x4" + table_suffix
    primary_metric = "rootMeanSquaredError"

    training_assets_path = f"{pipeline_files_gcs_path}/training/assets"
    # VAE training script
    # vae_train_script_uri = f"{training_assets_path}/train_vae.py"
    # vae_train_script_uri = f"{training_assets_path}/tftrain_vae.py"
    vae_train_script_uri = f"{training_assets_path}/tftrain_vae_nov8.py"
    # Forecasting training script
    # forecast_train_script_uri = f"{training_assets_path}/train_forecast.py"
    # forecast_train_script_uri = f"{training_assets_path}/train_tf_fast_model.py"
    forecast_train_script_uri = f"{training_assets_path}/tftrain_tf_fast_model.py"
    # Latent forecasting training script
    # latent_forecast_train_script_uri = (
    #     f"{training_assets_path}/train_latent_forecast.py"
    # )
    latent_forecast_train_script_uri = (
        f"{training_assets_path}/train_latent_forecast_nov10_tfrecords.py"
    )
    # Cross-threshold latent forecasting training script
    cross_threshold_latent_forecast_train_script_uri = (
        f"{training_assets_path}/train_cross_threshold_latent_forecast.py"
    )
    # Dataflow pipeline script
    # dataflow_pipeline_uri = f"{training_assets_path}/beam_tft_pipeline.py"

    hparams = dict(
        batch_size=100,
        epochs=5,
        loss_fn="MeanSquaredError",
        optimizer="Adam",
        learning_rate=0.01,
        hidden_units=[{"units": 64, "activation": "relu"}, {"units": 32, "activation": "relu"}],
        distribute_strategy="single",
        early_stopping_epochs=5,
        threshold=0.001,
    )

    # Step 1: Extract raw data from BigQuery
    queries_folder = pathlib.Path(__file__).parent / "queries"

    extract_query = generate_query(
        queries_folder / "extract.sql",
        project=ingestion_project_id,
        dataset_id=ingestion_dataset_id,
        table_id=raw_ticks_table,
        instrument=instrument,
        start_time=start_time,
        end_time=end_time,
    )

    # Create the extracted data table
    extracted_data_table = "extracted_ticks" + table_suffix
    extracted_data_bq = bq_query_to_table(
        query=extract_query,
        table_id=extracted_data_table,
        bq_client_project_id=project_id,
        destination_project_id=project_id,
        dataset_id=dataset_id,
        dataset_location=dataset_location,
        query_job_config={"write_disposition": "WRITE_TRUNCATE"},
    ).set_display_name("Save Extracted Ticks dataset to BQ")
    # .set_caching_options(enable_caching=False)

    # Extract the data to KFP managed GCS space
    extracted_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=extracted_data_table,
            dataset_location=dataset_location,
        )
        .after(extracted_data_bq)
        .set_display_name("Extract raw ticks to KFP managed storage")
        # .set_caching_options(enable_caching=False)
    ).outputs["dataset"]

    # dc_thresholds = [0.001, 0.005, 0.010, 0.015]
    # dc_config_gcs_uri = f"{pipeline_files_gcs_path}/training/assets/dc_config.json"
    # It means that by default up and down thresholds are the same
    # dc_thresholds = [(0.001,0.001), (0.005,0.005), (0.010,0.010), (0.015,0.015)]

    with dsl.ParallelFor(dc_thresholds, parallelism=0) as thr:
        dc_task = directional_change_detector(
            df=extracted_dataset,
            thresholds=thr,
            # experiment_name=cfg.exp_name,
            price_col="PRICE",
            # time_col="load_time_toronto",
            time_col="trade_ts",
            project_id=project_id,
            dataset_id=dataset_id,
            # table_id=cfg.dc_table,
            load_to_bq=True,
            fast_plot=True,
            max_event_markers=10,
        ).set_display_name("DC Detection")

        ticks = (
            feature_engineering(
                dc_summary=dc_task.outputs["directional_change_events"],
                raw_ticks=extracted_dataset,
                # threshold=cfg.thr,
                threshold=thr,
            )
            .after(dc_task)
            .set_display_name("Feature Engineer DC indicators")
        ).outputs["dc_ticks"]

        x_splits = tf_data_splitter(dataset=ticks).set_display_name(
            "Train/Validation/Test Split"
        )

        windowing_task = window_dataset(
            train_data=x_splits.outputs["train_data"],
            valid_data=x_splits.outputs["valid_data"],
            test_data=x_splits.outputs["test_data"],
            input_width=50,
            label_width=1,
            shift=50,
            batch_size=32,
            feature_names = [
                "PRICE",
                "vol_quote",
                "cvd_quote",
                "PDCC_Down",
                "OSV_Down",
                "OSV_Up",
                "PDCC2_UP",
                "regime_up",
                "regime_down",
            ],
            label_columns=["PRICE"],
        ).set_display_name("Windowing DC")

        # windowing_no_dc_task = window_dataset(
        #     train_data=x_splits.outputs["train_data"],
        #     valid_data=x_splits.outputs["valid_data"],
        #     test_data=x_splits.outputs["test_data"],
        #     input_width=50,
        #     label_width=1,
        #     shift=50,
        #     batch_size=32,
        #     feature_names = [
        #         "PRICE",
        #         "vol_quote",
        #         "cvd_quote",
        #     ],
        #     label_columns=["PRICE"],
        # ).set_display_name("Windowing No DC")


    # Get DC X-Threshold Features
    joint_threshold_features = concat_threshold_features(
        train_datasets=dsl.Collected(windowing_task.outputs["train_windowed"]),
        valid_datasets=dsl.Collected(windowing_task.outputs["valid_windowed"]),
        test_datasets=dsl.Collected(windowing_task.outputs["test_windowed"]),
        feature_names=[
            "PRICE",
            "vol_quote",
            "cvd_quote",
            "PDCC_Down",
            "OSV_Down",
            "OSV_Up",
            "PDCC2_UP",
            "regime_up",
            "regime_down",
        ],
        common_feature_names=[
            "PRICE_std",
            "vol_quote_std",
            "cvd_quote_std" 
        ],
        per_threshold_feature_names=[
            "PDCC_Down",
            "OSV_Down",
            "OSV_Up",
            "PDCC2_UP",
            "regime_up",
            "regime_down",
        ],
    ).set_caching_options(enable_caching=True).set_display_name("X-Thr Features")

    # Get common features
    no_dc_no_vae_features = concat_threshold_features(
            train_datasets=dsl.Collected(windowing_task.outputs["train_windowed"]),
            valid_datasets=dsl.Collected(windowing_task.outputs["valid_windowed"]),
            test_datasets=dsl.Collected(windowing_task.outputs["test_windowed"]),
            feature_names=["PRICE","vol_quote","cvd_quote"],
            common_feature_names=[
                "PRICE",
                "vol_quote",
                "cvd_quote" 
            ],
            per_threshold_feature_names=[],                   # no distinct features
        ).set_caching_options(enable_caching=True).set_display_name("Common Features")
    

    # Scale data

    # Train DC-VAE

    # train_vae = (
    #         custom_vae_train_job(
    #             train_script_uri=vae_train_script_uri,
    #             train_data=windowing_task.outputs["train_windowed"],
    #             valid_data=windowing_task.outputs["valid_windowed"],
    #             test_data=windowing_task.outputs["test_windowed"],
    #             project_id=project_id,
    #             project_location=project_location,
    #             # model_display_name=thr,
    #             train_container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
    #             serving_container_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
    #             hparams=hparams,
    #             threshold=thr,
    #             staging_bucket=staging_bucket,
    #             requirements=["google-cloud-logging", "google-cloud-aiplatform[cloud_profiler]","tensorboard_plugin_profile"],
    #         )
    #         .set_display_name("Train VAE")
    #         .set_caching_options(enable_caching=True)
    #     )

    # Train DC NoVAE Forecast
    # dc_no_vae_forecast = custom_forecast_train_job(
    #         train_script_uri=forecast_train_script_uri,
    #         train_data=joint_threshold_features.outputs["joint_train"],
    #         valid_data=joint_threshold_features.outputs["joint_valid"],
    #         test_data=joint_threshold_features.outputs["joint_test"],
    #         project_id=project_id,
    #         project_location=project_location,
    #         # model_display_name=f"{model_name}_forecast_engine_x1",
    #         train_container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
    #         serving_container_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
    #         hparams=hparams,
    #         threshold=7.7,
    #         staging_bucket=staging_bucket,
    #         requirements=["google-cloud-logging", "google-cloud-aiplatform[cloud_profiler]","tensorboard_plugin_profile"],
    #     ).set_caching_options(enable_caching=True).set_display_name("Train Forecast DC No VAE")
    
    # Train Baseline (No DC No VAE Forecast

    no_dc_no_vae_forecast = custom_forecast_train_job(
            train_script_uri=forecast_train_script_uri,
            train_data=no_dc_no_vae_features.outputs["joint_train"],
            valid_data=no_dc_no_vae_features.outputs["joint_valid"],
            test_data=no_dc_no_vae_features.outputs["joint_test"],
            project_id=project_id,
            project_location=project_location,
            # model_display_name=f"{model_name}_forecast_engine_x1",
            train_container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
            serving_container_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
            hparams=hparams,
            threshold=6.6,
            staging_bucket=staging_bucket,
            requirements=["google-cloud-logging", "google-cloud-aiplatform[cloud_profiler]","tensorboard_plugin_profile"],
        ).set_caching_options(enable_caching=True).set_display_name("Train No DC No VAE")

    load_dataset_to_bigquery(
        dataset=joint_latent_forecast.outputs["predictions"],
        destination_table_id=f"{project_id}.{dataset_id}.cross_threshold_model_predictions_10112025",
        destination_project_id=project_id,
        destination_dataset_id=dataset_id,
        destination_table_name="cross_threshold_model_predictions_10112025",
        bq_client_project_id=project_id,
        location=dataset_location,
        write_disposition="WRITE_TRUNCATE",
        create_disposition="CREATE_IF_NEEDED",
        source_format="CSV",
        autodetect=True,
        # schema_json='[{"name":"sample_index","type":"INTEGER","mode":"REQUIRED"},{"name":"horizon","type":"INTEGER","mode":"REQUIRED"},{"name":"y_true_raw","type":"FLOAT","mode":"REQUIRED"},{"name":"y_pred_raw","type":"FLOAT","mode":"REQUIRED"},{"name":"y_true_scaled","type":"FLOAT","mode":"REQUIRED"},{"name":"y_pred_scaled","type":"FLOAT","mode":"REQUIRED"},{"name":"rmse_raw","type":"FLOAT","mode":"REQUIRED"},{"name":"mae_raw","type":"FLOAT","mode":"REQUIRED"},{"name":"smape_raw","type":"FLOAT","mode":"REQUIRED"},{"name":"rmse_scaled","type":"FLOAT","mode":"REQUIRED"},{"name":"mae_scaled","type":"FLOAT","mode":"REQUIRED"},{"name":"smape_scaled","type":"FLOAT","mode":"REQUIRED"}]',
        csv_field_delimiter=",",
        csv_quote='"',
        csv_allow_quoted_newlines=True,
        csv_skip_leading_rows=1,
    ).after(joint_latent_forecast).set_display_name("Load Predictions to BQ")

    # cross_threshold_latent_forecast = cross_threshold_latent_forecast_train_job(
    #     train_script_uri=cross_threshold_latent_forecast_train_script_uri,
    #     train_data=joint_latent_concat.outputs["joint_train"],
    #     valid_data=joint_latent_concat.outputs["joint_valid"],
    #     test_data=joint_latent_concat.outputs["joint_test"],
    #     project_id=project_id,
    #     project_location=project_location,
    #     train_container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
    #     serving_container_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
    #     hparams={
    #         **hparams,
    #         "forecast_steps": 200,
    #         "compression": "GZIP",
    #         "latent_key": "z",
    #         "label_key": "y",
    #     },
    #     threshold=thr,
    #     staging_bucket=staging_bucket,
    #     scaler_artifact=joint_latent_concat.outputs["reference_scaler"],
    #     requirements=["google-cloud-logging"],
    # ).set_display_name("Train Cross-Threshold Latent Forecast")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=tensorflow_pipeline,
        package_path="training.json",
        type_check=False,
    )
