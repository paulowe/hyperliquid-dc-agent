# DC Feature Ablation Pipeline (v2: Equal Capacity + Regularization + More Data)
#
# 3-arm study comparing identical model architectures on different feature sets:
#   Baseline  : [PRICE_std, vol_quote_std, cvd_quote_std]             -> [50, 3]
#   Single-DC : 3 common + 6 DC features (threshold=0.001)           -> [50, 9]
#   Multi-DC  : 3 common + 6x4 DC features (all 4 thresholds)        -> [50, 27]
#
# All arms share: same ticks, same 70/20/10 split, same windowing params
# (input_width=50, shift=50, label_width=1), same target (PRICE_std).
#
# v2 improvements over v1:
# - Bottleneck projection layer (Dense(32)) after Flatten equalizes capacity
#   across all arms regardless of input feature count
# - Dropout (0.3) and L2 regularization (1e-4) to prevent overfitting
# - 3 months of data (May-Aug 2025) instead of 7 days
# - 20 epochs with early stopping (patience=5)

import os
import pathlib

from kfp.v2 import compiler, dsl
from pipelines import generate_query
from bigquery_components import (
    bq_query_to_table,
    extract_bq_to_dataset,
)
from vertex_components import (
    directional_change_detector,
    feature_engineering,
    tf_data_splitter,
    window_dataset,
    concat_threshold_features,
    custom_forecast_train_job,
    compare_forecasts,
)

# ---------------------------------------------------------------------------
# Feature name constants (matching pipeline-v1-nov20.py convention)
# ---------------------------------------------------------------------------
# Common features (used by all arms)
COMMON_FEATURES = ["PRICE_std", "vol_quote_std", "cvd_quote_std"]

# DC-specific features (per threshold)
DC_FEATURES = [
    "PDCC_Down",
    "OSV_Down_std",
    "OSV_Up_std",
    "PDCC2_UP",
    "regime_up",
    "regime_down",
]

# Full single-threshold feature set (common + DC = 9 features)
SINGLE_DC_FEATURES = COMMON_FEATURES + DC_FEATURES

# Label column
LABEL_COLUMNS = ["PRICE_std"]


@dsl.pipeline(name="dc-ablation-pipeline")
def ablation_pipeline(
    project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    project_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    dataset_id: str = "coindesk",
    dataset_location: str = os.environ.get("DATASET_LOCATION"),
    ingestion_dataset_id: str = "coindesk",
    staging_bucket: str = os.environ.get("VERTEX_PIPELINE_ROOT"),
    pipeline_files_gcs_path: str = os.environ.get("PIPELINE_FILES_GCS_PATH"),
    instrument: str = "BTC-USD",
    start_time: str = "2025-05-01 00:00:00",
    end_time: str = "2025-08-01 00:00:00",
    dc_thresholds: list = [0.001, 0.005, 0.010, 0.015],
    single_dc_threshold: float = 0.001,
    input_width: int = 50,
    shift: int = 50,
    label_width: int = 1,
):
    """3-arm DC feature ablation pipeline (v2: equal capacity + regularization).

    Trains models with bottleneck projection layer on:
      1. Baseline: 3 common features only (no DC)
      2. Single-DC: 3 common + 6 DC features from one threshold
      3. Multi-DC: 3 common + 6*4 DC features from all thresholds

    All arms use: Flatten -> Dense(32, bottleneck) -> Dropout(0.3) -> Dense(64, l2) ->
    Dropout(0.3) -> Dense(32, l2) -> Dense(1). The bottleneck equalizes effective
    capacity regardless of input feature count.

    Then compares all three arms with statistical tests.
    """
    # ---------------------------------------------------------------
    # Shared configuration
    # ---------------------------------------------------------------
    table_suffix = "_ablation"
    training_assets_path = f"{pipeline_files_gcs_path}/training/assets"
    forecast_train_script_uri = f"{training_assets_path}/tftrain_tf_fast_model.py"

    hparams = dict(
        batch_size=100,
        epochs=20,
        # NOTE: loss_fn and hidden_units are NOT read by the training script.
        # Actual loss = MeanAbsoluteError (hardcoded in tftrain_tf_fast_model.py:861)
        # Actual hidden = Dense(64) -> Dense(32) (hardcoded in build_multi_step_dense)
        # Actual optimizer = AdamW (not Adam)
        loss_fn="MeanAbsoluteError",
        optimizer="AdamW",
        learning_rate=0.01,
        hidden_units=[
            {"units": 64, "activation": "relu"},
            {"units": 32, "activation": "relu"},
        ],
        distribute_strategy="single",
        patience=5,
        # exp 012: re-run exp 005 config with seed=42 for reproducibility verification
        bottleneck_dim=128,
        dropout_rate=0.0,
        l2_reg=0.0,
        random_seed=42,
    )

    train_container_uri = (
        "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
    )
    serving_container_uri = (
        "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
    )
    train_requirements = [
        "google-cloud-logging",
        "google-cloud-aiplatform[cloud_profiler]",
        "tensorboard_plugin_profile",
    ]

    # ---------------------------------------------------------------
    # Step 1: Extract raw ticks from BigQuery
    # ---------------------------------------------------------------
    queries_folder = pathlib.Path(__file__).parent / "queries"

    extract_query = generate_query(
        queries_folder / "extract.sql",
        project=ingestion_project_id,
        dataset_id=ingestion_dataset_id,
        table_id="coindesk_latest_tick",
        instrument=instrument,
        start_time=start_time,
        end_time=end_time,
    )

    extracted_data_table = "extracted_ticks" + table_suffix
    extracted_data_bq = bq_query_to_table(
        query=extract_query,
        table_id=extracted_data_table,
        bq_client_project_id=project_id,
        destination_project_id=project_id,
        dataset_id=dataset_id,
        dataset_location=dataset_location,
        query_job_config={"write_disposition": "WRITE_TRUNCATE"},
    ).set_display_name("Extract Ticks to BQ")

    extracted_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=extracted_data_table,
            dataset_location=dataset_location,
        )
        .after(extracted_data_bq)
        .set_display_name("Extract Ticks to GCS")
    ).outputs["dataset"]

    # ---------------------------------------------------------------
    # Step 2: DC detection + feature engineering for all thresholds
    # (ParallelFor produces windowed datasets per threshold)
    # ---------------------------------------------------------------
    with dsl.ParallelFor(dc_thresholds, parallelism=0) as thr:
        dc_task = directional_change_detector(
            df=extracted_dataset,
            thresholds=thr,
            price_col="PRICE",
            time_col="trade_ts",
            project_id=project_id,
            dataset_id=dataset_id,
            load_to_bq=True,
            fast_plot=True,
            max_event_markers=10,
        ).set_display_name("DC Detection")

        ticks = (
            feature_engineering(
                dc_summary=dc_task.outputs["directional_change_events"],
                raw_ticks=extracted_dataset,
                threshold=thr,
            )
            .after(dc_task)
            .set_display_name("Feature Engineering")
        ).outputs["dc_ticks"]

        x_splits = tf_data_splitter(dataset=ticks).set_display_name(
            "Train/Val/Test Split"
        )

        # Window with full DC feature set (9 features per threshold)
        windowing_task = window_dataset(
            train_data=x_splits.outputs["train_data"],
            valid_data=x_splits.outputs["valid_data"],
            test_data=x_splits.outputs["test_data"],
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            batch_size=32,
            feature_names=SINGLE_DC_FEATURES,
            label_columns=LABEL_COLUMNS,
        ).set_display_name("Window (9 features)")

    # ---------------------------------------------------------------
    # Step 3: Baseline arm — common features only (3 features)
    # Uses concat_threshold_features with per_threshold_feature_names=[]
    # which keeps only the 3 common features.
    # ---------------------------------------------------------------
    baseline_features = concat_threshold_features(
        train_datasets=dsl.Collected(windowing_task.outputs["train_windowed"]),
        valid_datasets=dsl.Collected(windowing_task.outputs["valid_windowed"]),
        test_datasets=dsl.Collected(windowing_task.outputs["test_windowed"]),
        feature_names=SINGLE_DC_FEATURES,
        common_feature_names=COMMON_FEATURES,
        per_threshold_feature_names=[],
    ).set_caching_options(enable_caching=True).set_display_name(
        "Baseline Features (3)"
    )

    baseline_train = custom_forecast_train_job(
        train_script_uri=forecast_train_script_uri,
        train_data=baseline_features.outputs["joint_train"],
        valid_data=baseline_features.outputs["joint_valid"],
        test_data=baseline_features.outputs["joint_test"],
        project_id=project_id,
        project_location=project_location,
        train_container_uri=train_container_uri,
        serving_container_uri=serving_container_uri,
        hparams={**hparams, "model_label": "baseline"},
        threshold=0.0,
        staging_bucket=staging_bucket,
        model_display_name="ablation_baseline",
        requirements=train_requirements,
    ).set_caching_options(enable_caching=False).set_display_name(
        "Train Baseline (3 features)"
    )

    # ---------------------------------------------------------------
    # Step 4: Single-DC arm — one threshold's DC features (9 features)
    # Runs its own DC detection path for threshold=0.001.
    # ---------------------------------------------------------------
    single_dc_detect = directional_change_detector(
        df=extracted_dataset,
        thresholds=single_dc_threshold,
        price_col="PRICE",
        time_col="trade_ts",
        project_id=project_id,
        dataset_id=dataset_id,
        load_to_bq=False,
        fast_plot=True,
        max_event_markers=10,
    ).set_display_name("DC Detection (single)")

    single_dc_ticks = (
        feature_engineering(
            dc_summary=single_dc_detect.outputs["directional_change_events"],
            raw_ticks=extracted_dataset,
            threshold=single_dc_threshold,
        )
        .after(single_dc_detect)
        .set_display_name("Feature Eng (single)")
    ).outputs["dc_ticks"]

    single_dc_splits = tf_data_splitter(dataset=single_dc_ticks).set_display_name(
        "Split (single DC)"
    )

    single_dc_windowing = window_dataset(
        train_data=single_dc_splits.outputs["train_data"],
        valid_data=single_dc_splits.outputs["valid_data"],
        test_data=single_dc_splits.outputs["test_data"],
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        batch_size=32,
        feature_names=SINGLE_DC_FEATURES,
        label_columns=LABEL_COLUMNS,
    ).set_display_name("Window Single-DC (9 features)")

    single_dc_train = custom_forecast_train_job(
        train_script_uri=forecast_train_script_uri,
        train_data=single_dc_windowing.outputs["train_windowed"],
        valid_data=single_dc_windowing.outputs["valid_windowed"],
        test_data=single_dc_windowing.outputs["test_windowed"],
        project_id=project_id,
        project_location=project_location,
        train_container_uri=train_container_uri,
        serving_container_uri=serving_container_uri,
        hparams={**hparams, "model_label": "single_dc", "threshold": single_dc_threshold},
        threshold=single_dc_threshold,
        staging_bucket=staging_bucket,
        model_display_name="ablation_single_dc",
        requirements=train_requirements,
    ).set_caching_options(enable_caching=False).set_display_name(
        "Train Single-DC (9 features)"
    )

    # ---------------------------------------------------------------
    # Step 5: Multi-DC arm — all thresholds concatenated (27 features)
    # Uses concat_threshold_features to merge DC features across thresholds.
    # ---------------------------------------------------------------
    multi_dc_features = concat_threshold_features(
        train_datasets=dsl.Collected(windowing_task.outputs["train_windowed"]),
        valid_datasets=dsl.Collected(windowing_task.outputs["valid_windowed"]),
        test_datasets=dsl.Collected(windowing_task.outputs["test_windowed"]),
        feature_names=SINGLE_DC_FEATURES,
        common_feature_names=COMMON_FEATURES,
        per_threshold_feature_names=DC_FEATURES,
    ).set_caching_options(enable_caching=True).set_display_name(
        "Multi-DC Features (27)"
    )

    multi_dc_train = custom_forecast_train_job(
        train_script_uri=forecast_train_script_uri,
        train_data=multi_dc_features.outputs["joint_train"],
        valid_data=multi_dc_features.outputs["joint_valid"],
        test_data=multi_dc_features.outputs["joint_test"],
        project_id=project_id,
        project_location=project_location,
        train_container_uri=train_container_uri,
        serving_container_uri=serving_container_uri,
        hparams={**hparams, "model_label": "multi_dc"},
        threshold=9.9,  # sentinel: multi-threshold arm
        staging_bucket=staging_bucket,
        model_display_name="ablation_multi_dc",
        requirements=train_requirements,
    ).set_caching_options(enable_caching=False).set_display_name(
        "Train Multi-DC (27 features)"
    )

    # ---------------------------------------------------------------
    # Step 6: Compare all 3 arms
    # ---------------------------------------------------------------
    comparison = compare_forecasts(
        baseline_predictions=baseline_train.outputs["predictions"],
        single_dc_predictions=single_dc_train.outputs["predictions"],
        multi_dc_predictions=multi_dc_train.outputs["predictions"],
    ).set_display_name("Compare Forecasts")

    # ---------------------------------------------------------------
    # Step 7: Load comparison report to BigQuery
    # NOTE: Disabled — the compare_forecasts report is nested JSON (not tabular),
    # and the load_dataset_to_bigquery component hardcodes /*.csv URI patterns.
    # Results are available in GCS at the compare-forecasts task output path.
    # ---------------------------------------------------------------


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ablation_pipeline,
        package_path=str(pathlib.Path(__file__).parent.parent.parent.parent / "ablation.json"),
        type_check=False,
    )
