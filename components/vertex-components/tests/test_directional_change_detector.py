import pandas as pd
import pytest
import json
from pathlib import Path
from unittest import mock
from kfp.v2.dsl import Dataset

import vertex_components

# Get the plain Python function from the KFP component
directional_change_detector = vertex_components.directional_change_detector.python_func

# tmp_path is a pytest fixture
def test_directional_change_detector_basic(tmp_path):
    """End-to-end smoke test for the directional_change_detector component.

    The goal is to exercise the component exactly as Kubeflow Pipelines would,
    while staying local:
    1. Read a CSV from the input dataset path provided by the pipeline runtime.
    2. Generate the directional-change events CSV artifact.
    3. Persist figure metadata describing at least one saved Plotly figure.
    4. Invoke BigQuery's `load_table_from_dataframe` once, proving that summary
       rows would be uploaded when `load_to_bq=True`.

    All cloud SDKs are patched so the test never reaches external services; it
    only verifies that our code would make the expected calls.
    """

    # 1) Build a minimal price/timestamp dataset that reproduces a few DC events.
    df = pd.DataFrame(
        {
            "PRICE": [100, 99, 98, 101, 102, 100, 99, 97, 103],
            "load_time_toronto": pd.date_range("2024-01-01", periods=9, freq="T"),
        }
    )
    input_csv_path = tmp_path / "input.csv"
    df.to_csv(input_csv_path, index=False)

    # 2) Mirror the KFP contract: wrap file paths in lightweight Dataset objects.
    class DummyDataset:
        def __init__(self, path):
            self.path = str(path)

    input_dataset = DummyDataset(input_csv_path)
    output_events_path = tmp_path / "events.csv"
    output_figures_path = tmp_path / "figures"
    directional_change_events = DummyDataset(output_events_path)
    figures_output = DummyDataset(output_figures_path)

    # 3) Stub out BigQuery and GCS so the component stays offline.
    with mock.patch("google.cloud.bigquery.Client") as mock_bq_client, \
     mock.patch("google.cloud.storage.Client") as mock_storage_client:


        # Mock BigQuery client methods
        mock_bq_instance = mock_bq_client.return_value
        mock_bq_instance.get_table.side_effect = Exception("NotFound")  # simulate table missing
        mock_bq_instance.create_table.return_value = None
        mock_bq_instance.load_table_from_dataframe.return_value.result.return_value = None

        # Mock GCS client
        mock_storage_instance = mock_storage_client.return_value
        mock_bucket = mock_storage_instance.bucket.return_value
        mock_bucket.blob.return_value.upload_from_string.return_value = None

        # 4) Execute the component exactly as the pipeline would.
        directional_change_detector(
            df=input_dataset,
            thresholds=[0.01], 
            experiment_name="unit_test_exp",
            price_col="PRICE",
            time_col="load_time_toronto",
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
            load_to_bq=True,
            fast_plot=True,
            max_event_markers=100,
            directional_change_events=directional_change_events,
            figures_output=figures_output,
        )

    # 5) Assert that every promised side effect actually happened.
    # Events CSV should be materialized with the directional-change annotations.
    assert output_events_path.exists(), "Events CSV was not created"
    events_df = pd.read_csv(output_events_path)
    assert "event_type" in events_df.columns
    assert not events_df.empty

    # Figures metadata should capture the experiment context and saved figures.
    metadata_path = Path(output_figures_path) / "figures_metadata.json"
    assert metadata_path.exists(), "Figures metadata JSON was not created"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["experiment_name"] == "unit_test_exp"
    assert "experiment_id" in metadata

    # BigQuery load must be invoked once to publish the summary rows.
    mock_bq_instance.load_table_from_dataframe.assert_called_once()

    # At least one Plotly figure should have been exported.
    assert len(metadata["saved_figures"]) >= 1
