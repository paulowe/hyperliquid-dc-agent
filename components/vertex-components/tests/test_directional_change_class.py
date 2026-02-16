"""Unit tests that validate the behaviour of the DirectionalChange class.

This module focuses exclusively on the algorithm that powers directional change
analysis.  Each test verifies a distinct aspect of the class so we can detect
behavioural regressions quickly without needing live BigQuery or GCS access.
"""

# Standard library imports
import importlib.util  # Dynamically load the implementation module from the workspace
import types  # Build lightweight stand-ins for BigQuery and GCS clients
from pathlib import Path  # Resolve the path to the module under test

# Third-party imports
import pandas as pd  # Provide deterministic tabular data for the algorithm
import pytest  # Pytest gives us fixtures and assertion helpers


@pytest.fixture(scope="module")
def directional_change_module():
    """Return the directional_change_algorithm module with cloud clients stubbed."""
    # Resolve the absolute path to the implementation file inside the repo.
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vertex_components"
        / "directional_change_algorithm.py"
    )
    # Create an import specification that lets us load the module without touching
    # the package loader (important because vertex_components/__init__ has KFP side-effects).
    spec = importlib.util.spec_from_file_location(
        "directional_change_algorithm_under_test", module_path
    )
    # Materialise a new module object from the spec.
    module = importlib.util.module_from_spec(spec)
    # Execute the module so its contents populate the object we just created.
    spec.loader.exec_module(module)

    # The production code catches google.cloud.exceptions.NotFound; we replace it
    # with this lightweight variant so the branch behaves identically in tests.
    class StubNotFound(Exception):
        """Sentinel exception that mimics google.cloud.exceptions.NotFound."""

    # BigQuery produces table reference objects; this minimal clone carries the
    # same attributes so downstream assertions can inspect them.
    class FakeTableRef:
        """Store identifying components of a BigQuery table reference."""

        def __init__(self, project, dataset_id, table_id):
            self.project = project
            self.dataset_id = dataset_id
            self.table_id = table_id

    # load_table_from_dataframe returns a job whose .result() synchronises the
    # upload; here we simulate that contract with an empty shell.
    class StubJob:
        """Mimic a BigQuery load job that completes immediately."""

        def result(self):
            return None

    # The algorithm instantiates bigquery.Client and then calls dataset(),
    # get_table(), create_table(), load_table_from_dataframe(), and query().  This
    # fake client records every interaction so the tests can make precise assertions.
    class StubBigQueryClient:
        """Capture BigQuery traffic initiated by the DirectionalChange class."""

        def __init__(self, project=None):
            self.project = project
            self.created_tables = []  # Track schema creation attempts
            self.loaded_dfs = []  # Record dataframes pushed through load_table_from_dataframe
            self.query_calls = []  # Remember ad-hoc SQL queries

        def dataset(self, dataset_id):
            """Return an object whose table() method builds FakeTableRef values."""
            client = self

            class Dataset:
                def table(self, table_id):
                    return FakeTableRef(client.project, dataset_id, table_id)

                @property
                def dataset_id(self):  # pragma: no cover - simple attribute bridge
                    return dataset_id

            return Dataset()

        def get_table(self, table_ref):
            """Force the code path that creates the table when it is missing."""
            raise StubNotFound()

        def create_table(self, table):
            """Record that table creation was requested."""
            self.created_tables.append(table)

        def load_table_from_dataframe(self, df, table_ref):
            """Snapshot the dataframe that would be sent to BigQuery."""
            self.loaded_dfs.append((df.copy(), table_ref))
            return StubJob()

        def query(self, query):  # pragma: no cover - not exercised in these tests
            """Collect ad-hoc queries so future tests can assert on them."""
            self.query_calls.append(query)

            class EmptyResult:
                def to_dataframe(self):
                    return pd.DataFrame()

            return EmptyResult()

    # Plug the stubs into the module so every new DirectionalChange instance uses them.
    module.NotFound = StubNotFound
    module.bigquery = types.SimpleNamespace(
        Client=StubBigQueryClient,
        SchemaField=lambda *args, **kwargs: (args, kwargs),
        Table=lambda table_ref, schema=None: {"table_ref": table_ref, "schema": schema},
    )

    # Storage uploads are also intercepted; the tests do not rely on the return
    # value, so these stand-ins simply swallow the call.
    class FakeBlob:
        """Ignore GCS upload requests but keep the method signature intact."""

        def upload_from_string(self, *args, **kwargs):
            return None

    class FakeBucket:
        """Return FakeBlob for every blob() lookup."""

        def blob(self, *args, **kwargs):
            return FakeBlob()

    class FakeStorageClient:
        """Satisfy storage.Client(project=...) instantiation."""

        def bucket(self, *args, **kwargs):
            return FakeBucket()

    module.storage = types.SimpleNamespace(Client=FakeStorageClient)

    return module


@pytest.fixture
def price_series_df():
    """Provide a deterministic price series that triggers both down and up events."""
    # Prices deliberately fall by more than 5% then recover so both PDCC_Down and
    # PDCC2_UP events fire inside a short window.
    prices = [100, 102, 95, 94, 100]
    # Minute-by-minute timeline keeps the maths straightforward and reproducible.
    timeline = pd.date_range("2024-01-01", periods=len(prices), freq="min")
    # Build the DataFrame matching the schema DirectionalChange expects.
    return pd.DataFrame({"PRICE": prices, "load_time_toronto": timeline})


@pytest.fixture
def make_directional_change(directional_change_module, price_series_df):
    """Factory fixture that yields configured DirectionalChange instances."""

    def _builder(*, thresholds=None, load_to_bq=False):
        """Instantiate DirectionalChange with stubs and the sample dataframe."""
        return directional_change_module.DirectionalChange(
            df=price_series_df.copy(),  # keep the fixture immutable
            thresholds=thresholds or [0.05],  # default symmetric threshold
            experiment_name="unit-test",  # label test runs in outputs
            project_id="test-project",  # feed deterministic BigQuery identifiers
            dataset_id="unit_dataset",
            table_id="unit_table",
            load_to_bq=load_to_bq,  # toggle to reuse the factory for load tests
            fast_plot=False,  # avoid ScatterGL dependency during unit tests
            figures_output_path=None,  # disable figure exports
        )

    return _builder


def _events_by_type(events, label):
    """Filter event tuples to the ones matching a specific event_type label."""
    return [event for event in events if event[4] == label]


def test_run_all_detects_expected_events(make_directional_change, price_series_df):
    """DirectionalChange should produce the expected PDCC/OSV events for our data."""
    # Arrange: build the component with default thresholds against our sample series.
    dc = make_directional_change()
    # Act: run the detection workflow for every threshold (only one in this case).
    dc.run_all()

    # Assert: retrieve the computed event bundles and inspect their contents.
    result = dc.get_events(0.05)
    assert result is not None

    # Validate the single downward confirmation event: timestamps and prices should
    # match the pivot where the trend reversed downward.
    downward = _events_by_type(result["downward"], "PDCC_Down")
    assert len(downward) == 1
    start, end, start_price, end_price, _ = downward[0]
    assert start == price_series_df.loc[1, "load_time_toronto"]
    assert end == price_series_df.loc[2, "load_time_toronto"]
    assert start_price == price_series_df.loc[1, "PRICE"]
    assert end_price == price_series_df.loc[2, "PRICE"]

    # Overshoot updates should also align with the subsequent candle.
    osv_down = _events_by_type(result["downward"], "OSV_Down")
    assert len(osv_down) == 1
    _, osv_end, _, osv_end_price, _ = osv_down[0]
    assert osv_end == price_series_df.loc[3, "load_time_toronto"]
    assert osv_end_price == price_series_df.loc[3, "PRICE"]

    # The recovery path must yield one PDCC2_UP event covering the rebound.
    upward = _events_by_type(result["upward"], "PDCC2_UP")
    assert len(upward) == 1
    up_start, up_end, up_start_price, up_end_price, _ = upward[0]
    assert up_start == price_series_df.loc[3, "load_time_toronto"]
    assert up_end == price_series_df.loc[4, "load_time_toronto"]
    assert up_start_price == price_series_df.loc[3, "PRICE"]
    assert up_end_price == price_series_df.loc[4, "PRICE"]


def test_max_overshoot_metrics(make_directional_change):
    """max_OSV and max_OSV2 should reflect the peak overshoots observed."""
    # Running the pipeline populates the overshoot metrics we want to verify.
    dc = make_directional_change()
    dc.run_all()

    metrics = dc.get_events(0.05)
    assert metrics is not None
    # Downward overshoot should be a stable numeric value derived from the series.
    assert metrics["max_OSV"] == pytest.approx(0.21052631578947367)
    # Upward overshoot remains zero because the rebound does not exceed the
    # threshold once the up-trend is confirmed.
    assert metrics["max_OSV2"] == pytest.approx(0.0)


def test_event_annotations_written_to_dataframe(make_directional_change, price_series_df):
    """The working dataframe should be annotated with event labels for plotting."""
    dc = make_directional_change()
    dc.run_all()

    frame = dc.get_events(0.05)["df"]
    # Confirm the PDCC row is tagged correctly so Plotly renders the marker.
    pdcc_row = frame.loc[
        frame["load_time_toronto"] == price_series_df.loc[2, "load_time_toronto"]
    ]
    assert pdcc_row.iloc[0]["event"] == "PDCC_Down"
    assert pdcc_row.iloc[0]["event_type"] == "Downward Start"

    # Likewise ensure the upward confirmation receives the right labels.
    up_row = frame.loc[
        frame["load_time_toronto"] == price_series_df.loc[4, "load_time_toronto"]
    ]
    assert up_row.iloc[0]["event"] == "PDCC2_UP"
    assert up_row.iloc[0]["event_type"] == "Upward Start"


def test_load_dc_events_writes_expected_rows(directional_change_module, price_series_df):
    """When load_to_bq=True, summaries should be written with metadata included."""
    # Construct the detector with load_to_bq enabled so the BigQuery upload path runs.
    dc = directional_change_module.DirectionalChange(
        df=price_series_df.copy(),
        thresholds=[0.05],
        experiment_name="load-test",
        project_id="load-project",
        dataset_id="load_dataset",
        table_id="load_table",
        load_to_bq=True,
        fast_plot=False,
        figures_output_path=None,
    )
    dc.run_all()

    client = dc.client
    # First-time execution should create the table before loading rows.
    assert client.created_tables, "Expected DirectionalChange to create the BQ table"
    # After run_all, we should have exactly one dataframe queued for ingestion.
    assert client.loaded_dfs, "Expected DirectionalChange to load summary rows"

    loaded_df, table_ref = client.loaded_dfs[0]
    # The dataframe should contain one row for each of the three events we expect.
    assert list(loaded_df["event_type"]) == ["PDCC_Down", "OSV_Down", "PDCC2_UP"]
    # Timestamps must be normalised to UTC before ingestion.
    assert str(loaded_df["start_time"].dt.tz) == "UTC"
    # Experiment metadata ties the rows back to the pipeline run.
    assert loaded_df["experiment_name"].nunique() == 1
    assert loaded_df["experiment_id"].nunique() == 1
    # Table reference details should match the constructor inputs exactly.
    assert table_ref.project == "load-project"
    assert table_ref.dataset_id == "load_dataset"
    assert table_ref.table_id == "load_table"
