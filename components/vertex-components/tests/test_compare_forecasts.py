# Tests for the compare_forecasts KFP component.
# Validates metric computation, statistical tests, and edge cases.

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: create a predictions CSV on disk
# ---------------------------------------------------------------------------
def _write_predictions_csv(tmp_dir: str, name: str, y_true: list, y_pred: list) -> str:
    """Write a predictions CSV and return the file path."""
    path = os.path.join(tmp_dir, f"{name}_predictions.csv")
    df = pd.DataFrame({
        "sample_index": list(range(len(y_true))),
        "y_true_scaled": y_true,
        "y_pred_scaled": y_pred,
    })
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Unit tests for the pure metric helpers (extracted from component)
# ---------------------------------------------------------------------------
class TestMetricHelpers:
    """Tests for individual metric computation functions."""

    def test_rmse(self):
        """RMSE of perfect predictions is 0."""
        from vertex_components.compare_forecasts_helpers import compute_rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        """RMSE with known error."""
        from vertex_components.compare_forecasts_helpers import compute_rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # errors = [1, 1, 1], MSE = 1, RMSE = 1
        assert compute_rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mae(self):
        """MAE of perfect predictions is 0."""
        from vertex_components.compare_forecasts_helpers import compute_mae

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(0.0)

    def test_mae_known_value(self):
        """MAE with known error."""
        from vertex_components.compare_forecasts_helpers import compute_mae

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 1.0, 5.0])
        # abs errors = [1, 1, 2], MAE = 4/3
        assert compute_mae(y_true, y_pred) == pytest.approx(4.0 / 3.0)

    def test_r_squared_perfect(self):
        """R² of perfect predictions is 1."""
        from vertex_components.compare_forecasts_helpers import compute_r_squared

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r_squared(y_true, y_pred) == pytest.approx(1.0)

    def test_r_squared_mean_predictor(self):
        """R² of a mean predictor is 0."""
        from vertex_components.compare_forecasts_helpers import compute_r_squared

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        mean_pred = np.full_like(y_true, y_true.mean())
        assert compute_r_squared(y_true, mean_pred) == pytest.approx(0.0)

    def test_smape_perfect(self):
        """sMAPE of perfect predictions is 0."""
        from vertex_components.compare_forecasts_helpers import compute_smape

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert compute_smape(y_true, y_pred) == pytest.approx(0.0)

    def test_smape_symmetric(self):
        """sMAPE is symmetric (swapping true/pred gives same result)."""
        from vertex_components.compare_forecasts_helpers import compute_smape

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        assert compute_smape(y_true, y_pred) == pytest.approx(
            compute_smape(y_pred, y_true)
        )

    def test_directional_accuracy_perfect(self):
        """Directional accuracy is 1.0 when all directions match."""
        from vertex_components.compare_forecasts_helpers import compute_directional_accuracy

        # Monotonically increasing: direction always positive
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_directional_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_directional_accuracy_worst(self):
        """Directional accuracy is 0.0 when all directions are opposite."""
        from vertex_components.compare_forecasts_helpers import compute_directional_accuracy

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert compute_directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_mape_known_value(self):
        """MAPE with known values."""
        from vertex_components.compare_forecasts_helpers import compute_mape

        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        # abs pct errors = [0.2, 0.1, 0.1], MAPE = 0.4/3
        expected = np.mean([0.2, 0.1, 0.1])
        assert compute_mape(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests for compute_all_metrics (the single-model summary)
# ---------------------------------------------------------------------------
class TestComputeAllMetrics:
    """Tests for the per-model metric aggregation."""

    def test_returns_all_expected_keys(self):
        """compute_all_metrics returns all expected metric keys."""
        from vertex_components.compare_forecasts_helpers import compute_all_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        result = compute_all_metrics(y_true, y_pred)
        expected_keys = {
            "rmse", "mae", "mape", "r_squared", "smape", "directional_accuracy",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_finite(self):
        """All metric values must be finite floats."""
        from vertex_components.compare_forecasts_helpers import compute_all_metrics

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = y_true + rng.standard_normal(100) * 0.1
        result = compute_all_metrics(y_true, y_pred)
        for k, v in result.items():
            assert math.isfinite(v), f"Metric {k} is not finite: {v}"


# ---------------------------------------------------------------------------
# Tests for head-to-head statistical comparison
# ---------------------------------------------------------------------------
class TestStatisticalComparison:
    """Tests for pairwise statistical tests between model arms."""

    def test_identical_predictions_high_pvalue(self):
        """When two models make identical predictions, p-values should be 1.0 (or NaN)."""
        from vertex_components.compare_forecasts_helpers import compare_two_models

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        result = compare_two_models(y_true, y_pred, y_pred)
        # When errors are identical, Wilcoxon test should give p-value of 1.0 or NaN
        assert result["wilcoxon_p"] is None or result["wilcoxon_p"] >= 0.05

    def test_significantly_different_models(self):
        """When one model is clearly better, relative improvement should be positive."""
        from vertex_components.compare_forecasts_helpers import compare_two_models

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(200)
        # Model A: very good
        y_pred_a = y_true + rng.standard_normal(200) * 0.01
        # Model B: much worse
        y_pred_b = y_true + rng.standard_normal(200) * 1.0
        result = compare_two_models(y_true, y_pred_a, y_pred_b)
        # A is better than B, so relative RMSE improvement (A vs B) should be positive
        assert result["rmse_relative_improvement_pct"] > 0

    def test_comparison_returns_all_keys(self):
        """Pairwise comparison returns all expected keys."""
        from vertex_components.compare_forecasts_helpers import compare_two_models

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_a = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        y_pred_b = np.array([1.5, 2.5, 2.5, 3.5, 5.5])
        result = compare_two_models(y_true, y_pred_a, y_pred_b)
        expected_keys = {
            "rmse_relative_improvement_pct",
            "wilcoxon_p",
            "paired_ttest_p",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests for the full comparison report builder
# ---------------------------------------------------------------------------
class TestBuildComparisonReport:
    """Tests for the full 3-arm comparison report."""

    @pytest.fixture
    def three_arm_csvs(self, tmp_path):
        """Create 3 prediction CSVs with known data."""
        rng = np.random.default_rng(123)
        n = 50
        y_true = rng.standard_normal(n).tolist()

        # Baseline: moderate error
        y_baseline = (np.array(y_true) + rng.standard_normal(n) * 0.5).tolist()
        # Single DC: small error
        y_single_dc = (np.array(y_true) + rng.standard_normal(n) * 0.1).tolist()
        # Multi DC: medium error
        y_multi_dc = (np.array(y_true) + rng.standard_normal(n) * 0.3).tolist()

        base_path = _write_predictions_csv(str(tmp_path), "baseline", y_true, y_baseline)
        single_path = _write_predictions_csv(str(tmp_path), "single_dc", y_true, y_single_dc)
        multi_path = _write_predictions_csv(str(tmp_path), "multi_dc", y_true, y_multi_dc)

        return base_path, single_path, multi_path

    def test_report_structure(self, three_arm_csvs):
        """Report has per_model and comparisons sections."""
        from vertex_components.compare_forecasts_helpers import build_comparison_report

        base_path, single_path, multi_path = three_arm_csvs
        report = build_comparison_report(base_path, single_path, multi_path)

        assert "per_model" in report
        assert "comparisons" in report
        assert set(report["per_model"].keys()) == {"baseline", "single_dc", "multi_dc"}

    def test_report_per_model_metrics(self, three_arm_csvs):
        """Each model entry in the report has all expected metrics."""
        from vertex_components.compare_forecasts_helpers import build_comparison_report

        base_path, single_path, multi_path = three_arm_csvs
        report = build_comparison_report(base_path, single_path, multi_path)

        expected_metrics = {"rmse", "mae", "mape", "r_squared", "smape", "directional_accuracy"}
        for arm_name in ["baseline", "single_dc", "multi_dc"]:
            assert set(report["per_model"][arm_name].keys()) == expected_metrics

    def test_report_comparisons_keys(self, three_arm_csvs):
        """Comparisons section has the expected pairwise entries."""
        from vertex_components.compare_forecasts_helpers import build_comparison_report

        base_path, single_path, multi_path = three_arm_csvs
        report = build_comparison_report(base_path, single_path, multi_path)

        expected_comparisons = {
            "single_dc_vs_baseline",
            "multi_dc_vs_baseline",
            "multi_dc_vs_single_dc",
        }
        assert set(report["comparisons"].keys()) == expected_comparisons

    def test_report_is_json_serializable(self, three_arm_csvs):
        """Report must be JSON serializable."""
        from vertex_components.compare_forecasts_helpers import build_comparison_report

        base_path, single_path, multi_path = three_arm_csvs
        report = build_comparison_report(base_path, single_path, multi_path)
        # Should not raise
        serialized = json.dumps(report)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_single_sample(self):
        """Metrics work (or gracefully degrade) with a single sample."""
        from vertex_components.compare_forecasts_helpers import compute_all_metrics

        y_true = np.array([1.0])
        y_pred = np.array([1.5])
        result = compute_all_metrics(y_true, y_pred)
        assert math.isfinite(result["rmse"])
        assert math.isfinite(result["mae"])

    def test_constant_predictions(self):
        """Constant predictions should produce finite metrics."""
        from vertex_components.compare_forecasts_helpers import compute_all_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, 2.5)
        result = compute_all_metrics(y_true, y_pred)
        for k, v in result.items():
            assert math.isfinite(v), f"{k} is not finite: {v}"

    def test_zero_true_values_mape(self):
        """MAPE with zero true values should not be infinite."""
        from vertex_components.compare_forecasts_helpers import compute_mape

        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.5, 2.5])
        result = compute_mape(y_true, y_pred)
        # Should handle zero gracefully (skip or cap)
        assert math.isfinite(result)

    def test_wilcoxon_with_few_samples(self):
        """Wilcoxon test with very few samples should return None gracefully."""
        from vertex_components.compare_forecasts_helpers import compare_two_models

        y_true = np.array([1.0, 2.0])
        y_pred_a = np.array([1.1, 2.1])
        y_pred_b = np.array([1.5, 2.5])
        result = compare_two_models(y_true, y_pred_a, y_pred_b)
        # Should not crash; p-values may be None
        assert "wilcoxon_p" in result
