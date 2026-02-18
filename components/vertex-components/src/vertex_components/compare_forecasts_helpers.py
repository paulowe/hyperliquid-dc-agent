# Pure-Python helper functions for computing forecast comparison metrics.
# Kept separate from the KFP component so they can be unit-tested directly.

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-sample metric helpers
# ---------------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, skipping samples where y_true == 0."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-2 scale)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero when both true and pred are 0
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of consecutive steps where pred direction matches true direction."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


# ---------------------------------------------------------------------------
# Aggregate all metrics for one model
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all per-model metrics and return as a dict."""
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "r_squared": compute_r_squared(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Pairwise statistical comparison
# ---------------------------------------------------------------------------

def compare_two_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compare model A against model B.

    Returns:
        rmse_relative_improvement_pct: positive means A is better
        wilcoxon_p: Wilcoxon signed-rank test p-value on squared errors
        paired_ttest_p: paired t-test p-value on absolute errors
    """
    from scipy import stats

    rmse_a = compute_rmse(y_true, y_pred_a)
    rmse_b = compute_rmse(y_true, y_pred_b)

    # Relative RMSE improvement: positive means A is better than B
    if rmse_b == 0:
        rel_improvement = 0.0
    else:
        rel_improvement = (rmse_b - rmse_a) / rmse_b * 100.0

    # Squared errors per sample
    sq_err_a = (y_true - y_pred_a) ** 2
    sq_err_b = (y_true - y_pred_b) ** 2

    # Absolute errors per sample
    abs_err_a = np.abs(y_true - y_pred_a)
    abs_err_b = np.abs(y_true - y_pred_b)

    # Wilcoxon signed-rank test on squared errors
    wilcoxon_p: Optional[float] = None
    try:
        diff = sq_err_a - sq_err_b
        # Wilcoxon requires at least 1 non-zero difference
        if np.any(diff != 0) and len(diff) >= 5:
            _, wilcoxon_p = stats.wilcoxon(sq_err_a, sq_err_b)
            wilcoxon_p = float(wilcoxon_p)
    except Exception:
        wilcoxon_p = None

    # Paired t-test on absolute errors
    ttest_p: Optional[float] = None
    try:
        if len(abs_err_a) >= 2:
            _, ttest_p = stats.ttest_rel(abs_err_a, abs_err_b)
            ttest_p = float(ttest_p)
    except Exception:
        ttest_p = None

    return {
        "rmse_relative_improvement_pct": float(rel_improvement),
        "wilcoxon_p": wilcoxon_p,
        "paired_ttest_p": ttest_p,
    }


# ---------------------------------------------------------------------------
# Full 3-arm comparison report
# ---------------------------------------------------------------------------

def build_comparison_report(
    baseline_csv: str,
    single_dc_csv: str,
    multi_dc_csv: str,
) -> dict:
    """Build a complete comparison report from 3 prediction CSV files.

    Each CSV must have columns: sample_index, y_true_scaled, y_pred_scaled.
    """
    # Read CSVs
    df_base = pd.read_csv(baseline_csv)
    df_single = pd.read_csv(single_dc_csv)
    df_multi = pd.read_csv(multi_dc_csv)

    y_true_base = df_base["y_true_scaled"].to_numpy()
    y_pred_base = df_base["y_pred_scaled"].to_numpy()
    y_true_single = df_single["y_true_scaled"].to_numpy()
    y_pred_single = df_single["y_pred_scaled"].to_numpy()
    y_true_multi = df_multi["y_true_scaled"].to_numpy()
    y_pred_multi = df_multi["y_pred_scaled"].to_numpy()

    # Per-model metrics
    per_model = {
        "baseline": compute_all_metrics(y_true_base, y_pred_base),
        "single_dc": compute_all_metrics(y_true_single, y_pred_single),
        "multi_dc": compute_all_metrics(y_true_multi, y_pred_multi),
    }

    # Pairwise comparisons (A vs B means "how much better is A than B")
    comparisons = {
        "single_dc_vs_baseline": compare_two_models(
            y_true_base, y_pred_single, y_pred_base,
        ),
        "multi_dc_vs_baseline": compare_two_models(
            y_true_base, y_pred_multi, y_pred_base,
        ),
        "multi_dc_vs_single_dc": compare_two_models(
            y_true_single, y_pred_multi, y_pred_single,
        ),
    }

    return {
        "per_model": per_model,
        "comparisons": comparisons,
    }
