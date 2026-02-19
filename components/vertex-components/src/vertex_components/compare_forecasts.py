# KFP component that compares 3 forecast model arms (baseline, single-DC, multi-DC).
# Computes per-model metrics and head-to-head statistical tests, outputs a JSON
# report, an HTML visualization, and logs KFP Metrics.

from kfp.v2.dsl import Input, Output, Dataset, HTML, Metrics, component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.10.0",
    ],
)
def compare_forecasts(
    baseline_predictions: Input[Dataset],
    single_dc_predictions: Input[Dataset],
    multi_dc_predictions: Input[Dataset],
    report: Output[Dataset],
    visualization: Output[HTML],
    metrics: Output[Metrics],
):
    """Compare 3 ablation arms and produce a statistical comparison report.

    Each input Dataset should be a CSV file with columns:
        sample_index, y_true_scaled, y_pred_scaled
        Optional: reference_price (last PRICE_std from input window, for
        meaningful directional accuracy vs current price)

    Outputs:
        report: JSON file with per-model metrics and pairwise comparisons
        visualization: HTML table summarizing results
        metrics: KFP Metrics logged for the pipeline UI
    """
    import json
    import math
    import os
    from typing import Dict, List, Optional

    import numpy as np
    import pandas as pd
    from scipy import stats

    # -----------------------------------------------------------------
    # Metric helpers (inlined for KFP component isolation)
    # -----------------------------------------------------------------
    def compute_rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def compute_mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    def compute_mape(y_true, y_pred):
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    def compute_smape(y_true, y_pred):
        denom = np.abs(y_true) + np.abs(y_pred)
        mask = denom != 0
        if not mask.any():
            return 0.0
        return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

    def compute_r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def compute_directional_accuracy(y_true, y_pred, reference=None):
        """Compute directional accuracy.

        If reference prices are provided, measures whether the model correctly
        predicts UP/DOWN relative to the current price (useful for trading).
        Otherwise falls back to tick-by-tick direction of consecutive predictions.
        """
        if reference is not None:
            # Meaningful metric: does model predict price direction from current price?
            true_dir = np.sign(y_true - reference)
            pred_dir = np.sign(y_pred - reference)
            # Exclude samples where true direction is exactly 0 (no change)
            mask = true_dir != 0
            if not mask.any():
                return 0.5
            return float(np.mean(true_dir[mask] == pred_dir[mask]))
        # Fallback: tick-by-tick direction of consecutive predictions
        if len(y_true) < 2:
            return 0.0
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        return float(np.mean(true_dir == pred_dir))

    def compute_all_metrics(y_true, y_pred, reference=None):
        return {
            "rmse": compute_rmse(y_true, y_pred),
            "mae": compute_mae(y_true, y_pred),
            "mape": compute_mape(y_true, y_pred),
            "r_squared": compute_r_squared(y_true, y_pred),
            "smape": compute_smape(y_true, y_pred),
            "directional_accuracy": compute_directional_accuracy(
                y_true, y_pred, reference=reference
            ),
        }

    def compare_two(y_true, y_pred_a, y_pred_b):
        """Compare A vs B. Positive improvement means A is better."""
        rmse_a = compute_rmse(y_true, y_pred_a)
        rmse_b = compute_rmse(y_true, y_pred_b)
        rel_imp = (rmse_b - rmse_a) / rmse_b * 100.0 if rmse_b != 0 else 0.0

        sq_a = (y_true - y_pred_a) ** 2
        sq_b = (y_true - y_pred_b) ** 2
        abs_a = np.abs(y_true - y_pred_a)
        abs_b = np.abs(y_true - y_pred_b)

        wilcoxon_p = None
        try:
            diff = sq_a - sq_b
            if np.any(diff != 0) and len(diff) >= 5:
                _, wilcoxon_p = stats.wilcoxon(sq_a, sq_b)
                wilcoxon_p = float(wilcoxon_p)
        except Exception:
            pass

        ttest_p = None
        try:
            if len(abs_a) >= 2:
                _, ttest_p = stats.ttest_rel(abs_a, abs_b)
                ttest_p = float(ttest_p)
        except Exception:
            pass

        return {
            "rmse_relative_improvement_pct": float(rel_imp),
            "wilcoxon_p": wilcoxon_p,
            "paired_ttest_p": ttest_p,
        }

    # -----------------------------------------------------------------
    # Read prediction CSVs
    # -----------------------------------------------------------------
    df_base = pd.read_csv(baseline_predictions.path)
    df_single = pd.read_csv(single_dc_predictions.path)
    df_multi = pd.read_csv(multi_dc_predictions.path)

    yt_b = df_base["y_true_scaled"].to_numpy()
    yp_b = df_base["y_pred_scaled"].to_numpy()
    yt_s = df_single["y_true_scaled"].to_numpy()
    yp_s = df_single["y_pred_scaled"].to_numpy()
    yt_m = df_multi["y_true_scaled"].to_numpy()
    yp_m = df_multi["y_pred_scaled"].to_numpy()

    # Extract reference prices if available (for meaningful directional accuracy)
    ref_b = df_base["reference_price"].to_numpy() if "reference_price" in df_base.columns else None
    ref_s = df_single["reference_price"].to_numpy() if "reference_price" in df_single.columns else None
    ref_m = df_multi["reference_price"].to_numpy() if "reference_price" in df_multi.columns else None

    # -----------------------------------------------------------------
    # Per-model metrics
    # -----------------------------------------------------------------
    per_model = {
        "baseline": compute_all_metrics(yt_b, yp_b, reference=ref_b),
        "single_dc": compute_all_metrics(yt_s, yp_s, reference=ref_s),
        "multi_dc": compute_all_metrics(yt_m, yp_m, reference=ref_m),
    }

    # -----------------------------------------------------------------
    # Pairwise comparisons
    # -----------------------------------------------------------------
    comparisons = {
        "single_dc_vs_baseline": compare_two(yt_b, yp_s, yp_b),
        "multi_dc_vs_baseline": compare_two(yt_b, yp_m, yp_b),
        "multi_dc_vs_single_dc": compare_two(yt_s, yp_m, yp_s),
    }

    full_report = {
        "per_model": per_model,
        "comparisons": comparisons,
    }

    # -----------------------------------------------------------------
    # Write JSON report
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(report.path) or ".", exist_ok=True)
    with open(report.path, "w") as f:
        json.dump(full_report, f, indent=2)

    # -----------------------------------------------------------------
    # Log KFP metrics
    # -----------------------------------------------------------------
    for arm_name, arm_metrics in per_model.items():
        for metric_name, metric_val in arm_metrics.items():
            if metric_val is not None and math.isfinite(metric_val):
                metrics.log_metric(f"{arm_name}_{metric_name}", float(metric_val))

    for comp_name, comp_vals in comparisons.items():
        for k, v in comp_vals.items():
            if v is not None and math.isfinite(v):
                metrics.log_metric(f"{comp_name}_{k}", float(v))

    # -----------------------------------------------------------------
    # Build HTML visualization
    # -----------------------------------------------------------------
    def _fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.6f}"

    rows_per_model = ""
    for arm in ["baseline", "single_dc", "multi_dc"]:
        m = per_model[arm]
        rows_per_model += f"""
        <tr>
            <td style="border:1px solid #ccc;padding:6px 12px;font-weight:bold;">{arm}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(m['rmse'])}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(m['mae'])}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(m['mape'])}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(m['r_squared'])}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(m['smape'])}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(m['directional_accuracy'])}</td>
        </tr>"""

    rows_comp = ""
    for comp_name, comp_vals in comparisons.items():
        rows_comp += f"""
        <tr>
            <td style="border:1px solid #ccc;padding:6px 12px;font-weight:bold;">{comp_name}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(comp_vals['rmse_relative_improvement_pct'])}%</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(comp_vals['wilcoxon_p'])}</td>
            <td style="border:1px solid #ccc;padding:6px 12px;">{_fmt(comp_vals['paired_ttest_p'])}</td>
        </tr>"""

    html_content = f"""
    <html><body style="font-family:Inter,system-ui,Arial,sans-serif;padding:16px;">
    <h2>DC Feature Ablation Study - Comparison Report</h2>

    <h3>Per-Model Metrics</h3>
    <table style="border-collapse:collapse;margin:10px 0;">
        <tr>
            <th style="border:1px solid #ccc;padding:6px 12px;">Model</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">RMSE</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">MAE</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">MAPE</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">R²</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">sMAPE</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">Dir. Acc.</th>
        </tr>
        {rows_per_model}
    </table>

    <h3>Pairwise Comparisons</h3>
    <p><em>Positive RMSE improvement means the first model is better.</em></p>
    <table style="border-collapse:collapse;margin:10px 0;">
        <tr>
            <th style="border:1px solid #ccc;padding:6px 12px;">Comparison</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">RMSE Improvement</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">Wilcoxon p</th>
            <th style="border:1px solid #ccc;padding:6px 12px;">Paired t-test p</th>
        </tr>
        {rows_comp}
    </table>

    <h3>Architecture Note</h3>
    <p>With bottleneck projection (Dense(128)), all arms have equal effective capacity after
    the bottleneck regardless of input feature count. The bottleneck acts as implicit feature
    selection, projecting different input dimensions to the same representation size.</p>
    </body></html>
    """
    os.makedirs(os.path.dirname(visualization.path) or ".", exist_ok=True)
    with open(visualization.path, "w", encoding="utf-8") as f:
        f.write(html_content)
