"""
Hyperliquid leaderboard → longevity-weighted scoring + self-tuned anchor selection.

Fixes you requested:
✅ Keep percentile floors as a guide, but enforce hard positivity for anchor month PnL/ROI.
✅ Make longevity more central by adding an explicit “consistency” term.
✅ Optional auto-tuning of LONGEVITY_WEIGHT based on what the top results look like.

Schema assumption:
row["windowPerformances"] = [["day",{pnl,roi,vlm}],["week",{...}],["month",{...}],["allTime",{...}]]

Notes:
- Volume sufficiency is NOT double-counted: volume is only a hard filter.
- Turnover penalty is integrated into score.
- Dispersion is log-based (roiLogDispersion).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple, List

import httpx
import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

WINDOW_WEIGHTS = {
    "day": 0.05,
    "week": 0.15,
    "month": 0.35,
    "allTime": 0.45,
}

# Hard eligibility filters (copyability + reduce noise)
MIN_MONTH_VOLUME = 10_000_000.0
MIN_ACCOUNT_VALUE = 100_000.0

# Shape / penalties (some auto-tuned later)
DISPERSION_PENALTY_K = 3.0
TURNOVER_PENALTY_K = 1.2
LONG_TERM_K = 3.0

# Final score blend (can be auto-tuned)
LONGEVITY_WEIGHT_INIT = 0.55
CORE_WEIGHT_INIT = 1.0 - LONGEVITY_WEIGHT_INIT

# Auto-tune targets (you can tweak)
AUTO_TUNE_LONGEVITY = True
TARGET_TOP10_MED_LON_TERM = 0.25   # if top10 median longevity_term is below this, bump longevity weight
MAX_LONGEVITY_WEIGHT = 0.80
MIN_LONGEVITY_WEIGHT = 0.35
TUNE_STEPS = 4


@dataclass(frozen=True)
class AnchorPercentiles:
    roi_allTime_p: float = 60.0
    roi_month_p: float = 45.0
    disp_p: float = 60.0
    turnover_p: float = 80.0


# -----------------------------
# Data
# -----------------------------

def fetch_leaderboard(url: str = LEADERBOARD_URL, timeout: float = 60.0) -> Dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def leaderboard_to_frame(data: Dict[str, Any]) -> pd.DataFrame:
    rows = data.get("leaderboardRows", [])
    out = []

    for row in rows:
        windows = {w: m for w, m in row.get("windowPerformances", [])}

        def f(window: str, key: str) -> float:
            try:
                return float(windows[window][key])
            except Exception:
                return float("nan")

        out.append(
            {
                "ethAddress": row.get("ethAddress"),
                "displayName": row.get("displayName"),
                "accountValue": float(row.get("accountValue", 0.0)),
                "roi_day": f("day", "roi"),
                "roi_week": f("week", "roi"),
                "roi_month": f("month", "roi"),
                "roi_allTime": f("allTime", "roi"),
                "pnl_day": f("day", "pnl"),
                "pnl_week": f("week", "pnl"),
                "pnl_month": f("month", "pnl"),
                "pnl_allTime": f("allTime", "pnl"),
                "vlm_day": f("day", "vlm"),
                "vlm_week": f("week", "vlm"),
                "vlm_month": f("month", "vlm"),
                "vlm_allTime": f("allTime", "vlm"),
            }
        )

    df = pd.DataFrame(out)
    df = df.dropna(
        subset=[
            "ethAddress",
            "accountValue",
            "roi_month",
            "roi_allTime",
            "vlm_month",
            "pnl_allTime",
            "pnl_month",
        ]
    )
    return df


# -----------------------------
# Diagnostics
# -----------------------------

def qline(name: str, s: pd.Series) -> str:
    qs = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_dict()
    return (
        f"{name}: "
        f"p5={qs[0.05]:.6g}, p25={qs[0.25]:.6g}, p50={qs[0.50]:.6g}, "
        f"p75={qs[0.75]:.6g}, p95={qs[0.95]:.6g}"
    )


def universe_diagnostics(scored: pd.DataFrame) -> None:
    print("\n--- Universe diagnostics (after hard filters + scoring) ---")
    print(f"rows: {len(scored)}")
    if scored.empty:
        return
    print(qline("roi_allTime", scored["roi_allTime"]))
    print(qline("roi_month  ", scored["roi_month"]))
    print(qline("dispersion ", scored["roiLogDispersion"]))
    print(qline("turnover   ", scored["turnover"]))
    print(qline("score      ", scored["score"]))
    print(qline("lon_term   ", scored["longevity_term"]))
    print(qline("consistency", scored["consistency_term"]))


# -----------------------------
# Scoring
# -----------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.clip(-60, 60)))


def add_features_and_score(
    df: pd.DataFrame,
    *,
    longevity_weight: float,
    core_weight: float,
) -> pd.DataFrame:
    d = df.copy()

    # Hard filters: copyable + real enough
    d = d[
        (d["accountValue"] >= MIN_ACCOUNT_VALUE)
        & (d["vlm_month"] >= MIN_MONTH_VOLUME)
        & (d["pnl_allTime"] > 0)
        & (d["roi_allTime"].notna())
        & (d["roi_month"].notna())
        & (d["pnl_month"].notna())
    ].copy()

    if d.empty:
        d["score"] = []
        return d

    # Turnover
    d["turnover"] = d["vlm_month"] / d["accountValue"].replace(0, np.nan)

    # Log-based dispersion across windows (clamp negative ROI to 0 before log1p)
    roi_cols = ["roi_day", "roi_week", "roi_month", "roi_allTime"]
    roi_mat = d[roi_cols].to_numpy(dtype=float)
    log_roi = np.log1p(np.maximum(roi_mat, 0.0))
    d["roiLogDispersion"] = log_roi.std(axis=1, ddof=0)

    # Convexity helpers
    d["pop"] = d[["roi_day", "roi_week"]].max(axis=1) - d["roi_month"]
    d["convex_score"] = d[["roi_day", "roi_week"]].max(axis=1) / (1e-6 + d["roiLogDispersion"])

    # -----------------------------
    # Auto-tune scale params (robust to regime changes)
    # -----------------------------
    pos_month = d.loc[d["roi_month"] > 0, "roi_month"]
    if len(pos_month) >= 50:
        roi_norm = float(pos_month.quantile(0.75))
    else:
        roi_norm = float(d["roi_week"].quantile(0.75))
    roi_norm = float(np.clip(roi_norm, 0.05, 0.60))

    turnover_target = float(d["turnover"].quantile(0.75))
    turnover_target = float(np.clip(turnover_target, 20.0, 250.0))

    # -----------------------------
    # Core score (ROI + PnL)
    # -----------------------------
    roi_scaled = {w: np.tanh(d[f"roi_{w}"] / roi_norm) for w in WINDOW_WEIGHTS}
    S_roi = sum(WINDOW_WEIGHTS[w] * roi_scaled[w] for w in WINDOW_WEIGHTS)

    pnl_scaled = {w: np.log1p(np.maximum(d[f"pnl_{w}"], 0.0)) for w in WINDOW_WEIGHTS}
    S_pnl = sum(WINDOW_WEIGHTS[w] * pnl_scaled[w] for w in WINDOW_WEIGHTS)

    S_core = 0.70 * S_roi + 0.30 * S_pnl

    # Smoothness multiplier from log-dispersion (consistency)
    M_smooth = 1.0 / (1.0 + DISPERSION_PENALTY_K * d["roiLogDispersion"])

    # Long-term dominance multiplier: reward allTime strength relative to day
    lt_gap = (d["roi_allTime"] - d["roi_day"]).clip(-60, 60)
    M_long_term = 1.0 / (1.0 + np.exp(-LONG_TERM_K * lt_gap))

    # Turnover penalty multiplier (churn control)
    turnover_ratio = (d["turnover"] / max(turnover_target, 1e-9)).replace([np.inf, -np.inf], np.nan)
    M_turnover = 1.0 / (1.0 + TURNOVER_PENALTY_K * np.log1p(turnover_ratio.clip(lower=0)))

    # Mild penalties for negative shorter windows
    M_penalty = np.ones(len(d), dtype=float)
    M_penalty *= np.where(d["roi_week"] < 0, 0.70, 1.0)
    M_penalty *= np.where(d["roi_day"] < 0, 0.85, 1.0)

    # -----------------------------
    # Longevity + Consistency (explicit, central)
    # -----------------------------
    # Positivity signals (bounded)
    alltime_pos = _sigmoid(2.5 * d["roi_allTime"].to_numpy(dtype=float).clip(-5, 5))
    month_pos   = _sigmoid(2.0 * d["roi_month"].to_numpy(dtype=float).clip(-2, 2))
    week_pos    = _sigmoid(1.5 * d["roi_week"].to_numpy(dtype=float).clip(-2, 2))

    # NEW: explicit consistency term
    # 1) percent of windows with ROI > 0
    pos_rate = (roi_mat > 0).mean(axis=1)  # in [0,1]
    # 2) stability: reuse smoothness (already in [0,1-ish])
    stability = M_smooth.to_numpy(dtype=float)
    # 3) drawdown-ish proxy: punish if any window is deeply negative (softly)
    worst_roi = np.nanmin(roi_mat, axis=1)
    no_crash = _sigmoid(3.0 * (worst_roi + 0.05))  # if worst_roi < -5%, this drops quickly

    d["consistency_term"] = (0.55 * pos_rate + 0.30 * stability + 0.15 * no_crash)

    # Longevity term now includes consistency explicitly
    d["longevity_term"] = (
        (0.50 * alltime_pos + 0.30 * month_pos + 0.20 * week_pos)
        * d["consistency_term"]
        * M_turnover.to_numpy(dtype=float)
        * M_long_term.to_numpy(dtype=float)
    )

    # Final score blend
    d["score"] = (core_weight * (S_core * M_penalty)) + (longevity_weight * d["longevity_term"])

    # debug params
    d["roi_normalizer"] = roi_norm
    d["turnover_target"] = turnover_target
    d["longevity_weight"] = longevity_weight

    keep_front = [
        "ethAddress",
        "displayName",
        "score",
        "longevity_term",
        "consistency_term",
        "accountValue",
        "vlm_month",
        "roi_day",
        "roi_week",
        "roi_month",
        "roi_allTime",
        "pnl_month",
        "pnl_allTime",
        "roiLogDispersion",
        "turnover",
        "pop",
        "convex_score",
        "roi_normalizer",
        "turnover_target",
        "longevity_weight",
    ]
    rest = [c for c in d.columns if c not in keep_front]
    d = d[keep_front + rest]

    return d.sort_values("score", ascending=False).reset_index(drop=True)


def auto_tune_longevity_weight(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    If the top of the ranked list still has weak longevity_term, increase longevity weight.
    """
    w = LONGEVITY_WEIGHT_INIT
    for i in range(1, TUNE_STEPS + 1):
        scored = add_features_and_score(raw_df, longevity_weight=w, core_weight=(1.0 - w))
        if scored.empty:
            return scored, w

        top10_med = float(scored.head(10)["longevity_term"].median())
        print(f"[AUTO] step {i}: longevity_weight={w:.2f} | top10 median longevity_term={top10_med:.4f}")

        if top10_med >= TARGET_TOP10_MED_LON_TERM:
            return scored, w

        # increase longevity emphasis
        w = min(MAX_LONGEVITY_WEIGHT, w + 0.08)

    return add_features_and_score(raw_df, longevity_weight=w, core_weight=(1.0 - w)), w


# -----------------------------
# Anchors via percentile floors (WITH hard positivity)
# -----------------------------

def compute_anchor_thresholds(scored: pd.DataFrame, ps: AnchorPercentiles) -> Dict[str, float]:
    return {
        "roi_allTime_min": float(scored["roi_allTime"].quantile(ps.roi_allTime_p / 100.0)),
        "roi_month_min": float(scored["roi_month"].quantile(ps.roi_month_p / 100.0)),
        "roiLogDispersion_max": float(scored["roiLogDispersion"].quantile(ps.disp_p / 100.0)),
        "turnover_max": float(scored["turnover"].quantile(ps.turnover_p / 100.0)),
    }


def eligible_anchor_set(
    pool: pd.DataFrame,
    thr: Dict[str, float],
    *,
    require_positive_week: bool,
    min_account_value: float,
) -> pd.DataFrame:
    """
    IMPORTANT FIX:
    - Percentile floors guide the “quality bar”
    - BUT anchors must be hard-positive on month ROI and month PnL (boring winners).
    """
    elig = pool[
        (pool["roi_allTime"] >= thr["roi_allTime_min"])
        & (pool["roi_month"] >= thr["roi_month_min"])
        & (pool["roiLogDispersion"] <= thr["roiLogDispersion_max"])
        & (pool["turnover"] <= thr["turnover_max"])
        & (pool["accountValue"] >= min_account_value)
        & (pool["pnl_allTime"] > 0)
        # HARD positivity for anchors:
        & (pool["roi_month"] > 0)
        & (pool["pnl_month"] > 0)
    ].copy()

    if require_positive_week:
        elig = elig[(elig["roi_week"] > 0) & (elig["pnl_week"] > 0)].copy()

    # Anchors should be consistently positive: require 3/4 windows positive
    roi_cols = ["roi_day", "roi_week", "roi_month", "roi_allTime"]
    pos_rate = (elig[roi_cols].to_numpy(dtype=float) > 0).mean(axis=1)
    elig = elig[pos_rate >= 0.75].copy()

    # Extra durability: longevity_term above median of pool
    if not elig.empty:
        med = float(pool["longevity_term"].median())
        elig = elig[elig["longevity_term"] >= med].copy()

    return elig


def tune_anchor_percentiles(
    scored: pd.DataFrame,
    *,
    n_anchors: int,
    top_k: int,
    require_positive_week: bool,
    min_account_value: float,
    steps: Optional[List[AnchorPercentiles]] = None,
) -> Tuple[AnchorPercentiles, pd.DataFrame, Dict[str, float]]:
    pool = scored.head(top_k).copy()
    if pool.empty:
        return AnchorPercentiles(), pool, {}

    if steps is None:
        steps = [
            AnchorPercentiles(60, 45, 60, 80),
            AnchorPercentiles(55, 40, 65, 85),
            AnchorPercentiles(50, 35, 70, 90),
            AnchorPercentiles(45, 30, 75, 92),
            AnchorPercentiles(40, 25, 80, 94),
            AnchorPercentiles(35, 20, 85, 96),
        ]

    print("\n[TUNE] Trying anchor percentile settings...")
    best_ps = steps[-1]
    best_elig = pd.DataFrame()
    best_thr: Dict[str, float] = {}

    for i, ps in enumerate(steps, start=1):
        thr = compute_anchor_thresholds(pool, ps)
        elig = eligible_anchor_set(
            pool, thr,
            require_positive_week=require_positive_week,
            min_account_value=min_account_value,
        )
        anchors_found = min(len(elig), n_anchors)
        print(f"  step {i}: ps={ps}  | eligible={len(elig)}  | anchors_found={anchors_found}")
        print(
            f"          floors: roi_allTime>={thr['roi_allTime_min']:.4f}, "
            f"roi_month>={thr['roi_month_min']:.4f}, "
            f"disp<={thr['roiLogDispersion_max']:.4f}, "
            f"turnover<={thr['turnover_max']:.2f}"
        )

        if len(elig) >= n_anchors:
            best_ps, best_elig, best_thr = ps, elig, thr
            break

        if len(elig) > len(best_elig):
            best_ps, best_elig, best_thr = ps, elig, thr

    return best_ps, best_elig, best_thr


def pick_anchors_from_elig(elig: pd.DataFrame, *, n: int) -> pd.DataFrame:
    if elig.empty:
        return elig
    return elig.sort_values("score", ascending=False).head(n).reset_index(drop=True)


# -----------------------------
# Convexity satellite
# -----------------------------

def pick_convexity_trader(
    scored: pd.DataFrame,
    anchor_addresses: set[str],
    *,
    top_k: int = 3000,
    disp_range: Tuple[float, float] = (0.18, 0.45),
    turnover_max: float = 300.0,
    min_month_roi: float = 0.01,
    min_week_roi: float = 0.0,
) -> Optional[pd.Series]:
    pool = scored.head(top_k).copy()
    pool = pool[~pool["ethAddress"].isin(anchor_addresses)].copy()

    dmin, dmax = disp_range
    pool = pool[
        (pool["roiLogDispersion"] >= dmin)
        & (pool["roiLogDispersion"] <= dmax)
        & (pool["turnover"] <= turnover_max)
        & (pool["roi_month"] >= min_month_roi)
        & (pool["roi_week"] >= min_week_roi)
        & (pool["roi_allTime"] > 0)
        & (pool["pop"] > 0)
    ].copy()

    if pool.empty:
        return None

    pop_norm = np.tanh(pool["pop"] / 0.25)
    convex_norm = np.tanh(pool["convex_score"] / 5.0)
    long_norm = np.tanh(pool["longevity_term"] / max(float(pool["longevity_term"].median()), 1e-6))

    pool["satellite_rank"] = 0.60 * pop_norm + 0.20 * convex_norm + 0.20 * long_norm
    return pool.sort_values("satellite_rank", ascending=False).iloc[0]


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    data = fetch_leaderboard()
    raw_df = leaderboard_to_frame(data)

    if AUTO_TUNE_LONGEVITY:
        scored, used_w = auto_tune_longevity_weight(raw_df)
        print(f"[AUTO] final longevity_weight={used_w:.2f}, core_weight={1.0 - used_w:.2f}")
    else:
        scored = add_features_and_score(raw_df, longevity_weight=LONGEVITY_WEIGHT_INIT, core_weight=CORE_WEIGHT_INIT)

    universe_diagnostics(scored)

    if scored.empty:
        print("No scored traders after filters. Lower MIN_MONTH_VOLUME / MIN_ACCOUNT_VALUE.")
        return

    # Tune anchors
    n_anchors = 2
    top_k = 5000
    require_positive_week = True
    min_account_value = 250_000.0

    best_ps, elig, thr = tune_anchor_percentiles(
        scored,
        n_anchors=n_anchors,
        top_k=top_k,
        require_positive_week=require_positive_week,
        min_account_value=min_account_value,
    )

    anchors = pick_anchors_from_elig(elig, n=n_anchors)
    anchor_addresses = set(anchors["ethAddress"].tolist())

    convex = pick_convexity_trader(scored, anchor_addresses, top_k=8000)

    print("\n--- Anchor percentile thresholds (computed) ---")
    for k, v in (thr or {}).items():
        print(f"{k}: {v:.6f}")

    print("\n--- Percentiles used (self-tuned) ---")
    print(best_ps)

    print("\n--- Anchors (hard-positive month ROI & PnL) ---")
    if anchors.empty:
        print("No anchors found. Try lowering min_account_value or require_positive_week=False.")
    else:
        cols = [
            "ethAddress", "score", "longevity_term", "consistency_term", "accountValue", "vlm_month",
            "roi_day", "roi_week", "roi_month", "roi_allTime",
            "pnl_week", "pnl_month", "roiLogDispersion", "turnover",
        ]
        print(anchors[cols].to_string(index=False))

    print("\n--- Convexity (satellite) trader ---")
    if convex is None:
        print("No convexity trader found (try widening disp_range or relaxing turnover_max).")
    else:
        cols = [
            "ethAddress", "score", "satellite_rank", "accountValue",
            "roi_day", "roi_week", "roi_month", "roi_allTime",
            "pop", "roiLogDispersion", "turnover", "longevity_term", "consistency_term",
        ]
        print(pd.DataFrame([convex])[cols].to_string(index=False))

    print("\n--- Top 10 overall by score (for sanity) ---")
    cols = [
        "ethAddress", "score", "longevity_term", "consistency_term",
        "accountValue", "roi_month", "roi_allTime",
        "roiLogDispersion", "turnover", "vlm_month",
    ]
    print(scored.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
