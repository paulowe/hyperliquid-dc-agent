import httpx
import math
import statistics
from typing import Dict, Any, Optional, List


# -----------------------------
# Helper math functions
# -----------------------------

def tanh(x: float) -> float:
    """
    Hyperbolic tangent used to smoothly cap extreme values.
    Useful for ROI so that extreme returns do not dominate the score.
    """
    return math.tanh(x)


def sigmoid(x: float) -> float:
    """
    Sigmoid squashing function mapping (-inf, +inf) -> (0, 1).
    Used for soft thresholds (volume sufficiency, long-term dominance).
    """
    return 1 / (1 + math.exp(-x))


def safe_float(x: Any) -> float:
    """
    Convert string / numeric values safely to float.
    Returns 0.0 on failure.
    """
    try:
        return float(x)
    except Exception:
        return 0.0


# -----------------------------
# Configuration / weights
# -----------------------------

# Window importance weights — longer horizons matter more
WINDOW_WEIGHTS = {
    "day": 0.05,
    "week": 0.15,
    "month": 0.35,
    "allTime": 0.45,
}

# -----------------------------
# Core scoring logic
# -----------------------------

def row_to_windows(row: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Convert windowPerformances list into a dict keyed by window name.

    Input:
        [["day", {...}], ["week", {...}], ...]

    Output:
        {
            "day": {...},
            "week": {...},
            ...
        }
    """
    return {window: metrics for window, metrics in row["windowPerformances"]}


def _log_roi(x: float) -> float:
    """
    Log-based ROI transform for dispersion.
    Assumes ROI >= 0 due to filters; log1p keeps small ROIs meaningful
    and prevents one big ROI window from dominating dispersion.
    """
    return math.log1p(max(x, 0.0))


def score_row(
    row: Dict[str, Any],
    *,
    # hard eligibility filters
    MIN_MONTH_VOLUME: float = 10_000_000,
    MIN_ACCOUNT_VALUE: float = 100_000,

    # ROI / PnL shaping
    ROI_NORMALIZER: float = 0.25,

    # Dispesion (log-based) -> Smoothness
    DISPERSION_PENALTY_K: float = 3.0,

    # Turnover penalty inside main score
    TURNOVER_TARGET: float = 100.0,          # "ok-ish" monthly turnover
    TURNOVER_PENALTY_K: float = 1.2,         # strength of turnover penalty

    # long-term dominance
    LONG_TERM_K: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """
    Compute a longevity-weighted profitability score for a leaderboard trader.

    The score prioritizes:
    - Long-term profitability
    - Consistency across time windows
    - Sufficient trading activity for copy-trading
    - Penalization of short-term spikes and volatile equity curves

    Returns:
        Dict with score + diagnostics, or None if trader fails hard filters.
    """

    # ----------------------------------
    # Extract and sanitize raw values
    # ----------------------------------

    windows = row_to_windows(row)

    roi = {t: safe_float(windows[t]["roi"]) for t in WINDOW_WEIGHTS}
    pnl = {t: safe_float(windows[t]["pnl"]) for t in WINDOW_WEIGHTS}
    vlm = {t: safe_float(windows[t]["vlm"]) for t in WINDOW_WEIGHTS}
    account_value = safe_float(row.get("accountValue", 0))

    # -----------------------------
    # Hard filters (eligibility)
    # -----------------------------

    # Avoid tiny accounts (noise / lucky runs)
    if account_value < MIN_ACCOUNT_VALUE:
        return None

    # Must be profitable over meaningful horizons
    if roi["month"] <= 0 or roi["allTime"] <= 0:
        return None

    if pnl["allTime"] <= 0:
        return None

    # Must be active enough to copy
    if vlm["month"] < MIN_MONTH_VOLUME:
        return None

    # ----------------------------------
    # Scaled metrics (normalization)
    # ----------------------------------

    # ROI scaled with tanh so large returns saturate
    roi_scaled = {
        t: tanh(roi[t] / ROI_NORMALIZER)
        for t in WINDOW_WEIGHTS
    }

    # PnL scaled logarithmically to prevent whales dominating
    pnl_scaled = {
        t: math.log1p(max(pnl[t], 0.0))
        for t in WINDOW_WEIGHTS
    }

    # ----------------------------------
    # Longevity-weighted core score
    # ----------------------------------

    S_roi = sum(
        WINDOW_WEIGHTS[t] * roi_scaled[t]
        for t in WINDOW_WEIGHTS
    )

    S_pnl = sum(
        WINDOW_WEIGHTS[t] * pnl_scaled[t]
        for t in WINDOW_WEIGHTS
    )

    # ROI dominates — PnL is supportive, not primary
    S_core = 0.70 * S_roi + 0.30 * S_pnl

    # ----------------------------------
    # Consistency (log-based dispersion)
    # ----------------------------------

    # Dispersion across time windows
    # High dispersion = spiky / unstable trader
    log_rois = [_log_roi(roi[t]) for t in WINDOW_WEIGHTS]
    roi_log_dispersion = statistics.pstdev(log_rois)

    # low dispersion => multiplier closer to 1
    M_smooth = 1 / (1 + DISPERSION_PENALTY_K * roi_log_dispersion)

    # -----------------------------
    # Long-term dominance
    # -----------------------------

    # Reward traders whose long-term edge dominates short-term noise
    M_long_term = sigmoid(LONG_TERM_K * (roi["allTime"] - roi["day"]))


    # -----------------------------
    # Turnover penalty (inside main score)
    # turnover = monthVolume / accountValue
    # Use log scaling so extreme churn gets punished smoothly, not abruptly.
    # -----------------------------
    turnover = (vlm["month"] / account_value) if account_value > 0 else float("inf")
    turnover_ratio = turnover / max(TURNOVER_TARGET, 1e-9)
    M_turnover = 1 / (1 + TURNOVER_PENALTY_K * math.log1p(turnover_ratio))

    # ----------------------------------
    # Explicit penalties for bad patterns
    # ----------------------------------

    M_penalty = 1.0

    # Losing week = caution
    if roi["week"] < 0:
        M_penalty *= 0.60

    # Losing month = strong red flag
    if roi["month"] < 0:
        M_penalty *= 0.25

    # Losing all-time = basically disqualified
    if roi["allTime"] < 0:
        M_penalty *= 0.10


    # ----------------------------------
    # Final score
    # ----------------------------------

    final_score = (
        S_core
        * M_smooth
        * M_long_term
        * M_turnover
        * M_penalty
    )

    # -----------------------------
    # Convexity helpers (for satellite selection)
    # (kept as diagnostics; you can use them later in pick_convexity_trader)
    # -----------------------------
    pop = max(roi["day"], roi["week"]) - roi["month"]
    convex_score = max(roi["day"], roi["week"]) / (1e-6 + roi_log_dispersion)


    return {
        "ethAddress": row["ethAddress"],
        "displayName": row.get("displayName"),
        "score": final_score,
        "accountValue": account_value,
        "roi": roi,
        "pnl": pnl,
        "monthVolume": vlm["month"],
        # Stability metric (log-based)
        "roiLogDispersion": roi_log_dispersion,
        # Churn proxy
        "turnover": turnover,
        # Convexity metrics
        "pop": pop,
        "convex_score": convex_score,
    }

def score_all(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    scored = []
    for row in data["leaderboardRows"]:
        r = score_row(row)
        if r:
            scored.append(r)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def pick_anchors(scored: List[Dict[str, Any]], n: int = 2, top_k: int = 200):
    pool = scored[:top_k]

    pool = [
        x for x in pool
        # MUST have real edge
        if x["roi"]["allTime"] >= 0.30
        and x["roi"]["month"] >= 0.05
        # Stability constraint (not objective)
        and x["roiLogDispersion"] <= 0.20
        # Copyability
        and x["accountValue"] >= 1_000_000
        and x["turnover"] <= 200
    ]

    # Now pick the best anchors by score
    pool.sort(key=lambda x: x["score"], reverse=True)
    return pool[:n]


def pick_convexity_trader(scored: List[Dict[str, Any]], anchor_addresses: set, top_k: int = 500):
    """
    Convexity (satellite) selection:
    - exclude anchors
    - avoid grenades: too stable (already anchors) or too unstable
    - require positive week/month/allTime
    - require a "pop" (day/week > month) so it can add convexity
    - keep churn reasonable (turnover cap)
    """
    pool = [x for x in scored[:top_k] if x["ethAddress"] not in anchor_addresses]

    candidates = []
    for x in pool:
        d = x["roiLogDispersion"]

        # “not an anchor, not a grenade”
        if not (0.18 <= d <= 0.40):
            continue

        # must still be “real”
        if x["roi"]["week"] <= 0 or x["roi"]["month"] <= 0 or x["roi"]["allTime"] <= 0:
            continue

        # avoid extreme leverage churn proxies
        if x["turnover"] > 300:
            continue

        # convexity: short-term pop relative to month
        pop = x["pop"]
        if pop <= 0:
            continue

        pop_norm = math.tanh(pop / 0.25)
        convex_norm = math.tanh(x["convex_score"] / 5.0)  # fixed key name

        satellite_rank = 0.7 * pop_norm + 0.3 * convex_norm
        candidates.append((satellite_rank, x))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1] if candidates else None


def hyperliquid_leaderboard():
    # Ordered by 30D volume by default
    URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

    with httpx.Client(timeout=60) as client:
        r = client.get(URL)
        r.raise_for_status()
        data = r.json()

    return data

if __name__ == "__main__":
    data = hyperliquid_leaderboard()
    scored = score_all(data)

    # Pick n anchors (n=2)
    anchors = pick_anchors(scored, n=2, top_k=200)
    anchor_addresses = {a["ethAddress"] for a in anchors}
    # Pick 1 convexity trader excluding anchors
    convexity_trader = pick_convexity_trader(scored, anchor_addresses, top_k=500)

    print("ANCHORS:", [a["ethAddress"] for a in anchors])
    print("CONVEXITY:", convexity_trader["ethAddress"] if convexity_trader else None)