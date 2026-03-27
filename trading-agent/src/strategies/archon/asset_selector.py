"""Dynamic asset selector for Archon scanner.

Queries Hyperliquid for all assets, ranks by momentum strength,
and returns the top N symbols with the strongest directional moves.
Designed to be called periodically (every 30-60 min) to rotate
the scanner's watchlist toward wherever the action is.

Criteria for selection:
1. Minimum 24h volume ($1M+) — needs liquidity for execution
2. Strong directional momentum (|24h change| > 1.5%)
3. Preference for assets where funding rate confirms direction
   (negative funding = shorts paying longs = downtrend confirmed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.hyperliquid.xyz/info"

# Assets to always include regardless of momentum
CORE_SYMBOLS = {"HYPE", "SOL"}

# Assets to never trade (too illiquid, too new, etc.)
EXCLUDED = {"PURR", "JEFF"}  # spot-only or problematic

# Threshold scales with volatility — higher vol needs wider threshold
DEFAULT_THRESHOLD = 0.02


@dataclass
class AssetCandidate:
    """A tradeable asset with momentum data."""
    symbol: str
    price: float
    change_24h_pct: float
    volume_24h_usd: float
    funding_rate: float  # per 8h
    open_interest_usd: float
    direction: str  # "long", "short", or "neutral"
    score: float  # momentum score for ranking
    threshold: float  # recommended DC threshold


def fetch_candidates(
    min_volume_usd: float = 1_000_000,
    min_change_pct: float = 1.5,
    max_assets: int = 5,
) -> List[AssetCandidate]:
    """Fetch and rank assets by momentum strength.

    Returns top N assets sorted by momentum score (strongest first).
    Always includes CORE_SYMBOLS if they meet minimum volume.
    """
    try:
        mids_resp = requests.post(API_URL, json={"type": "allMids"}, timeout=10)
        mids = mids_resp.json()

        meta_resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
        meta_data = meta_resp.json()
        meta, ctxs = meta_data[0], meta_data[1]
    except Exception as e:
        logger.error("Failed to fetch market data: %s", e)
        return []

    candidates = []
    for i, ctx in enumerate(ctxs):
        if i >= len(meta.get("universe", [])):
            break

        name = meta["universe"][i]["name"]
        if name in EXCLUDED:
            continue

        price = float(mids.get(name, 0))
        prev = float(ctx.get("prevDayPx", 0))
        vol = float(ctx.get("dayNtlVlm", 0))
        funding = float(ctx.get("funding", 0))
        oi = float(ctx.get("openInterest", 0))

        if price <= 0 or prev <= 0:
            continue

        change_pct = (price - prev) / prev * 100
        oi_usd = oi * price

        # Filter: minimum volume
        is_core = name in CORE_SYMBOLS
        if vol < min_volume_usd and not is_core:
            continue

        # Direction from momentum
        if change_pct > min_change_pct:
            direction = "long"
        elif change_pct < -min_change_pct:
            direction = "short"
        elif is_core:
            direction = "neutral"
        else:
            continue  # skip assets without clear momentum

        # Score: |momentum| * volume_factor * funding_alignment
        abs_change = abs(change_pct)
        vol_factor = min(vol / 10_000_000, 3.0)  # cap at 3x for mega-liquid
        # Funding alignment: bonus if funding confirms direction
        funding_bonus = 1.0
        if direction == "short" and funding < 0:
            funding_bonus = 1.2  # shorts paying, confirms downtrend
        elif direction == "long" and funding > 0:
            funding_bonus = 1.2  # longs paying, confirms uptrend

        score = abs_change * vol_factor * funding_bonus

        # Threshold: scale with volatility
        # Higher change = more volatile = needs wider threshold
        if abs_change > 6:
            threshold = 0.03
        elif abs_change > 4:
            threshold = 0.025
        elif abs_change > 2:
            threshold = 0.02
        else:
            threshold = 0.015

        candidates.append(AssetCandidate(
            symbol=name,
            price=price,
            change_24h_pct=change_pct,
            volume_24h_usd=vol,
            funding_rate=funding * 100,
            open_interest_usd=oi_usd,
            direction=direction,
            score=score,
            threshold=threshold,
        ))

    # Sort by score (strongest momentum first)
    candidates.sort(key=lambda x: x.score, reverse=True)

    # Ensure core symbols are included
    selected = []
    selected_names = set()

    # Add top momentum picks
    for c in candidates:
        if len(selected) >= max_assets:
            break
        if c.symbol not in selected_names:
            selected.append(c)
            selected_names.add(c.symbol)

    # Ensure core symbols
    for c in candidates:
        if c.symbol in CORE_SYMBOLS and c.symbol not in selected_names:
            if len(selected) >= max_assets:
                selected.pop()  # drop weakest to make room
            selected.append(c)
            selected_names.add(c.symbol)

    # Re-sort by score
    selected.sort(key=lambda x: x.score, reverse=True)
    return selected


def format_selection(candidates: List[AssetCandidate]) -> str:
    """Format selection as human-readable string."""
    lines = []
    for c in candidates:
        arrow = "↑" if c.direction == "long" else ("↓" if c.direction == "short" else "—")
        lines.append(
            f"  {c.symbol:<8} {arrow} {c.change_24h_pct:>+6.1f}% "
            f"vol=${c.volume_24h_usd/1e6:.0f}M "
            f"thresh={c.threshold*100:.1f}% "
            f"score={c.score:.1f}"
        )
    return "\n".join(lines)


def build_thresholds_arg(candidates: List[AssetCandidate]) -> str:
    """Build --thresholds CLI arg string from candidates."""
    return ",".join(f"{c.symbol}:{c.threshold}" for c in candidates)


def build_symbols_arg(candidates: List[AssetCandidate]) -> str:
    """Build --symbols CLI arg string from candidates."""
    return ",".join(c.symbol for c in candidates)
