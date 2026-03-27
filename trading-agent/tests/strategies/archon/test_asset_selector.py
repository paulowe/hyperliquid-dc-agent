"""Tests for dynamic asset selector."""

from unittest.mock import patch, MagicMock

import pytest

from strategies.archon.asset_selector import (
    AssetCandidate,
    fetch_candidates,
    format_selection,
    build_thresholds_arg,
    build_symbols_arg,
)


def mock_api_response():
    """Mock Hyperliquid API responses."""
    mids = {"BTC": "70000", "SOL": "86", "HYPE": "39", "FARTCOIN": "0.17", "LIT": "0.9"}
    meta = {
        "universe": [
            {"name": "BTC"}, {"name": "SOL"}, {"name": "HYPE"},
            {"name": "FARTCOIN"}, {"name": "LIT"},
        ]
    }
    ctxs = [
        {"prevDayPx": "71000", "dayNtlVlm": "2000000000", "funding": "0.0001", "openInterest": "28000"},
        {"prevDayPx": "90", "dayNtlVlm": "300000000", "funding": "-0.00003", "openInterest": "3800"},
        {"prevDayPx": "40", "dayNtlVlm": "200000000", "funding": "0.0001", "openInterest": "21000"},
        {"prevDayPx": "0.18", "dayNtlVlm": "18000000", "funding": "0.0001", "openInterest": "200000"},
        {"prevDayPx": "0.95", "dayNtlVlm": "8000000", "funding": "-0.00004", "openInterest": "35000"},
    ]
    return mids, [meta, ctxs]


class TestFetchCandidates:
    @patch("strategies.archon.asset_selector.requests.post")
    def test_returns_candidates(self, mock_post):
        mids, meta_ctxs = mock_api_response()
        mock_post.side_effect = [
            MagicMock(json=lambda: mids),
            MagicMock(json=lambda: meta_ctxs),
        ]
        candidates = fetch_candidates(min_volume_usd=1_000_000, max_assets=5)
        assert len(candidates) > 0
        assert all(isinstance(c, AssetCandidate) for c in candidates)

    @patch("strategies.archon.asset_selector.requests.post")
    def test_core_symbols_included(self, mock_post):
        mids, meta_ctxs = mock_api_response()
        mock_post.side_effect = [
            MagicMock(json=lambda: mids),
            MagicMock(json=lambda: meta_ctxs),
        ]
        candidates = fetch_candidates(max_assets=5)
        names = {c.symbol for c in candidates}
        # SOL has -4.4% change, should be included as core + momentum
        assert "SOL" in names

    @patch("strategies.archon.asset_selector.requests.post")
    def test_direction_assignment(self, mock_post):
        mids, meta_ctxs = mock_api_response()
        mock_post.side_effect = [
            MagicMock(json=lambda: mids),
            MagicMock(json=lambda: meta_ctxs),
        ]
        candidates = fetch_candidates()
        for c in candidates:
            assert c.direction in ("long", "short", "neutral")
            if c.change_24h_pct > 1.5:
                assert c.direction == "long"
            elif c.change_24h_pct < -1.5:
                assert c.direction == "short"

    @patch("strategies.archon.asset_selector.requests.post")
    def test_sorted_by_score(self, mock_post):
        mids, meta_ctxs = mock_api_response()
        mock_post.side_effect = [
            MagicMock(json=lambda: mids),
            MagicMock(json=lambda: meta_ctxs),
        ]
        candidates = fetch_candidates()
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)


class TestFormatSelection:
    def test_format(self):
        candidates = [
            AssetCandidate("SOL", 86.0, -4.5, 300e6, -0.003, 300e6, "short", 15.0, 0.025),
            AssetCandidate("HYPE", 39.0, -1.8, 200e6, 0.001, 800e6, "short", 8.0, 0.02),
        ]
        text = format_selection(candidates)
        assert "SOL" in text
        assert "HYPE" in text
        assert "↓" in text


class TestBuildArgs:
    def test_thresholds(self):
        candidates = [
            AssetCandidate("SOL", 86.0, -4.5, 300e6, 0, 0, "short", 15.0, 0.015),
            AssetCandidate("HYPE", 39.0, -2.0, 200e6, 0, 0, "short", 8.0, 0.02),
        ]
        assert build_thresholds_arg(candidates) == "SOL:0.015,HYPE:0.02"

    def test_symbols(self):
        candidates = [
            AssetCandidate("SOL", 86.0, -4.5, 300e6, 0, 0, "short", 15.0, 0.015),
            AssetCandidate("LIT", 0.9, -8.0, 8e6, 0, 0, "short", 5.0, 0.03),
        ]
        assert build_symbols_arg(candidates) == "SOL,LIT"
