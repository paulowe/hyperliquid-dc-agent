"""Tests for Archon config."""

import pytest

from strategies.archon.config import ArchonConfig


class TestArchonConfigDefaults:
    def test_defaults(self):
        cfg = ArchonConfig()
        assert cfg.symbol == "HYPE"
        assert cfg.dc_threshold == (0.02, 0.02)
        assert cfg.direction_filter == "long"
        assert cfg.use_ai is True
        assert cfg.min_confidence == 0.6

    def test_from_dict_basic(self):
        cfg = ArchonConfig.from_dict({"symbol": "SOL", "leverage": 3})
        assert cfg.symbol == "SOL"
        assert cfg.leverage == 3

    def test_from_dict_tuple_conversion(self):
        cfg = ArchonConfig.from_dict({"dc_threshold": [0.015, 0.015]})
        assert cfg.dc_threshold == (0.015, 0.015)

    def test_direction_filter_validation(self):
        ArchonConfig.from_dict({"direction_filter": "long"})
        ArchonConfig.from_dict({"direction_filter": "short"})
        ArchonConfig.from_dict({"direction_filter": "both"})
        with pytest.raises(ValueError, match="direction_filter"):
            ArchonConfig.from_dict({"direction_filter": "invalid"})
