"""Tests for StateManager — JSON file persistence."""

import json
import pytest
from pathlib import Path

from agents.observer.state import StateManager, ObserverState


class TestStateManagerBasics:
    """Load, save, and update cycle."""

    def test_load_returns_default_when_no_file(self, tmp_path):
        state_file = tmp_path / "state.json"
        mgr = StateManager(state_file)
        state = mgr.load()

        assert state.current_phase == "idle"
        assert state.exploration_round == 0
        assert state.active_sessions == {}
        assert state.completed_rounds == []

    def test_save_and_load_roundtrip(self, tmp_path):
        state_file = tmp_path / "state.json"
        mgr = StateManager(state_file)

        state = ObserverState(
            current_phase="exploring",
            exploration_round=3,
            active_sessions={"abc": {"threshold": 0.004}},
            completed_rounds=[{"round": 1, "best": 0.01}],
            current_live_config=None,
            live_session_id=None,
        )
        mgr.save(state)

        loaded = mgr.load()
        assert loaded.current_phase == "exploring"
        assert loaded.exploration_round == 3
        assert loaded.active_sessions == {"abc": {"threshold": 0.004}}
        assert len(loaded.completed_rounds) == 1

    def test_update_modifies_and_saves(self, tmp_path):
        state_file = tmp_path / "state.json"
        mgr = StateManager(state_file)

        mgr.update(current_phase="analyzing", exploration_round=5)

        loaded = mgr.load()
        assert loaded.current_phase == "analyzing"
        assert loaded.exploration_round == 5

    def test_update_preserves_unmodified_fields(self, tmp_path):
        state_file = tmp_path / "state.json"
        mgr = StateManager(state_file)

        mgr.update(current_phase="exploring", exploration_round=2)
        mgr.update(current_phase="analyzing")

        loaded = mgr.load()
        assert loaded.current_phase == "analyzing"
        assert loaded.exploration_round == 2  # Preserved

    def test_atomic_write(self, tmp_path):
        """Save should use atomic write (no .tmp file left behind)."""
        state_file = tmp_path / "state.json"
        mgr = StateManager(state_file)

        mgr.save(ObserverState(
            current_phase="idle",
            exploration_round=0,
            active_sessions={},
            completed_rounds=[],
            current_live_config=None,
            live_session_id=None,
        ))

        assert state_file.exists()
        assert not state_file.with_suffix(".tmp").exists()

    def test_state_file_is_valid_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        mgr = StateManager(state_file)
        mgr.update(current_phase="exploring")

        # Should be valid JSON
        data = json.loads(state_file.read_text())
        assert data["current_phase"] == "exploring"


class TestObserverState:
    """ObserverState dataclass."""

    def test_default_state(self):
        state = ObserverState.default()
        assert state.current_phase == "idle"
        assert state.exploration_round == 0
        assert state.active_sessions == {}

    def test_to_dict_and_from_dict_roundtrip(self):
        state = ObserverState(
            current_phase="exploiting",
            exploration_round=10,
            active_sessions={"x": {"pid": 123}},
            completed_rounds=[{"best": 0.015}],
            current_live_config={"threshold": 0.015},
            live_session_id="live_abc",
        )
        d = state.to_dict()
        restored = ObserverState.from_dict(d)

        assert restored.current_phase == state.current_phase
        assert restored.exploration_round == state.exploration_round
        assert restored.active_sessions == state.active_sessions
        assert restored.current_live_config == state.current_live_config
        assert restored.live_session_id == state.live_session_id
