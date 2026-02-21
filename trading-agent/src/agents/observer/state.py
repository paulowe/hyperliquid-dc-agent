"""StateManager: persistent observer state via JSON file."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ObserverState:
    """Persistent state of the observer agent."""

    current_phase: str  # "idle", "exploring", "analyzing", "exploiting"
    exploration_round: int
    active_sessions: dict  # session_id -> config/status dict
    completed_rounds: list[dict]  # History of exploration rounds + decisions
    current_live_config: dict | None
    live_session_id: str | None

    @classmethod
    def default(cls) -> ObserverState:
        """Create a default idle state."""
        return cls(
            current_phase="idle",
            exploration_round=0,
            active_sessions={},
            completed_rounds=[],
            current_live_config=None,
            live_session_id=None,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "current_phase": self.current_phase,
            "exploration_round": self.exploration_round,
            "active_sessions": self.active_sessions,
            "completed_rounds": self.completed_rounds,
            "current_live_config": self.current_live_config,
            "live_session_id": self.live_session_id,
            "last_updated": time.time(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObserverState:
        """Deserialize from a plain dict."""
        return cls(
            current_phase=d.get("current_phase", "idle"),
            exploration_round=d.get("exploration_round", 0),
            active_sessions=d.get("active_sessions", {}),
            completed_rounds=d.get("completed_rounds", []),
            current_live_config=d.get("current_live_config"),
            live_session_id=d.get("live_session_id"),
        )


class StateManager:
    """Persists observer state to a JSON file."""

    def __init__(self, state_file: Path):
        self._state_file = state_file

    def load(self) -> ObserverState:
        """Load state from disk, or return default if file doesn't exist."""
        if not self._state_file.exists():
            return ObserverState.default()
        try:
            data = json.loads(self._state_file.read_text())
            return ObserverState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt state file, returning default: %s", e)
            return ObserverState.default()

    def save(self, state: ObserverState) -> None:
        """Atomically write state to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state.to_dict(), indent=2))
        tmp_path.rename(self._state_file)

    def update(self, **kwargs) -> ObserverState:
        """Load current state, update specified fields, and save."""
        state = self.load()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning("Unknown state field: %s", key)
        self.save(state)
        return state
