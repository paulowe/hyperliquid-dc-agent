"""SessionManager: manages concurrent live_bridge.py subprocesses."""

from __future__ import annotations

import logging
import signal
import subprocess
import uuid
from pathlib import Path

from agents.observer.config import ObserverConfig, SessionConfig

logger = logging.getLogger(__name__)

# Path to the live_bridge.py script (relative to repo root)
_BRIDGE_SCRIPT = Path(__file__).resolve().parents[2] / "strategies" / "dc_overshoot" / "live_bridge.py"


class SessionManager:
    """Manages concurrent observe-only live_bridge.py sessions as subprocesses."""

    def __init__(self, config: ObserverConfig):
        self._config = config
        # session_id -> (process, session_config)
        self._sessions: dict[str, tuple[subprocess.Popen, SessionConfig]] = {}

        # Ensure report directory exists
        self._config.report_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_session_count(self) -> int:
        """Number of sessions currently tracked (running or completed)."""
        return len(self._sessions)

    def _build_command(self, session_cfg: SessionConfig, session_id: str) -> list[str]:
        """Build the CLI command to launch a live_bridge.py subprocess."""
        report_path = self._config.report_dir / f"report_{session_id}.json"
        cmd = [
            "uv", "run", "--package", "hyperliquid-trading-bot",
            "python", str(_BRIDGE_SCRIPT),
        ]
        # Add all session config args
        cmd.extend(session_cfg.to_bridge_args())
        # Add duration and report path
        cmd.extend([
            "--duration", str(self._config.observation_duration_minutes),
            "--json-report", str(report_path),
        ])
        return cmd

    def start_session(self, session_cfg: SessionConfig) -> str:
        """Start a single observe-only session. Returns session_id."""
        session_id = uuid.uuid4().hex[:8]
        cmd = self._build_command(session_cfg, session_id)

        logger.info(
            "Starting session %s: %s threshold=%.4f",
            session_id, session_cfg.symbol, session_cfg.threshold,
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[4]),  # Repo root
        )
        self._sessions[session_id] = (proc, session_cfg)
        return session_id

    def start_all(self) -> list[str]:
        """Start all sessions from config concurrently. Returns list of session IDs."""
        session_ids = []
        for session_cfg in self._config.sessions[:self._config.max_concurrent]:
            sid = self.start_session(session_cfg)
            session_ids.append(sid)
        return session_ids

    def get_session_status(self, session_id: str) -> dict:
        """Get status of a single session."""
        if session_id not in self._sessions:
            return {"error": f"Unknown session {session_id}"}

        proc, cfg = self._sessions[session_id]
        returncode = proc.poll()
        return {
            "session_id": session_id,
            "running": returncode is None,
            "pid": proc.pid,
            "returncode": returncode,
            "symbol": cfg.symbol,
            "threshold": cfg.threshold,
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all sessions."""
        return {sid: self.get_session_status(sid) for sid in self._sessions}

    def is_all_complete(self) -> bool:
        """Check if all sessions have completed."""
        return all(
            proc.poll() is not None
            for proc, _ in self._sessions.values()
        )

    def wait_all(self, timeout: int | None = None) -> dict[str, int]:
        """Wait for all sessions to complete. Returns {session_id: returncode}."""
        results = {}
        for sid, (proc, _) in self._sessions.items():
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Session %s timed out after %ds", sid, timeout)
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=10)
            results[sid] = proc.returncode
        return results

    def stop_session(self, session_id: str) -> None:
        """Send SIGINT to a session for graceful shutdown."""
        if session_id not in self._sessions:
            logger.warning("Unknown session %s", session_id)
            return
        proc, _ = self._sessions[session_id]
        if proc.poll() is None:
            logger.info("Stopping session %s (PID %d)", session_id, proc.pid)
            proc.send_signal(signal.SIGINT)

    def stop_all(self) -> None:
        """Stop all active sessions."""
        for sid in list(self._sessions):
            self.stop_session(sid)

    def get_session_ids(self) -> list[str]:
        """Return all tracked session IDs."""
        return list(self._sessions.keys())
