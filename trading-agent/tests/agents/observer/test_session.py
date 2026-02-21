"""Tests for SessionManager — subprocess lifecycle management."""

import asyncio
import signal
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from agents.observer.config import ObserverConfig, SessionConfig
from agents.observer.session import SessionManager


@pytest.fixture
def sample_config(tmp_path):
    return ObserverConfig(
        sessions=[
            SessionConfig(symbol="SOL", threshold=0.004, sl_pct=0.005, tp_pct=0.01),
            SessionConfig(symbol="SOL", threshold=0.01, sl_pct=0.01, tp_pct=0.03),
        ],
        report_dir=tmp_path / "reports",
        state_file=tmp_path / "state.json",
        observation_duration_minutes=5,
    )


class TestSessionManagerInit:
    """SessionManager initialization."""

    def test_creates_with_config(self, sample_config):
        mgr = SessionManager(sample_config)
        assert mgr.active_session_count == 0

    def test_report_dir_created(self, sample_config):
        mgr = SessionManager(sample_config)
        assert sample_config.report_dir.exists()


class TestBuildCommand:
    """Verify CLI command construction."""

    def test_build_command_for_session(self, sample_config):
        mgr = SessionManager(sample_config)
        session_cfg = sample_config.sessions[0]
        cmd = mgr._build_command(session_cfg, "test123")

        # Should be a list of strings
        assert isinstance(cmd, list)
        assert all(isinstance(c, str) for c in cmd)

        # Should include key arguments
        cmd_str = " ".join(cmd)
        assert "--observe-only" in cmd_str
        assert "--symbol SOL" in cmd_str
        assert "--threshold 0.004" in cmd_str
        assert "--json-report" in cmd_str
        assert "test123" in cmd_str
        assert "--duration 5" in cmd_str

    def test_build_command_includes_all_params(self, sample_config):
        mgr = SessionManager(sample_config)
        session_cfg = SessionConfig(
            symbol="BTC", threshold=0.008, sl_pct=0.01, tp_pct=0.03,
            trail_pct=0.7, min_profit_to_trail_pct=0.002,
            position_size_usd=50.0, leverage=3,
        )
        cmd = mgr._build_command(session_cfg, "abc")
        cmd_str = " ".join(cmd)
        assert "--trail-pct 0.7" in cmd_str
        assert "--min-profit-to-trail-pct 0.002" in cmd_str
        assert "--position-size 50.0" in cmd_str
        assert "--leverage 3" in cmd_str


class TestStartSession:
    """Start individual sessions."""

    @patch("agents.observer.session.subprocess.Popen")
    def test_start_session_returns_session_id(self, mock_popen, sample_config):
        mock_popen.return_value = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        mgr = SessionManager(sample_config)
        session_id = mgr.start_session(sample_config.sessions[0])

        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert mgr.active_session_count == 1

    @patch("agents.observer.session.subprocess.Popen")
    def test_start_all_launches_all_sessions(self, mock_popen, sample_config):
        mock_popen.return_value = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        mgr = SessionManager(sample_config)
        ids = mgr.start_all()

        assert len(ids) == 2
        assert mgr.active_session_count == 2
        assert mock_popen.call_count == 2


class TestStopSession:
    """Stop sessions gracefully."""

    @patch("agents.observer.session.subprocess.Popen")
    def test_stop_session_sends_sigint(self, mock_popen, sample_config):
        mock_proc = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        mock_popen.return_value = mock_proc
        mgr = SessionManager(sample_config)
        sid = mgr.start_session(sample_config.sessions[0])

        mgr.stop_session(sid)
        mock_proc.send_signal.assert_called_once_with(signal.SIGINT)

    @patch("agents.observer.session.subprocess.Popen")
    def test_stop_all_stops_all_sessions(self, mock_popen, sample_config):
        mock_proc = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        mock_popen.return_value = mock_proc
        mgr = SessionManager(sample_config)
        mgr.start_all()

        mgr.stop_all()
        # Should have sent SIGINT to each process
        assert mock_proc.send_signal.call_count == 2


class TestGetStatus:
    """Session status tracking."""

    @patch("agents.observer.session.subprocess.Popen")
    def test_get_session_status_running(self, mock_popen, sample_config):
        mock_proc = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        mock_popen.return_value = mock_proc
        mgr = SessionManager(sample_config)
        sid = mgr.start_session(sample_config.sessions[0])

        status = mgr.get_session_status(sid)
        assert status["running"] is True
        assert status["pid"] == 12345

    @patch("agents.observer.session.subprocess.Popen")
    def test_get_session_status_completed(self, mock_popen, sample_config):
        mock_proc = MagicMock(pid=12345, poll=MagicMock(return_value=0))
        mock_popen.return_value = mock_proc
        mgr = SessionManager(sample_config)
        sid = mgr.start_session(sample_config.sessions[0])

        status = mgr.get_session_status(sid)
        assert status["running"] is False
        assert status["returncode"] == 0

    @patch("agents.observer.session.subprocess.Popen")
    def test_get_all_status(self, mock_popen, sample_config):
        mock_proc = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        mock_popen.return_value = mock_proc
        mgr = SessionManager(sample_config)
        mgr.start_all()

        statuses = mgr.get_all_status()
        assert len(statuses) == 2
