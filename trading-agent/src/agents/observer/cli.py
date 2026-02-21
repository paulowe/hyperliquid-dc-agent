"""CLI entry point for the observer agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src/ to path for imports (follows live_bridge.py pattern)
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agents.observer.config import ObserverConfig, DEFAULT_THRESHOLDS
from agents.observer.session import SessionManager
from agents.observer.collector import ResultCollector
from agents.observer.reasoner import ClaudeReasoner
from agents.observer.state import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Default paths (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT_DIR = _REPO_ROOT / "trading-agent" / "state" / "reports"
DEFAULT_STATE_FILE = _REPO_ROOT / "trading-agent" / "state" / "observer_state.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the observer agent CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="DC Overshoot Observer Agent — multi-threshold concurrent trading orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run one exploration round (30 min, 6 thresholds)
  python -m agents.observer explore --symbol SOL --duration 30

  # Explore with custom thresholds
  python -m agents.observer explore --symbol BTC --thresholds 0.002 0.005 0.01

  # Continuous explore loop
  python -m agents.observer loop --symbol SOL

  # Check status
  python -m agents.observer status

  # Stop all sessions
  python -m agents.observer stop
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Observer agent commands")

    # explore: run a single exploration round
    explore_p = subparsers.add_parser("explore", help="Run one exploration round")
    explore_p.add_argument("--symbol", default="SOL", help="Trading symbol (default: SOL)")
    explore_p.add_argument("--duration", type=int, default=30, help="Minutes per session (default: 30)")
    explore_p.add_argument(
        "--thresholds", nargs="+", type=float, default=None,
        help="DC thresholds to test (default: 0.002 0.004 0.006 0.008 0.01 0.015)",
    )
    explore_p.add_argument("--no-ai", action="store_true", help="Skip Claude analysis, use heuristic ranking")
    explore_p.add_argument("--position-size", type=float, default=100.0, help="Position size USD (default: 100)")
    explore_p.add_argument("--leverage", type=int, default=10, help="Leverage (default: 10)")

    # loop: continuous explore/exploit loop
    loop_p = subparsers.add_parser("loop", help="Run continuous explore/exploit loop")
    loop_p.add_argument("--symbol", default="SOL", help="Trading symbol (default: SOL)")
    loop_p.add_argument("--duration", type=int, default=30, help="Minutes per session (default: 30)")
    loop_p.add_argument("--rounds", type=int, default=0, help="Number of rounds (0=infinite, default: 0)")
    loop_p.add_argument("--thresholds", nargs="+", type=float, default=None)
    loop_p.add_argument("--no-ai", action="store_true", help="Skip Claude analysis")
    loop_p.add_argument("--position-size", type=float, default=100.0)
    loop_p.add_argument("--leverage", type=int, default=10)

    # status: show current state
    subparsers.add_parser("status", help="Show observer agent status")

    # stop: stop all running sessions
    subparsers.add_parser("stop", help="Stop all running sessions")

    return parser


async def run_exploration_round(
    config: ObserverConfig,
    state_mgr: StateManager,
    reasoner: ClaudeReasoner,
    round_num: int,
) -> None:
    """Run a single exploration round: start sessions → wait → collect → analyze."""
    logger.info("=" * 70)
    logger.info("EXPLORATION ROUND %d", round_num)
    logger.info("=" * 70)
    logger.info("Sessions: %d | Duration: %d min | Thresholds: %s",
                len(config.sessions), config.observation_duration_minutes,
                [s.threshold for s in config.sessions])

    # 1. Start all sessions
    state_mgr.update(current_phase="exploring", exploration_round=round_num)
    session_mgr = SessionManager(config)
    session_ids = session_mgr.start_all()

    active = {sid: {"threshold": cfg.threshold, "pid": proc.pid}
              for sid, (proc, cfg) in session_mgr._sessions.items()}
    state_mgr.update(active_sessions=active)

    logger.info("Launched %d sessions: %s", len(session_ids), session_ids)

    # 2. Wait for all sessions to complete
    logger.info("Waiting for sessions to complete (up to %d minutes)...",
                config.observation_duration_minutes)
    timeout_seconds = (config.observation_duration_minutes + 2) * 60  # +2 min grace
    results = session_mgr.wait_all(timeout=timeout_seconds)

    logger.info("All sessions complete: %s",
                {sid: f"rc={rc}" for sid, rc in results.items()})

    # 3. Collect reports
    state_mgr.update(current_phase="analyzing")
    collector = ResultCollector(config.report_dir)
    reports = collector.collect(session_ids)
    logger.info("Collected %d reports (of %d sessions)", len(reports), len(session_ids))

    if not reports:
        logger.warning("No reports collected. Skipping analysis.")
        state_mgr.update(current_phase="idle", active_sessions={})
        return

    # 4. Compute simulated PnL for each report
    report_pnl_pairs = []
    for report in reports:
        pnl = collector.compute_simulated_pnl(report)
        report_pnl_pairs.append((report, pnl))
        logger.info(
            "  threshold=%.4f | signals=%d | trades=%d | net=$%+.2f | %dW/%dL",
            report.threshold, report.signal_count, pnl.total_trades,
            pnl.net_pnl_usd, pnl.wins, pnl.losses,
        )

    # 5. Analyze with Claude (or heuristic)
    analysis = await reasoner.analyze(report_pnl_pairs, {"round": round_num})

    logger.info("-" * 70)
    logger.info("ANALYSIS RESULT")
    logger.info("-" * 70)
    logger.info("Best threshold: %s", analysis.best_threshold)
    logger.info("Confidence: %.0f%%", analysis.confidence * 100)
    logger.info("Reasoning: %s", analysis.reasoning)
    if analysis.suggestions:
        for s in analysis.suggestions:
            logger.info("  Suggestion: %s", s)
    if analysis.best_config:
        logger.info("Recommended live command:")
        logger.info(
            "  uv run --package hyperliquid-trading-bot python "
            "trading-agent/src/strategies/dc_overshoot/live_bridge.py "
            "--symbol %s --threshold %s --sl-pct %s --tp-pct %s "
            "--trail-pct %s --position-size %s --leverage %s --yes",
            reports[0].symbol, analysis.best_config["threshold"],
            analysis.best_config["sl_pct"], analysis.best_config["tp_pct"],
            analysis.best_config.get("trail_pct", 0.5),
            reports[0].position_size_usd, reports[0].leverage,
        )

    # 6. Save to state
    round_result = {
        "round": round_num,
        "timestamp": time.time(),
        "best_threshold": analysis.best_threshold,
        "best_config": analysis.best_config,
        "reasoning": analysis.reasoning,
        "confidence": analysis.confidence,
        "session_count": len(reports),
    }
    state = state_mgr.load()
    state.completed_rounds.append(round_result)
    state.active_sessions = {}
    state.current_phase = "idle"
    state_mgr.save(state)

    logger.info("=" * 70)


def cmd_explore(args) -> None:
    """Handle the 'explore' command."""
    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    config = ObserverConfig.default_exploration(
        symbol=args.symbol,
        report_dir=DEFAULT_REPORT_DIR,
        state_file=DEFAULT_STATE_FILE,
        thresholds=thresholds,
        position_size_usd=args.position_size,
        leverage=args.leverage,
        observation_duration_minutes=args.duration,
    )
    state_mgr = StateManager(DEFAULT_STATE_FILE)
    reasoner = ClaudeReasoner(use_ai=not args.no_ai)

    asyncio.run(run_exploration_round(config, state_mgr, reasoner, round_num=1))


def cmd_loop(args) -> None:
    """Handle the 'loop' command."""
    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    config = ObserverConfig.default_exploration(
        symbol=args.symbol,
        report_dir=DEFAULT_REPORT_DIR,
        state_file=DEFAULT_STATE_FILE,
        thresholds=thresholds,
        position_size_usd=args.position_size,
        leverage=args.leverage,
        observation_duration_minutes=args.duration,
    )
    state_mgr = StateManager(DEFAULT_STATE_FILE)
    reasoner = ClaudeReasoner(use_ai=not args.no_ai)

    async def loop():
        round_num = 0
        while args.rounds == 0 or round_num < args.rounds:
            round_num += 1
            await run_exploration_round(config, state_mgr, reasoner, round_num)
            if args.rounds == 0 or round_num < args.rounds:
                logger.info("Next round starts in 10 seconds...")
                await asyncio.sleep(10)

    asyncio.run(loop())


def cmd_status(args) -> None:
    """Handle the 'status' command."""
    state_mgr = StateManager(DEFAULT_STATE_FILE)
    state = state_mgr.load()

    print("=" * 70)
    print("Observer Agent Status")
    print("=" * 70)
    print(f"  Phase:       {state.current_phase}")
    print(f"  Round:       {state.exploration_round}")
    print(f"  Active:      {len(state.active_sessions)} sessions")
    print(f"  Completed:   {len(state.completed_rounds)} rounds")

    if state.active_sessions:
        print("\n  Active Sessions:")
        for sid, info in state.active_sessions.items():
            print(f"    {sid}: threshold={info.get('threshold')} pid={info.get('pid')}")

    if state.completed_rounds:
        print("\n  Recent Rounds:")
        for r in state.completed_rounds[-5:]:
            print(f"    Round {r['round']}: best={r.get('best_threshold')} "
                  f"conf={r.get('confidence', 0):.0%} — {r.get('reasoning', '')[:60]}...")

    if state.current_live_config:
        print(f"\n  Live Config: {state.current_live_config}")
    print("=" * 70)


def cmd_stop(args) -> None:
    """Handle the 'stop' command."""
    state_mgr = StateManager(DEFAULT_STATE_FILE)
    state = state_mgr.load()

    if not state.active_sessions:
        print("No active sessions to stop.")
        return

    # Try to kill processes by PID
    import signal
    import os

    stopped = 0
    for sid, info in state.active_sessions.items():
        pid = info.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGINT)
                stopped += 1
                print(f"  Sent SIGINT to session {sid} (PID {pid})")
            except ProcessLookupError:
                print(f"  Session {sid} (PID {pid}) already terminated")

    state_mgr.update(active_sessions={}, current_phase="idle")
    print(f"Stopped {stopped} sessions.")


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "explore": cmd_explore,
        "loop": cmd_loop,
        "status": cmd_status,
        "stop": cmd_stop,
    }

    try:
        handlers[args.command](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
