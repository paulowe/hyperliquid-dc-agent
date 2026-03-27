"""Claude CLI proxy for Archon trade decisions.

Routes Claude calls through the `claude` CLI binary (which uses the
Claude Code Max subscription) instead of the Anthropic API. This
avoids needing separate API credits.

The proxy:
1. Writes system prompt + user prompt to temp files
2. Invokes `claude --print --system-prompt <file> <prompt>` with
   CLAUDECODE env var unset (to avoid nesting detection)
3. Parses the JSON response

This module is intentionally NOT committed to the repo — it's a
local runtime adapter that depends on the user's Claude Code
subscription and CLI installation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str | None:
    """Extract JSON object from text that may contain markdown or prose."""
    import re
    # Try raw JSON first
    text = text.strip()
    if text.startswith("{"):
        # Find the matching closing brace
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:i + 1]

    # Try code block extraction
    if "```json" in text:
        match = re.search(r"```json\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "```" in text:
        match = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith("{"):
                return candidate

    # Try finding JSON object anywhere in the text
    match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', text)
    if match:
        return match.group(0)

    return None


def claude_cli_query(
    user_prompt: str,
    system_prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout_seconds: int = 30,
) -> str | None:
    """Call Claude via the CLI binary.

    Returns the raw text response, or None on failure.
    """
    # Write system prompt to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix="archon_sys_"
    ) as f:
        f.write(system_prompt)
        sys_path = f.name

    try:
        # Build command — unset CLAUDECODE to avoid nesting check
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        cmd = [
            "claude",
            "--print",                    # output only, no interactive
            "--model", model,
            "--system-prompt", sys_path,
            "--max-turns", "1",
            "--output-format", "text",    # raw text output
            user_prompt,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent.parent),  # repo root
        )

        if result.returncode != 0:
            logger.warning("Claude CLI failed (rc=%d): stderr=%s stdout=%s",
                           result.returncode, result.stderr[:300], result.stdout[:200])
            return None

        response = result.stdout.strip()
        if not response:
            logger.warning("Claude CLI returned empty response")
            return None

        # Try to extract JSON if Claude wrapped it in markdown or extra text
        extracted = _extract_json(response)
        return extracted or response

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out after %ds", timeout_seconds)
        return None
    except FileNotFoundError:
        logger.warning("Claude CLI binary not found")
        return None
    except Exception as e:
        logger.warning("Claude CLI error: %s", e)
        return None
    finally:
        try:
            os.unlink(sys_path)
        except OSError:
            pass
