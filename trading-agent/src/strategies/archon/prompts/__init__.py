"""Prompt management for Archon strategy.

Loads system prompt and renders user prompts via Jinja2 templates.
All prompts live in this directory as .md and .md.j2 files.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_PROMPT_DIR = Path(__file__).parent
_env = Environment(
    loader=FileSystemLoader(str(_PROMPT_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_system_prompt() -> str:
    """Load the system prompt from system.md."""
    return (_PROMPT_DIR / "system.md").read_text().strip()


def render_decision_prompt(context_data: dict) -> str:
    """Render the decision prompt template with market context.

    Args:
        context_data: Dict with all template variables from MarketContext.

    Returns:
        Rendered markdown prompt string.
    """
    template = _env.get_template("decision.md.j2")
    return template.render(**context_data)
