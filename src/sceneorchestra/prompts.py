"""Prompt templates used consistently for training and inference."""

from __future__ import annotations

from collections.abc import Iterable

from .constants import TOOL_GUIDE
from .trajectory import ToolCall, format_trajectory


def orchestrator_prompt(instruction: str, history: Iterable[ToolCall] | None = None) -> str:
    history = list(history or [])
    parts = [f"Instruction: {instruction}", TOOL_GUIDE]
    if history:
        parts.append("History:\n" + format_trajectory(history))
        parts.append(
            "Based on the instruction, available tools, and history above, predict the next tool call with its arguments."
        )
    else:
        parts.append(
            "Based on the instruction and available tools above, generate the complete tool-call trajectory."
        )
    return "\n\n".join(parts)


def discriminator_prompt(instruction: str, candidates: list[list[ToolCall]]) -> str:
    rendered = [f"Candidate {index}:\n{format_trajectory(calls)}" for index, calls in enumerate(candidates)]
    return "\n\n".join(
        [
            f"Instruction: {instruction}",
            TOOL_GUIDE,
            *rendered,
            f"Return only the integer index (0-{len(candidates) - 1}) of the best trajectory.",
        ]
    )
