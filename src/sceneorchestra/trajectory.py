"""Safe parsing, validation, and formatting of model-generated tool trajectories."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .constants import ALL_TOOLS, INITIALIZER_TOOLS


_NUMBERING = re.compile(r"^\s*(?:\d+[\.:]\s*)?(?P<call>[A-Za-z_]\w*\s*\(.*\))\s*$")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCall:
        name = value.get("name", value.get("tool"))
        arguments = value.get("arguments", value.get("args", {}))
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError(f"Invalid tool call: {value!r}")
        return cls(name=name, arguments=arguments)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("Tool arguments must be Python literal values") from exc


def parse_call(text: str) -> ToolCall:
    match = _NUMBERING.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Invalid tool-call line: {text!r}")
    try:
        expression = ast.parse(match.group("call"), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid tool-call syntax: {text!r}") from exc
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError(f"Expected a direct function call: {text!r}")
    if expression.args:
        raise ValueError("Only keyword arguments are supported")
    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs are not supported")
        if keyword.arg in arguments:
            raise ValueError(f"Duplicate argument {keyword.arg!r}")
        arguments[keyword.arg] = _literal(keyword.value)
    return ToolCall(expression.func.id, arguments)


def parse_trajectory(text: str, *, require_complete: bool = True) -> list[ToolCall]:
    calls = [parse_call(line) for line in text.splitlines() if line.strip()]
    validate_trajectory(calls, require_complete=require_complete)
    return calls


def validate_trajectory(calls: Iterable[ToolCall], *, require_complete: bool = True) -> None:
    calls = list(calls)
    if not calls:
        raise ValueError("Trajectory is empty")
    unknown = [call.name for call in calls if call.name not in ALL_TOOLS]
    if unknown:
        raise ValueError(f"Unknown SceneWeaver tool(s): {unknown}")
    if require_complete and calls[0].name not in INITIALIZER_TOOLS:
        raise ValueError("A complete trajectory must start with an initializer")
    if require_complete and calls[-1].name != "terminate":
        raise ValueError("A complete trajectory must end with terminate")
    if any(call.name == "terminate" for call in calls[:-1]):
        raise ValueError("terminate may only appear as the final call")


def _format_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        suffix = "," if len(value) == 1 else ""
        return "(" + ", ".join(_format_literal(item) for item in value) + suffix + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{_format_literal(key)}: {_format_literal(item)}" for key, item in value.items()
        ) + "}"
    raise ValueError(f"Unsupported tool argument type: {type(value).__name__}")


def format_call(call: ToolCall) -> str:
    rendered = ", ".join(
        f"{key}={_format_literal(value)}" for key, value in call.arguments.items()
    )
    return f"{call.name}({rendered})"


def format_trajectory(calls: Iterable[ToolCall]) -> str:
    return "\n".join(f"{index}. {format_call(call)}" for index, call in enumerate(calls, 1))
