"""Portable JSON records shared by rollout, data construction, and execution."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trajectory import ToolCall


@dataclass
class StepRecord:
    index: int
    call: ToolCall
    cumulative_minutes: float
    metric: dict[str, Any]
    score: dict[str, float]
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "call": self.call.to_dict(),
            "result": self.result,
            "cumulative_minutes": self.cumulative_minutes,
            "metric": self.metric,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StepRecord:
        return cls(
            index=int(value["index"]),
            call=ToolCall.from_dict(value["call"]),
            result=value.get("result"),
            cumulative_minutes=float(value["cumulative_minutes"]),
            metric=dict(value.get("metric", {})),
            score=dict(value["score"]),
        )


@dataclass
class RolloutRecord:
    instruction: str
    rollout_id: str
    steps: list[StepRecord]
    source_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "rollout_id": self.rollout_id,
            "source_dir": self.source_dir,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RolloutRecord:
        return cls(
            instruction=str(value["instruction"]),
            rollout_id=str(value["rollout_id"]),
            source_dir=value.get("source_dir"),
            metadata=dict(value.get("metadata", {})),
            steps=[StepRecord.from_dict(item) for item in value["steps"]],
        )


def read_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    values = value if isinstance(value, list) else [value]
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object record")
        yield item


def read_rollouts(path: str | Path) -> list[RolloutRecord]:
    return [RolloutRecord.from_dict(item) for item in read_json_records(path)]


def write_jsonl(path: str | Path, values: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            count += 1
    return count
