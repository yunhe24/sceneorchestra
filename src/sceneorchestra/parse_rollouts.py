"""Normalize raw SceneWeaver output folders into scored rollout JSONL."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .records import RolloutRecord, StepRecord, read_json_records, write_jsonl
from .scoring import score_metric
from .trajectory import ToolCall


_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_STEP = re.compile(r"Executing step (\d+)/")
_TOOL_NAME = re.compile(r"Activating tool: ['\"]([^'\"]+)['\"]")
_TOOL_ARGS_MARKER = "Tool arguments:"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _step_cumulative_minutes(log_path: Path) -> dict[int, float]:
    events: list[tuple[datetime, int | None]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _ANSI.sub("", line)
        timestamp_match = _TIMESTAMP.match(line)
        if not timestamp_match:
            continue
        timestamp = datetime.fromisoformat(timestamp_match.group(1))
        step_match = _STEP.search(line)
        events.append((timestamp, int(step_match.group(1)) if step_match else None))
    starts = [(timestamp, step) for timestamp, step in events if step is not None]
    if not starts:
        raise ValueError(f"No timestamped 'Executing step' markers in {log_path}")
    origin = starts[0][0]
    final_time = events[-1][0]
    cumulative: dict[int, float] = {}
    for position, (timestamp, step) in enumerate(starts):
        later = next(
            (candidate_time for candidate_time, candidate_step in starts[position + 1 :] if candidate_step > step),
            final_time,
        )
        cumulative[step] = max((later - origin).total_seconds() / 60.0, 0.0)
    return cumulative


def _latest_trajectory(run_dir: Path) -> dict[str, Any]:
    files = list((run_dir / "pipeline").glob("trajs_*.json"))
    if not files:
        raise FileNotFoundError(f"No pipeline/trajs_*.json in {run_dir}")
    latest = max(files, key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    value = json.loads(latest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected trajectory object in {latest}")
    return value


def _calls_from_log(log_path: Path) -> dict[int, ToolCall]:
    """Recover full arguments (notably initializer roomtype) omitted by trajs JSON."""
    current_step: int | None = None
    pending_arguments: dict[str, Any] | None = None
    calls: dict[int, ToolCall] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _ANSI.sub("", line)
        step_match = _STEP.search(line)
        if step_match:
            current_step = int(step_match.group(1))
            pending_arguments = None
        if _TOOL_ARGS_MARKER in line:
            raw = line.split(_TOOL_ARGS_MARKER, 1)[1].strip()
            try:
                value = json.loads(raw)
                pending_arguments = value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pending_arguments = None
        name_match = _TOOL_NAME.search(line)
        if name_match and current_step is not None:
            calls[current_step] = ToolCall(name_match.group(1), pending_arguments or {})
    return calls


def _call_from_trajectory_item(item: dict[str, Any]) -> ToolCall:
    name = item.get("action", item.get("tool"))
    arguments = item.get("arguments", item.get("args"))
    if arguments is None:
        arguments = {}
        for key in ("ideas", "roomtype", "status"):
            if key in item:
                arguments[key] = item[key]
    return ToolCall.from_dict({"name": name, "arguments": arguments})


def parse_manifest_record(record: dict[str, Any]) -> RolloutRecord:
    run_dir = Path(record["run_dir"]).resolve()
    log_path = Path(record["log_file"]).resolve()
    runtimes = _step_cumulative_minutes(log_path)
    trajectory = _latest_trajectory(run_dir)
    logged_calls = _calls_from_log(log_path)
    steps = []
    for key in sorted(trajectory, key=lambda value: int(value)):
        index = int(key)
        item = trajectory[key]
        metric_path = run_dir / "pipeline" / f"metric_{index}.json"
        if not metric_path.is_file() or index not in runtimes:
            continue
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        layout_path = run_dir / "record_scene" / f"layout_{index}.json"
        object_count = None
        if layout_path.is_file():
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            object_count = len(layout.get("objects", {}))
        score = score_metric(metric, runtimes[index], object_count_fallback=object_count)
        steps.append(
            StepRecord(
                index=index,
                call=logged_calls.get(index, _call_from_trajectory_item(item)),
                result=item.get("results"),
                cumulative_minutes=runtimes[index],
                metric=metric,
                score=score.to_dict(),
            )
        )
    if not steps:
        raise ValueError(f"No complete, scored steps found in {run_dir}")
    return RolloutRecord(
        instruction=str(record["instruction"]),
        rollout_id=str(record["rollout_id"]),
        source_dir=str(run_dir),
        metadata={"log_file": str(log_path)},
        steps=steps,
    )


def parse_manifest(manifest_path: str | Path, output_path: str | Path, *, skip_failed: bool = False) -> int:
    rollouts = []
    for record in read_json_records(manifest_path):
        if int(record.get("returncode", 0)) != 0 or not record.get("run_dir"):
            if skip_failed:
                continue
            raise RuntimeError(f"Manifest contains a failed rollout: {record}")
        try:
            rollouts.append(parse_manifest_record(record).to_dict())
        except (ValueError, FileNotFoundError):
            if not skip_failed:
                raise
    return write_jsonl(output_path, rollouts)
