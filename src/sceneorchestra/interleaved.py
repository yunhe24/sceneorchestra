"""Execution and scoring utilities for the single interleaved cycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .prompts import discriminator_prompt
from .records import read_json_records, write_jsonl
from .sceneweaver import execute_trajectory
from .scoring import score_metric
from .trajectory import ToolCall


def execute_candidate_groups(
    candidates_path: str | Path,
    sceneweaver_root: str | Path,
    output_dir: str | Path,
    groups_path: str | Path,
    *,
    socket: bool = False,
) -> int:
    """Execute S2 candidates, score them, and label the best for discriminator SFT."""
    output_dir = Path(output_dir)
    groups = []
    for group_index, item in enumerate(read_json_records(candidates_path)):
        scores = []
        candidates = item["candidates"]
        for candidate_index, candidate in enumerate(candidates):
            calls = [ToolCall.from_dict(call) for call in candidate]
            run_dir = output_dir / f"group_{group_index:05d}" / f"candidate_{candidate_index:03d}"
            result = execute_trajectory(
                calls,
                str(item["instruction"]),
                sceneweaver_root,
                run_dir,
                socket=socket,
                evaluate_final=True,
            )
            if not result["metric"]:
                raise RuntimeError(f"No final metric for {run_dir}")
            cumulative = float(result["steps"][-1]["cumulative_minutes"])
            final_index = len(calls) - 1
            layout_path = run_dir / "record_scene" / f"layout_{final_index}.json"
            object_count = None
            if layout_path.is_file():
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
                object_count = len(layout.get("objects", {}))
            scores.append(
                score_metric(
                    result["metric"], cumulative, object_count_fallback=object_count
                ).composition
            )
        best_index = max(range(len(scores)), key=scores.__getitem__)
        groups.append({**item, "scores": scores, "best_index": best_index})
        write_jsonl(groups_path, groups)
    return len(groups)


def discriminator_records(groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = []
    for item in groups:
        candidates = [
            [ToolCall.from_dict(call) for call in candidate] for candidate in item["candidates"]
        ]
        records.append(
            {
                "prompt": discriminator_prompt(str(item["instruction"]), candidates),
                "response": str(item["best_index"]),
            }
        )
    return records
