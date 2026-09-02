"""LLaMAFactory-backed generation helpers."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

from .prompts import discriminator_prompt, orchestrator_prompt
from .records import read_json_records, write_jsonl
from .trajectory import ToolCall, parse_call, parse_trajectory


class Generator:
    """Load one LLaMAFactory ChatModel and reuse it across requests."""

    def __init__(self, config_path: str | Path):
        import yaml

        from llamafactory.chat import ChatModel

        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Inference YAML must contain a mapping")
        self.model = ChatModel(args=config)

    def generate(self, prompt: str) -> str:
        chunks = self.model.chat([{"role": "user", "content": prompt}])
        if not chunks:
            raise RuntimeError("Model returned no response")
        return chunks[0].response_text.strip()

    def close(self) -> None:
        loop = getattr(self.model, "_loop", None)
        thread = getattr(self.model, "_thread", None)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=2.0)
        gc.collect()

    def __enter__(self) -> Generator:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def generate_valid_trajectory(model: Generator, prompt: str, attempts: int = 3) -> list[ToolCall]:
    errors = []
    for _ in range(attempts):
        response = model.generate(prompt)
        try:
            return parse_trajectory(response)
        except ValueError as exc:
            errors.append(f"{exc}: {response}")
    raise RuntimeError("Trajectory validation failed after retries:\n" + "\n".join(errors))


def generate_candidate_groups(
    instructions: list[str],
    config_path: str | Path,
    output_path: str | Path,
    *,
    candidates: int = 4,
    validation_attempts: int = 3,
) -> int:
    records = []
    with Generator(config_path) as model:
        for instruction in instructions:
            prompt = orchestrator_prompt(instruction)
            calls = [
                generate_valid_trajectory(model, prompt, validation_attempts) for _ in range(candidates)
            ]
            records.append(
                {
                    "instruction": instruction,
                    "candidates": [[call.to_dict() for call in trajectory] for trajectory in calls],
                }
            )
    return write_jsonl(output_path, records)


def generate_stepwise_alternatives(
    tasks_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    validation_attempts: int = 3,
) -> int:
    records = []
    with Generator(config_path) as model:
        for task in read_json_records(tasks_path):
            errors = []
            alternative = None
            for _ in range(validation_attempts):
                response = model.generate(str(task["prompt"]))
                try:
                    alternative = parse_call(response.splitlines()[0])
                    break
                except ValueError as exc:
                    errors.append(str(exc))
            if alternative is None:
                raise RuntimeError(f"Could not sample alternative for {task['task_id']}: {errors}")
            records.append({**task, "alternative_call": alternative.to_dict()})
    return write_jsonl(output_path, records)


def rank_candidate_groups(
    candidates_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    validation_attempts: int = 3,
) -> int:
    records = []
    with Generator(config_path) as model:
        for item in read_json_records(candidates_path):
            candidates = [
                [ToolCall.from_dict(call) for call in trajectory] for trajectory in item["candidates"]
            ]
            prompt = discriminator_prompt(str(item["instruction"]), candidates)
            best_index = None
            raw_response = ""
            for _ in range(validation_attempts):
                raw_response = model.generate(prompt)
                match = re.fullmatch(r"\s*(\d+)\s*", raw_response)
                if match and int(match.group(1)) < len(candidates):
                    best_index = int(match.group(1))
                    break
            if best_index is None:
                raise RuntimeError(f"Invalid discriminator response: {raw_response!r}")
            records.append({**item, "best_index": best_index})
    return write_jsonl(output_path, records)
