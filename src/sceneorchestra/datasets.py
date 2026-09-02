"""Paper-aligned construction of LLaMAFactory SFT and DPO datasets."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import PAPER, PaperConfig
from .prompts import discriminator_prompt, orchestrator_prompt
from .records import RolloutRecord, write_jsonl
from .trajectory import ToolCall, format_call, format_trajectory


def _composition(step: Any) -> float:
    return float(step.score["composition"])


def augment_rollouts(
    rollouts: Iterable[RolloutRecord], rephrases: dict[str, list[str]]
) -> list[RolloutRecord]:
    """Duplicate trajectories under manually approved, semantics-preserving instructions."""
    augmented = []
    for rollout in rollouts:
        augmented.append(rollout)
        for variant_index, instruction in enumerate(rephrases.get(rollout.instruction, [])):
            copied = deepcopy(rollout)
            copied.instruction = instruction
            copied.rollout_id = f"{rollout.rollout_id}-rephrase-{variant_index}"
            copied.metadata = {**copied.metadata, "rephrased_from": rollout.instruction}
            augmented.append(copied)
    return augmented


def _prefix(rollout: RolloutRecord, stop: int, *, terminate: bool) -> list[ToolCall]:
    calls = [step.call for step in rollout.steps[: stop + 1] if step.call.name != "terminate"]
    if terminate:
        calls.append(ToolCall("terminate", {"status": "success"}))
    return calls


def build_stepwise_sft(
    rollouts: Iterable[RolloutRecord], *, config: PaperConfig = PAPER
) -> list[dict[str, str]]:
    records = []
    for rollout in rollouts:
        for index in range(1, len(rollout.steps)):
            if _composition(rollout.steps[index]) - _composition(rollout.steps[index - 1]) <= config.stepwise_sft_threshold:
                continue
            history = [step.call for step in rollout.steps[:index]]
            records.append(
                {
                    "prompt": orchestrator_prompt(rollout.instruction, history),
                    "response": f"{index + 1}. {format_call(rollout.steps[index].call)}",
                }
            )
    return records


def build_trajectory_sft(
    rollouts: Iterable[RolloutRecord], *, seed: int = 0, config: PaperConfig = PAPER
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records = []
    for rollout in rollouts:
        candidates = [index for index, step in enumerate(rollout.steps) if step.call.name != "terminate"]
        if not candidates:
            continue
        index = rng.choice(candidates)
        if _composition(rollout.steps[index]) <= config.trajectory_sft_threshold:
            continue
        records.append(
            {
                "prompt": orchestrator_prompt(rollout.instruction),
                "response": format_trajectory(_prefix(rollout, index, terminate=True)),
            }
        )
    return records


def build_stepwise_dpo_tasks(
    rollouts: Iterable[RolloutRecord], *, config: PaperConfig = PAPER
) -> list[dict[str, Any]]:
    """Select histories where |C_t-C_(t-1)| > tau_1 for alternative sampling."""
    records = []
    for rollout in rollouts:
        for index in range(1, len(rollout.steps)):
            delta = _composition(rollout.steps[index]) - _composition(rollout.steps[index - 1])
            if abs(delta) <= config.stepwise_sft_threshold:
                continue
            records.append(
                {
                    "task_id": f"{rollout.rollout_id}:{index}",
                    "rollout_id": rollout.rollout_id,
                    "source_dir": rollout.source_dir,
                    "step_index": index,
                    "next_number": index + 1,
                    "instruction": rollout.instruction,
                    "history": [step.call.to_dict() for step in rollout.steps[:index]],
                    "prompt": orchestrator_prompt(
                        rollout.instruction, [step.call for step in rollout.steps[:index]]
                    ),
                    "original_call": rollout.steps[index].call.to_dict(),
                    "original_composition": _composition(rollout.steps[index]),
                    "previous_cumulative_minutes": rollout.steps[index - 1].cumulative_minutes,
                }
            )
    return records


def build_stepwise_dpo(
    comparisons: Iterable[dict[str, Any]], *, config: PaperConfig = PAPER
) -> list[dict[str, str]]:
    """Convert executed original/alternative comparisons to DPO triplets."""
    records = []
    for item in comparisons:
        original_score = float(item["original_composition"])
        alternative_score = float(item["alternative_composition"])
        if abs(alternative_score - original_score) <= config.stepwise_dpo_threshold:
            continue
        original = ToolCall.from_dict(item["original_call"])
        alternative = ToolCall.from_dict(item["alternative_call"])
        chosen, rejected = (
            (alternative, original) if alternative_score > original_score else (original, alternative)
        )
        records.append(
            {
                "prompt": str(item["prompt"]),
                "chosen": f"{int(item.get('next_number', 1))}. {format_call(chosen)}",
                "rejected": f"{int(item.get('next_number', 1))}. {format_call(rejected)}",
            }
        )
    return records


def build_trajectory_dpo(
    rollouts: Iterable[RolloutRecord],
    *,
    seed: int = 0,
    pairs_per_instruction: int = 1,
    config: PaperConfig = PAPER,
) -> list[dict[str, str]]:
    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout.instruction].append(rollout)
    rng = random.Random(seed)
    records = []
    for instruction, group in grouped.items():
        if len(group) < 2:
            continue
        for _ in range(pairs_per_instruction):
            first, second = rng.sample(group, 2)
            first_indices = [i for i, step in enumerate(first.steps) if step.call.name != "terminate"]
            second_indices = [i for i, step in enumerate(second.steps) if step.call.name != "terminate"]
            if not first_indices or not second_indices:
                continue
            first_index, second_index = rng.choice(first_indices), rng.choice(second_indices)
            first_score, second_score = _composition(first.steps[first_index]), _composition(second.steps[second_index])
            if abs(first_score - second_score) <= config.trajectory_dpo_threshold:
                continue
            first_text = format_trajectory(_prefix(first, first_index, terminate=True))
            second_text = format_trajectory(_prefix(second, second_index, terminate=True))
            chosen, rejected = (
                (first_text, second_text) if first_score > second_score else (second_text, first_text)
            )
            records.append(
                {
                    "prompt": orchestrator_prompt(instruction),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
    return records


def sample_discriminator_groups(
    rollouts: Iterable[RolloutRecord], *, seed: int = 0, candidates: int = 4
) -> list[dict[str, Any]]:
    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout.instruction].append(rollout)
    rng = random.Random(seed)
    records = []
    for instruction, group in grouped.items():
        if len(group) < candidates:
            continue
        selected = rng.sample(group, candidates)
        trajectories: list[list[ToolCall]] = []
        scores: list[float] = []
        for rollout in selected:
            indices = [i for i, step in enumerate(rollout.steps) if step.call.name != "terminate"]
            if not indices:
                break
            index = rng.choice(indices)
            trajectories.append(_prefix(rollout, index, terminate=True))
            scores.append(_composition(rollout.steps[index]))
        if len(trajectories) != candidates:
            continue
        records.append(
            {
                "instruction": instruction,
                "candidates": [[call.to_dict() for call in calls] for calls in trajectories],
                "scores": scores,
                "best_index": max(range(candidates), key=scores.__getitem__),
            }
        )
    return records


def build_discriminator_sft(groups: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    records = []
    for group in groups:
        candidates = [[ToolCall.from_dict(call) for call in calls] for calls in group["candidates"]]
        records.append(
            {
                "prompt": discriminator_prompt(str(group["instruction"]), candidates),
                "response": str(group["best_index"]),
            }
        )
    return records


def build_interleaved_dpo(rankings: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Use discriminator rankings over unexecuted S3 candidates for trajectory DPO."""
    records = []
    for item in rankings:
        candidates = item["candidates"]
        best = int(item["best_index"])
        if best < 0 or best >= len(candidates):
            raise ValueError(f"Invalid best_index={best}")
        chosen = format_trajectory([ToolCall.from_dict(call) for call in candidates[best]])
        for index, candidate in enumerate(candidates):
            if index == best:
                continue
            records.append(
                {
                    "prompt": orchestrator_prompt(str(item["instruction"])),
                    "chosen": chosen,
                    "rejected": format_trajectory([ToolCall.from_dict(call) for call in candidate]),
                }
            )
    return records


def write_llamafactory_bundle(output_dir: str | Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {}
    for name, records in datasets.items():
        filename = f"{name}.jsonl"
        write_jsonl(output_dir / filename, records)
        is_dpo = bool(records and "chosen" in records[0]) or name.endswith("dpo")
        columns = {"prompt": "prompt"}
        if is_dpo:
            columns.update({"chosen": "chosen", "rejected": "rejected"})
        else:
            columns["response"] = "response"
        registry[name] = {"file_name": filename, "columns": columns}
        if is_dpo:
            registry[name]["ranking"] = True
    (output_dir / "dataset_info.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
