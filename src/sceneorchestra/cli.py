"""Command-line interface for the complete SceneOrchestra workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .datasets import (
    augment_rollouts,
    build_discriminator_sft,
    build_interleaved_dpo,
    build_stepwise_dpo,
    build_stepwise_dpo_tasks,
    build_stepwise_sft,
    build_trajectory_dpo,
    build_trajectory_sft,
    sample_discriminator_groups,
    write_llamafactory_bundle,
)
from .instructions import (
    generate_instruction_candidates,
    generate_rephrase_candidates,
    load_approved_rephrases,
)
from .interleaved import execute_candidate_groups
from .models import (
    Generator,
    generate_candidate_groups,
    generate_stepwise_alternatives,
    generate_valid_trajectory,
    rank_candidate_groups,
)
from .parse_rollouts import parse_manifest
from .prompts import orchestrator_prompt
from .records import read_json_records, read_rollouts, write_jsonl
from .sceneweaver import (
    execute_alternative_from_state,
    execute_trajectory,
    generate_sceneweaver_rollouts,
    load_instructions,
)
from .trajectory import ToolCall, format_trajectory, parse_trajectory


def _print_count(name: str, count: int) -> None:
    print(f"{name}: {count}")


def _build_independent(args: argparse.Namespace) -> None:
    original_rollouts = read_rollouts(args.rollouts)
    rollouts = original_rollouts
    if args.rephrases:
        rollouts = augment_rollouts(rollouts, load_approved_rephrases(args.rephrases))
    datasets = {
        "sceneorchestra_stepwise_sft": build_stepwise_sft(rollouts),
        "sceneorchestra_trajectory_sft": build_trajectory_sft(rollouts, seed=args.seed),
        "sceneorchestra_trajectory_dpo": build_trajectory_dpo(
            rollouts, seed=args.seed, pairs_per_instruction=args.pairs_per_instruction
        ),
    }
    groups = sample_discriminator_groups(
        rollouts, seed=args.seed, candidates=args.discriminator_candidates
    )
    datasets["sceneorchestra_discriminator"] = build_discriminator_sft(groups)
    write_llamafactory_bundle(args.output_dir, datasets)
    # Counterfactual tools are executed once against the original instruction/state.
    # Rephrases are added only after scoring, as training augmentation.
    tasks = build_stepwise_dpo_tasks(original_rollouts)
    write_jsonl(Path(args.output_dir) / "stepwise_dpo_tasks.jsonl", tasks)
    for name, records in datasets.items():
        _print_count(name, len(records))
    _print_count("stepwise_dpo_tasks", len(tasks))


def _execute_alternatives(args: argparse.Namespace) -> None:
    comparisons = []
    for task in read_json_records(args.alternatives):
        alternative = ToolCall.from_dict(task["alternative_call"])
        destination = Path(args.output_dir) / str(task["task_id"]).replace(":", "_")
        comparisons.append(
            execute_alternative_from_state(
                task, alternative, args.sceneweaver_root, destination, socket=args.socket
            )
        )
        write_jsonl(args.comparisons, comparisons)
    _print_count("executed comparisons", len(comparisons))


def _finalize_stepwise_dpo(args: argparse.Namespace) -> None:
    comparisons = list(read_json_records(args.comparisons))
    if args.rephrases:
        rephrases = load_approved_rephrases(args.rephrases)
        augmented = []
        for comparison in comparisons:
            augmented.append(comparison)
            history = [ToolCall.from_dict(call) for call in comparison["history"]]
            for instruction in rephrases.get(str(comparison["instruction"]), []):
                augmented.append(
                    {
                        **comparison,
                        "instruction": instruction,
                        "prompt": orchestrator_prompt(instruction, history),
                    }
                )
        comparisons = augmented
    records = build_stepwise_dpo(comparisons)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "sceneorchestra_stepwise_dpo.jsonl"
    write_jsonl(output_dir / filename, records)
    registry_path = output_dir / "dataset_info.json"
    registry: dict[str, Any] = {}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sceneorchestra_stepwise_dpo"] = {
        "file_name": filename,
        "ranking": True,
        "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    }
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    _print_count("sceneorchestra_stepwise_dpo", len(records))


def _build_discriminator(args: argparse.Namespace) -> None:
    rollouts = read_rollouts(args.rollouts)
    groups = sample_discriminator_groups(
        rollouts, seed=args.seed, candidates=args.candidates
    )
    records = build_discriminator_sft(groups)
    write_llamafactory_bundle(args.output_dir, {args.name: records})
    _print_count(args.name, len(records))


def _build_discriminator_from_groups(args: argparse.Namespace) -> None:
    records = build_discriminator_sft(read_json_records(args.groups))
    write_llamafactory_bundle(args.output_dir, {args.name: records})
    _print_count(args.name, len(records))


def _infer(args: argparse.Namespace) -> None:
    prompt = orchestrator_prompt(args.instruction)
    with Generator(args.config) as model:
        calls = generate_valid_trajectory(model, prompt, args.validation_attempts)
    print(format_trajectory(calls))
    execute_trajectory(
        calls,
        args.instruction,
        args.sceneweaver_root,
        args.output_dir,
        socket=args.socket,
        evaluate_final=args.evaluate_final,
    )


def _generate_trajectory(args: argparse.Namespace) -> None:
    prompt = orchestrator_prompt(args.instruction)
    with Generator(args.config) as model:
        calls = generate_valid_trajectory(model, prompt, args.validation_attempts)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(format_trajectory(calls) + "\n", encoding="utf-8")


def _execute_trajectory_file(args: argparse.Namespace) -> None:
    calls = parse_trajectory(Path(args.trajectory).read_text(encoding="utf-8"))
    execute_trajectory(
        calls,
        args.instruction,
        args.sceneweaver_root,
        args.output_dir,
        socket=args.socket,
        evaluate_final=args.evaluate_final,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sceneorchestra")
    commands = parser.add_subparsers(dest="command", required=True)

    rollout = commands.add_parser("generate-rollouts", help="Run original SceneWeaver rollouts for S1")
    rollout.add_argument("--instructions", required=True)
    rollout.add_argument("--sceneweaver-root", required=True)
    rollout.add_argument("--output-dir", required=True)
    rollout.add_argument("--repeats", type=int, default=2)
    rollout.add_argument("--python", default="python")
    rollout.add_argument("--socket", action="store_true")
    rollout.add_argument("--continue-on-error", action="store_true")

    normalize = commands.add_parser("parse-rollouts", help="Normalize and score SceneWeaver outputs")
    normalize.add_argument("--manifest", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--skip-failed", action="store_true")

    independent = commands.add_parser("build-independent", help="Build phase-one datasets and DPO tasks")
    independent.add_argument("--rollouts", required=True)
    independent.add_argument("--output-dir", required=True)
    independent.add_argument("--seed", type=int, default=0)
    independent.add_argument("--pairs-per-instruction", type=int, default=1)
    independent.add_argument("--discriminator-candidates", type=int, default=4)
    independent.add_argument("--rephrases", help="Manually approved instruction-rephrase JSON/JSONL")

    instructions = commands.add_parser("generate-instructions")
    instructions.add_argument("--num", required=True, type=int)
    instructions.add_argument("--model", required=True)
    instructions.add_argument("--existing", help="Existing JSON/JSONL/text instructions to avoid")
    instructions.add_argument("--output", required=True)

    rephrase = commands.add_parser("rephrase-instructions")
    rephrase.add_argument("--instructions", required=True)
    rephrase.add_argument("--model", required=True)
    rephrase.add_argument("--variants", type=int, default=3)
    rephrase.add_argument("--output", required=True)

    alternatives = commands.add_parser("sample-stepwise-alternatives")
    alternatives.add_argument("--tasks", required=True)
    alternatives.add_argument("--config", required=True)
    alternatives.add_argument("--output", required=True)
    alternatives.add_argument("--validation-attempts", type=int, default=3)

    execute_alt = commands.add_parser("execute-stepwise-alternatives")
    execute_alt.add_argument("--alternatives", required=True)
    execute_alt.add_argument("--sceneweaver-root", required=True)
    execute_alt.add_argument("--output-dir", required=True)
    execute_alt.add_argument("--comparisons", required=True)
    execute_alt.add_argument("--socket", action="store_true")

    finalize = commands.add_parser("finalize-stepwise-dpo")
    finalize.add_argument("--comparisons", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--rephrases", help="Manually approved instruction-rephrase JSON/JSONL")

    discriminator = commands.add_parser("build-discriminator")
    discriminator.add_argument("--rollouts", required=True)
    discriminator.add_argument("--output-dir", required=True)
    discriminator.add_argument("--name", default="sceneorchestra_discriminator_interleaved")
    discriminator.add_argument("--seed", type=int, default=0)
    discriminator.add_argument("--candidates", type=int, default=4)

    discriminator_groups = commands.add_parser("build-discriminator-from-groups")
    discriminator_groups.add_argument("--groups", required=True)
    discriminator_groups.add_argument("--output-dir", required=True)
    discriminator_groups.add_argument("--name", default="sceneorchestra_discriminator_interleaved")

    candidates = commands.add_parser("generate-candidates")
    candidates.add_argument("--instructions", required=True)
    candidates.add_argument("--config", required=True)
    candidates.add_argument("--output", required=True)
    candidates.add_argument("--candidates", type=int, default=4)
    candidates.add_argument("--validation-attempts", type=int, default=3)

    execute_candidates = commands.add_parser("execute-candidates")
    execute_candidates.add_argument("--candidates", required=True)
    execute_candidates.add_argument("--sceneweaver-root", required=True)
    execute_candidates.add_argument("--output-dir", required=True)
    execute_candidates.add_argument("--groups", required=True)
    execute_candidates.add_argument("--socket", action="store_true")

    rank = commands.add_parser("rank-candidates")
    rank.add_argument("--candidates", required=True)
    rank.add_argument("--config", required=True)
    rank.add_argument("--output", required=True)
    rank.add_argument("--validation-attempts", type=int, default=3)

    interleaved_dpo = commands.add_parser("build-interleaved-dpo")
    interleaved_dpo.add_argument("--rankings", required=True)
    interleaved_dpo.add_argument("--output-dir", required=True)

    generate = commands.add_parser("generate-trajectory", help="Generate and validate one full trajectory")
    generate.add_argument("--instruction", required=True)
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--validation-attempts", type=int, default=3)

    execute = commands.add_parser("execute-trajectory", help="Execute a trajectory with external SceneWeaver")
    execute.add_argument("--trajectory", required=True)
    execute.add_argument("--instruction", required=True)
    execute.add_argument("--sceneweaver-root", required=True)
    execute.add_argument("--output-dir", required=True)
    execute.add_argument("--socket", action="store_true")
    execute.add_argument("--evaluate-final", action="store_true")

    infer = commands.add_parser("infer", help="Generate once, validate, and execute end-to-end")
    infer.add_argument("--instruction", required=True)
    infer.add_argument("--config", required=True)
    infer.add_argument("--sceneweaver-root", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--validation-attempts", type=int, default=3)
    infer.add_argument("--socket", action="store_true")
    infer.add_argument("--evaluate-final", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate-rollouts":
        records = generate_sceneweaver_rollouts(
            load_instructions(args.instructions),
            args.sceneweaver_root,
            args.output_dir,
            repeats=args.repeats,
            python=args.python,
            socket=args.socket,
            continue_on_error=args.continue_on_error,
        )
        _print_count("rollouts", len(records))
    elif args.command == "generate-instructions":
        _print_count(
            "instruction candidates",
            generate_instruction_candidates(
                args.output,
                model=args.model,
                count=args.num,
                existing_path=args.existing,
            ),
        )
    elif args.command == "rephrase-instructions":
        _print_count(
            "rephrase groups",
            generate_rephrase_candidates(
                args.instructions,
                args.output,
                model=args.model,
                variants=args.variants,
            ),
        )
    elif args.command == "parse-rollouts":
        _print_count("parsed rollouts", parse_manifest(args.manifest, args.output, skip_failed=args.skip_failed))
    elif args.command == "build-independent":
        _build_independent(args)
    elif args.command == "sample-stepwise-alternatives":
        _print_count(
            "alternatives",
            generate_stepwise_alternatives(
                args.tasks, args.config, args.output, validation_attempts=args.validation_attempts
            ),
        )
    elif args.command == "execute-stepwise-alternatives":
        _execute_alternatives(args)
    elif args.command == "finalize-stepwise-dpo":
        _finalize_stepwise_dpo(args)
    elif args.command == "build-discriminator":
        _build_discriminator(args)
    elif args.command == "build-discriminator-from-groups":
        _build_discriminator_from_groups(args)
    elif args.command == "generate-candidates":
        _print_count(
            "candidate groups",
            generate_candidate_groups(
                load_instructions(args.instructions),
                args.config,
                args.output,
                candidates=args.candidates,
                validation_attempts=args.validation_attempts,
            ),
        )
    elif args.command == "execute-candidates":
        _print_count(
            "executed candidate groups",
            execute_candidate_groups(
                args.candidates,
                args.sceneweaver_root,
                args.output_dir,
                args.groups,
                socket=args.socket,
            ),
        )
    elif args.command == "rank-candidates":
        _print_count(
            "ranked groups",
            rank_candidate_groups(
                args.candidates,
                args.config,
                args.output,
                validation_attempts=args.validation_attempts,
            ),
        )
    elif args.command == "build-interleaved-dpo":
        records = build_interleaved_dpo(read_json_records(args.rankings))
        write_llamafactory_bundle(args.output_dir, {"sceneorchestra_interleaved_dpo": records})
        _print_count("sceneorchestra_interleaved_dpo", len(records))
    elif args.command == "generate-trajectory":
        _generate_trajectory(args)
    elif args.command == "execute-trajectory":
        _execute_trajectory_file(args)
    elif args.command == "infer":
        _infer(args)


if __name__ == "__main__":
    main()
