"""Thin integration layer for an external, official SceneWeaver checkout.

This module intentionally does not vendor SceneWeaver. Imports are delayed so that
dataset-only commands work in a lightweight Python environment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .records import write_jsonl
from .scoring import score_metric
from .trajectory import ToolCall, format_trajectory


def resolve_sceneweaver_root(value: str | Path | None) -> Path:
    candidate = Path(value or os.getenv("SCENEWEAVER_ROOT", "")).expanduser().resolve()
    required = candidate / "Pipeline" / "app" / "agent" / "scenedesigner.py"
    if not str(value or os.getenv("SCENEWEAVER_ROOT", "")) or not required.is_file():
        raise FileNotFoundError(
            "SceneWeaver checkout not found. Pass --sceneweaver-root or set "
            "SCENEWEAVER_ROOT; expected Pipeline/app/agent/scenedesigner.py."
        )
    return candidate


def _slug(text: str, limit: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", text[:limit]).strip("_")
    return value or "scene"


def load_instructions(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    else:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(values, list):
        raise ValueError("Instruction JSON must be an array or JSONL")
    instructions = []
    for value in values:
        instruction = value if isinstance(value, str) else value.get("instruction", value.get("prompt"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Invalid instruction record: {value!r}")
        instructions.append(instruction.strip())
    return instructions


def generate_sceneweaver_rollouts(
    instructions: Iterable[str],
    sceneweaver_root: str | Path,
    output_dir: str | Path,
    *,
    repeats: int = 2,
    python: str = sys.executable,
    socket: bool = False,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Run the original SceneWeaver execute-review-reflect loop and write a manifest."""
    root = resolve_sceneweaver_root(sceneweaver_root)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest = []
    if manifest_path.is_file():
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_ids = {
        str(item["rollout_id"])
        for item in manifest
        if int(item.get("returncode", 1)) == 0 and item.get("run_dir")
    }
    for instruction_index, instruction in enumerate(instructions):
        for repeat in range(repeats):
            rollout_id = f"i{instruction_index:05d}-r{repeat:03d}"
            if rollout_id in completed_ids:
                continue
            manifest = [item for item in manifest if str(item.get("rollout_id")) != rollout_id]
            job_root = output_dir / rollout_id
            if job_root.exists() and any(job_root.iterdir()):
                raise FileExistsError(
                    f"Incomplete rollout directory already exists: {job_root}. Inspect it, then move or remove it before retrying."
                )
            job_root.mkdir(parents=True, exist_ok=True)
            log_path = job_root / "sceneweaver.log"
            command = [
                python,
                "main.py",
                "--prompt",
                instruction,
                "--cnt",
                "1",
                "--basedir",
                str(job_root),
                "--socket",
                str(socket),
            ]
            existing_logs = set((root / "Pipeline" / "logs").glob("*.log"))
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=root / "Pipeline",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if "Executing step " not in log_path.read_text(encoding="utf-8", errors="replace"):
                new_logs = set((root / "Pipeline" / "logs").glob("*.log")) - existing_logs
                if new_logs:
                    source_log = max(new_logs, key=lambda path: path.stat().st_mtime_ns)
                    shutil.copyfile(source_log, log_path)
            scene_dirs = sorted(path for path in job_root.iterdir() if path.is_dir())
            record = {
                "instruction": instruction,
                "rollout_id": rollout_id,
                "run_dir": str(scene_dirs[0]) if len(scene_dirs) == 1 else None,
                "log_file": str(log_path),
                "wall_minutes": (time.time() - started) / 60.0,
                "returncode": result.returncode,
            }
            manifest.append(record)
            write_jsonl(manifest_path, manifest)
            if (result.returncode != 0 or record["run_dir"] is None) and not continue_on_error:
                raise RuntimeError(f"SceneWeaver rollout failed: {record}")
    return manifest


def _import_sceneweaver(root: Path):
    pipeline = root / "Pipeline"
    for path in (root, pipeline):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from app.agent.scenedesigner import SceneDesigner  # type: ignore
    from app.evaluation import eval_scene  # type: ignore
    from app.schema import Function  # type: ignore
    from app.schema import ToolCall as SWToolCall

    return SceneDesigner, eval_scene, Function, SWToolCall


def _prepare_run(root: Path, save_dir: Path, instruction: str, socket: bool) -> None:
    for name in ("pipeline", "args", "record_files", "record_scene"):
        (save_dir / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "save_dir": str(save_dir.resolve()),
            "UserDemand": instruction,
            "sceneweaver_dir": str(root),
            "socket": str(socket),
        }
    )


def _execute_call(agent: Any, call: ToolCall, index: int, function_cls: Any, call_cls: Any) -> Any:
    agent.current_step = index
    os.environ["iter"] = str(index)
    if index == 0 and "roomtype" in call.arguments:
        os.environ["roomtype"] = str(call.arguments["roomtype"])
    if index == 0:
        agent.available_tools = agent.available_tools0
    elif call.name == "terminate":
        agent.available_tools = agent.available_tools2
    else:
        agent.available_tools = agent.available_tools1
    agent.tool_calls = [
        call_cls(
            id=f"sceneorchestra_{index}",
            function=function_cls(name=call.name, arguments=json.dumps(call.arguments)),
        )
    ]
    return agent.act()


def _execute_trajectory(
    calls: list[ToolCall],
    instruction: str,
    sceneweaver_root: str | Path,
    output_dir: str | Path,
    *,
    socket: bool = False,
    evaluate_final: bool = False,
) -> dict[str, Any]:
    """Execute a complete predicted trajectory without intermediate review/evaluation."""
    root = resolve_sceneweaver_root(sceneweaver_root)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse a non-empty execution directory: {output_dir}")
    _prepare_run(root, output_dir, instruction, socket)
    SceneDesigner, eval_scene, Function, SWToolCall = _import_sceneweaver(root)
    try:
        import dill
    except ImportError as exc:
        raise RuntimeError("SceneWeaver execution requires dill in its environment") from exc
    agent = SceneDesigner()
    results = []
    started = time.monotonic()
    for index, call in enumerate(calls):
        step_started = time.monotonic()
        result = _execute_call(agent, call, index, Function, SWToolCall)
        with (output_dir / "pipeline" / f"memory_{index}.pkl").open("wb") as handle:
            dill.dump(agent.memory, handle)
        (output_dir / "pipeline" / f"sceneorchestra_tool_{index}.json").write_text(
            json.dumps(call.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results.append(
            {
                "index": index,
                "call": call.to_dict(),
                "result": str(result),
                "step_minutes": (time.monotonic() - step_started) / 60.0,
                "cumulative_minutes": (time.monotonic() - started) / 60.0,
            }
        )
    final_metric = None
    if evaluate_final:
        final_metric = eval_scene(iter=len(calls) - 1, user_demand=instruction)
    summary = {
        "instruction": instruction,
        "trajectory": format_trajectory(calls),
        "steps": results,
        "metric": final_metric,
    }
    (output_dir / "sceneorchestra_execution.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def execute_trajectory(
    calls: list[ToolCall],
    instruction: str,
    sceneweaver_root: str | Path,
    output_dir: str | Path,
    *,
    socket: bool = False,
    evaluate_final: bool = False,
) -> dict[str, Any]:
    """Run SceneWeaver from its Pipeline directory, as required by the upstream project."""
    root = resolve_sceneweaver_root(sceneweaver_root)
    resolved_output = Path(output_dir).resolve()
    previous_cwd = Path.cwd()
    os.chdir(root / "Pipeline")
    try:
        return _execute_trajectory(
            calls,
            instruction,
            root,
            resolved_output,
            socket=socket,
            evaluate_final=evaluate_final,
        )
    finally:
        os.chdir(previous_cwd)


def _execute_alternative_from_state(
    task: dict[str, Any],
    alternative: ToolCall,
    sceneweaver_root: str | Path,
    output_dir: str | Path,
    *,
    socket: bool = False,
) -> dict[str, Any]:
    """Restore the copied state before t, execute an alternative t, and evaluate it."""
    if not task.get("source_dir"):
        raise ValueError("Stepwise DPO task has no source_dir")
    source = Path(task["source_dir"]).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite candidate workspace: {destination}")
    shutil.copytree(source, destination)
    root = resolve_sceneweaver_root(sceneweaver_root)
    _prepare_run(root, destination, str(task["instruction"]), socket)
    SceneDesigner, eval_scene, Function, SWToolCall = _import_sceneweaver(root)
    try:
        import dill
    except ImportError as exc:
        raise RuntimeError("SceneWeaver execution requires dill in its environment") from exc
    index = int(task["step_index"])
    memory_path = destination / "pipeline" / f"memory_{index - 1}.pkl"
    if not memory_path.is_file():
        raise FileNotFoundError(f"State snapshot is required for stepwise DPO: {memory_path}")
    agent = SceneDesigner()
    with memory_path.open("rb") as handle:
        agent.memory = dill.load(handle)
    roomtype_path = destination / "pipeline" / "roomtype.txt"
    if roomtype_path.is_file():
        os.environ["roomtype"] = roomtype_path.read_text(encoding="utf-8").strip()
    started = time.monotonic()
    result = _execute_call(agent, alternative, index, Function, SWToolCall)
    metric = eval_scene(iter=index, user_demand=str(task["instruction"]))
    elapsed = (time.monotonic() - started) / 60.0
    layout_path = destination / "record_scene" / f"layout_{index}.json"
    object_count = None
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        object_count = len(layout.get("objects", {}))
    cumulative = float(task["previous_cumulative_minutes"]) + elapsed
    score = score_metric(metric, cumulative, object_count_fallback=object_count)
    comparison = {
        **task,
        "alternative_call": alternative.to_dict(),
        "alternative_result": str(result),
        "alternative_metric": metric,
        "alternative_composition": score.composition,
        "alternative_cumulative_minutes": cumulative,
        "candidate_dir": str(destination),
    }
    (destination / "sceneorchestra_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return comparison


def execute_alternative_from_state(
    task: dict[str, Any],
    alternative: ToolCall,
    sceneweaver_root: str | Path,
    output_dir: str | Path,
    *,
    socket: bool = False,
) -> dict[str, Any]:
    """Execute a counterfactual from SceneWeaver's required Pipeline working directory."""
    root = resolve_sceneweaver_root(sceneweaver_root)
    resolved_output = Path(output_dir).resolve()
    resolved_task = dict(task)
    if resolved_task.get("source_dir"):
        resolved_task["source_dir"] = str(Path(resolved_task["source_dir"]).resolve())
    previous_cwd = Path.cwd()
    os.chdir(root / "Pipeline")
    try:
        return _execute_alternative_from_state(
            resolved_task,
            alternative,
            root,
            resolved_output,
            socket=socket,
        )
    finally:
        os.chdir(previous_cwd)
