"""GPT-assisted instruction generation and semantics-preserving rephrasing.

The generation prompt and batching behavior are retained from the development
code. Outputs are review candidates: the paper reports manual filtering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .records import read_json_records, write_jsonl
from .sceneweaver import load_instructions


SYSTEM_PROMPT = """You are an expert at writing user instructions for an indoor 3D scene synthesis system.
The system can create rooms with furniture, decorations, and small daily objects.

Generate exactly {n} diverse user instructions. Vary room type, complexity, object counts, layout focus, activities, spatial relations, and opening imperative verb. Every scene must be a real enclosed room. Layouts must be physically plausible and must not block access or contain contradictory relations.

Do not mention style, theme, or aesthetic descriptors (for example modern, minimalist, cozy, rustic, vintage, luxury, industrial, Scandinavian, boho, or colorful). Describe only room type, objects, counts, placement, relations, and functionality.

Return a JSON array of strings and nothing else."""

USER_PROMPT = """Generate {n} new indoor scene synthesis instructions. Use these only as format references and do not repeat them:

- Create a bedroom with abundant furniture, wall decorations, and small daily objects.
- First create a garage with a car in the center. Then add a work bench and shelf with related tools.
- Create a laundromat with 10 machines. Add washing supplies on each machine and baskets elsewhere in the room.

Use diverse opening verbs, no style/theme/aesthetic descriptors, and only reasonable feasible layouts. Return only a JSON array of strings."""


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install instruction-generation support with: pip install -e '.[data]'") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment")
    return OpenAI()


def _json_array(response: str) -> list[str]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.removesuffix("```").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON array: {response!r}")
    values = json.loads(text[start : end + 1])
    if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("Expected a non-empty JSON string array")
    return [value.strip() for value in values]


def _complete(client: Any, model: str, system: str, messages: list[dict[str, str]]) -> list[str]:
    from openai import BadRequestError

    request: dict[str, Any] = {
        "model": model,
        "temperature": 1.0,
        "messages": [{"role": "system", "content": system}, *messages],
    }
    try:
        completion = client.chat.completions.create(
            **request, extra_body={"max_completion_tokens": 4096}
        )
    except BadRequestError as exc:
        if "max_completion_tokens" not in str(exc):
            raise
        completion = client.chat.completions.create(**request, max_tokens=4096)
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Instruction model returned empty content")
    return _json_array(content)


def generate_instruction_candidates(
    output_path: str | Path,
    *,
    model: str,
    count: int,
    existing_path: str | Path | None = None,
    max_attempts_per_batch: int = 5,
) -> int:
    """Generate the paper's diverse detailed instruction candidates in batches."""
    existing = load_instructions(existing_path) if existing_path else []
    client = _client()
    generated: list[str] = []
    batch_size = min(count, 30)
    while len(generated) < count:
        requested = min(count - len(generated), batch_size)
        avoid = existing + generated
        messages = [{"role": "user", "content": USER_PROMPT.format(n=requested)}]
        if avoid:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Do not generate a semantically duplicate scene. Reusing common room types is encouraged, "
                        "but the objects, quantities, arrangement, and functional focus must differ from these existing "
                        "instructions:\n" + json.dumps(avoid, indent=2, ensure_ascii=False)
                    ),
                }
            )
        error: Exception | None = None
        for _ in range(max_attempts_per_batch):
            try:
                generated.extend(
                    _complete(client, model, SYSTEM_PROMPT.format(n=requested), messages)[:requested]
                )
                error = None
                break
            except (ValueError, json.JSONDecodeError) as exc:
                error = exc
        if error is not None:
            raise RuntimeError(f"Failed to generate a valid instruction batch: {error}")
        records = [
            {
                "instruction": instruction,
                "approved": False,
                "generator_model": model,
            }
            for instruction in generated[:count]
        ]
        write_jsonl(output_path, records)
    return min(len(generated), count)


def generate_rephrase_candidates(
    instructions_path: str | Path,
    output_path: str | Path,
    *,
    model: str,
    variants: int,
) -> int:
    client = _client()
    records = []
    for instruction in load_instructions(instructions_path):
        prompt = f"""Rephrase the following 3D indoor-scene instruction in {variants} distinct ways.
Preserve every object count, requested object, position, spatial relation, and constraint exactly. Do not add or remove requirements.
Instruction: {instruction!r}
Return exactly {variants} strings as a JSON array."""
        records.append(
            {
                "instruction": instruction,
                "rephrases": _complete(
                    client,
                    model,
                    "Return only a valid JSON array of strings. Do not use Markdown.",
                    [{"role": "user", "content": prompt}],
                )[:variants],
                "approved": False,
                "generator_model": model,
            }
        )
        write_jsonl(output_path, records)
    return len(records)


def load_approved_rephrases(path: str | Path) -> dict[str, list[str]]:
    mapping = {}
    for record in read_json_records(path):
        if not record.get("approved", False):
            continue
        mapping[str(record["instruction"])] = [str(value) for value in record["rephrases"]]
    return mapping
