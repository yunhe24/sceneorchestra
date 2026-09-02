"""SceneOrchestra quality and composition scores from the paper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Number
from typing import Any

from .constants import PAPER, PaperConfig


@dataclass(frozen=True)
class Score:
    object_count: float
    out_of_bounds: float
    collisions: float
    realism: float
    functionality: float
    layout: float
    completeness: float
    cumulative_minutes: float
    physical: float
    visual: float
    quality: float
    composition: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _number(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("grade")
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"Missing or non-numeric metric: {label}={value!r}")


def _pick(mapping: Mapping[str, Any], fragments: tuple[str, ...], label: str) -> float:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key, value in lowered.items():
        if all(fragment in key for fragment in fragments):
            return _number(value, label)
    raise ValueError(f"Metric {label!r} not found in keys: {list(mapping)}")


def score_metric(
    metric: Mapping[str, Any],
    cumulative_minutes: float,
    *,
    object_count_fallback: int | None = None,
    config: PaperConfig = PAPER,
) -> Score:
    """Compute Eqs. (3)-(4); runtime is cumulative minutes, as reported in the paper."""
    physics = metric.get("Physics score", metric.get("physics", {}))
    visual = metric.get("GPT score (0-10, higher is better)", metric.get("visual", {}))
    if not isinstance(physics, Mapping) or not isinstance(visual, Mapping):
        raise ValueError("Metric must contain physical and visual sections")
    try:
        object_count = _pick(physics, ("object", "number"), "object_count")
    except ValueError:
        if object_count_fallback is None:
            raise
        object_count = float(object_count_fallback)
    out_of_bounds = _pick(physics, ("object", "not", "inside"), "out_of_bounds")
    collisions = _pick(physics, ("collision",), "collisions")
    realism = _pick(visual, ("real",), "realism")
    functionality = _pick(visual, ("func",), "functionality")
    layout = _pick(visual, ("layout",), "layout")
    completeness = _pick(visual, ("complet",), "completeness")
    physical_score = object_count - config.alpha * (out_of_bounds + collisions)
    visual_score = (realism + functionality + layout + completeness) / 4.0
    quality = config.quality_weight * physical_score + visual_score
    composition = quality - config.time_weight * float(cumulative_minutes)
    return Score(
        object_count=object_count,
        out_of_bounds=out_of_bounds,
        collisions=collisions,
        realism=realism,
        functionality=functionality,
        layout=layout,
        completeness=completeness,
        cumulative_minutes=float(cumulative_minutes),
        physical=physical_score,
        visual=visual_score,
        quality=quality,
        composition=composition,
    )
