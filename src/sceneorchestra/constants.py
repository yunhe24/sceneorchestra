"""Paper constants and the SceneWeaver tool interface used by SceneOrchestra."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    """Hyperparameters reported in SceneOrchestra arXiv v2, Section 10."""

    alpha: float = 4.0
    quality_weight: float = 0.1
    time_weight: float = 0.05
    stepwise_sft_threshold: float = 3.0
    trajectory_sft_threshold: float = 7.5
    stepwise_dpo_threshold: float = 3.0
    trajectory_dpo_threshold: float = 3.0


PAPER = PaperConfig()

INITIALIZER_TOOLS = ("init_gpt", "init_metascene", "init_physcene")
IMPLEMENTER_TOOLS = ("add_gpt", "add_acdc", "add_crowd")
MODIFIER_TOOLS = (
    "remove_object",
    "update_layout",
    "update_rotation",
    "update_size",
    "add_relation",
)
TERMINATOR_TOOLS = ("terminate",)
ALL_TOOLS = INITIALIZER_TOOLS + IMPLEMENTER_TOOLS + MODIFIER_TOOLS + TERMINATOR_TOOLS

TOOL_GUIDE = """Available tools and typical usage:
- init_gpt(ideas, roomtype): initialize a scene with a flexible GPT-designed base layout.
- init_metascene(ideas, roomtype): initialize by loading a realistic MetaScene layout.
- init_physcene(ideas, roomtype): initialize with a physics-oriented scene generator.
- add_gpt(ideas): add objects using GPT guidance.
- add_acdc(ideas): add retrieved small objects or tabletop assets.
- add_crowd(ideas): add human actors or a crowd.
- remove_object(ideas): remove specified objects.
- update_layout(ideas): move objects to improve placement, spacing, or collisions.
- update_rotation(ideas): adjust object orientations.
- update_size(ideas): adjust object scales.
- add_relation(ideas): add spatial relations between objects and supports.
- terminate(status=\"success\"): finish when the scene meets the instruction.

Output rules:
1. Output only numbered tool calls, one call per line.
2. Use Python keyword arguments with literal string/number/bool/list/dict values.
3. The first call must be an initializer and the last call must be terminate.
4. Do not use tools outside the list above."""
