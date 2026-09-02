import unittest

from sceneorchestra.datasets import (
    build_stepwise_dpo,
    build_stepwise_dpo_tasks,
    build_stepwise_sft,
)
from sceneorchestra.records import RolloutRecord, StepRecord
from sceneorchestra.trajectory import ToolCall


def _step(index: int, name: str, composition: float) -> StepRecord:
    return StepRecord(
        index=index,
        call=ToolCall(name, {"ideas": str(index)}),
        cumulative_minutes=float(index),
        metric={},
        score={"composition": composition},
    )


class DatasetTest(unittest.TestCase):
    def test_stepwise_thresholds_are_strict(self) -> None:
        rollout = RolloutRecord(
            instruction="make a room",
            rollout_id="r0",
            source_dir="/tmp/private-rollout",
            steps=[
                _step(0, "init_gpt", 1),
                _step(1, "add_gpt", 4),
                _step(2, "add_crowd", 7.1),
                _step(3, "update_layout", 2),
            ],
        )
        sft = build_stepwise_sft([rollout])
        self.assertEqual(len(sft), 1)
        self.assertTrue(sft[0]["response"].startswith("3. add_crowd"))
        tasks = build_stepwise_dpo_tasks([rollout])
        self.assertEqual([task["step_index"] for task in tasks], [2, 3])

    def test_stepwise_dpo_prefers_executed_higher_score(self) -> None:
        comparison = {
            "prompt": "p",
            "original_call": {"name": "add_gpt", "arguments": {"ideas": "a"}},
            "alternative_call": {"name": "update_layout", "arguments": {"ideas": "b"}},
            "original_composition": 2,
            "alternative_composition": 6,
        }
        records = build_stepwise_dpo([comparison])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["chosen"].endswith('update_layout(ideas="b")'))


if __name__ == "__main__":
    unittest.main()
