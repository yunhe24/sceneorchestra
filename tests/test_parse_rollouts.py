import json
import tempfile
import unittest
from pathlib import Path

from sceneorchestra.parse_rollouts import parse_manifest_record


class ParseRolloutTest(unittest.TestCase):
    def test_normalizes_sceneweaver_files_and_recovers_roomtype(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            (run / "pipeline").mkdir(parents=True)
            (run / "record_scene").mkdir()
            (run / "pipeline" / "trajs_0.json").write_text(
                json.dumps(
                    {
                        "0": {
                            "iter": 0,
                            "action": "init_gpt",
                            "ideas": "base scene",
                            "results": "ok",
                        }
                    }
                )
            )
            metric = {
                "Physics score": {
                    "object number (higher is better)": "Unknown",
                    "object not inside the room (lower is better)": 0,
                    "object has collision (lower is better)": 0,
                },
                "GPT score (0-10, higher is better)": {
                    key: {"grade": 8}
                    for key in ("realism", "functionality", "layout", "completion")
                },
            }
            (run / "pipeline" / "metric_0.json").write_text(json.dumps(metric))
            (run / "record_scene" / "layout_0.json").write_text(
                json.dumps({"objects": {"chair": {}, "desk": {}}})
            )
            log = root / "rollout.log"
            log.write_text(
                "\n".join(
                    [
                        "2026-01-01 10:00:00.000 | INFO | Executing step 0/15 for run",
                        '2026-01-01 10:00:01.000 | INFO | Tool arguments: {"ideas":"base scene","roomtype":"office"}',
                        "2026-01-01 10:00:02.000 | INFO | Activating tool: 'init_gpt'...",
                        "2026-01-01 10:10:00.000 | INFO | completed",
                    ]
                )
            )
            rollout = parse_manifest_record(
                {
                    "instruction": "make an office",
                    "rollout_id": "r0",
                    "run_dir": str(run),
                    "log_file": str(log),
                }
            )
            self.assertEqual(rollout.steps[0].call.arguments["roomtype"], "office")
            self.assertEqual(rollout.steps[0].score["object_count"], 2)
            self.assertAlmostEqual(rollout.steps[0].cumulative_minutes, 10)


if __name__ == "__main__":
    unittest.main()
