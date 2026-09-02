import unittest

from sceneorchestra.scoring import score_metric


class ScoringTest(unittest.TestCase):
    def test_paper_score(self) -> None:
        metric = {
            "Physics score": {
                "object number (higher is better)": 10,
                "object not inside the room (lower is better)": 1,
                "object has collision (lower is better)": 1,
            },
            "GPT score (0-10, higher is better)": {
                "realism": {"grade": 8},
                "functionality": {"grade": 8},
                "layout": {"grade": 8},
                "completion": {"grade": 8},
            },
        }
        score = score_metric(metric, cumulative_minutes=4)
        self.assertEqual(score.physical, 2)
        self.assertEqual(score.visual, 8)
        self.assertAlmostEqual(score.quality, 8.2)
        self.assertAlmostEqual(score.composition, 8.0)

    def test_object_count_layout_fallback(self) -> None:
        metric = {
            "Physics score": {
                "object number (higher is better)": "Unknown",
                "object not inside the room (lower is better)": 0,
                "object has collision (lower is better)": 0,
            },
            "GPT score (0-10, higher is better)": dict.fromkeys(("realism", "functionality", "layout", "completion"), 5),
        }
        self.assertEqual(score_metric(metric, 0, object_count_fallback=7).object_count, 7)


if __name__ == "__main__":
    unittest.main()
