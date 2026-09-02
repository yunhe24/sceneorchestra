import unittest

from sceneorchestra.trajectory import ToolCall, format_trajectory, parse_call, parse_trajectory


class TrajectoryTest(unittest.TestCase):
    def test_parse_and_format_round_trip(self) -> None:
        calls = [
            ToolCall("init_gpt", {"ideas": "bedroom", "roomtype": "bedroom"}),
            ToolCall("add_gpt", {"ideas": "add lamps", "enabled": True}),
            ToolCall("terminate", {"status": "success"}),
        ]
        self.assertEqual(parse_trajectory(format_trajectory(calls)), calls)

    def test_parser_rejects_code_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "literal"):
            parse_call("add_gpt(ideas=__import__('os').system('false'))")

    def test_complete_trajectory_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "initializer"):
            parse_trajectory("1. add_gpt(ideas='x')\n2. terminate(status='success')")
        with self.assertRaisesRegex(ValueError, "terminate"):
            parse_trajectory("1. init_gpt(ideas='x')")


if __name__ == "__main__":
    unittest.main()
