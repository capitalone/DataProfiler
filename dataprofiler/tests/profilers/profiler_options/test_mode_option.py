import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import ModeOption
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBooleanOption,
)


class TestModeOptions(TestBooleanOption):
    def test_json_encode(self):
        option = ModeOption(is_enabled=False, max_k_modes=5)

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected_class = "ModeOption"
        expected_options_attributes = {"is_enabled", "top_k_modes"}
        expected_is_enabled = option.is_enabled
        expected_top_k_modes = option.top_k_modes

        actual_option_json = json.loads(serialized)

        self.assertIn("class", actual_option_json)
        self.assertEqual(expected_class, actual_option_json["class"])
        self.assertIn("data", actual_option_json)
        self.assertEqual(
            expected_options_attributes, set(actual_option_json["data"].keys())
        )
        self.assertIn("is_enabled", actual_option_json["data"])
        self.assertEqual(expected_is_enabled, actual_option_json["data"]["is_enabled"])
        self.assertIn("top_k_modes", actual_option_json["data"])
        self.assertEqual(
            expected_top_k_modes, actual_option_json["data"]["top_k_modes"]
        )
