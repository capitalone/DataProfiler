import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import ModeOption
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBooleanOption,
)


class TestCategoricalOptions(TestBooleanOption):
    def test_json_encode_after_update(self):
        option = ModeOption(is_enabled=False, max_k_modes=5)

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected_options_attributes = {"is_enabled", "top_k_modes"}
        expected_is_enabled = option.is_enabled
        expected_top_k_modes = option.top_k_modes

        self.assertEqual(
            expected_options_attributes, set(json.loads(serialized)["data"].keys())
        )
        self.assertEqual(
            expected_is_enabled, json.loads(serialized)["data"]["is_enabled"]
        )
        self.assertEqual(
            expected_top_k_modes, json.loads(serialized)["data"]["top_k_modes"]
        )
