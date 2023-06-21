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

        expected = {
            "class": "ModeOption",
            "data": {"is_enabled": False, "top_k_modes": 5},
        }

        self.assertDictEqual(expected, json.loads(serialized))
