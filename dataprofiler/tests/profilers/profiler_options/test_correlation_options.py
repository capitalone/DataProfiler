import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import CorrelationOptions
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options import (
    TestBaseInspectorOptions,
)


class TestCorrelationOptions(TestBaseInspectorOptions):
    def test_json_encode(self):
        option = CorrelationOptions(
            is_enabled=False, columns=["name", "age", "location"]
        )

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected = {
            "class": "CorrelationOptions",
            "data": {"is_enabled": False, "columns": ["name", "age", "location"]},
        }

        self.assertDictEqual(expected, json.loads(serialized))
