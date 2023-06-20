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

        expected_options_attributes = {"is_enabled", "columns"}
        expected_is_enabled = option.is_enabled
        expected_columns = option.columns

        self.assertEqual(
            expected_options_attributes, set(json.loads(serialized)["data"].keys())
        )
        self.assertEqual(
            expected_is_enabled, json.loads(serialized)["data"]["is_enabled"]
        )
        self.assertEqual(expected_columns, json.loads(serialized)["data"]["columns"])
