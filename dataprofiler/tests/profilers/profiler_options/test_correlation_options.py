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

        expected_class = "CorrelationOptions"
        expected_options_attributes = {"is_enabled", "columns"}
        expected_is_enabled = option.is_enabled
        expected_columns = option.columns

        actual_option_json = json.loads(serialized)

        self.assertIn("class", actual_option_json)
        self.assertEqual(expected_class, actual_option_json["class"])
        self.assertIn("data", actual_option_json)
        self.assertEqual(
            expected_options_attributes, set(actual_option_json["data"].keys())
        )
        self.assertIn("is_enabled", actual_option_json["data"])
        self.assertEqual(expected_is_enabled, actual_option_json["data"]["is_enabled"])
        self.assertIn("columns", actual_option_json["data"])
        self.assertEqual(expected_columns, actual_option_json["data"]["columns"])
