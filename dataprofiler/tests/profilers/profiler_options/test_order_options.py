import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import OrderOptions
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options import (
    TestBaseInspectorOptions,
)


class TestOrderOptions(TestBaseInspectorOptions):

    options_class = OrderOptions

    def test_init(self):
        super().test_init()

    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()

    def test_validate_helper(self):
        super().test_validate_helper()

    def test_validate(self):
        super().test_validate()

    def test_is_prop_enabled(self):
        super().test_is_prop_enabled()

    def test_eq(self):
        super().test_eq()

    def test_json_encode(self):
        option = OrderOptions()

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected_class = "OrderOptions"
        expected_options_attributes = {"is_enabled"}

        actual_option_json = json.loads(serialized)

        self.assertIn("class", actual_option_json)
        self.assertEqual(expected_class, actual_option_json["class"])
        self.assertIn("data", actual_option_json)
        self.assertEqual(
            expected_options_attributes, set(actual_option_json["data"].keys())
        )
