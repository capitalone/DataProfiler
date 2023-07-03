import json
from unittest import mock

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import RowStatisticsOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBooleanOption,
)


class TestRowStatisticsOptions(TestBooleanOption):

    option_class = RowStatisticsOptions
    keys = ["unique_count", "null_count"]

    def get_options(self, **params):
        options = RowStatisticsOptions()
        options.set(params)
        return options

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.properties["is_enabled"])
        option = self.get_options(is_enabled=False)
        self.assertFalse(option.properties["is_enabled"])

        options = self.get_options()
        for key in self.keys:
            self.assertIn(key, options.properties)

    def test_set_helper(self):
        super().test_set_helper()
        option = self.get_options()
        optpth = self.get_options_path()
        for key in self.keys:
            # Enable and Disable Option
            option._set_helper({f"{key}.is_enabled": False}, "")
            self.assertFalse(option.properties[key].is_enabled)
            option._set_helper({f"{key}.is_enabled": True}, "")
            self.assertTrue(option.properties[key].is_enabled)

            # Treat is_enabled as a BooleanOption
            expected_error = (
                "type object '{}.is_enabled' has no attribute "
                "'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option._set_helper({f"{key}.is_enabled.is_enabled": True}, "")

    def test_set(self):
        super().test_set_helper()
        option = self.get_options()
        optpth = self.get_options_path()
        for key in self.keys:
            # Enable and Disable Option
            option._set_helper({f"{key}.is_enabled": False}, "")
            self.assertFalse(option.properties[key].is_enabled)
            option._set_helper({f"{key}.is_enabled": True}, "")
            self.assertTrue(option.properties[key].is_enabled)

            # Treat is_enabled as a BooleanOption
            expected_error = (
                "type object '{}.is_enabled' has no attribute "
                "'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option._set_helper({f"{key}.is_enabled.is_enabled": True}, "")

    def test_validate_helper(self):
        super().test_validate_helper()

    def test_validate(self):
        super().test_validate()

    def test_eq(self):
        options = self.get_options()
        options2 = self.get_options()
        options.is_enabled = False
        self.assertNotEqual(options, options2)
        options2.is_enabled = False
        self.assertEqual(options, options2)

    def test_json_encode(self):
        option = self.get_options(is_enabled=False)

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected = {
            "class": "RowStatisticsOptions",
            "data": {
                "is_enabled": False,
                "unique_count": {"class": "UniqueCountOptions", "data": mock.ANY},
                "null_count": {"class": "BooleanOption", "data": {"is_enabled": True}},
            },
        }
        self.assertDictEqual(expected, json.loads(serialized))
