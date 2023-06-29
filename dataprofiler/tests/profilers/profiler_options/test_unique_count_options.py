import json
from unittest import mock

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import UniqueCountOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBooleanOption,
)


class TestUniqueCountOptions(TestBooleanOption):

    option_class = UniqueCountOptions

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
        optpth = self.get_options_path()

        # hashing_method must be a string
        option = self.get_options(hashing_method=5)
        expected_error = [f"{optpth}.full_hashing must be a String."]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # hashing_method must be "hll" or "full"
        option = self.get_options(hashing_method="other")
        expected_error = [f"{optpth}.hashing_method must be 'full' or 'hll'."]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

    def test_validate(self):
        super().test_validate()

        optpth = self.get_options_path()

        # hashing_method must be a string
        option = self.get_options(hashing_method=5)
        expected_error = "UniqueCountOptions.full_hashing must be a String."
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # hashing_method must be "hll" or "full"
        option = self.get_options(hashing_method="other")
        expected_error = "UniqueCountOptions.hashing_method must be 'full' or 'hll'."
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

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
            "class": "UniqueCountOptions",
            "data": {
                "is_enabled": False,
                "hashing_method": "full",
                "hll": {"class": "HyperLogLogOptions", "data": mock.ANY},
            },
        }
        self.assertDictEqual(expected, json.loads(serialized))
