import json
from unittest import mock

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import ProfilerOptions
from dataprofiler.tests.profilers.profiler_options.abstract_test_options import (
    JSONDecodeTestMixin,
)
from dataprofiler.tests.profilers.profiler_options.test_base_option import (
    TestBaseOption,
)


class TestProfilerOptions(TestBaseOption, JSONDecodeTestMixin):

    option_class = ProfilerOptions
    keys = ["structured_options", "unstructured_options"]

    @classmethod
    def get_options(self, **params):
        options = ProfilerOptions()
        options.set(params)
        return options

    def test_init(self):
        options = self.get_options()
        for key in self.keys:
            self.assertIn(key, options.properties)

    def test_set_helper(self):
        super().test_set_helper()
        option = self.get_options()
        optpth = self.get_options_path()

        # validate can set lower properties
        self.assertTrue(option.structured_options.text.is_enabled)
        self.assertTrue(option.unstructured_options.text.is_enabled)

        # set default state for test
        option.structured_options.text.is_enabled = True
        option.unstructured_options.text.is_enabled = True

        # test set False
        option._set_helper({"structured_options.text.is_enabled": False}, "")
        self.assertFalse(option.structured_options.text.is_enabled)
        option._set_helper({"unstructured_options.text.is_enabled": False}, "")
        self.assertFalse(option.unstructured_options.text.is_enabled)

        # test set True
        option._set_helper({"structured_options.text.is_enabled": True}, "")
        self.assertTrue(option.structured_options.text.is_enabled)
        option._set_helper({"unstructured_options.text.is_enabled": True}, "")
        self.assertTrue(option.unstructured_options.text.is_enabled)

    def test_set(self):
        super().test_set()
        option = self.get_options()
        optpth = self.get_options_path()

        # validate can set lower properties
        self.assertTrue(option.structured_options.text.is_enabled)
        self.assertTrue(option.unstructured_options.text.is_enabled)

        # set default state for test
        option.structured_options.text.is_enabled = True
        option.unstructured_options.text.is_enabled = True

        # test set False
        option.set({"structured_options.text.is_enabled": False})
        self.assertFalse(option.structured_options.text.is_enabled)
        option.set({"unstructured_options.text.is_enabled": False})
        self.assertFalse(option.unstructured_options.text.is_enabled)

        # test set True
        option.set({"structured_options.text.is_enabled": True})
        self.assertTrue(option.structured_options.text.is_enabled)
        option.set({"unstructured_options.text.is_enabled": True})
        self.assertTrue(option.unstructured_options.text.is_enabled)

        # validate raises assert error in lower properties due to bad calls
        for key in ["structured_options", "unstructured_options"]:
            expected_error = (
                "type object '{}.text.is_enabled' "
                "has no attribute 'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option.set({f"{key}.text.is_enabled.is_enabled": True})

    def test_validate_helper(self):
        # Valid cases should return [] while invalid cases
        # should return a list of errors
        option = self.get_options()
        optpth = self.get_options_path()

        # Default Configuration Is Valid
        self.assertEqual([], option._validate_helper())

        # Variable Path Is Not A String
        expected_error = "The variable path must be a string."
        with self.assertRaisesRegex(ValueError, expected_error):
            option._validate_helper(1)

        # Option is_enabled is not a boolean
        for key in self.keys:
            option.set({f"{key}.text.is_enabled": "Hello World"})
        expected_error = [
            "{}.{}.text.is_enabled must be a " "Boolean.".format(optpth, key)
            for key in self.keys
        ]
        expected_error = set(expected_error)

        # Verify expected errors are a subset of all errors
        self.assertSetEqual(
            expected_error, expected_error.intersection(set(option._validate_helper()))
        )

        # Wrong Class Type
        option = self.get_options()
        option.structured_options = ProfilerOptions()
        option.unstructured_options = ProfilerOptions()
        expected_error = [
            f"{optpth}.structured_options must be a StructuredOptions.",
            f"{optpth}.unstructured_options must be an UnstructuredOptions.",
        ]
        self.assertEqual(expected_error, option._validate_helper())

    def test_validate(self):
        # Valid cases should return None while invalid cases
        # should return or throw a list of errors
        option = self.get_options()
        optpth = self.get_options_path()

        # Default Configuration Is Valid
        self.assertEqual(None, option.validate())

        # Option is_enabled is not a boolean
        for key in self.keys:
            option.set({f"{key}.text.is_enabled": "Hello World"})
        expected_error = [
            "{}.{}.text.is_enabled must be a " "Boolean.".format(optpth, key)
            for key in self.keys
        ]
        expected_error = set(expected_error)

        # Verify expected errors are a subset of all errors
        with self.assertRaises(ValueError) as cm:
            option.validate(raise_error=True)
        raised_error = set(str(cm.exception).split("\n"))
        self.assertSetEqual(expected_error, expected_error.intersection(raised_error))
        self.assertSetEqual(
            expected_error,
            expected_error.intersection(set(option.validate(raise_error=False))),
        )

        # Wrong Class Type
        option = self.get_options()
        option.structured_options = ProfilerOptions()
        option.unstructured_options = ProfilerOptions()
        expected_error = [
            f"{optpth}.structured_options must be a StructuredOptions.",
            f"{optpth}.unstructured_options must be an UnstructuredOptions.",
        ]
        with self.assertRaisesRegex(ValueError, "\n".join(expected_error)):
            option.validate()
        self.assertListEqual(expected_error, option.validate(raise_error=False))

    def test_json_encode(self):
        option = ProfilerOptions(presets="complete")

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected = {
            "class": "ProfilerOptions",
            "data": {
                "structured_options": {
                    "class": "StructuredOptions",
                    "data": mock.ANY,
                },
                "unstructured_options": {
                    "class": "UnstructuredOptions",
                    "data": mock.ANY,
                },
                "presets": "complete",
            },
        }

        self.assertDictEqual(expected, json.loads(serialized))
