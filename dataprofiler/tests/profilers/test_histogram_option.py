from dataprofiler.profilers.profiler_options import HistogramOption
from dataprofiler.tests.profilers.test_boolean_option import TestBooleanOption
import re


class TestHistogramOption(TestBooleanOption):

    option_class = HistogramOption
    keys = []

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.is_enabled)
        self.assertIsNone(option.method)

    def test_set_helper(self):
        option = self.get_options()
        option._set_helper({"method": "sturges"}, "")
        self.assertEqual("sturges", option.method)
        option._set_helper({"is_enabled": False}, "")
        self.assertFalse(option.is_enabled)

        # Treat method as a BooleanOption
        expected_error = "type object 'method' has no attribute 'is_enabled'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'method.is_enabled': True}, '')

    def test_set(self):
        option = self.get_options()
        option.set({"method": "sturges"})
        self.assertEqual("sturges", option.method)
        option.set({"is_enabled": False})
        self.assertFalse(option.is_enabled)

        # Treat method as a BooleanOption
        expected_error = "type object 'method' has no attribute 'is_enabled'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'method.is_enabled': True})

    def test_validate_helper(self):
        option = self.get_options()

        # Default configuration is valid
        self.assertEqual([], option._validate_helper())

        # Set method to something that isn't a string
        option.method = 3
        errs = ["HistogramOption.method must be a string.",
                "HistogramOption.method must be one of the following: "
                "['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']."]
        self.assertEqual(errs, option._validate_helper())

        # Set method to a string that isn't a correct method
        option.method = "whoops"
        self.assertEqual(errs[1:], option._validate_helper())

    def test_validate(self):
        option = self.get_options()

        # Default configuration is valid
        self.assertEqual([], option._validate_helper())

        # Set method to something that isn't a string
        option.method = 3
        errs = ["HistogramOption.method must be a string.",
                "HistogramOption.method must be one of the following: "
                "['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']."]
        with self.assertRaisesRegex(ValueError, re.escape('\n'.join(errs))):
            option.validate()

        # Set method to a string that isn't a correct method
        option.method = "whoops"
        with self.assertRaisesRegex(ValueError, re.escape(errs[1])):
            option.validate()
