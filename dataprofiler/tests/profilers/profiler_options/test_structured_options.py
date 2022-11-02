import re

from dataprofiler.profilers.profiler_options import StructuredOptions
from dataprofiler.tests.profilers.profiler_options.test_base_option import (
    TestBaseOption,
)


class TestStructuredOptions(TestBaseOption):

    option_class = StructuredOptions
    other_keys = ["null_values", "column_null_values"]
    boolean_keys = [
        "int",
        "float",
        "datetime",
        "text",
        "order",
        "category",
        "data_labeler",
        "multiprocess",
        "correlation",
        "chi2_homogeneity",
    ]
    keys = boolean_keys + other_keys

    @classmethod
    def get_options(self, **params):
        options = StructuredOptions()
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
        # Enable and Disable Option
        for key in self.boolean_keys:
            option._set_helper({"{}.is_enabled".format(key): False}, "")
            self.assertFalse(option.properties[key].is_enabled)
            option._set_helper({"{}.is_enabled".format(key): True}, "")
            self.assertTrue(option.properties[key].is_enabled)

        # Treat is_enabled as a BooleanOption
        for key in self.boolean_keys:
            expected_error = (
                "type object '{}.is_enabled' has no attribute "
                "'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option._set_helper({"{}.is_enabled.is_enabled".format(key): True}, "")

    def test_set(self):
        super().test_set()
        option = self.get_options()
        optpth = self.get_options_path()

        # Enable and Disable Options
        for key in self.boolean_keys:
            option.set({"{}.is_enabled".format(key): False})
            self.assertFalse(option.properties[key].is_enabled)
            option.set({"{}.is_enabled".format(key): True})
            self.assertTrue(option.properties[key].is_enabled)

        # Treat is_enabled as a BooleanOption
        for key in self.boolean_keys:
            expected_error = (
                "type object '{}.is_enabled' has no attribute "
                "'is_enabled'".format(key)
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option.set({"{}.is_enabled.is_enabled".format(key): True})

        for key in self.other_keys:
            expected_error = "type object '{}' has no attribute " "'is_enabled'".format(
                key
            )
            with self.assertRaisesRegex(AttributeError, expected_error):
                option.set({"{}.is_enabled".format(key): True})

        for test_dict in ({"a": 0}, {"a": re.IGNORECASE}, None):
            option.set({"null_values": test_dict})
            self.assertEqual(test_dict, option.null_values)

        for test_dict in ({0: {"a": 0}}, {0: {"a": re.IGNORECASE}}, None):
            option.set({"column_null_values": test_dict})
            self.assertEqual(test_dict, option.column_null_values)

    def test_validate_helper(self):
        # Valid cases should return [] while invalid cases
        # should return a list of errors
        option = self.get_options()
        optpth = self.get_options_path()

        # Default Configuration Is Valid
        self.assertEqual([], option._validate_helper())

        # Disable categories and enable chi2
        option.set({"category.is_enabled": False, "chi2_homogeneity.is_enabled": True})
        expected_error = [
            "Categorical statistics must be enabled if " "Chi-squared test in enabled."
        ]
        self.assertEqual(expected_error, option._validate_helper())
        option.set({"category.is_enabled": True, "chi2_homogeneity.is_enabled": True})

        # Variable Path Is Not A String
        expected_error = "The variable path must be a string."
        with self.assertRaisesRegex(ValueError, expected_error):
            option._validate_helper(1)

        # Option is_enabled is not a boolean
        for key in self.boolean_keys:
            option.set({"{}.is_enabled".format(key): "Hello World"})
        expected_error = [
            "{}.{}.is_enabled must be a Boolean.".format(optpth, key)
            for key in self.boolean_keys
        ]
        expected_error = set(expected_error)
        # Verify expected errors are a subset of all errors
        self.assertSetEqual(
            expected_error, expected_error.intersection(set(option._validate_helper()))
        )

        # Wrong Class Type
        option = self.get_options()
        option.int = StructuredOptions()
        option.float = StructuredOptions()
        option.datetime = StructuredOptions()
        option.text = StructuredOptions()
        option.order = StructuredOptions()
        option.category = StructuredOptions()
        option.data_labeler = StructuredOptions()
        option.multiprocess = StructuredOptions()
        option.correlation = StructuredOptions()
        option.chi2_homogeneity = StructuredOptions()

        expected_error = set()
        for key in self.boolean_keys:
            ckey = key.capitalize()
            if key == "data_labeler":
                ckey = "DataLabeler"
            elif key == "category":
                ckey = "Categorical"
            elif key == "datetime":
                ckey = "DateTime"
            if key == "multiprocess" or key == "chi2_homogeneity":
                expected_error.add(
                    "{}.{} must be a(n) BooleanOption.".format(optpth, key, ckey)
                )
            else:
                expected_error.add(
                    "{}.{} must be a(n) {}Options.".format(optpth, key, ckey)
                )
        self.assertSetEqual(expected_error, set(option._validate_helper()))

    def test_validate(self):
        # Valid cases should return None while invalid cases
        # should return or throw a list of errors
        option = self.get_options()
        optpth = self.get_options_path()

        # Default Configuration Is Valid
        self.assertEqual(None, option.validate())

        # Disable categories and enable chi2
        option.set({"category.is_enabled": False, "chi2_homogeneity.is_enabled": True})
        expected_error = (
            "Categorical statistics must be enabled if " "Chi-squared test in enabled."
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate(raise_error=True)
        option.set({"category.is_enabled": True, "chi2_homogeneity.is_enabled": True})

        # Option is_enabled is not a boolean
        for key in self.boolean_keys:
            option.set({"{}.is_enabled".format(key): "Hello World"})

        expected_error = [
            "{}.{}.is_enabled must be a Boolean.".format(optpth, key)
            for key in self.boolean_keys
        ]
        expected_error = set(expected_error)
        # Verify expected errors are a subset of all errors
        with self.assertRaises(ValueError) as cm:
            option.validate(raise_error=True)
        raised_error = set(str(cm.exception).split("\n"))
        self.assertEqual(expected_error, expected_error.intersection(raised_error))
        self.assertSetEqual(
            expected_error,
            expected_error.intersection(set(option.validate(raise_error=False))),
        )

        # Wrong Class Type
        option = self.get_options()
        option.int = StructuredOptions()
        option.float = StructuredOptions()
        option.datetime = StructuredOptions()
        option.text = StructuredOptions()
        option.order = StructuredOptions()
        option.category = StructuredOptions()
        option.data_labeler = StructuredOptions()
        option.multiprocess = StructuredOptions()
        option.correlation = StructuredOptions()
        option.chi2_homogeneity = StructuredOptions()

        expected_error = set()
        for key in self.boolean_keys:
            ckey = key.capitalize()
            if key == "data_labeler":
                ckey = "DataLabeler"
            elif key == "category":
                ckey = "Categorical"
            elif key == "datetime":
                ckey = "DateTime"
            if key == "multiprocess" or key == "chi2_homogeneity":
                expected_error.add(
                    "{}.{} must be a(n) BooleanOption.".format(optpth, key, ckey)
                )
            else:
                expected_error.add(
                    "{}.{} must be a(n) {}Options.".format(optpth, key, ckey)
                )
        # Verify expected errors are a subset of all errors
        self.assertSetEqual(expected_error, set(option.validate(raise_error=False)))
        with self.assertRaises(ValueError) as cm:
            option.validate(raise_error=True)
        raised_error = set(str(cm.exception).split("\n"))
        self.assertEqual(expected_error, raised_error)
        option = self.get_options()
        expected_error = [
            "{}.null_values must be either None or a "
            "dictionary that contains keys of type str "
            "and values == 0 or are instances of "
            "a re.RegexFlag".format(optpth)
        ]
        # Test key is not a string
        option.set({"null_values": {0: 0}})
        self.assertEqual(expected_error, option._validate_helper())
        # Test value is not correct type (0 or regex)
        option.set({"null_values": {"a": 1}})
        self.assertEqual(expected_error, option._validate_helper())
        # Test variable is not correct variable type
        option.set({"null_values": 1})
        self.assertEqual(expected_error, option._validate_helper())
        # Test 0 works for option set
        option.set({"null_values": {"a": 0}})
        self.assertEqual([], option._validate_helper())
        # Test a regex flag works for option set
        option.set({"null_values": {"a": re.IGNORECASE}})
        self.assertEqual([], option._validate_helper())
        # Test None works for option set
        option.set({"null_values": None})
        self.assertEqual([], option._validate_helper())

        expected_error = [
            "{}.column_null_values must be either None or "
            "a dictionary that contains keys of type int "
            "that map to dictionaries that contains keys "
            "of type str and values == 0 or are instances of "
            "a re.RegexFlag".format(optpth)
        ]
        # Test column key is not an int
        option.set({"column_null_values": {"a": {"a": 0}}})
        self.assertEqual(expected_error, option._validate_helper())
        # Test key is not a str
        option.set({"column_null_values": {0: {0: 0}}})
        self.assertEqual(expected_error, option._validate_helper())
        # Test value is not correct type (0 or regex)
        option.set({"column_null_values": {0: {"a": 1}}})
        self.assertEqual(expected_error, option._validate_helper())
        # Test variable is not correct variable type
        option.set({"column_null_values": 1})
        self.assertEqual(expected_error, option._validate_helper())
        # Test 0 works for option set
        option.set({"column_null_values": {0: {"a": 0}}})
        self.assertEqual([], option._validate_helper())
        # Test a regex flag works for option set
        option.set({"column_null_values": {0: {"a": re.IGNORECASE}}})
        self.assertEqual([], option._validate_helper())
        # Test None works for option set
        option.set({"column_null_values": None})
        self.assertEqual([], option._validate_helper())

    def test_enabled_profilers(self):
        options = self.get_options()
        self.assertNotIn("null_values", options.enabled_profiles)
        self.assertNotIn("column_null_values", options.enabled_profiles)

        # All Columns Enabled
        for key in self.boolean_keys:
            options.set({"{}.is_enabled".format(key): True})
        self.assertSetEqual(set(self.boolean_keys), set(options.enabled_profiles))

        # No Columns Enabled
        for key in self.boolean_keys:
            options.set({"{}.is_enabled".format(key): False})
        self.assertEqual([], options.enabled_profiles)

        # One Column Enabled
        for key in self.boolean_keys:
            options.set({"{}.is_enabled".format(key): True})
            self.assertSetEqual(set([key]), set(options.enabled_profiles))
            options.set({"{}.is_enabled".format(key): False})

    def test_eq(self):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.multiprocess.is_enabled = False
        self.assertNotEqual(options, options2)
        options2.multiprocess.is_enabled = False
        self.assertEqual(options, options2)

        options.float.precision.sample_ratio = 0.1
        self.assertNotEqual(options, options2)
        options2.float.precision.sample_ratio = 0.15
        self.assertNotEqual(options, options2)
        options2.float.precision.sample_ratio = 0.1
        self.assertEqual(options, options2)
