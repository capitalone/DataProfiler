import json

from dataprofiler.profilers.json_decoder import load_option
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import HistogramAndQuantilesOption

from .. import utils as test_utils
from .test_boolean_option import TestBooleanOption


class TestHistogramAndQuantilesOption(TestBooleanOption):

    option_class = HistogramAndQuantilesOption
    keys = []

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.is_enabled)
        self.assertEqual(option.bin_count_or_method, "auto")
        self.assertEqual(option.num_quantiles, 1000)

    def test_set_helper(self):
        option = self.get_options()

        # validate, variable path being passed
        expected_error = (
            "type object 'test.bin_count_or_method' has no attribute 'is_enabled'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({"bin_count_or_method.is_enabled": True}, "test")

        # validate, variable path being passed
        expected_error = (
            "type object 'test.num_quantiles' has no attribute 'is_enabled'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({"num_quantiles.is_enabled": True}, "test")

    def test_set(self):
        option = self.get_options()

        params_to_check = [
            dict(prop="is_enabled", value_list=[False, True]),
            dict(
                prop="bin_count_or_method",
                value_list=[
                    None,
                    "auto",
                    "fd",
                    "doane",
                    "scott",
                    "rice",
                    "sturges",
                    "sqrt",
                    ["sturges", "doane"],
                    1,
                    10,
                    100,
                    1000,
                    99,
                    10000000,
                ],
            ),
        ]

        # this code can be abstracted to limit code everywhere else
        # AKA, params_to_check would be the only needed code plus raise errors
        def _assert_set_helper(prop, value):
            option.set({prop: value})
            self.assertEqual(value, getattr(option, prop), msg=prop)

        for params in params_to_check:
            prop, value_list = params["prop"], params["value_list"]
            for value in value_list:
                _assert_set_helper(prop, value)

        # Treat bin_count_or_method as a BooleanOption
        expected_error = (
            "type object 'bin_count_or_method' has no attribute 'is_enabled'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({"bin_count_or_method.is_enabled": True})

        # Treat num_quantiles as a BooleanOption
        expected_error = "type object 'num_quantiles' has no attribute 'is_enabled'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({"num_quantiles.is_enabled": True})

        # Test set option for num_quantiles
        option.set({"num_quantiles": 50})
        self.assertEqual(option.num_quantiles, 50)

    def test_validate_helper(self):
        super().test_validate_helper()

        optpth = self.get_options_path()

        # Default configuration
        option = self.get_options(num_quantiles=1000)
        self.assertEqual([], option._validate_helper())

        # Valid configurations
        option = self.get_options(num_quantiles=50)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(num_quantiles=2000)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(num_quantiles=1)
        self.assertEqual([], option._validate_helper())

        # Option num_quantiles
        option = self.get_options(num_quantiles="Hello World")
        expected_error = [f"{optpth}.num_quantiles must be a positive integer."]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option num_quantiles cannot be a float, must be an int
        option = self.get_options(num_quantiles=1.1)
        expected_error = [f"{optpth}.num_quantiles must be a positive integer."]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option num_quantiles may not be zero, must be greater than one(1)
        option = self.get_options(num_quantiles=0)
        expected_error = [f"{optpth}.num_quantiles must be a positive integer."]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option num_quantiles cannot be a negative integer
        option = self.get_options(num_quantiles=-5)
        expected_error = [f"{optpth}.num_quantiles must be a positive integer."]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

    def test_validate(self):

        super().test_validate()

        optpth = self.get_options_path()

        params_to_check = [
            # non errors
            dict(prop="is_enabled", value_list=[False, True], errors=[]),
            dict(
                prop="bin_count_or_method",
                value_list=[
                    "auto",
                    "fd",
                    "doane",
                    "scott",
                    "rice",
                    "sturges",
                    "sqrt",
                    ["sturges", "doane"],
                    1,
                    10,
                    100,
                    1000,
                    99,
                    10000000,
                ],
                errors=[],
            ),
            # errors
            dict(
                prop="bin_count_or_method",
                value_list=[
                    -1,
                    1.2,
                    1.0,
                    [],
                    False,
                    "whoops",
                    ["doane", "incorrect"],
                    "1",
                ],
                errors=[
                    "HistogramAndQuantilesOption.bin_count_or_method must be an integer "
                    "more than 1, a string, or list of strings from the "
                    "following: ['auto', 'fd', 'doane', 'scott', 'rice', "
                    "'sturges', 'sqrt']."
                ],
            ),
        ]

        # this code can be abstracted to limit code everywhere else
        # AKA, for loop below could be abstracted to a utils func

        # Default configuration is valid
        option = self.get_options()
        self.assertIsNone(option.validate(raise_error=False))

        for params in params_to_check:
            prop, value_list, expected_errors = (
                params["prop"],
                params["value_list"],
                params["errors"],
            )
            option = self.get_options()
            for value in value_list:
                setattr(option, prop, value)
                validate_errors = option.validate(raise_error=False)
                if expected_errors:
                    self.assertListEqual(
                        expected_errors,
                        validate_errors,
                        msg=f"Errored for prop: {prop}, value: {value}.",
                    )
                else:
                    self.assertIsNone(
                        validate_errors,
                        msg=f"Errored for prop: {prop}, value: {value}.",
                    )

        # this time testing raising an error
        option.bin_count_or_method = "fake method"
        expected_error = (
            r"HistogramAndQuantilesOption.bin_count_or_method must be an integer more than "
            r"1, a string, or list of strings from the following: "
            r"\['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']."
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Valid configurations
        option = self.get_options(num_quantiles=50)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(num_quantiles=2000)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(num_quantiles=1)
        self.assertEqual([], option._validate_helper())

        # Option num_quantiles cannot be a string, must be an int
        option = self.get_options(num_quantiles="Hello World")
        expected_error = f"{optpth}.num_quantiles must be a positive integer"
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option num_quantiles cannot be a float, must be an int
        option = self.get_options(num_quantiles=1.1)
        expected_error = f"{optpth}.num_quantiles must be a positive integer"
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option num_quantiles must be a positive integer
        option = self.get_options(num_quantiles=0)
        expected_error = f"{optpth}.num_quantiles must be a positive integer"
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option num_quantiles cannot be a negative integer
        option = self.get_options(num_quantiles=-5)
        expected_error = f"{optpth}.num_quantiles must be a positive integer"
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

    def test_eq(self):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.bin_count_or_method = "sturges"
        self.assertNotEqual(options, options2)
        options2.bin_count_or_method = "doane"
        self.assertNotEqual(options, options2)
        options2.bin_count_or_method = "sturges"
        self.assertEqual(options, options2)
        options.num_quantiles = 30
        self.assertNotEqual(options, options2)
        options2.num_quantiles = 50
        self.assertNotEqual(options, options2)
        options2.num_quantiles = 30
        self.assertEqual(options, options2)

    def test_json_encode(self):
        option = HistogramAndQuantilesOption(
            is_enabled=False, bin_count_or_method="doane"
        )

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected = {
            "class": "HistogramAndQuantilesOption",
            "data": {
                "bin_count_or_method": "doane",
                "num_quantiles": 1000,
                "is_enabled": False,
            },
        }

        self.assertDictEqual(expected, json.loads(serialized))

    def test_json_decode_warn(self):
        old_histogram = {
            "class": "HistogramOption",
            "data": {
                "bin_count_or_method": "doane",
                "is_enabled": False,
            },
        }

        expected = HistogramAndQuantilesOption(
            is_enabled=False, bin_count_or_method="doane"
        )

        expected_string = json.dumps(old_histogram, cls=ProfileEncoder)

        expected_warning = (
            "HistogramOption will be deprecated in the future. During the JSON encode "
            "process, HistogramOption is mapped to HistogramAndQuantilesOption. "
            "Please begin utilizing the new HistogramAndQuantilesOption class."
        )

        with self.assertWarnsRegex(DeprecationWarning, expected_warning):
            deserialized = load_option(json.loads(expected_string))
            test_utils.assert_profiles_equal(deserialized, expected)
