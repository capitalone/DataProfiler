import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import HyperLogLogOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBaseOption,
)


class TestHyperLogLogOptions(TestBaseOption):

    option_class = HyperLogLogOptions

    def test_init(self):
        option = self.get_options()
        self.assertDictEqual({"seed": 0, "register_count": 15}, option.properties)
        option = self.get_options(seed=10, register_count=9)
        self.assertDictEqual({"seed": 10, "register_count": 9}, option.properties)

    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
        option = self.get_options()
        option.set({"seed": 5, "register_count": 8})
        self.assertDictEqual({"seed": 5, "register_count": 8}, option.properties)

    def test_validate_helper(self):
        optpth = self.get_options_path()

        # Default configuration
        option = self.get_options()
        self.assertEqual([], option._validate_helper())

        # Valid configurations
        option = self.get_options(seed=1, register_count=8)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(seed=-34, register_count=1)
        self.assertEqual([], option._validate_helper())

        expected_error = [f"{optpth}.seed must be an integer."]

        # Option seed cannot be a string, must be an int
        option = self.get_options(seed="Hello World")
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option seed cannot be a float, must be an int
        option = self.get_options(seed=1.1)
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option seed cannot be None, must be an int
        option = self.get_options(seed=None)
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        expected_error = [f"{optpth}.register_count must be an integer."]

        # Option register_count cannot be a float
        option = self.get_options(register_count=1.1)
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option register_count cannot be a str
        option = self.get_options(register_count="hello")
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        expected_error = [f"{optpth}.register_count must be greater than 0."]

        # Option register_count cannot be 0
        option = self.get_options(register_count=0)
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option register_count cannot be negative
        option = self.get_options(register_count=-1)
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        expected_error = [
            f"{optpth}.seed must be an integer.",
            f"{optpth}.register_count must be greater than 0.",
        ]

        # Testing multiple errors
        option = self.get_options(seed="hello", register_count=0)
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

    def test_validate(self):
        optpth = self.get_options_path()

        # Default configuration
        option = self.get_options(seed=0, register_count=10)
        self.assertEqual([], option._validate_helper())

        # Valid configurations
        option = self.get_options(seed=1)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(seed=-34)
        self.assertEqual([], option._validate_helper())

        expected_error = "HyperLogLogOptions.seed must be an int."

        # Option seed cannot be a string, must be a float
        option = self.get_options(seed="Hello World")
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option seed cannot be a float, must be an int
        option = self.get_options(seed=1.1)
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option seed cannot be None, must be an int
        option = self.get_options(seed=None)
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        expected_error = "HyperLogLogOptions.register_count must be an integer."

        # Option register_count cannot be a float
        option = self.get_options(register_count=1.1)
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option register_count cannot be a str
        option = self.get_options(register_count="hello")
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        expected_error = "HyperLogLogOptions.register_count must be greater than 0."

        # Option register_count cannot be 0
        option = self.get_options(register_count=0)
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option register_count cannot be negative
        option = self.get_options(register_count=-1)
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        expected_error = [
            "HyperLogLogOptions.seed must be an integer.",
            "HyperLogLogOptions.register_count must be greater than 0.",
        ]

        # Option raise warning if register_count greater than or equal to 20
        option = self.get_options(register_count=20)
        with self.assertWarnsRegex(
            UserWarning,
            "HyperLogLogOptions.register_count is greater than or equal "
            "to 20, so the row hashing object is greater than 5 MB.",
        ):
            option.validate()

        # Testing multiple errors
        option = self.get_options(seed="hello", register_count=0)
        self.assertSetEqual(
            set(expected_error), set(option.validate(raise_error=False))
        )

    def test_eq(self):
        options = self.get_options()
        options2 = self.get_options()
        options.seed = 1
        self.assertNotEqual(options, options2)
        options2.seed = -5
        self.assertNotEqual(options, options2)
        options2.seed = 1
        self.assertEqual(options, options2)

        options = self.get_options()
        options2 = self.get_options()
        options.register_count = 1
        self.assertNotEqual(options, options2)
        options2.register_count = 9
        self.assertNotEqual(options, options2)
        options2.register_count = 1
        self.assertEqual(options, options2)

    def test_json_encode(self):
        option = self.get_options()

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected = {
            "class": "HyperLogLogOptions",
            "data": {"seed": 0, "register_count": 15},
        }

        self.assertDictEqual(expected, json.loads(serialized))
