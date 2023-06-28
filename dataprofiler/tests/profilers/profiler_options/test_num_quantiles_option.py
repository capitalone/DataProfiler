from dataprofiler.profilers.profiler_options import NumQuantilesOption
from dataprofiler.tests.profilers.profiler_options.test_boolean_option import (
    TestBooleanOption,
)


class TestNumQuantilesOption(TestBooleanOption):

    option_class = NumQuantilesOption
    keys = []

    def test_init(self):
        option = self.get_options()

        self.assertDictEqual(
            {"is_enabled": True, "num_quantiles": 1000}, option.properties
        )
        option = self.get_options(is_enabled=False)
        self.assertDictEqual(
            {"is_enabled": False, "num_quantiles": 1000}, option.properties
        )
        option = self.get_options(is_enabled=False, num_quantiles=50)
        self.assertDictEqual(
            {"is_enabled": False, "num_quantiles": 50}, option.properties
        )

    def test_set_helper(self):
        super().test_set_helper
        option = self.get_options()

        # validate, variable path being passed
        expected_error = (
            "type object 'test.num_quantiles' has no " "attribute 'is_enabled'"
        )
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({"num_quantiles.is_enabled": True}, "test")

    def test_set(self):
        super().test_set()
        option = self.get_options()
        option.set({"num_quantiles": 50})
        self.assertDictEqual(
            {"is_enabled": True, "num_quantiles": 50}, option.properties
        )

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

        # Option num_quantiles must be a positive integer
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

        # Option num_quantiles cannot be a string, must be an int
        option = self.get_options(num_quantiles="Hello World")
        expected_error = (
            "NumQuantilesOption.num_quantiles must be a " "positive integer"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option num_quantiles cannot be a float, must be an int
        option = self.get_options(num_quantiles=1.1)
        expected_error = (
            "NumQuantilesOption.num_quantiles must be a " "positive integer"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option num_quantiles must be a positive integer
        option = self.get_options(num_quantiles=0)
        expected_error = (
            "NumQuantilesOption.num_quantiles must be a " "positive integer"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option num_quantiles must be a positive integer
        option = self.get_options(num_quantiles=-5)
        expected_error = (
            "NumQuantilesOption.num_quantiles must be a " "positive integer"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

    def test_eq(self):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.num_quantiles = 30
        self.assertNotEqual(options, options2)
        options2.num_quantiles = 50
        self.assertNotEqual(options, options2)
        options2.num_quantiles = 30
        self.assertEqual(options, options2)
