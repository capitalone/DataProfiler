from dataprofiler.profilers.profiler_options import PrecisionOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option \
     import TestBooleanOption


class TestPrecisionOptions(TestBooleanOption):
    
    option_class = PrecisionOptions
        
    def test_init(self):
        option = self.get_options()
        self.assertDictEqual(
            {"is_enabled": True, "sample_ratio": None}, option.properties)
        option = self.get_options(is_enabled=False)
        self.assertDictEqual(
            {"is_enabled": False, "sample_ratio": None}, option.properties)
        option = self.get_options(is_enabled=False, sample_ratio=0.5)
        self.assertDictEqual(
            {"is_enabled": False, "sample_ratio": 0.5}, option.properties)
    
    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
        option = self.get_options()
        option.set({"sample_ratio": 0.5})
        self.assertDictEqual(
            {"is_enabled": True, "sample_ratio": 0.5}, option.properties)
    
    def test_validate_helper(self):
        super().test_validate_helper()

        optpth = self.get_options_path()

        # Default configuration
        option = self.get_options(sample_ratio=None)
        self.assertEqual([], option._validate_helper())

        # Valid configurations
        option = self.get_options(sample_ratio=0.5)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(sample_ratio=1)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(sample_ratio=0)
        self.assertEqual([], option._validate_helper())

        # Option sample_ratio cannot be a string, must be a float
        option = self.get_options(sample_ratio="Hello World")
        expected_error = ["{}.sample_ratio must be a float."
                          .format(optpth)]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option sample_ratio must be between 0 and 1
        option = self.get_options(sample_ratio=1.1)
        expected_error = ["{}.sample_ratio must be a float between 0 and 1."
                          .format(optpth)]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))

        # Option sample_ratio must be between 0 and 1
        option = self.get_options(sample_ratio=-0.1)
        expected_error = ["{}.sample_ratio must be a float between 0 and 1."
                          .format(optpth)]
        self.assertSetEqual(set(expected_error), set(option._validate_helper()))
    
    def test_validate(self):
        super().test_validate()

        optpth = self.get_options_path()

        # Default configuration
        option = self.get_options(sample_ratio=None)
        self.assertEqual([], option._validate_helper())

        # Valid configurations
        option = self.get_options(sample_ratio=0.5)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(sample_ratio=1)
        self.assertEqual([], option._validate_helper())
        option = self.get_options(sample_ratio=0)
        self.assertEqual([], option._validate_helper())

        # Option sample_ratio cannot be a string, must be a float
        option = self.get_options(sample_ratio="Hello World")
        expected_error = ("PrecisionOptions.sample_ratio must be a float.")
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option sample_ratio must be between 0 and 1
        option = self.get_options(sample_ratio=1.1)
        expected_error = (
            "PrecisionOptions.sample_ratio must be a float between 0 and 1."
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

        # Option sample_ratio must be between 0 and 1
        option = self.get_options(sample_ratio=-1)
        expected_error = (
            "PrecisionOptions.sample_ratio must be a float between 0 and 1."
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            option.validate()

    def test_eq(self):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.sample_ratio = 0.3
        self.assertNotEqual(options, options2)
        options2.sample_ratio = 0.5
        self.assertNotEqual(options, options2)
        options2.sample_ratio = 0.3
        self.assertEqual(options, options2)
