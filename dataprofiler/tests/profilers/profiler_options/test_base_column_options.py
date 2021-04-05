from dataprofiler.profilers.profiler_options import BaseColumnOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option \
     import TestBooleanOption


class TestBaseColumnOptions(TestBooleanOption):
    
    option_class = BaseColumnOptions

    @classmethod
    def get_options(cls, *args, **params):
        cls.validate_option_class()
        options = cls.option_class()
        options.set(params)
        return options
            
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
        options = self.get_options()
        optpth = self.get_options_path()

        # Check is prop enabled for valid property
        options.set({"is_enabled": True})
        self.assertTrue(options.is_prop_enabled("is_enabled"))
        options.set({"is_enabled": False})
        self.assertFalse(options.is_prop_enabled("is_enabled"))
    
        # Check is prop enabled for invalid property    
        expected_error = 'Property "Hello World" does not exist in {}.' \
                         .format(optpth)
        with self.assertRaisesRegex(AttributeError, expected_error):
            options.is_prop_enabled("Hello World")

