from dataprofiler.profilers.profiler_options import FloatOptions
from dataprofiler.tests.profilers.profiler_options.test_numerical_options \
     import TestNumericalOptions


class TestFloatOptions(TestNumericalOptions):
    
    option_class = FloatOptions    
    keys = TestNumericalOptions.keys + ["precision"] 
        
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
    
    def test_is_numeric_stats_enabled(self):
        super().test_is_numeric_stats_enabled()

    def test_eq(self):
        super().test_eq()

        options = self.get_options()
        options2 = self.get_options()
        options.precision.is_enabled = False
        self.assertNotEqual(options, options2)
        options2.precision.is_enabled = False
        self.assertEqual(options, options2)
