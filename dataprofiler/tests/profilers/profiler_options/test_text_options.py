from dataprofiler.profilers.profiler_options import TextOptions
from dataprofiler.tests.profilers.profiler_options.test_numerical_options \
     import TestNumericalOptions


class TestTextOptions(TestNumericalOptions):
    
    option_class = TextOptions
    keys = TestNumericalOptions.keys + ["vocab"]

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
