from dataprofiler.profilers.profiler_options import IntOptions
from dataprofiler.tests.profilers.test_numerical_options import TestNumericalOptions


class TestIntOptions(TestNumericalOptions):
    
    option_class = IntOptions

    def test_init(self, *mocks):
        super().test_init(*mocks)
    
    def test_set_helper(self, *mocks):
        super().test_set_helper(*mocks)

    def test_set(self, *mocks):
        super().test_set(*mocks)
    
    def test_validate_helper(self, *mocks):
        super().test_validate_helper(*mocks)
    
    def test_validate(self, *mocks):
        super().test_validate(*mocks)
    
    def test_is_numeric_stats_enabled(self, *mocks):
        super().test_is_numeric_stats_enabled(*mocks)
