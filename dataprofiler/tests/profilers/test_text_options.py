from dataprofiler.profilers.profiler_options import TextOptions
from dataprofiler.tests.profilers.test_numerical_options import TestNumericalOptions


class TestTextOptions(TestNumericalOptions):
    
    option_class = TextOptions
    keys = TestNumericalOptions.keys + ["vocab"]

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
