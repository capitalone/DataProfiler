from dataprofiler import Data, ProfilerOptions, Profiler
from dataprofiler.profilers.profiler_options import OrderOptions
from dataprofiler.tests.profilers.test_base_column_options import TestBaseColumnOptions


class TestOrderOptions(TestBaseColumnOptions):
    
    options_class = OrderOptions
    
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

    def test_is_prop_enabled(self, *mocks):
        super().test_is_prop_enabled(*mocks)
