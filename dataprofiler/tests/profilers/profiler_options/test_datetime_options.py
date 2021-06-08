from dataprofiler.profilers.profiler_options import DateTimeOptions
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options \
     import TestBaseInspectorOptions


class TestDateTimeOptions(TestBaseInspectorOptions):

    option_class = DateTimeOptions
        
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
        super().test_is_prop_enabled()

    def test_eq(self):
        super().test_eq()
