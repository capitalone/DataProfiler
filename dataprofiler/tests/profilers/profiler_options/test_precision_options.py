from dataprofiler.profilers.profiler_options import PrecisionOptions
from dataprofiler.tests.profilers.profiler_options.test_boolean_option \
     import TestBooleanOption

class TestPrecisionOptions(TestBooleanOption):
    
    option_class = PrecisionOptions
    keys=[]
        
    def test_init(self):
        option = self.get_options()
        self.assertDictEqual(
            {"is_enabled": True, "sample_ratio": None}, option.properties)
        option = self.get_options(is_enabled=False)
        self.assertDictEqual(
            {"is_enabled": False, "sample_ratio": None}, option.properties)
    
    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
    
    def test_validate_helper(self):
        super().test_validate_helper()
    
    def test_validate(self):
        super().test_validate()
