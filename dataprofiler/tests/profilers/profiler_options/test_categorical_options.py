from dataprofiler.profilers.profiler_options import CategoricalOptions
from dataprofiler.tests.profilers.profiler_options.test_base_inspector_options \
     import TestBaseInspectorOptions


class TestCategoricalOptions(TestBaseInspectorOptions):
    
    option_class = CategoricalOptions
    
    def test_init(self):
        option = self.get_options()
        self.assertDictEqual({"is_enabled": True, "top_k_categories": None}, option.properties)
        option = self.get_options(is_enabled=False)
        self.assertDictEqual({"is_enabled": False, "top_k_categories": None}, option.properties)
        option = self.get_options(top_k_categories=2)
        self.assertDictEqual({"is_enabled": True, "top_k_categories": 2}, option.properties)
    
    def test_set_helper(self):
        super().test_set_helper()

    def test_set(self):
        super().test_set()
    
    def test_validate_helper(self):
        super().test_validate_helper()
        # These ones should throw an error
        expected_error = "top_k_categories must be either None" \
                         " or a positive integer"
        options = self.get_options(top_k_categories=-2)
        self.assertEqual([expected_error], options._validate_helper())
        options = self.get_options(top_k_categories="E")
        self.assertEqual([expected_error], options._validate_helper())
        options = self.get_options(top_k_categories=2.0)
        self.assertEqual([expected_error], options._validate_helper())
        # These ones should not
        options = self.get_options(top_k_categories=6)
        self.assertEqual([], options._validate_helper())
        options = self.get_options(top_k_categories=None)
        self.assertEqual([], options._validate_helper())
    
    def test_validate(self):
        super().test_validate()

    def test_is_prop_enabled(self):
        super().test_is_prop_enabled()

    def test_eq(self):
        super().test_eq()
