from dataprofiler.profilers.profiler_options import HistogramOption
from dataprofiler.tests.profilers.test_boolean_option import TestBooleanOption


class TestHistogramOption(TestBooleanOption):

    option_class = HistogramOption
    keys = []

    def test_init(self):
        option = self.get_options()
        self.assertTrue(option.is_enabled)
        self.assertIsNone(option.method)

    def test_set_helper(self):
        option = self.get_options()
        option._set_helper({"method": "sturges"}, "")
        self.assertEqual("sturges", option.method)
        option._set_helper({"is_enabled": False}, "")
        self.assertFalse(option.is_enabled)

        # Treat method as a BooleanOption
        expected_error = "type object 'method' has no attribute 'is_enabled'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            option._set_helper({'method.is_enabled': True}, '')

    def test_set(self):
        option = self.get_options()
        option.set({"method": "sturges"})
        self.assertEqual("sturges", option.method)
        option.set({"is_enabled": False})
        self.assertFalse(option.is_enabled)

        # Treat method as a BooleanOption
        expected_error = "type object 'method' has no attribute 'is_enabled'"
        with self.assertRaisesRegex(AttributeError, expected_error):
            option.set({'method.is_enabled': True})

    def test_validate_helper(self):
        pass

    def test_validate(self):
        pass
