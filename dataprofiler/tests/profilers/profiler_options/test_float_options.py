import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import FloatOptions
from dataprofiler.tests.profilers.profiler_options.test_numerical_options import (
    TestNumericalOptions,
)


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

    def test_json_encode(self):
        option = FloatOptions()

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected_class = "FloatOptions"
        expected_options_attributes = {
            "is_enabled",
            "min",
            "max",
            "mode",
            "median",
            "sum",
            "variance",
            "skewness",
            "kurtosis",
            "median_abs_deviation",
            "num_zeros",
            "num_negatives",
            "histogram_and_quantiles",
            "bias_correction",
            "precision",
        }

        actual_option_json = json.loads(serialized)

        self.assertIn("class", actual_option_json)
        self.assertEqual(expected_class, actual_option_json["class"])
        self.assertIn("data", actual_option_json)
        self.assertEqual(
            expected_options_attributes, set(actual_option_json["data"].keys())
        )
