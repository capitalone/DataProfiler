import json

from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import IntOptions
from dataprofiler.tests.profilers.profiler_options.test_numerical_options import (
    TestNumericalOptions,
)


class TestIntOptions(TestNumericalOptions):

    option_class = IntOptions

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

    def test_json_encode_after_update(self):
        option = IntOptions()

        serialized = json.dumps(option, cls=ProfileEncoder)

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
        }

        self.assertEqual(
            expected_options_attributes, set(json.loads(serialized)["data"].keys())
        )
