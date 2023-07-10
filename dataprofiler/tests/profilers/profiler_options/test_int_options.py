import json
from unittest import mock

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

    def test_json_encode(self):
        option = IntOptions()

        serialized = json.dumps(option, cls=ProfileEncoder)

        expected = {
            "class": "IntOptions",
            "data": {
                "min": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "max": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "mode": {
                    "class": "ModeOption",
                    "data": mock.ANY,
                },
                "median": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "sum": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "variance": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "skewness": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "kurtosis": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "median_abs_deviation": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "num_zeros": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "num_negatives": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "histogram_and_quantiles": {
                    "class": "HistogramAndQuantilesOption",
                    "data": mock.ANY,
                },
                "bias_correction": {
                    "class": "BooleanOption",
                    "data": {"is_enabled": True},
                },
                "is_enabled": True,
            },
        }

        self.assertDictEqual(expected, json.loads(serialized))
