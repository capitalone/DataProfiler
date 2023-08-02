import json
from unittest import mock

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

        expected = {
            "class": "FloatOptions",
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
                "precision": {
                    "class": "PrecisionOptions",
                    "data": mock.ANY,
                },
            },
        }

        self.assertDictEqual(expected, json.loads(serialized))
