import os
import sys
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.profilers import NumericStatsMixin
from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler
from dataprofiler.profilers.profiler_options import NumericalOptions

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestColumn(NumericStatsMixin):
    def __init__(self):
        NumericStatsMixin.__init__(self)
        self.match_count = 0
        self.times = defaultdict(float)

    def update(self, df_series):
        pass

    def _filter_properties_w_options(self, calculations, options):
        pass


class TestColumnWProps(TestColumn):
    # overrides the property func
    median = None
    mode = None
    median_abs_deviation = None

    def __init__(self):
        super().__init__()
        self.median = None
        self.mode = None
        self.median_abs_deviation = None


class TestNumericStatsMixin(unittest.TestCase):
    @mock.patch.multiple(
        NumericStatsMixin,
        __abstractmethods__=set(),
        _filter_properties_w_options=mock.MagicMock(return_value=None),
        create=True,
    )
    def test_base(self):

        # validate requires NumericalOptions
        with self.assertRaisesRegex(
            ValueError,
            "NumericalStatsMixin parameter 'options' "
            "must be of type NumericalOptions.",
        ):
            profile = NumericStatsMixin(options="bad options")

        try:
            # validate doesn't fail
            profile = NumericStatsMixin()
            profile = NumericStatsMixin(NumericalOptions())
        except Exception as e:
            self.fail(e)

    def test_check_float(self):
        """
        Checks if number is float.
        :return:
        """
        true_asserts = [
            1.3,
            1.345,
            -1.3,
            0.03,
            0.0,
            -0.0,
            1,  # numeric values
            float("nan"),
            np.nan,  # nan values
            "1.3",
            "nan",  # strings
        ]
        for assert_val in true_asserts:
            self.assertTrue(NumericStatsMixin.is_float(assert_val))

        false_asserts = ["1.3a", "abc", "", "1.23.45"]
        for assert_val in false_asserts:
            self.assertFalse(NumericStatsMixin.is_float(assert_val))

    def test_check_int(self):
        """
        Checks if number is integer.
        :return:
        """
        true_asserts = [1, 1345, -13, 0, -0, "1"]  # numeric values  # strings
        for assert_val in true_asserts:
            self.assertTrue(NumericStatsMixin.is_int(assert_val))

        false_asserts = [
            1.3,  # float
            float("nan"),
            np.nan,  # nan value
            "nan",
            "1a",
            "abc",
            "",
            "1.3",  # strings
        ]
        for assert_val in false_asserts:
            self.assertFalse(NumericStatsMixin.is_int(assert_val))

    def test_update_variance(self):
        """
        Checks update variance
        :return:
        """
        num_profiler = TestColumn()

        # test update variance
        data1 = [-3.0, 2.0, 11.0]
        mean1 = (-3.0 + 2.0 + 11.0) / 3
        var1 = ((-3.0 - mean1) ** 2 + (2.0 - mean1) ** 2 + (11.0 - mean1) ** 2) / 2
        count1 = len(data1)
        num_profiler._biased_variance = num_profiler._update_variance(
            mean1, var1 * 2 / 3, count1
        )
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)
        self.assertAlmostEqual(var1, num_profiler.variance)

        # test streaming update variance with new data
        data2 = [-5.0, 5.0, 11.0]
        mean2 = (-5.0 + 5.0 + 11.0) / 3
        var2 = ((-5.0 - mean2) ** 2 + (5.0 - mean2) ** 2 + (11.0 - mean2) ** 2) / 2
        count2 = len(data2)
        num_profiler._biased_variance = num_profiler._update_variance(
            mean2, var2 * 2 / 3, count2
        )
        num_profiler.match_count += count2
        num_profiler.sum += sum(data2)
        var_from_profile_updated = num_profiler.variance

        data_all = [-5.0, 5.0, 11.0, -3.0, 2.0, 11.0]
        mean_all = (-5.0 + 5.0 + 11.0 - 3.0 + 2.0 + 11.0) / 6
        var_all = (
            (-5.0 - mean_all) ** 2
            + (5.0 - mean_all) ** 2
            + (11.0 - mean_all) ** 2
            + (-3.0 - mean_all) ** 2
            + (2.0 - mean_all) ** 2
            + (11.0 - mean_all) ** 2
        ) / 5

        self.assertAlmostEqual(var_all, var_from_profile_updated)

    def test_update_variance_with_varying_data_length(self):
        """
        Checks update variance
        :return:
        """
        # empty data
        data1 = []
        mean1, var1, count1 = 0, np.nan, 0

        num_profiler = TestColumn()
        num_profiler._biased_variance = num_profiler._update_variance(
            mean1, var1, count1
        )
        num_profiler.match_count = count1
        num_profiler.sum = 0
        self.assertTrue(num_profiler.variance is np.nan)

        # data with 1 element
        data2 = [5.0]
        mean2, var2, count2 = 5.0, 0, 1

        num_profiler = TestColumn()
        num_profiler._biased_variance = num_profiler._update_variance(
            mean2, var2, count2
        )
        num_profiler.match_count += count2
        num_profiler.sum += 5.0
        self.assertTrue(num_profiler.variance is np.nan)

        # data with multiple elements
        data3 = [-5.0, 5.0, 11.0, -11.0]
        mean3, count3 = 0, 4
        var3 = (
            (-5.0 - mean3) ** 2
            + (5.0 - mean3) ** 2
            + (11.0 - mean3) ** 2
            + (-11.0 - mean3) ** 2
        ) / 3

        num_profiler = TestColumn()
        num_profiler._biased_variance = num_profiler._update_variance(
            mean3, var3 * 3 / 4, count3
        )
        num_profiler.match_count += count3
        num_profiler.sum += sum(data3)
        self.assertEqual(var3, num_profiler.variance)

    def test_update_variance_with_empty_data(self):
        """
        Checks update variance
        :return:
        """
        num_profiler = TestColumn()

        data1 = [-3.0, 2.0, 11.0]
        mean1 = (-3.0 + 2.0 + 11.0) / 3
        var1 = ((-3.0 - mean1) ** 2 + (2.0 - mean1) ** 2 + (11.0 - mean1) ** 2) / 2
        count1 = len(data1)
        num_profiler._biased_variance = num_profiler._update_variance(
            mean1, var1 * 2 / 3, count1
        )
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)
        self.assertEqual(var1, num_profiler.variance)

        # test adding data which would not have anything
        # data + empty
        mean2, var2, count2 = 0, 0, 0
        num_profiler._biased_variance = num_profiler._update_variance(
            mean2, var2, count2
        )
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)
        var_from_profile_updated = num_profiler.variance

        # simulate not having data
        mean_all, var_all = mean1, var1
        self.assertEqual(var_all, var_from_profile_updated)

    def test_timeit_merge(self):
        """
        Checks profiles have been merged and timed
        :return:
        """
        num_profiler, other1, other2 = TestColumn(), TestColumn(), TestColumn()
        mock_histogram = {
            "bin_counts": np.array([1, 1, 1, 1]),
            "bin_edges": np.array([2.0, 5.25, 8.5, 11.75, 15.0]),
        }

        (
            other1.min,
            other1.max,
            other1._biased_variance,
            other1.sum,
            other1.num_zeros,
            other1.num_negatives,
        ) = (0, 0, 0, 0, 0, 0)
        (
            other2.min,
            other2.max,
            other2._biased_variance,
            other2.sum,
            other2.num_zeros,
            other2.num_negatives,
        ) = (1, 1, 1, 1, 1, 1)

        # set auto as only histogram to merge
        other1.histogram_selection = "auto"
        other2.histogram_selection = "auto"
        other1.histogram_bin_method_names = ["auto"]
        other2.histogram_bin_method_names = ["auto"]
        other1._stored_histogram["histogram"] = mock_histogram
        other2._stored_histogram["histogram"] = mock_histogram
        other1.histogram_selection = "auto"

        time_array = [float(i) for i in range(2, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            # Validate profiles are merged and timed.
            expected = defaultdict(float, {"histogram_and_quantiles": 1.0})
            num_profiler._add_helper(other1, other2)
            self.assertEqual(expected, num_profiler.times)

    def test_timeit(self):
        """
        Checks stat properties have been timed
        :return:
        """
        num_profiler = TestColumn()

        # Dummy data to make min call
        prev_dependent_properties = {
            "mean": 0,
            "biased_variance": 0,
            "biased_skewness": 0,
        }
        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        subset_properties = {"min": 0, "match_count": 0}

        time_array = [float(i) for i in range(24, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):

            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            # Validate _get_min is timed.
            expected = defaultdict(float, {"min": 1.0})
            num_profiler._get_min(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_max is timed.
            expected["max"] = 1.0
            num_profiler._get_max(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_sum is timed.
            expected["sum"] = 1.0
            num_profiler._get_sum(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_variance is timed.
            expected["variance"] = 1.0
            num_profiler._get_variance(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_skewness is timed
            expected["skewness"] = 1.0
            num_profiler._get_skewness(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_kurtosis is timed
            expected["kurtosis"] = 1.0
            num_profiler._get_kurtosis(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_histogram_and_quantiles is timed.
            expected["histogram_and_quantiles"] = 1.0
            num_profiler._get_histogram_and_quantiles(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

    def test_histogram_bin_error(self):
        num_profiler = TestColumn()

        # Dummy data for calculating bin error
        num_profiler._stored_histogram = {
            "histogram": {"bin_edges": np.array([0.0, 4.0, 8.0, 12.0, 16.0])}
        }

        input_array = [0, 3, 5, 9, 11, 17]

        sum_error = num_profiler._histogram_bin_error(input_array)

        # Sum of errors should be difference of each input value to midpoint of bin squared
        # bin_midpoints = [2, 6, 10, 14]   ids = [1, 1, 2, 3, 3, 4]
        assert (
            sum_error
            == (2 - 0) ** 2
            + (2 - 3) ** 2
            + (6 - 5) ** 2
            + (10 - 9) ** 2
            + (10 - 11) ** 2
            + (17 - 14) ** 2
        )

        # Max value test
        input_array = [sys.float_info.max, 1.2e308, 1.3e308, 1.5e308]

        num_profiler._stored_histogram = {
            "histogram": {"bin_edges": np.array([1e308, 1.2e308, 1.4e308, 1.6e308])}
        }

        sum_error = num_profiler._histogram_bin_error(input_array)

        assert sum_error == np.inf

        # Min value test
        input_array = [sys.float_info.min, -1.2e308, -1.3e308, -1.5e308]

        num_profiler._stored_histogram = {
            "histogram": {"bin_edges": np.array([-1.6e308, -1.4e308, -1.2e308, -1e308])}
        }

        sum_error = num_profiler._histogram_bin_error(input_array)

        assert sum_error == np.inf

    def test_get_best_histogram_profile(self):
        num_profiler = TestColumn()

        num_profiler._histogram_for_profile = mock.MagicMock(
            side_effect=[("hist_1", 3), ("hist_2", 2), ("hist_3", 1)]
        )

        num_profiler.histogram_selection = None

        num_profiler.histogram_methods = {
            "method_1": {
                "total_loss": 0,
                "current_loss": 0,
                "histogram": None,
                "suggested_bin_count": 3,
            },
            "method_2": {
                "total_loss": 0,
                "current_loss": 0,
                "histogram": None,
                "suggested_bin_count": 3,
            },
            "method_3": {
                "total_loss": 0,
                "current_loss": 0,
                "histogram": None,
                "suggested_bin_count": 3,
            },
        }

        best_histogram = num_profiler._get_best_histogram_for_profile()

        assert best_histogram == "hist_3"

    def test_get_best_histogram_profile_infinite_loss(self):
        num_profiler = TestColumn()

        num_profiler._histogram_for_profile = mock.MagicMock(return_value=("hist_1", 3))

        num_profiler.histogram_selection = None

        num_profiler.histogram_methods = {
            "method_1": {
                "total_loss": np.inf,
                "current_loss": np.inf,
                "histogram": None,
                "suggested_bin_count": 3,
            },
        }

        best_histogram = num_profiler._get_best_histogram_for_profile()

        assert best_histogram == "hist_1"

    def test_get_percentile_median(self):
        num_profiler = TestColumn()
        # Dummy data for calculating bin error
        num_profiler._stored_histogram = {
            "histogram": {
                "bin_counts": np.array([1, 2, 0, 2, 1]),
                "bin_edges": np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0]),
            }
        }
        median = NumericStatsMixin._get_percentile(num_profiler, percentiles=[50, 50])
        self.assertListEqual([10, 10], median)

    def test_num_zeros(self):
        num_profiler = TestColumn()

        # Dummy data to make num_zeros call
        prev_dependent_properties = {"mean": 0}
        subset_properties = {"num_zeros": 0}

        df_series = pd.Series([])
        num_profiler._get_num_zeros(
            df_series, prev_dependent_properties, subset_properties
        )
        self.assertEqual(subset_properties["num_zeros"], 0)

        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        num_profiler._get_num_zeros(
            df_series, prev_dependent_properties, subset_properties
        )
        self.assertEqual(subset_properties["num_zeros"], 5)

        data = np.array([000.0, 0.00, 0.000, 1.11234, 0, -1])
        df_series = pd.Series(data)
        num_profiler._get_num_zeros(
            df_series, prev_dependent_properties, subset_properties
        )
        self.assertEqual(subset_properties["num_zeros"], 4)

    def test_num_negatives(self):
        num_profiler = TestColumn()

        # Dummy data to make num_negatives call
        prev_dependent_properties = {"mean": 0}
        subset_properties = {"num_negatives": 0}

        df_series = pd.Series([])
        num_profiler._get_num_negatives(
            df_series, prev_dependent_properties, subset_properties
        )
        self.assertEqual(subset_properties["num_negatives"], 0)

        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        num_profiler._get_num_negatives(
            df_series, prev_dependent_properties, subset_properties
        )
        self.assertEqual(subset_properties["num_negatives"], 0)

        data = np.array([1, 0, -0.003, -16, -1.0, -24.45])
        df_series = pd.Series(data)
        num_profiler._get_num_negatives(
            df_series, prev_dependent_properties, subset_properties
        )
        self.assertEqual(subset_properties["num_negatives"], 4)

    def test_fold_histogram(self):
        num_profiler = TestColumn()

        # the break point is at the mid point of a bin
        bin_counts = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        bin_edges = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        value = 0.35

        histogram_pos, histogram_neg = num_profiler._fold_histogram(
            bin_counts, bin_edges, value
        )
        bin_counts_pos, bin_edges_pos = histogram_pos[0], histogram_pos[1]
        bin_counts_neg, bin_edges_neg = histogram_neg[0], histogram_neg[1]
        self.assertCountEqual(
            np.round([1 / 12, 1 / 6, 1 / 6, 1 / 6], 10), np.round(bin_counts_pos, 10)
        )
        self.assertCountEqual(
            np.round([0, 0.05, 0.15, 0.25, 0.35], 10), np.round(bin_edges_pos, 10)
        )
        self.assertCountEqual(
            np.round([1 / 12, 1 / 6, 1 / 6], 10), np.round(bin_counts_neg, 10)
        )
        self.assertCountEqual(
            np.round([0, 0.05, 0.15, 0.25], 10), np.round(bin_edges_neg, 10)
        )

        # the break point is at the middle of a bin, and divides the bin
        # into two parts with the ratio 1/4, 3/4
        bin_counts = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        bin_edges = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        value = 0.325

        histogram_pos, histogram_neg = num_profiler._fold_histogram(
            bin_counts, bin_edges, value
        )
        bin_counts_pos, bin_edges_pos = histogram_pos[0], histogram_pos[1]
        bin_counts_neg, bin_edges_neg = histogram_neg[0], histogram_neg[1]
        self.assertCountEqual(
            np.round([3 / 24, 1 / 6, 1 / 6, 1 / 6], 10), np.round(bin_counts_pos, 10)
        )
        self.assertCountEqual(
            np.round([0, 0.075, 0.175, 0.275, 0.375], 10), np.round(bin_edges_pos, 10)
        )
        self.assertCountEqual(
            np.round([1 / 24, 1 / 6, 1 / 6], 10), np.round(bin_counts_neg, 10)
        )
        self.assertCountEqual(
            np.round([0, 0.025, 0.125, 0.225], 10), np.round(bin_edges_neg, 10)
        )

        # the break point is at the edge of a bin
        bin_counts = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        bin_edges = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        value = 0.3

        histogram_pos, histogram_neg = num_profiler._fold_histogram(
            bin_counts, bin_edges, value
        )
        bin_counts_pos, bin_edges_pos = histogram_pos[0], histogram_pos[1]
        bin_counts_neg, bin_edges_neg = histogram_neg[0], histogram_neg[1]
        self.assertCountEqual(
            np.round([1 / 6, 1 / 6, 1 / 6, 1 / 6], 10), np.round(bin_counts_pos, 10)
        )
        self.assertCountEqual(
            np.round([0, 0.1, 0.2, 0.3, 0.4], 10), np.round(bin_edges_pos, 10)
        )
        self.assertCountEqual(
            np.round([1 / 6, 1 / 6], 10), np.round(bin_counts_neg, 10)
        )
        self.assertCountEqual(np.round([0, 0.1, 0.2], 10), np.round(bin_edges_neg, 10))

    def test_timeit_num_zeros_and_negatives(self):
        """
        Checks num_zeros and num_negatives have been timed
        :return:
        """
        num_profiler = TestColumn()

        # Dummy data to make min call
        prev_dependent_properties = {"mean": 0}
        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        subset_properties = {"num_zeros": 0, "num_negatives": 0}

        time_array = [float(i) for i in range(4, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            # Validate _get_min is timed.
            expected = defaultdict(float, {"num_zeros": 1.0})
            num_profiler._get_num_zeros(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_max is timed.
            expected["num_negatives"] = 1.0
            num_profiler._get_num_negatives(
                df_series, prev_dependent_properties, subset_properties
            )
            self.assertEqual(expected, num_profiler.times)

    def test_merge_num_zeros_and_negatives(self):
        """
        Checks num_zeros and num_negatives can be merged
        :return:
        """
        num_profiler, other1, other2 = TestColumn(), TestColumn(), TestColumn()
        other1.num_zeros, other1.num_negatives = 3, 1
        other2.num_zeros, other2.num_negatives = 7, 1
        num_profiler._add_helper(other1, other2)
        self.assertEqual(num_profiler.num_zeros, 10)
        self.assertEqual(num_profiler.num_negatives, 2)

        num_profiler, other1, other2 = TestColumn(), TestColumn(), TestColumn()
        other1.num_zeros, other1.num_negatives = 0, 0
        other2.num_zeros, other2.num_negatives = 0, 0
        num_profiler._add_helper(other1, other2)
        self.assertEqual(num_profiler.num_zeros, 0)
        self.assertEqual(num_profiler.num_negatives, 0)

    def test_profile(self):
        num_profiler = TestColumn()

        mock_profile = dict(
            min=1.0,
            max=1.0,
            median=np.nan,  # default
            mode=[np.nan],  # default
            sum=1.0,
            mean=0,  # default
            variance=np.nan,  # default
            skewness=np.nan,  # default
            kurtosis=np.nan,  # default
            median_abs_deviation=np.nan,  # default
            stddev=np.nan,  # default
            histogram={
                "bin_counts": np.array([1, 1, 1]),
                "bin_edges": np.array([1.0, 2.0, 3.0, 4.0]),
            },
            quantiles={
                0: 2.0,
                1: 3.0,
                2: 4.0,
            },
            num_zeros=0,  # default
            num_negatives=0,  # default
            times=defaultdict(float),  # default
        )

        num_profiler.match_count = 0
        num_profiler.min = mock_profile["min"]
        num_profiler.max = mock_profile["max"]
        num_profiler.sum = mock_profile["sum"]
        num_profiler.histogram_selection = "auto"
        num_profiler.histogram_methods["auto"]["histogram"] = mock_profile["histogram"]
        num_profiler.quantiles = mock_profile["quantiles"]
        num_profiler.times = mock_profile["times"]

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            profile = num_profiler.profile()
            # pop out the histogram and quartiles to test separately from the
            # rest of the dict as we need comparison with some precision
            histogram = profile.pop("histogram")
            expected_histogram = mock_profile.pop("histogram")
            quartiles = profile.pop("quantiles")
            expected_quartiles = mock_profile.pop("quantiles")

            self.assertDictEqual(mock_profile, profile)
            self.assertEqual(
                expected_histogram["bin_counts"].tolist(),
                histogram["bin_counts"].tolist(),
            )
            self.assertCountEqual(
                np.round(expected_histogram["bin_edges"], 12),
                np.round(histogram["bin_edges"], 12),
            )
            self.assertAlmostEqual(expected_quartiles[0], quartiles[0])
            self.assertAlmostEqual(expected_quartiles[1], quartiles[1])
            self.assertAlmostEqual(expected_quartiles[2], quartiles[2])

    @mock.patch.multiple(
        NumericStatsMixin,
        __abstractmethods__=set(),
        _filter_properties_w_options=mock.MagicMock(
            side_effect=BaseColumnProfiler._filter_properties_w_options
        ),
        create=True,
    )
    def test_report(self):
        options = NumericalOptions()
        options.max.is_enabled = False
        options.min.is_enabled = False
        options.histogram_and_quantiles.is_enabled = False
        options.variance.is_enabled = False

        num_profiler = NumericStatsMixin(options=options)

        num_profiler.match_count = 0
        num_profiler.times = defaultdict(float)

        report = num_profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())

        for disabled_key in ["max", "min", "variance", "histogram", "quantiles"]:
            self.assertNotIn(disabled_key, report_keys)

        # test report default `remove_disabled_flag`
        # value and no NumericalOptions
        report = num_profiler.report()
        report_keys = list(report.keys())

        for disabled_key in ["max", "min", "variance", "histogram", "quantiles"]:
            self.assertIn(disabled_key, report_keys)

    def test_report_no_numerical_options(self):
        num_profiler = TestColumn()

        num_profiler.match_count = 0
        num_profiler.times = defaultdict(float)

        report = num_profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())

        # now test to see if the keys that were disabled and ensure
        # they keys are not poped without specifying `NumericalOptions`
        for disabled_key in ["max", "min", "variance", "histogram", "quantiles"]:
            self.assertIn(disabled_key, report_keys)

    def test_diff(self):
        """
        Checks _diff_helper() works appropriately.
        """

        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.min = 3
        other1.max = 4
        other1._biased_variance = 1
        other1.sum = 6
        other1.match_count = 10
        other1.median = 5
        other1.mode = [3]
        other1.median_abs_deviation = 4

        other2.min = 3
        other2.max = None
        other2._biased_variance = 9
        other2.sum = 6
        other2.match_count = 20
        other2.median = 6
        other2.mode = [2]
        other2.median_abs_deviation = 3

        # T-stat and Welch's df calculation can be found here:
        #    https://en.wikipedia.org/wiki/Welch%27s_t-test#Calculations
        # Conservative df = min(count1, count2) - 1
        # P-value is found using scipy:  (1 - CDF(abs(t-stat))) * 2
        expected_diff = {
            "min": "unchanged",
            "max": [4, None],
            "sum": "unchanged",
            "mean": 0.3,
            "median": -1,
            "mode": [[3], [], [2]],
            "median_absolute_deviation": 1,
            "variance": 10 / 9 - (9 * 20 / 19),
            "stddev": np.sqrt(10 / 9) - np.sqrt(9 * 20 / 19),
            "t-test": {
                "t-statistic": 0.3923009049186606,
                "conservative": {"df": 9, "p-value": 0.7039643545772609},
                "welch": {"df": 25.945257024943864, "p-value": 0.6980401261750298},
            },
            "psi": None,
        }

        difference = other1.diff(other2)
        self.assertDictEqual(expected_diff, difference)

        # Invalid statistics
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.min = 3
        other1.max = 4
        other1._biased_variance = np.nan  # NaN variance
        other1.sum = 6
        other1.match_count = 10
        other1.median = 5
        other1.mode = [3]
        other1.median_abs_deviation = 4

        other2.min = 3
        other2.max = None
        other2._biased_variance = 9
        other2.sum = 6
        other2.match_count = 20
        other2.median = 6
        other2.mode = [2]
        other2.median_abs_deviation = 3

        expected_diff = {
            "min": "unchanged",
            "max": [4, None],
            "sum": "unchanged",
            "mean": 0.3,
            "median": -1,
            "mode": [[3], [], [2]],
            "median_absolute_deviation": 1,
            "variance": np.nan,
            "stddev": np.nan,
            "t-test": {
                "t-statistic": None,
                "conservative": {"df": None, "p-value": None},
                "welch": {"df": None, "p-value": None},
            },
            "psi": None,
        }
        expected_var = expected_diff.pop("variance")
        expected_stddev = expected_diff.pop("stddev")
        with self.assertWarns(
            RuntimeWarning,
            msg="Null value(s) found in mean and/or variance values. "
            "T-test cannot be performed",
        ):
            difference = other1.diff(other2)
        var = difference.pop("variance")
        stddev = difference.pop("stddev")
        self.assertDictEqual(expected_diff, difference)
        self.assertTrue(np.isnan([expected_var, var, expected_stddev, stddev]).all())

        # Insufficient match count
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.min = 3
        other1.max = 4
        other1._biased_variance = 1
        other1.sum = 6
        other1.match_count = 10
        other1.median = 5
        other1.mode = [3]
        other1.median_abs_deviation = 4

        other2.min = 3
        other2.max = None
        other2._biased_variance = 9
        other2.sum = 6
        other2.match_count = 1  # Insufficient count
        other2.median = 6
        other2.mode = [2]
        other2.median_abs_deviation = 3

        expected_diff = {
            "min": "unchanged",
            "max": [4, None],
            "sum": "unchanged",
            "mean": -5.4,
            "median": -1,
            "mode": [[3], [], [2]],
            "median_absolute_deviation": 1,
            "variance": np.nan,
            "stddev": np.nan,
            "t-test": {
                "t-statistic": None,
                "conservative": {"df": None, "p-value": None},
                "welch": {"df": None, "p-value": None},
            },
            "psi": None,
        }
        expected_var = expected_diff.pop("variance")
        expected_stddev = expected_diff.pop("stddev")
        with self.assertWarns(
            RuntimeWarning,
            msg="Insufficient sample size. T-test cannot be performed.",
        ):
            difference = other1.diff(other2)
        var = difference.pop("variance")
        stddev = difference.pop("stddev")
        self.assertDictEqual(expected_diff, difference)
        self.assertTrue(np.isnan([expected_var, var, expected_stddev, stddev]).all())

        # Constant values
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.min = 3
        other1.max = 4
        other1._biased_variance = 0  # constant value has 0 variance
        other1.sum = 6
        other1.match_count = 3
        other1.median = 5
        other1.mode = [3]
        other1.median_abs_deviation = 4

        other2.min = 3
        other2.max = None
        other2._biased_variance = 0  # constant value has 0 variance
        other2.sum = 6
        other2.match_count = 3
        other2.median = 6
        other2.mode = [2]
        other2.median_abs_deviation = 3

        expected_diff = {
            "min": "unchanged",
            "max": [4, None],
            "sum": "unchanged",
            "mean": "unchanged",
            "median": -1,
            "mode": [[3], [], [2]],
            "median_absolute_deviation": 1,
            "variance": 0,
            "stddev": 0,
            "t-test": {
                "t-statistic": None,
                "conservative": {"df": None, "p-value": None},
                "welch": {"df": None, "p-value": None},
            },
            "psi": None,
        }
        expected_var = expected_diff.pop("variance")
        expected_stddev = expected_diff.pop("stddev")
        with self.assertWarns(
            RuntimeWarning,
            msg="Data were essentially constant. T-test cannot be performed.",
        ):
            difference = other1.diff(other2)
        var = difference.pop("variance")
        stddev = difference.pop("stddev")
        self.assertDictEqual(expected_diff, difference)

        # Small p-value
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.min = 3
        other1.max = 4
        other1._biased_variance = 1
        other1.sum = 6
        other1.match_count = 10
        other1.median = 5
        other1.mode = [3]
        other1.median_abs_deviation = 4

        other2.min = 3
        other2.max = None
        other2._biased_variance = 9
        other2.sum = 60
        other2.match_count = 20
        other2.median = 6
        other2.mode = [2]
        other2.median_abs_deviation = 3

        expected_diff = {
            "min": "unchanged",
            "max": [4, None],
            "sum": -54,
            "mean": -2.4,
            "median": -1,
            "mode": [[3], [], [2]],
            "median_absolute_deviation": 1,
            "variance": 10 / 9 - (9 * 20 / 19),
            "stddev": np.sqrt(10 / 9) - np.sqrt(9 * 20 / 19),
            "t-test": {
                "t-statistic": -3.138407239349285,
                "conservative": {"df": 9, "p-value": 0.011958658754358975},
                "welch": {"df": 25.945257024943864, "p-value": 0.004201616692122823},
            },
            "psi": None,
        }
        difference = other1.diff(other2)
        self.assertDictEqual(expected_diff, difference)

        # Assert type error is properly called
        with self.assertRaises(TypeError) as exc:
            other1.diff("Inproper input")
        self.assertEqual(
            str(exc.exception),
            "Unsupported operand type(s) for diff: 'TestColumnWProps' and" " 'str'",
        )

        # PSI same distribution test
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.match_count = 55
        other1._stored_histogram = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": 10,
            "histogram": {
                "bin_counts": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "bin_edges": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            },
        }

        other2.match_count = 550
        other2._stored_histogram = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": 10,
            "histogram": {
                "bin_counts": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 10,
                "bin_edges": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            },
        }

        expected_psi_value = 0
        psi_value = other1._calculate_psi(
            self_match_count=other1.match_count,
            self_histogram=other1._stored_histogram["histogram"],
            other_match_count=other2.match_count,
            other_histogram=other2._stored_histogram["histogram"],
        )
        self.assertEquals(expected_psi_value, psi_value)

        # PSI min_min_edge == max_max_edge
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.match_count = 10
        other1._stored_histogram = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": 10,
            "histogram": {"bin_counts": np.array([10]), "bin_edges": np.array([1, 1])},
        }

        other2.match_count = 20
        other2._stored_histogram = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": 10,
            "histogram": {"bin_counts": np.array([20]), "bin_edges": np.array([1, 1])},
        }

        expected_psi_value = 0
        psi_value = other1._calculate_psi(
            self_match_count=other1.match_count,
            self_histogram=other1._stored_histogram["histogram"],
            other_match_count=other2.match_count,
            other_histogram=other2._stored_histogram["histogram"],
        )
        self.assertEquals(expected_psi_value, psi_value)

        # PSI regen other / not self
        other1, other2 = TestColumnWProps(), TestColumnWProps()
        other1.match_count = 55
        other1._stored_histogram = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": 10,
            "histogram": {
                "bin_counts": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "bin_edges": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            },
        }

        other2.match_count = 20
        other2._stored_histogram = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": 10,
            "histogram": {
                "bin_counts": np.array([5, 5, 10]),
                "bin_edges": np.array([1, 3, 5, 7]),
            },
        }

        expected_psi_value = 0.6617899380349177
        psi_value = other1._calculate_psi(
            self_match_count=other1.match_count,
            self_histogram=other1._stored_histogram["histogram"],
            other_match_count=other2.match_count,
            other_histogram=other2._stored_histogram["histogram"],
        )
        self.assertEquals(expected_psi_value, psi_value)

        # PSI regen self / not other
        expected_psi_value = 0.6617899380349177
        psi_value = other1._calculate_psi(
            self_match_count=other2.match_count,
            self_histogram=other2._stored_histogram["histogram"],
            other_match_count=other1.match_count,
            other_histogram=other1._stored_histogram["histogram"],
        )
        self.assertEquals(expected_psi_value, psi_value)
