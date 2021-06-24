import os
import unittest
from unittest import mock
from collections import defaultdict
import sys

import pandas as pd
import numpy as np

from dataprofiler.profilers import NumericStatsMixin
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


class TestNumericStatsMixin(unittest.TestCase):

    @mock.patch.multiple(NumericStatsMixin, __abstractmethods__=set(),
                         _filter_properties_w_options=mock.MagicMock(
                             return_value=None),
                         create=True)
    def test_base(self):

        # validate requires NumericalOptions
        with self.assertRaisesRegex(ValueError,
                                    "NumericalStatsMixin parameter 'options' "
                                    "must be of type NumericalOptions."):
            profile = NumericStatsMixin(options='bad options')

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
        true_asserts = [1.3, 1.345, -1.3, 0.03, 0.0, -0.0, 1,  # numeric values
                        float("nan"), np.nan,  # nan values
                        "1.3", "nan"  # strings
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
        true_asserts = [1, 1345, -13, 0, -0,  # numeric values
                        "1"  # strings
                        ]
        for assert_val in true_asserts:
            self.assertTrue(NumericStatsMixin.is_int(assert_val))

        false_asserts = [1.3,  # float
                         float("nan"), np.nan,  # nan value
                         "nan", "1a", "abc", "", "1.3"  # strings
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
        var1 = ((-3.0 - mean1) ** 2 + (2.0 - mean1)
                ** 2 + (11.0 - mean1) ** 2) / 2
        count1 = len(data1)
        num_profiler._biased_variance = num_profiler._update_variance(
            mean1, var1 * 2 / 3, count1)
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)
        self.assertAlmostEqual(var1, num_profiler.variance)

        # test streaming update variance with new data
        data2 = [-5.0, 5.0, 11.0]
        mean2 = (-5.0 + 5.0 + 11.0) / 3
        var2 = ((-5.0 - mean2) ** 2 + (5.0 - mean2)
                ** 2 + (11.0 - mean2) ** 2) / 2
        count2 = len(data2)
        num_profiler._biased_variance = num_profiler._update_variance(
            mean2, var2 * 2 / 3, count2)
        num_profiler.match_count += count2
        num_profiler.sum += sum(data2)
        var_from_profile_updated = num_profiler.variance

        data_all = [-5.0, 5.0, 11.0, -3.0, 2.0, 11.0]
        mean_all = (-5.0 + 5.0 + 11.0 - 3.0 + 2.0 + 11.0) / 6
        var_all = ((-5.0 - mean_all) ** 2 + (5.0 - mean_all) ** 2 + \
                   (11.0 - mean_all) ** 2 + (-3.0 - mean_all) ** 2 + \
                   (2.0 - mean_all) ** 2 + (11.0 - mean_all) ** 2) / 5

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
            mean1, var1, count1)
        num_profiler.match_count = count1
        num_profiler.sum = 0
        self.assertTrue(num_profiler.variance is np.nan)

        # data with 1 element
        data2 = [5.0]
        mean2, var2, count2 = 5.0, 0, 1

        num_profiler = TestColumn()
        num_profiler._biased_variance = num_profiler._update_variance(
            mean2, var2, count2)
        num_profiler.match_count += count2
        num_profiler.sum += 5.0
        self.assertTrue(num_profiler.variance is np.nan)

        # data with multiple elements
        data3 = [-5.0, 5.0, 11.0, -11.0]
        mean3, count3 = 0, 4
        var3 = ((-5.0 - mean3) ** 2 + (5.0 - mean3) ** 2 +
                (11.0 - mean3) ** 2 + (-11.0 - mean3) ** 2) / 3

        num_profiler = TestColumn()
        num_profiler._biased_variance = num_profiler._update_variance(
            mean3, var3 * 3 / 4, count3)
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
        var1 = ((-3.0 - mean1) ** 2 + (2.0 - mean1)
                ** 2 + (11.0 - mean1) ** 2) / 2
        count1 = len(data1)
        num_profiler._biased_variance = num_profiler._update_variance(
            mean1, var1 * 2 / 3, count1)
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)
        self.assertEqual(var1, num_profiler.variance)

        # test adding data which would not have anything
        # data + empty
        mean2, var2, count2 = 0, 0, 0
        num_profiler._biased_variance = num_profiler._update_variance(
            mean2, var2, count2)
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
            'bin_counts': np.array([1, 1, 1, 1]),
            'bin_edges': np.array([2., 5.25, 8.5, 11.75, 15.])
        }

        other1.min, other1.max, other1._biased_variance, other1.sum, \
        other1.num_zeros, other1.num_negatives = 0, 0, 0, 0, 0, 0
        other2.min, other2.max, other2._biased_variance, other2.sum, \
        other2.num_zeros, other2.num_negatives = 1, 1, 1, 1, 1, 1

        # set auto as only histogram to merge
        other1.histogram_selection = "auto"
        other2.histogram_selection = "auto"
        other1.histogram_bin_method_names = ['auto']
        other2.histogram_bin_method_names = ['auto']
        other1._stored_histogram['histogram'] = mock_histogram
        other2._stored_histogram['histogram'] = mock_histogram
        other1.histogram_selection = 'auto'

        time_array = [float(i) for i in range(2, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            # Validate profiles are merged and timed.
            expected = defaultdict(float, {'histogram_and_quantiles': 1.0})
            num_profiler._add_helper(other1, other2)
            self.assertEqual(expected, num_profiler.times)

    def test_timeit(self):
        """
        Checks stat properties have been timed
        :return:
        """
        num_profiler = TestColumn()

        # Dummy data to make min call
        prev_dependent_properties = {"mean": 0,
                                     "biased_variance": 0,
                                     "biased_skewness": 0}
        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        subset_properties = {"min": 0, "match_count": 0}

        time_array = [float(i) for i in range(24, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):

            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            # Validate _get_min is timed.
            expected = defaultdict(float, {'min': 1.0})
            num_profiler._get_min(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_max is timed.
            expected['max'] = 1.0
            num_profiler._get_max(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_sum is timed.
            expected['sum'] = 1.0
            num_profiler._get_sum(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_variance is timed.
            expected['variance'] = 1.0
            num_profiler._get_variance(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_skewness is timed
            expected['skewness'] = 1.0
            num_profiler._get_skewness(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_kurtosis is timed
            expected['kurtosis'] = 1.0
            num_profiler._get_kurtosis(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_histogram_and_quantiles is timed.
            expected['histogram_and_quantiles'] = 1.0
            num_profiler._get_histogram_and_quantiles(
                df_series, prev_dependent_properties, subset_properties)
            self.assertEqual(expected, num_profiler.times)

    def test_histogram_bin_error(self):
        num_profiler = TestColumn()

        # Dummy data for calculating bin error
        num_profiler._stored_histogram = {
            "histogram": {
                "bin_edges": np.array([0.0, 4.0, 8.0, 12.0, 16.0])
            }
        }

        input_array = [0, 3, 5, 9, 11, 17]

        sum_error = num_profiler._histogram_bin_error(input_array)

        # Sum of errors should be difference of each input value to midpoint of bin squared
        # bin_midpoints = [2, 6, 10, 14]   ids = [1, 1, 2, 3, 3, 4]
        assert sum_error == (2-0)**2 + (2-3)**2 + (6-5)**2 + \
               (10-9)**2 + (10-11)**2 + (17-14)**2

        # Max value test
        input_array = [sys.float_info.max, 1.2e308, 1.3e308, 1.5e308]

        num_profiler._stored_histogram = {
            "histogram": {
                "bin_edges": np.array([1e308, 1.2e308, 1.4e308, 1.6e308])
            }
        }

        sum_error = num_profiler._histogram_bin_error(input_array)

        assert sum_error == np.inf

        # Min value test
        input_array = [sys.float_info.min, -1.2e308, -1.3e308, -1.5e308]

        num_profiler._stored_histogram = {
            "histogram": {
                "bin_edges": np.array([-1.6e308, -1.4e308, -1.2e308, -1e308])
            }
        }

        sum_error = num_profiler._histogram_bin_error(input_array)

        assert sum_error == np.inf

    def test_get_best_histogram_profile(self):
        num_profiler = TestColumn()

        num_profiler._histogram_for_profile = mock.MagicMock(side_effect=[
            ("hist_1", 3),
            ("hist_2", 2),
            ("hist_3", 1)
        ])

        num_profiler.histogram_selection = None

        num_profiler.histogram_methods = {
            'method_1': {
                'total_loss': 0,
                'current_loss': 0,
                'histogram': None,
                'suggested_bin_count': 3
            },
            'method_2': {
                'total_loss': 0,
                'current_loss': 0,
                'histogram': None,
                'suggested_bin_count': 3
            },
            'method_3': {
                'total_loss': 0,
                'current_loss': 0,
                'histogram': None,
                'suggested_bin_count': 3
            }
        }

        best_histogram = num_profiler._get_best_histogram_for_profile()

        assert best_histogram == "hist_3"

    def test_get_best_histogram_profile_infinite_loss(self):
        num_profiler = TestColumn()

        num_profiler._histogram_for_profile = mock.MagicMock(return_value=("hist_1", 3))

        num_profiler.histogram_selection = None

        num_profiler.histogram_methods = {
            'method_1': {
                'total_loss': np.inf,
                'current_loss': np.inf,
                'histogram': None,
                'suggested_bin_count': 3
            },
        }

        best_histogram = num_profiler._get_best_histogram_for_profile()

        assert best_histogram == "hist_1"

    def test_num_zeros(self):
        num_profiler = TestColumn()

        # Dummy data to make num_zeros call
        prev_dependent_properties = {"mean": 0}
        subset_properties = {"num_zeros": 0}

        df_series = pd.Series([])
        num_profiler._get_num_zeros(df_series, prev_dependent_properties,
                                    subset_properties)
        self.assertEqual(subset_properties["num_zeros"], 0)

        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        num_profiler._get_num_zeros(df_series, prev_dependent_properties,
                                    subset_properties)
        self.assertEqual(subset_properties["num_zeros"], 5)

        data = np.array([000., 0.00, .000, 1.11234, 0, -1])
        df_series = pd.Series(data)
        num_profiler._get_num_zeros(df_series, prev_dependent_properties,
                                    subset_properties)
        self.assertEqual(subset_properties["num_zeros"], 4)

    def test_num_negatives(self):
        num_profiler = TestColumn()

        # Dummy data to make num_negatives call
        prev_dependent_properties = {"mean": 0}
        subset_properties = {"num_negatives": 0}

        df_series = pd.Series([])
        num_profiler._get_num_negatives(df_series, prev_dependent_properties,
                                        subset_properties)
        self.assertEqual(subset_properties["num_negatives"], 0)

        data = np.array([0, 0, 0, 0, 0])
        df_series = pd.Series(data)
        num_profiler._get_num_negatives(df_series, prev_dependent_properties,
                                        subset_properties)
        self.assertEqual(subset_properties["num_negatives"], 0)

        data = np.array([1, 0, -.003, -16, -1., -24.45])
        df_series = pd.Series(data)
        num_profiler._get_num_negatives(df_series, prev_dependent_properties,
                                        subset_properties)
        self.assertEqual(subset_properties["num_negatives"], 4)

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
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), num_profiler.times)

            # Validate _get_min is timed.
            expected = defaultdict(float, {'num_zeros': 1.0})
            num_profiler._get_num_zeros(
                df_series,
                prev_dependent_properties,
                subset_properties)
            self.assertEqual(expected, num_profiler.times)

            # Validate _get_max is timed.
            expected['num_negatives'] = 1.0
            num_profiler._get_num_negatives(
                df_series,
                prev_dependent_properties,
                subset_properties)
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

    def test_diff(self):
        """
        Checks _diff_helper() works appropriately.
        """
        other1, other2 = TestColumn(), TestColumn()
        other1.min = 3
        other1.max = 4
        other1._biased_variance = 1
        other1.sum = 6
        other1.match_count = 10
        
        other2.min = 3
        other2.max = None
        other2._biased_variance = 9
        other2.sum = 6
        other2.match_count = 20
        
        expected_diff = {
            'min': 'unchanged',
            'max': [4, None],
            'sum': 'unchanged',
            'mean': 0.3,
            'variance': -8.362573099415204,
            'stddev': -2.0238425028660023
        }
        self.assertDictEqual(expected_diff, other1.diff(other2))
        
        # Assert type error is properly called
        with self.assertRaises(TypeError) as exc:
            other1.diff("Inproper input")
        self.assertEqual(str(exc.exception),
                         "Unsupported operand type(s) for diff: 'TestColumn' and"
                         " 'str'")
