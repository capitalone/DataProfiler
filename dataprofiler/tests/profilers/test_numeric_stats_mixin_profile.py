import os
import unittest
from unittest import mock
from collections import defaultdict

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
        num_profiler.variance = num_profiler._update_variance(
            mean1, var1, count1)
        self.assertEqual(var1, num_profiler.variance)
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)

        # test streaming update variance with new data
        data2 = [-5.0, 5.0, 11.0]
        mean2 = (-5.0 + 5.0 + 11.0) / 3
        var2 = ((-5.0 - mean2) ** 2 + (5.0 - mean2)
                ** 2 + (11.0 - mean2) ** 2) / 2
        count2 = len(data2)
        num_profiler.variance = num_profiler._update_variance(
            mean2, var2, count2)
        var_from_profile_updated = num_profiler.variance

        data_all = [-5.0, 5.0, 11.0, -3.0, 2.0, 11.0]
        mean_all = (-5.0 + 5.0 + 11.0 - 3.0 + 2.0 + 11.0) / 6
        var_all = ((-5.0 - mean_all) ** 2 + (5.0 - mean_all) ** 2 + \
                   (11.0 - mean_all) ** 2 + (-3.0 - mean_all) ** 2 + \
                   (2.0 - mean_all) ** 2 + (11.0 - mean_all) ** 2) / 5

        self.assertEqual(var_all, var_from_profile_updated)

    def test_update_variance_with_varying_data_length(self):
        """
        Checks update variance
        :return:
        """
        # empty data
        data1 = []
        mean1, var1, count1 = 0, 0, 0

        num_profiler = TestColumn()
        num_profiler.variance = num_profiler._update_variance(
            mean1, var1, count1)
        self.assertEqual(var1, num_profiler.variance)

        # data with 1 element
        data2 = [5.0]
        mean2, var2, count2 = 5.0, 0, 1

        num_profiler = TestColumn()
        num_profiler.variance = num_profiler._update_variance(
            mean2, var2, count2)
        self.assertEqual(var2, num_profiler.variance)

        # data with multiple elements
        data3 = [-5.0, 5.0, 11.0, -11.0]
        mean3, count3 = 0, 4
        var3 = ((-5.0 - mean3) ** 2 + (5.0 - mean3) ** 2 +
                (11.0 - mean3) ** 2 + (-11.0 - mean3) ** 2) / 3

        num_profiler = TestColumn()
        num_profiler.variance = num_profiler._update_variance(
            mean3, var3, count3)
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
        num_profiler.variance = num_profiler._update_variance(
            mean1, var1, count1)
        self.assertEqual(var1, num_profiler.variance)
        num_profiler.match_count = count1
        num_profiler.sum = sum(data1)

        # test adding data which would not have anything
        # data + empty
        mean2, var2, count2 = 0, 0, 0
        num_profiler.variance = num_profiler._update_variance(
            mean2, var2, count2)
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

        other1.min, other1.max, other1.variance, other1.sum = 0, 0, 0, 0
        other2.min, other2.max, other2.variance, other2.sum = 1, 1, 1, 1

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
        prev_dependent_properties = {"mean": 0}
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

            # Validate _get_histogram_and_quantiles is timed.
            expected['histogram_and_quantiles'] = 1.0
            num_profiler._get_histogram_and_quantiles(
                df_series, prev_dependent_properties, subset_properties)
            self.assertEqual(expected, num_profiler.times)
