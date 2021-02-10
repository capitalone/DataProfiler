import unittest
import os
from collections import defaultdict
from unittest import mock

import pandas as pd
import numpy as np

from data_profiler.profilers import FloatColumn
from data_profiler.profilers.profiler_options import FloatOptions


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestFloatColumn(unittest.TestCase):

    def test_base_case(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)

        self.assertEqual(profiler.match_count, 0)
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)
        self.assertEqual(profiler.mean, 0)
        self.assertEqual(profiler.variance, 0)
        self.assertTrue(profiler.stddev is np.nan)
        self.assertIsNone(profiler.histogram_selection)
        self.assertEqual(len(profiler.quantiles), 1000)
        self.assertIsNone(profiler.data_type_ratio)

    def test_single_data_variance_case(self):
        data = pd.Series([1.5]).apply(str)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.match_count, 1.0)
        self.assertEqual(profiler.mean, 1.5)
        self.assertEqual(profiler.variance, 0.0)
        
        data = pd.Series([2.5]).apply(str)
        profiler.update(data)
        self.assertEqual(profiler.match_count, 2)
        self.assertEqual(profiler.mean, 2.0)
        self.assertEqual(profiler.variance, 0.5)

    def test_profiled_precision(self):
        """
        Checks whether the precision for the profiler is correct.
        :return:
        """
        df_1 = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1]).apply(str)
        df_2 = pd.Series([0.11, 0.11, 0.12, 2.11]).apply(str)
        df_3 = pd.Series([4.114, 3.161, 2.512, 2.131]).apply(str)
        df_mix = pd.Series([4.1, 3., 2.52, 2.13143]).apply(str)

        float_profiler = FloatColumn(df_1.name)
        float_profiler.update(df_1)
        self.assertEqual(1, float_profiler.precision)

        float_profiler.update(df_2)
        self.assertEqual(2, float_profiler.precision)

        float_profiler.update(df_3)
        self.assertEqual(3, float_profiler.precision)

        float_profiler.update(df_2)
        self.assertEqual(3, float_profiler.precision)

        float_profiler.update(df_mix)
        self.assertEqual(5, float_profiler.precision)

        float_profiler = FloatColumn(df_mix.name)
        float_profiler.update(df_mix)
        self.assertEqual(5, float_profiler.precision)

    def test_profiled_min(self):
        # test with multiple values
        data = np.linspace(-5, 5, 11)
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)
        profiler.update(df[1:])
        self.assertEqual(profiler.min, -4)

        profiler.update(df)
        self.assertEqual(profiler.min, -5)

        profiler.update(pd.Series(['-4']))
        self.assertEqual(profiler.min, -5)

        # empty data
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.min, None)

        # data with None value
        df = pd.Series([2.0, 3.0, None, np.nan]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2.0)

        # data with one value
        df = pd.Series([2.0]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2.0)

        # data with unique value
        df = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2.0)

        # data with unique value as zero
        df = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 0.0)

    def test_profiled_max(self):
        data = np.linspace(-5, 5, 11)
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)
        profiler.update(df[:-1])
        self.assertEqual(profiler.max, 4)

        profiler.update(df)
        self.assertEqual(profiler.max, 5)

        profiler.update(pd.Series(['4']))
        self.assertEqual(profiler.max, 5)

        # empty data
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.max, None)

        # data with None value
        df = pd.Series([2.0, 3.0, None, np.nan]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 3.0)

        # data with one value
        df = pd.Series([2.0]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 2.0)

        # data with unique value
        df = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 2.0)

        # data with unique value as zero
        df = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 0.0)

    def test_profiled_mean_and_variance(self):
        """
        Checks the mean and variance of profiled numerical columns.
        :return:
        """

        def mean(df):
            total = 0
            for item in df:
                total += item
            return total / len(df)

        def var(df):
            var = 0
            mean_df = mean(df)
            for item in df:
                var += (item - mean_df) ** 2
            return var / (len(df) - 1)

        def batch_variance(mean_a, var_a, count_a, mean_b, var_b, count_b):
            delta = mean_b - mean_a
            m_a = var_a * (count_a - 1)
            m_b = var_b * (count_b - 1)
            M2 = m_a + m_b + delta ** 2 * count_a * count_b / (
                    count_a + count_b)
            return M2 / (count_a + count_b - 1)

        data = np.linspace(-5, 5, 11).tolist()
        df1 = pd.Series(data)

        data = np.linspace(-3, 2, 11).tolist()
        df2 = pd.Series(data)

        data = np.full((10,), 1)
        df3 = pd.Series(data)

        num_profiler = FloatColumn(df1.name)
        num_profiler.update(df1.apply(str))

        self.assertEqual(mean(df1), num_profiler.mean)
        self.assertEqual(var(df1), num_profiler.variance)
        self.assertEqual(np.sqrt(var(df1)), num_profiler.stddev)

        variance = batch_variance(
            mean_a=num_profiler.mean, var_a=num_profiler.variance,
            count_a=num_profiler.match_count,
            mean_b=mean(df2), var_b=var(df2), count_b=df2.count()
        )
        num_profiler.update(df2.apply(str))
        df = pd.concat([df1, df2])
        self.assertEqual(mean(df), num_profiler.mean)
        self.assertEqual(variance, num_profiler.variance)
        self.assertEqual(np.sqrt(variance), num_profiler.stddev)

        variance = batch_variance(
            mean_a=num_profiler.mean, var_a=num_profiler.variance,
            count_a=num_profiler.match_count,
            mean_b=mean(df3), var_b=var(df3), count_b=df3.count()
        )
        num_profiler.update(df3.apply(str))

        df = pd.concat([df1, df2, df3])
        self.assertEqual(mean(df), num_profiler.mean)
        self.assertEqual(variance, num_profiler.variance)
        self.assertEqual(np.sqrt(variance), num_profiler.stddev)

    def test_null_values_for_histogram(self):
        data = pd.Series(['-inf', 'inf'])
        profiler = FloatColumn(data.name)
        profiler.update(data)

        profile = profiler.profile
        histogram = profile['histogram']

        self.assertEqual(histogram['bin_counts'], None)
        self.assertEqual(histogram['bin_edges'], None)

        data = pd.Series(['-2', '-1', '1', '2', '-inf', 'inf'])
        profiler = FloatColumn(data.name)
        profiler.update(data)

        profile = profiler.profile
        histogram = profile['histogram']

        expected_histogram = {
            'bin_counts': np.array([1, 1, 0, 2]),
            'bin_edges': np.array([-2., -1.,  0.,  1.,  2.]),
        }

        self.assertCountEqual(histogram['bin_counts'], expected_histogram['bin_counts'])
        self.assertCountEqual(histogram['bin_edges'], expected_histogram['bin_edges'])

    def test_profiled_histogram(self):
        """
        Checks the histogram of profiled numerical columns.
        :return:
        """

        list_data_test = []
        # this data has 4 bins, range of 3
        # with equal bin size, each bin has the width of 0.75
        df1 = pd.Series(["1.0", "2.0", "3.0", "4.0"])
        expected_histogram1 = {
            'bin_counts': np.array([1, 1, 1, 1]),
            'bin_edges': np.array([1.0, 1.75, 2.5, 3.25, 4.0]),
        }
        list_data_test.append([df1, expected_histogram1])

        # this data has 4 bins, range of 12
        # with equal bin size, each bin has the width of 3.0
        df2 = pd.Series(["1.0", "5.0", "8.0", "13.0"])
        expected_histogram2 = {
            'bin_counts': np.array([1, 1, 1, 1]),
            'bin_edges': np.array([1.0, 4.0, 7.0, 10.0, 13.0]),
        }
        list_data_test.append([df2, expected_histogram2])

        # this data has 3 bins, range of 3
        # with equal bin size, each bin has the width of 1
        df3 = pd.Series(["1.0", "1.0", "3.0", "4.0"])
        expected_histogram3 = {
            'bin_counts': np.array([2, 0, 2]),
            'bin_edges': np.array([1.0, 2.0, 3.0, 4.0]),
        }
        list_data_test.append([df3, expected_histogram3])

        # this data has only one unique value, not overflow
        df4 = pd.Series([-10.0, -10.0, -10.0]).apply(str)
        expected_histogram4 = {
            'bin_counts': np.array([3]),
            'bin_edges': np.array([-10.0, -10.0]),
        }
        list_data_test.append([df4, expected_histogram4])

        # this data has only one unique value, overflow
        df5 = pd.Series([-10.0 ** 20]).apply(str)
        expected_histogram5 = {
            'bin_counts': np.array([1]),
            'bin_edges': np.array([-10.0 ** 20, -10.0 ** 20]),
        }
        list_data_test.append([df5, expected_histogram5])

        for i, (df, expected_histogram) in enumerate(list_data_test):
            profiler = FloatColumn(df.name)
            profiler.update(df)

            profile = profiler.profile
            histogram = profile['histogram']

            self.assertCountEqual(expected_histogram['bin_counts'],
                                  histogram['bin_counts'])
            if i != 4:
                self.assertCountEqual(np.round(expected_histogram['bin_edges'], 12),
                                  np.round(histogram['bin_edges'], 12))
            else: # for overflow, dont use np.round
                self.assertCountEqual(expected_histogram['bin_edges'],
                                      histogram['bin_edges'])

    def test_histogram_with_varying_number_of_bin(self):
        """
        Checks the histogram with large number of bins
        """
        # this data use number of bins less than the max limit
        df1 = pd.Series([1, 2, 3, 4]).apply(str)
        profiler1 = FloatColumn(df1.name)
        profiler1.max_histogram_bin = 50
        profiler1.update(df1)
        num_bins = len(profiler1.profile['histogram']['bin_counts'])
        self.assertEqual(num_bins, 4)

        # this data uses large number of bins, which will be set to
        # the max limit
        df2 = pd.Series([3.195103249264023e+18, 9999995.0, 9999999.0,
                         0.0, -10 ** 10]).apply(str)
        profiler2 = FloatColumn(df2.name)
        profiler2.max_histogram_bin = 50
        profiler2.update(df2)
        num_bins = len(profiler2.profile['histogram']['bin_counts'])
        self.assertEqual(num_bins, 50)

        # max number of bin is increased to 10000
        profiler2 = FloatColumn(df2.name)
        profiler2.max_histogram_bin = 10000
        profiler2.update(df2)
        num_bins = len(profiler2.profile['histogram']['bin_counts'])
        self.assertEqual(num_bins, 10000)

    def test_estimate_stats_from_histogram(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler.histogram_methods['auto']['histogram']['bin_counts'] = \
            np.array([1, 2, 1])
        profiler.histogram_methods['auto']['histogram']['bin_edges'] = \
            np.array([1.0, 3.0, 5.0, 7.0])
        expected_mean = (2.0 * 1 + 4.0 * 2 + 6.0 * 1) / 4
        expected_var = (1 * (2.0 - expected_mean) ** 2
                        + 2 * (4.0 - expected_mean) ** 2
                        + 1 * (6.0 - expected_mean) ** 2) / 4
        expected_std = np.sqrt(expected_var)
        est_mean, est_var, est_std = profiler._estimate_stats_from_histogram(
            method='auto')
        self.assertEqual(expected_mean, est_mean)
        self.assertEqual(expected_var, est_var)
        self.assertEqual(expected_std, est_std)

    def test_total_histogram_bin_variance(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler.histogram_methods['auto']['histogram']['bin_counts'] = \
            np.array([3, 2, 1])
        profiler.histogram_methods['auto']['histogram']['bin_edges'] = \
            np.array([1.0, 3.0, 5.0, 7.0])
        input_array = np.array([1.1, 1.5, 2.3, 3.5, 4.0, 6.5])
        expected_total_var = np.array([1.1, 1.5, 2.3]).var() \
                             + np.array([3.5, 4.0]).var() \
                             + np.array([6.5]).var()
        est_total_var = profiler._total_histogram_bin_variance(
            input_array, method='auto')
        self.assertEqual(expected_total_var, est_total_var)

    def test_histogram_loss(self):
        # run time is small
        diff_var, avg_diffvar, total_var, avg_totalvar, run_time, avg_runtime =\
            0.3, 0.2, 0.1, 0.05, 0.0014, 0.0022
        expected_loss = 0.1 / 0.2 + 0.05 / 0.05
        est_loss = FloatColumn._histogram_loss(
            diff_var, avg_diffvar, total_var, avg_totalvar, run_time,
            avg_runtime)
        self.assertEqual(expected_loss, est_loss)

        # run time is big
        diff_var, avg_diffvar, total_var, avg_totalvar, run_time, avg_runtime =\
            0.3, 0.2, 0.1, 0.05, 22, 14
        expected_loss = 0.1 / 0.2 + 0.05 / 0.05 + 8 / 14
        est_loss = FloatColumn._histogram_loss(
            diff_var, avg_diffvar, total_var, avg_totalvar, run_time,
            avg_runtime)
        self.assertEqual(expected_loss, est_loss)

    def test_select_method_for_histogram(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        list_method = ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges',
                       'sqrt']
        current_exact_var = 0
        # sqrt has the least current loss
        current_est_var = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005])
        current_total_var = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        current_run_time = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        # all methods have the same total loss
        list_total_loss = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        for i, method in enumerate(list_method):
            profiler.histogram_methods[method]['total_loss'] = \
                list_total_loss[i]
        selected_method = profiler._select_method_for_histogram(
            current_exact_var, current_est_var,
            current_total_var, current_run_time)
        self.assertEqual(selected_method, 'sqrt')

        # another test
        current_exact_var = 0

        # sqrt has the least current loss
        current_est_var = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.029])
        current_total_var = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        current_run_time = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

        # but sturges has the least total loss
        list_total_loss = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1])
        for i, method in enumerate(list_method):
            profiler.histogram_methods[method]['total_loss'] = \
                list_total_loss[i]
        selected_method = profiler._select_method_for_histogram(
            current_exact_var, current_est_var,
            current_total_var, current_run_time)
        self.assertEqual(selected_method, 'sturges')

    def test_histogram_to_array(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler.histogram_methods['auto']['histogram']['bin_counts'] = \
            np.array([3, 2, 1])
        profiler.histogram_methods['auto']['histogram']['bin_edges'] = \
            np.array([1.0, 3.0, 5.0, 7.0])
        array_from_histogram = profiler._histogram_to_array('auto')
        expected_array = [1.0, 1.0, 1.0, 3.0, 3.0, 7.0]
        self.assertCountEqual(array_from_histogram, expected_array)

    def test_merge_histogram(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler.histogram_methods['sqrt']['histogram']['bin_counts'] = \
            np.array([3, 2])
        profiler.histogram_methods['sqrt']['histogram']['bin_edges'] = \
            np.array([1.0, 3.0, 5.0])
        input_array = [0.5, 1.0, 2.0, 5.0]

        profiler._merge_histogram(input_array, 'sqrt')
        merged_bin_counts = \
            profiler.histogram_methods['sqrt']['histogram']['bin_counts']
        merged_bin_edges = \
            profiler.histogram_methods['sqrt']['histogram']['bin_edges']
        expected_bin_counts, expected_bin_edges = \
            [5, 2, 2], [0.5, 2.0, 3.5, 5.0]
        self.assertCountEqual(merged_bin_counts, expected_bin_counts)
        self.assertCountEqual(merged_bin_edges, expected_bin_edges)

    def test_profiled_quantiles(self):
        """
        Checks the quantiles of profiled numerical columns.
        :return:
        """

        # this data has 4 bins, range of 3
        # with equal bin size, each bin has the width of 0.75

        data = ["1.0", "2.0", "3.0", "4.0"]
        df = pd.Series(data)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        profile = profiler.profile

        est_quartiles = profile['quantiles']
        est_Q1 = est_quartiles[249]
        est_Q2 = est_quartiles[499]
        est_Q3 = est_quartiles[749]

        data_to_num = [float(item) for item in data]
        exact_Q1 = np.percentile(data_to_num, 25)
        exact_Q2 = np.percentile(data_to_num, 50)
        exact_Q3 = np.percentile(data_to_num, 75)

        self.assertEqual(est_Q1, exact_Q1)
        self.assertEqual(est_Q2, exact_Q2)
        self.assertEqual(est_Q3, exact_Q3)

    def test_data_type_ratio(self):
        data = np.linspace(-5, 5, 4)
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 1.0)

        df = pd.Series(['not a float'])
        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 0.8)

    def test_profile(self):
        data = [2.5, 12.5, 'not a float', 5, 'not a float']
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)

        expected_profile = dict(
            min=2.5,
            max=12.5,
            mean=20/3.0,
            median=None,
            variance=27 + 1/12.0,
            stddev=np.sqrt(27+1/12.0),
            histogram={
                'bin_counts': np.array([1, 1, 0, 1]),
                'bin_edges': np.array([2.5, 5.0, 7.5, 10.0, 12.5]),
            },
            quantiles={
                0: 3.75,
                1: 5.0,
                2: 10.0
            },
            times=defaultdict(float, {'histogram_and_quantiles': 15.0,\
                                      'precision': 1.0, 'max': 1.0, 'min': 1.0,\
                                      'sum': 1.0, 'variance': 1.0}),
            precision=1.0
        )
        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile['times'])
            profiler.update(df)

            profile = profiler.profile
            # pop out the histogram to test separately from the rest of the dict
            # as we need comparison with some precision
            histogram = profile.pop('histogram')
            expected_histogram = expected_profile.pop('histogram')
            quantiles = profile.pop('quantiles')
            expected_quantiles = expected_profile.pop('quantiles')
            actual_quartiles = {0: quantiles[249], 1: quantiles[499], 2: quantiles[749]}

            self.assertDictEqual(expected_profile, profile)
            self.assertEqual(expected_profile['precision'], 1.0)
            self.assertCountEqual(expected_histogram['bin_counts'],
                                  histogram['bin_counts'])
            self.assertCountEqual(np.round(expected_histogram['bin_edges'], 12),
                                  np.round(histogram['bin_edges'], 12))

            self.assertDictEqual(actual_quartiles, expected_quantiles)

            # Validate time in datetime class has expected time after second update
            profiler.update(df)
            expected = defaultdict(float, {'min': 2.0, 'max': 2.0, 'sum': 2.0,\
                                           'variance': 2.0, 'precision': 2.0,\
                                           'histogram_and_quantiles': 30.0})
            self.assertEqual(expected, profiler.profile['times'])

    def test_option_timing(self):
        data = [2.0, 12.5, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)

        options = FloatOptions()
        options.set({"min.is_enabled": False})

        profiler = FloatColumn(df.name, options=options)

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile['times'])
            profiler.update(df)

            # Validate the time in the datetime class has the expected time.
            profile = profiler.profile

            expected = defaultdict(float, {'max': 1.0, 'sum': 1.0,\
                                           'variance': 1.0, 'precision': 1.0,\
                                           'histogram_and_quantiles': 15.0})
            self.assertEqual(expected, profile['times'])

            # Validate time in datetime class has expected time after second update
            profiler.update(df)
            expected = defaultdict(float, {'max': 2.0, 'sum': 2.0,\
                                           'variance': 2.0, 'precision': 2.0,\
                                           'histogram_and_quantiles': 30.0})
            self.assertEqual(expected, profiler.profile['times'])

    def test_profile_merge(self):
        data = [2.0, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)

        data2 = [10.0, 'not a float', 15.0, 'not a float']
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        expected_profile = dict(
            min=2.0,
            max=15.0,
            mean=8.25,
            variance=30.916666666666668,
            stddev=np.sqrt(30.916),
            histogram={
                'bin_counts': np.array([1, 1, 1, 1]),
                'bin_edges': np.array([2.,5.25,8.5,11.75,15.])
            },
        )

        profiler3 = profiler1 + profiler2

        expected_histogram = expected_profile.pop('histogram')
        profile3 = profiler3.profile
        histogram = profile3.pop('histogram')

        self.assertAlmostEqual(profiler3.stddev,
                               expected_profile.pop('stddev'), places=3)
        self.assertAlmostEqual(profiler3.variance,
                               expected_profile.pop('variance'), places=3)
        self.assertEqual(profiler3.mean, expected_profile.pop('mean'))
        self.assertEqual(profiler3.histogram_selection, 'rice')
        self.assertEqual(profiler3.min, expected_profile.pop('min'))
        self.assertEqual(profiler3.max, expected_profile.pop('max'))
        self.assertCountEqual(histogram['bin_counts'],
                              expected_histogram['bin_counts'])
        self.assertCountEqual(histogram['bin_edges'],
                              expected_histogram['bin_edges'])

    def test_profile_merge_edge_case(self):
        data = [2.0, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)
        profiler1.match_count = 0

        data2 = [10.0, 'not a float', 15.0, 'not a float']
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        profiler3 = profiler1 + profiler2
        self.assertEqual(profiler3.stddev,profiler2.stddev)

        # test merge with empty data
        df1 = pd.Series([], dtype=object)
        profiler1 = FloatColumn("Float")
        profiler1.update(df1)

        df2 = pd.Series([], dtype=object)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        profiler = profiler1 + profiler2
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)

        df3 = pd.Series([2.0, 3.0]).apply(str)
        profiler3 = FloatColumn("Float")
        profiler3.update(df3)

        profiler = profiler1 + profiler3
        self.assertEqual(profiler.min, 2.0)
        self.assertEqual(profiler.max, 3.0)

        df4 = pd.Series([4.0, 5.0]).apply(str)
        profiler4 = FloatColumn("Float")
        profiler4.update(df4)

        profiler = profiler3 + profiler4
        self.assertEqual(profiler.min, 2.0)
        self.assertEqual(profiler.max, 5.0)

    def test_profile_merge_no_bin_overlap(self):

        data = [2.0, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)
        profiler1.match_count = 0

        data2 = [10.0, 'not a float', 15.0, 'not a float']
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        # set bin names so no overlap
        profiler1.histogram_bin_method_names = ['No overlap 1']
        profiler2.histogram_bin_method_names = ['No overlap 2']

        with self.assertRaisesRegex(ValueError,
                                    'Profiles have no overlapping bin methods '
                                    'and therefore cannot be added together.'):
            profiler1 + profiler2
