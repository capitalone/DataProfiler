import os
import unittest
from unittest import mock
from collections import defaultdict

import pandas as pd
import numpy as np

from dataprofiler.profilers import IntColumn
from dataprofiler.profilers.profiler_options import IntOptions


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestIntColumn(unittest.TestCase):

    def test_base_case(self):
        data = pd.Series([], dtype=object)
        profiler = IntColumn(data.name)
        profiler.update(data)

        self.assertEqual(profiler.match_count, 0)
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)
        self.assertEqual(profiler.mean, 0)
        self.assertEqual(profiler.variance, 0)
        self.assertTrue(profiler.stddev is np.nan)
        self.assertIsNone(profiler.histogram_selection)
        self.assertDictEqual({k: profiler.quantiles.get(k, 'fail')
                              for k in (0, 1, 2)}, {0: None, 1: None, 2: None})
        self.assertIsNone(profiler.data_type_ratio)

    def test_single_data_variance_case(self):
        data = pd.Series([1])
        profiler = IntColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.match_count, 1)
        self.assertEqual(profiler.mean, 1)
        self.assertEqual(profiler.variance, 0)

        data = pd.Series([2])
        profiler.update(data)
        self.assertEqual(profiler.match_count, 2)
        self.assertEqual(profiler.mean, 1.5)
        self.assertEqual(profiler.variance, 0.5)

    def test_profiled_min(self):
        data = np.linspace(-5, 5, 11)
        df = pd.Series(data).apply(str)

        profiler = IntColumn(df.name)
        profiler.update(df[1:])
        self.assertEqual(profiler.min, -4)

        profiler.update(df)
        self.assertEqual(profiler.min, -5)

        profiler.update(pd.Series(['-4']))
        self.assertEqual(profiler.min, -5)

        # empty data
        data = pd.Series([], dtype=object)
        profiler = IntColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.min, None)

        # data with None value
        df = pd.Series([2, 3, None, np.nan]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2)

        # data with one value
        df = pd.Series([2]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2)

        # data with unique value
        df = pd.Series([2, 2, 2, 2, 2]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2)

        # data with unique value as zero
        df = pd.Series([0, 0, 0, 0, 0]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 0)

    def test_profiled_max(self):
        data = np.linspace(-5, 5, 11)
        df = pd.Series(data).apply(str)

        profiler = IntColumn(df.name)
        profiler.update(df[:-1])
        self.assertEqual(profiler.max, 4)

        profiler.update(df)
        self.assertEqual(profiler.max, 5)

        profiler.update(pd.Series(['4']))
        self.assertEqual(profiler.max, 5)

        # empty data
        data = pd.Series([], dtype=object)
        profiler = IntColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.max, None)

        # data with None value
        df = pd.Series([2, 3, None, np.nan]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 3)

        # data with one value
        df = pd.Series([2]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 2)

        # data with unique value
        df = pd.Series([2, 2, 2, 2, 2]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 2)

        # data with unique value as zero
        df = pd.Series([0, 0, 0, 0, 0]).apply(str)
        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 0)

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

        num_profiler = IntColumn(df1.name)
        num_profiler.update(df1.apply(str))

        self.assertEqual(mean(df1), num_profiler.mean)
        self.assertEqual(var(df1), num_profiler.variance)
        self.assertEqual(np.sqrt(var(df1)), num_profiler.stddev)

        df2_ints = df2[df2 == df2.round()]
        variance = batch_variance(
            mean_a=num_profiler.mean, var_a=num_profiler.variance,
            count_a=num_profiler.match_count,
            mean_b=mean(df2_ints), var_b=var(df2_ints), count_b=df2_ints.count()
        )
        num_profiler.update(df2.apply(str))
        df = pd.concat([df1, df2_ints])
        self.assertEqual(mean(df), num_profiler.mean)
        self.assertEqual(variance, num_profiler.variance)
        self.assertEqual(np.sqrt(variance), num_profiler.stddev)

        df3_ints = df3[df3 == df3.round()]
        variance = batch_variance(
            mean_a=num_profiler.mean, var_a=num_profiler.variance,
            count_a=num_profiler.match_count,
            mean_b=mean(df3_ints), var_b=var(df3_ints), count_b=df3_ints.count()
        )
        num_profiler.update(df3.apply(str))

        df = pd.concat([df1, df2_ints, df3_ints])
        self.assertEqual(mean(df), num_profiler.mean)
        self.assertEqual(variance, num_profiler.variance)
        self.assertEqual(np.sqrt(variance), num_profiler.stddev)

    def test_profiled_histogram(self):
        """
        Checks the histogram of profiled numerical columns.
        :return:
        """

        list_data_test = []
        # this data has 4 bins, range of 3
        # with equal bin size, each bin has the width of 0.75
        data1 = ["1", "2", "3", "4"]
        expected_histogram1 = {
            'bin_counts': np.array([1, 1, 1, 1]),
            'bin_edges': np.array([1.0, 1.75, 2.5, 3.25, 4.0]),
        }
        list_data_test.append([data1, expected_histogram1])

        # this data has 4 bins, range of 12
        # with equal bin size, each bin has the width of 3.0
        data2 = ["1", "5", "8", "13"]
        expected_histogram2 = {
            'bin_counts': np.array([1, 1, 1, 1]),
            'bin_edges': np.array([1.0, 4.0, 7.0, 10.0, 13.0]),
        }
        list_data_test.append([data2, expected_histogram2])

        # this data has 3 bins, range of 3
        # with equal bin size, each bin has the width of 1
        data3 = ["1", "1", "3", "4"]  # 3 bins, range of 3
        expected_histogram3 = {
            'bin_counts': np.array([2, 0, 1, 1]),
            'bin_edges': np.array([1.0, 1.75, 2.5, 3.25, 4.0]),
        }
        list_data_test.append([data3, expected_histogram3])

        for data, expected_histogram in list_data_test:
            df = pd.Series(data)
            profiler = IntColumn(df.name)
            profiler.update(df)

            profile = profiler.profile
            histogram = profile['histogram']

            self.assertEqual(expected_histogram['bin_counts'].tolist(),
                             histogram['bin_counts'].tolist())
            self.assertCountEqual(np.round(expected_histogram['bin_edges'], 12),
                                  np.round(histogram['bin_edges'], 12))

    def test_data_type_ratio(self):
        data = np.linspace(-5, 5, 11)
        df = pd.Series(data).apply(str)

        profiler = IntColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 1.0)

        df = pd.Series(['not a float', '0.1'])
        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 11/13.0)

    def test_profile(self):
        data = [2.0, 12.5, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)

        profiler = IntColumn(df.name)

        expected_profile = dict(
            min=2.0,
            max=6.0,
            mean=4.0,
            variance=8.0,
            stddev=np.sqrt(8.0),
            histogram={
                'bin_counts': np.array([1, 0, 1]),
                'bin_edges': np.array([2.0, 10.0/3.0, 14.0/3.0, 6.0])
            },
            quantiles={
                0: 2.002,
                1: 4,
                2: 5.998,
            },
            times=defaultdict(
                float, {'histogram_and_quantiles': 1.0, 'max': 1.0, 'min': 1.0,
                        'sum': 1.0, 'variance': 1.0})
            
        )
        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile['times'])
            profiler.update(df)

            # Validate the time in the datetime class has the expected time.
            profile = profiler.profile
            # pop out the histogram and quartiles to test separately from the
            # rest of the dict as we need comparison with some precision
            histogram = profile.pop('histogram')
            expected_histogram = expected_profile.pop('histogram')
            quartiles = profile.pop('quantiles')
            expected_quartiles = expected_profile.pop('quantiles')

            self.assertDictEqual(expected_profile, profile)
            self.assertEqual(expected_histogram['bin_counts'].tolist(),
                             histogram['bin_counts'].tolist())
            self.assertCountEqual(np.round(expected_histogram['bin_edges'], 12),
                                  np.round(histogram['bin_edges'], 12))

            self.assertAlmostEqual(expected_quartiles[0], quartiles[249])
            self.assertAlmostEqual(expected_quartiles[1], quartiles[499])
            self.assertAlmostEqual(expected_quartiles[2], quartiles[749])

            expected = defaultdict(
                float, {'min': 1.0, 'max': 1.0, 'sum': 1.0, 'variance': 1.0,
                        'histogram_and_quantiles': 1.0})
            self.assertEqual(expected, profile['times'])

            # Validate time in datetime class has expected time after second
            # update
            profiler.update(df)
            expected = defaultdict(
                float, {'min': 2.0, 'max': 2.0, 'sum': 2.0, 'variance': 2.0,
                        'histogram_and_quantiles': 2.0})
            self.assertEqual(expected, profiler.profile['times'])

    def test_option_timing(self):
        data = [2.0, 12.5, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)

        options = IntOptions()
        options.set({"min.is_enabled": False})

        profiler = IntColumn(df.name, options=options)

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertCountEqual(defaultdict(float), profiler.profile['times'])
            profiler.update(df)

            # Validate the time in the datetime class has the expected time.
            profile = profiler.profile

            expected = defaultdict(float, {'max': 1.0, 'sum': 1.0, 'variance': 1.0, \
                                           'histogram_and_quantiles': 1.0})
            self.assertCountEqual(expected, profile['times'])

            # Validate time in datetime class has expected time after second update
            profiler.update(df)
            expected = defaultdict(float, {'max': 2.0, 'sum': 2.0, 'variance': 2.0, \
                                           'histogram_and_quantiles': 2.0})
            self.assertCountEqual(expected, profiler.profile['times'])

    def test_profile_merge(self):
        data = [2.0, 12.5, 'not an int', 6.0, 'not an int']
        df = pd.Series(data).apply(str)
        profiler1 = IntColumn("Int")
        profiler1.update(df)

        data2 = [10.0, 3.5, 'not an int', 15.0, 'not an int']
        df2 = pd.Series(data2).apply(str)
        profiler2 = IntColumn("Int")
        profiler2.update(df2)

        expected_profile = dict(
            min=2.0,
            max=15.0,
            mean=8.25,
            variance=30.916666666666668,
            stddev=np.sqrt(30.916),
            histogram={
                'bin_counts': np.array([1, 1, 1, 1]),
                'bin_edges': np.array([2., 5.25, 8.5, 11.75, 15.])
            },
        )

        profiler3 = profiler1 + profiler2

        expected_histogram = expected_profile.pop('histogram')
        profile3 = profiler3.profile
        histogram = profile3.pop('histogram')

        self.assertAlmostEqual(profiler3.stddev,
                               expected_profile.pop('stddev'),places=3)
        self.assertAlmostEqual(profiler3.variance,
                               expected_profile.pop('variance'), places=3)
        self.assertEqual(profiler3.mean,expected_profile.pop('mean'))
        self.assertEqual(profiler3.histogram_selection, 'doane')
        self.assertEqual(profiler3.min,expected_profile.pop('min'))
        self.assertEqual(profiler3.max,expected_profile.pop('max'))
        self.assertEqual(histogram['bin_counts'].tolist(),
                         expected_histogram['bin_counts'].tolist())
        self.assertCountEqual(histogram['bin_edges'],
                              expected_histogram['bin_edges'])

    def test_profile_merge_edge_case(self):
        data = [2.0, 12.5, 'not a float', 6.0, 'not a float']
        df = pd.Series(data).apply(str)
        profiler1 = IntColumn(name="Int")
        profiler1.update(df)
        profiler1.match_count = 0

        data2 = [10.0, 3.5, 'not a float', 15.0, 'not a float']
        df2 = pd.Series(data2).apply(str)
        profiler2 = IntColumn(name="Int")
        profiler2.update(df2)

        profiler3 = profiler1 + profiler2
        self.assertEqual(profiler3.stddev, profiler2.stddev)

        # test merge with empty data
        df1 = pd.Series([], dtype=object)
        profiler1 = IntColumn("Int")
        profiler1.update(df1)

        df2 = pd.Series([], dtype=object)
        profiler2 = IntColumn("Int")
        profiler2.update(df2)

        profiler = profiler1 + profiler2
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)
        self.assertIsNone(profiler.histogram_selection)

        df3 = pd.Series([2, 3]).apply(str)
        profiler3 = IntColumn("Int")
        profiler3.update(df3)

        profiler = profiler1 + profiler3
        self.assertEqual(profiler.min, 2)
        self.assertEqual(profiler.max, 3)

        df4 = pd.Series([4, 5]).apply(str)
        profiler4 = IntColumn("Int")
        profiler4.update(df4)

        profiler = profiler3 + profiler4
        self.assertEqual(profiler.min, 2)
        self.assertEqual(profiler.max, 5)

    def test_custom_bin_count_merge(self):

        options = IntOptions()
        options.histogram_and_quantiles.bin_count_or_method = 10

        data = [2, 'not an int', 6, 'not an int']
        df = pd.Series(data).apply(str)
        profiler1 = IntColumn("Int", options)
        profiler1.update(df)

        data2 = [10, 'not an int', 15, 'not an int']
        df2 = pd.Series(data2).apply(str)
        profiler2 = IntColumn("Int", options)
        profiler2.update(df2)

        # no warning should occur
        import warnings
        with warnings.catch_warnings(record=True) as w:
            merge_profile = profiler1 + profiler2
        self.assertListEqual([], w)
        self.assertEqual(10, merge_profile.user_set_histogram_bin)

        # make bin counts different and get warning
        profiler2.user_set_histogram_bin = 120
        with self.assertWarnsRegex(UserWarning,
                                   'User set histogram bin counts did not '
                                   'match. Choosing the larger bin count.'):
            merged_profile = profiler1 + profiler2
        self.assertEqual(120, merged_profile.user_set_histogram_bin)

    def test_profile_merge_no_bin_overlap(self):

        data = [2, 'not an int', 6, 'not an int']
        df = pd.Series(data).apply(str)
        profiler1 = IntColumn("Int")
        profiler1.update(df)
        profiler1.match_count = 0

        data2 = [10, 'not an int', 15, 'not an int']
        df2 = pd.Series(data2).apply(str)
        profiler2 = IntColumn("Int")
        profiler2.update(df2)

        # set bin names so no overlap
        profiler1.histogram_bin_method_names = ['No overlap 1']
        profiler2.histogram_bin_method_names = ['No overlap 2']

        with self.assertRaisesRegex(ValueError,
                                    'Profiles have no overlapping bin methods '
                                    'and therefore cannot be added together.'):
            profiler1 + profiler2
            
    def test_profile_merge_with_different_options(self):
        # Creating first profiler with default options
        options = IntOptions()
        options.max.is_enabled = False
        options.min.is_enabled = False

        data = [2, 4, 6, 8]
        df = pd.Series(data).apply(str)
        profiler1 = IntColumn("Int", options=options)
        profiler1.update(df)
        profiler1.match_count = 0

        # Creating second profiler with separate options
        options = IntOptions()
        options.min.is_enabled = False
        data2 = [10, 15]
        df2 = pd.Series(data2).apply(str)
        profiler2 = IntColumn("Int", options=options)
        profiler2.update(df2)

        # Asserting warning when adding 2 profilers with different options
        with self.assertWarnsRegex(RuntimeWarning,
                                   "max is disabled because it is not enabled in"
                                   " both profiles."):
            profiler3 = profiler1 + profiler2
        
        # Assert that these features are still merged
        profile = profiler3.profile
        self.assertIsNotNone(profiler3.histogram_selection)
        self.assertIsNotNone(profile['variance'])
        self.assertIsNotNone(profiler3.sum)
        
        # Assert that these features are not calculated
        self.assertIsNone(profiler3.max)
        self.assertIsNone(profiler3.min)

    def test_int_column_with_wrong_options(self):
        with self.assertRaisesRegex(ValueError,
                                   "IntColumn parameter 'options' must be of"
                                   " type IntOptions."):
            profiler = IntColumn("Int", options="wrong_data_type")

    def test_histogram_option_integration(self):
        # test setting bin methods
        options = IntOptions()
        options.histogram_and_quantiles.bin_count_or_method = "sturges"
        num_profiler = IntColumn(name="test", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["sturges"], num_profiler.histogram_bin_method_names)

        options.histogram_and_quantiles.bin_count_or_method = ["sturges",
                                                               "doane"]
        num_profiler = IntColumn(name="test2", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["sturges", "doane"],
                         num_profiler.histogram_bin_method_names)

        options.histogram_and_quantiles.bin_count_or_method = 100
        num_profiler = IntColumn(name="test3", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(['custom'], num_profiler.histogram_bin_method_names)

        # case when just 1 unique value, should just set bin size to be 1
        num_profiler.update(pd.Series(['1', '1']))
        self.assertEqual(
            1,
            len(num_profiler.histogram_methods['custom']['histogram'][
                    'bin_counts'])
        )

        # case when more than 1 unique value, by virtue of a streaming update
        num_profiler.update(pd.Series(['2']))
        self.assertEqual(
            100, len(num_profiler._stored_histogram['histogram']['bin_counts']))

        histogram, _ = num_profiler._histogram_for_profile('custom')
        self.assertEqual(100, len(histogram['bin_counts']))
