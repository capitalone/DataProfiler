import json
import os
import unittest
import warnings
from collections import defaultdict
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.profilers import FloatColumn
from dataprofiler.profilers.profiler_options import FloatOptions

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestFloatColumn(unittest.TestCase):
    def test_base_case(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)

        self.assertEqual(profiler.match_count, 0)
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)
        self.assertEqual(profiler.sum, 0)
        self.assertEqual(profiler.mean, 0)
        self.assertTrue(profiler.median is np.nan)
        self.assertEqual([np.nan], profiler.mode)
        self.assertTrue(profiler.variance is np.nan)
        self.assertTrue(profiler.skewness is np.nan)
        self.assertTrue(profiler.kurtosis is np.nan)
        self.assertTrue(profiler.stddev is np.nan)
        self.assertIsNone(profiler.histogram_selection)
        self.assertEqual(len(profiler.quantiles), 999)
        self.assertIsNone(profiler.data_type_ratio)

    def test_single_data_variance_case(self):
        data = pd.Series([1.5]).apply(str)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        self.assertEqual(profiler.match_count, 1.0)
        self.assertEqual(profiler.mean, 1.5)
        self.assertTrue(profiler.variance is np.nan)

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
        df_mix = pd.Series([4.1, "3.", 2.52, 2.13143]).apply(str)

        float_profiler = FloatColumn("Name")
        float_profiler.update(df_3)
        self.assertEqual(4, float_profiler.precision["min"])
        self.assertEqual(4, float_profiler.precision["max"])

        float_profiler.update(df_2)
        self.assertEqual(2, float_profiler.precision["min"])
        self.assertEqual(4, float_profiler.precision["max"])

        float_profiler.update(df_1)
        self.assertEqual(1, float_profiler.precision["min"])
        self.assertEqual(4, float_profiler.precision["max"])

        float_profiler = FloatColumn("Name")
        float_profiler.update(df_mix)
        self.assertEqual(1, float_profiler.precision["min"])
        self.assertEqual(6, float_profiler.precision["max"])

        # edge cases #
        # integer with 0s on right and left side
        df_ints = pd.Series(["0013245678", "123456700", "0012345600"])
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_ints)
        self.assertEqual(6, float_profiler.precision["min"])
        self.assertEqual(8, float_profiler.precision["max"])

        # scientific
        df_scientific = pd.Series(["1.23e-3", "2.2344", "1.244e4"])
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_scientific)
        self.assertEqual(3, float_profiler.precision["min"])
        self.assertEqual(5, float_profiler.precision["max"])

        # plus
        df_plus = pd.Series(["+1.3e-3", "+2.244", "+1.3324e4"])
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_plus)
        self.assertEqual(2, float_profiler.precision["min"])
        self.assertEqual(5, float_profiler.precision["max"])

        # minus
        df_minus = pd.Series(["-1.3234e-3", "-0.244", "-1.3324e4"])
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_minus)
        self.assertEqual(3, float_profiler.precision["min"])
        self.assertEqual(5, float_profiler.precision["max"])

        # spaces around values
        df_spaces = pd.Series(["  -1.3234e-3  ", "  -0.244  "])
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_spaces)
        self.assertEqual(3, float_profiler.precision["min"])
        self.assertEqual(5, float_profiler.precision["max"])

        # constant precision
        df_constant = pd.Series(
            [
                "1.34",
                "+1.23e-4",
                "00101",
                "+100.",
                "0.234",
                "-432",
                ".954",
                "+.342",
                "-123e1",
                "23.1",
            ]
        )
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_constant)
        self.assertEqual(3, float_profiler.precision["min"])
        self.assertEqual(3, float_profiler.precision["max"])
        self.assertEqual(3, float_profiler.precision["mean"])
        self.assertEqual(10, float_profiler.precision["sample_size"])
        self.assertEqual(0, float_profiler.precision["var"])
        self.assertEqual(0, float_profiler.precision["std"])

        # random precision
        df_random = pd.Series(
            [
                "+ 9",
                "-.3",
                "-1e-3",
                "3.2343",
                "0",
                "1230",
                "0.33",
                "4.3",
                "302.1",
                "-4.322",
            ]
        )
        float_profiler = FloatColumn("Name")
        float_profiler.update(df_random)
        self.assertEqual(0, float_profiler.precision["min"])
        self.assertEqual(5, float_profiler.precision["max"])
        self.assertEqual(2.4444, float_profiler.precision["mean"])
        self.assertEqual(9, float_profiler.precision["sample_size"])
        self.assertEqual(2.7778, float_profiler.precision["var"])
        self.assertEqual(1.6667, float_profiler.precision["std"])

        # Ensure order doesn't change anything
        df_random_order = pd.Series(
            [
                "1230",
                "0.33",
                "4.3",
                "302.1",
                "-4.322",
                "+ 9",
                "-.3",
                "-1e-3",
                "3.2343",
                "0",
            ]
        )
        float_profiler_order = FloatColumn("Name")
        float_profiler_order.update(df_random)

        self.assertDictEqual(float_profiler.precision, float_profiler_order.precision)

        # check to make sure all formats of precision are correctly predicted
        samples = [
            # value, min expected precision
            ["10.01", 4],
            [".01", 1],
            ["0.01", 1],
            ["-0.01", 1],
            ["+0.01", 1],
            [" +0.013", 2],
            ["  -1.3234e-3  ", 5],
            ["  0012345600  ", 6],
            ["  0012345600.  ", 8],
            ["  -0012345600.  ", 8],
        ]

        for sample in samples:
            df_series = pd.Series([sample[0]])
            min_expected_precision = sample[1]
            precision = FloatColumn._get_float_precision(df_series)
            self.assertEqual(
                min_expected_precision,
                precision["min"],
                msg="Errored for: {}".format(sample[0]),
            )

    def test_profiled_min(self):
        # test with multiple values
        data = np.linspace(-5, 5, 11)
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)
        profiler.update(df[1:])
        self.assertEqual(profiler.min, -4)

        profiler.update(df)
        self.assertEqual(profiler.min, -5)

        profiler.update(pd.Series(["-4"]))
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

        profiler.update(pd.Series(["4"]))
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

    def test_profiled_mode(self):
        # disabled mode
        df = pd.Series([1, 1, 1, 1, 1, 1, 1]).apply(str)
        options = FloatOptions()
        options.mode.is_enabled = False
        profiler = FloatColumn(df.name, options)
        profiler.update(df)
        self.assertListEqual([np.nan], profiler.mode)

        # same values
        df = pd.Series([1, 1, 1, 1, 1, 1, 1]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertListEqual([1], profiler.mode)

        # multiple modes
        df = pd.Series([1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.1, 4.1]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        np.testing.assert_array_almost_equal(
            [1.5, 2.5, 3.5, 4.1], profiler.mode, decimal=2
        )

        # with different values
        df = pd.Series([1.25, 1.25, 1.25, 1.25, 2.9]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        np.testing.assert_array_almost_equal([1.25], profiler.mode, decimal=2)

        # with negative values
        df = pd.Series([-1.1, 1.9, 1.9, 1.9, 2.1, 2.01, 2.01, 2.01]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        np.testing.assert_array_almost_equal([1.9, 2.01], profiler.mode, decimal=2)

        # all unique values
        df = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        # By default, returns 5 of the possible modes
        np.testing.assert_array_almost_equal([1, 2, 3, 4, 5], profiler.mode, decimal=2)

        # Edge case where mode appears later in the dataset
        df = pd.Series([1, 2, 3, 4, 5, 6.2, 6.2]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        np.testing.assert_array_almost_equal([6.2], profiler.mode, decimal=2)

        df = pd.Series([2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7.1, 7.1, 7.1]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        np.testing.assert_array_almost_equal([7.1], profiler.mode, decimal=2)

    def test_top_k_modes(self):
        # Default options
        options = FloatOptions()
        df = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).apply(str)
        profiler = FloatColumn(df.name, options)
        profiler.update(df)
        self.assertEqual(5, len(profiler.mode))

        # Test if top_k_modes is less than the number of modes
        options = FloatOptions()
        options.mode.top_k_modes = 2
        df = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).apply(str)
        profiler = FloatColumn(df.name, options)
        profiler.update(df)
        self.assertEqual(2, len(profiler.mode))

        # Test if top_k_mode is greater than the number of modes
        options = FloatOptions()
        options.mode.top_k_modes = 8
        df = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).apply(str)
        profiler = FloatColumn(df.name, options)
        profiler.update(df)
        # Only 5 possible modes so return 5
        self.assertEqual(5, len(profiler.mode))

    def test_profiled_median(self):
        # disabled median
        df = pd.Series([1, 1, 1, 1, 1, 1, 1]).apply(str)
        options = FloatOptions()
        options.median.is_enabled = False
        profiler = FloatColumn(df.name, options)
        profiler.update(df)
        self.assertTrue(profiler.median is np.nan)

        # same values
        df = pd.Series([1, 1, 1, 1, 1, 1, 1]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(1, profiler.median)

        # median lies between two values (2.5 and 3.5)
        df = pd.Series([1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.1, 4.1]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertAlmostEqual(3, profiler.median, places=2)

        # with different values
        df = pd.Series([1.25, 1.25, 1.25, 1.25, 2.9]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertAlmostEqual(1.25, profiler.median, places=2)

        # with negative values, median lies in between values
        df = pd.Series([-1.1, 1.9, 1.9, 1.9, 2.1, 2.1, 2.1, 2.1]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertAlmostEqual(2, profiler.median, places=2)

        # all unique values
        df = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9]).apply(str)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertAlmostEqual(5, profiler.median, places=2)

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
            M2 = m_a + m_b + delta**2 * count_a * count_b / (count_a + count_b)
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
            mean_a=num_profiler.mean,
            var_a=num_profiler.variance,
            count_a=num_profiler.match_count,
            mean_b=mean(df2),
            var_b=var(df2),
            count_b=df2.count(),
        )
        num_profiler.update(df2.apply(str))
        df = pd.concat([df1, df2])
        self.assertEqual(mean(df), num_profiler.mean)
        self.assertEqual(variance, num_profiler.variance)
        self.assertEqual(np.sqrt(variance), num_profiler.stddev)

        variance = batch_variance(
            mean_a=num_profiler.mean,
            var_a=num_profiler.variance,
            count_a=num_profiler.match_count,
            mean_b=mean(df3),
            var_b=var(df3),
            count_b=df3.count(),
        )
        num_profiler.update(df3.apply(str))

        df = pd.concat([df1, df2, df3])
        self.assertEqual(mean(df), num_profiler.mean)
        self.assertEqual(variance, num_profiler.variance)
        self.assertEqual(np.sqrt(variance), num_profiler.stddev)

    def test_profiled_skewness(self):
        data = np.linspace(-5, 5, 11).tolist()
        df1 = pd.Series(data)

        data = np.linspace(-3, 2, 11).tolist()
        df2 = pd.Series(data)

        data = np.full((10,), 1)
        df3 = pd.Series(data)

        num_profiler = FloatColumn(df1.name)
        num_profiler.update(df1.apply(str))

        self.assertEqual(0, num_profiler.skewness)

        num_profiler.update(df2.apply(str))
        self.assertAlmostEqual(np.sqrt(22 * 21) / 20 * 133 / 750, num_profiler.skewness)

        num_profiler.update(df3.apply(str))
        self.assertAlmostEqual(-0.3109967, num_profiler.skewness)

    def test_profiled_kurtosis(self):
        data = np.linspace(-5, 5, 11).tolist()
        df1 = pd.Series(data)

        data = np.linspace(-3, 2, 11).tolist()
        df2 = pd.Series(data)

        data = np.full((10,), 1)
        df3 = pd.Series(data)

        num_profiler = FloatColumn(df1.name)
        num_profiler.update(df1.apply(str))

        self.assertAlmostEqual(-6 / 5, num_profiler.kurtosis)

        num_profiler.update(df2.apply(str))
        self.assertAlmostEqual(-0.390358, num_profiler.kurtosis)

        num_profiler.update(df3.apply(str))
        self.assertAlmostEqual(0.3311739, num_profiler.kurtosis)

    def test_bias_correction_option(self):
        # df1 = [-5, -4, ..., 3, 4, 5]
        data = np.linspace(-5, 5, 11).tolist()
        df1 = pd.Series(data)

        # df2 = [-3, -2.5, -2, ..., 1.5, 2]
        data = np.linspace(-3, 2, 11).tolist()
        df2 = pd.Series(data)

        # df3 = [1, 1, ... , 1] (ten '1's)
        data = np.full((10,), 1)
        df3 = pd.Series(data)

        # Disable bias correction
        options = FloatOptions()
        options.bias_correction.is_enabled = False
        num_profiler = FloatColumn(df1.name, options=options)
        num_profiler.update(df1.apply(str))
        # Test biased values of variance, skewness, kurtosis
        self.assertAlmostEqual(10, num_profiler.variance)
        self.assertAlmostEqual(0, num_profiler.skewness)
        self.assertAlmostEqual(89 / 50 - 3, num_profiler.kurtosis)

        df2_ints = df2[df2 == df2.round()]
        num_profiler.update(df2.apply(str))
        df = pd.concat([df1, df2_ints])
        self.assertAlmostEqual(6.3125, num_profiler.variance)
        self.assertAlmostEqual(0.17733336, num_profiler.skewness)
        self.assertAlmostEqual(-0.56798353, num_profiler.kurtosis)

        df3_ints = df3[df3 == df3.round()]
        num_profiler.update(df3.apply(str))
        df = pd.concat([df1, df2_ints, df3_ints])
        self.assertAlmostEqual(4.6755371, num_profiler.variance)
        self.assertAlmostEqual(-0.29622465, num_profiler.skewness)
        self.assertAlmostEqual(0.099825352, num_profiler.kurtosis)

    def test_bias_correction_merge(self):
        data = np.linspace(-5, 5, 11).tolist()
        df1 = pd.Series(data)

        data = np.linspace(-3, 2, 11).tolist()
        df2 = pd.Series(data)

        data = np.full((10,), 1)
        df3 = pd.Series(data)

        # Disable bias correction
        options = FloatOptions()
        options.bias_correction.is_enabled = False
        num_profiler1 = FloatColumn(df1.name, options=options)
        num_profiler1.update(df1.apply(str))
        self.assertAlmostEqual(10, num_profiler1.variance)
        self.assertAlmostEqual(0, num_profiler1.skewness)
        self.assertAlmostEqual(89 / 50 - 3, num_profiler1.kurtosis)

        num_profiler2 = FloatColumn(df2.name)
        num_profiler2.update(df2.apply(str))
        num_profiler = num_profiler1 + num_profiler2
        self.assertFalse(num_profiler.bias_correction)
        self.assertAlmostEqual(6.3125, num_profiler.variance)
        self.assertAlmostEqual(0.17733336, num_profiler.skewness)
        self.assertAlmostEqual(-0.56798353, num_profiler.kurtosis)

        num_profiler3 = FloatColumn(df3.name)
        num_profiler3.update(df3.apply(str))
        num_profiler = num_profiler1 + num_profiler2 + num_profiler3
        self.assertFalse(num_profiler.bias_correction)
        self.assertAlmostEqual(4.6755371, num_profiler.variance)
        self.assertAlmostEqual(-0.29622465, num_profiler.skewness)
        self.assertAlmostEqual(0.099825352, num_profiler.kurtosis)

    def test_null_values_for_histogram(self):
        data = pd.Series(["-inf", "inf"])
        profiler = FloatColumn(data.name)
        profiler.update(data)

        profile = profiler.profile
        histogram = profile["histogram"]

        self.assertEqual(histogram["bin_counts"], None)
        self.assertEqual(histogram["bin_edges"], None)

        data = pd.Series(["-2", "-1", "1", "2", "-inf", "inf"])
        profiler = FloatColumn(data.name)
        profiler.update(data)

        profile = profiler.profile
        histogram = profile["histogram"]

        expected_histogram = {
            "bin_counts": np.array([1, 1, 0, 2]),
            "bin_edges": np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        }

        self.assertEqual(
            expected_histogram["bin_counts"].tolist(), histogram["bin_counts"].tolist()
        )
        self.assertCountEqual(expected_histogram["bin_edges"], histogram["bin_edges"])

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
            "bin_counts": np.array([1, 1, 1, 1]),
            "bin_edges": np.array([1.0, 1.75, 2.5, 3.25, 4.0]),
        }
        list_data_test.append([df1, expected_histogram1])

        # this data has 4 bins, range of 12
        # with equal bin size, each bin has the width of 3.0
        df2 = pd.Series(["1.0", "5.0", "8.0", "13.0"])
        expected_histogram2 = {
            "bin_counts": np.array([1, 1, 1, 1]),
            "bin_edges": np.array([1.0, 4.0, 7.0, 10.0, 13.0]),
        }
        list_data_test.append([df2, expected_histogram2])

        # this data has 3 bins, range of 3
        # with equal bin size, each bin has the width of 1
        df3 = pd.Series(["1.0", "1.0", "3.0", "4.0"])
        expected_histogram3 = {
            "bin_counts": np.array([2, 0, 1, 1]),
            "bin_edges": np.array([1.0, 1.75, 2.5, 3.25, 4.0]),
        }
        list_data_test.append([df3, expected_histogram3])

        # this data has only one unique value, not overflow
        df4 = pd.Series([-10.0, -10.0, -10.0]).apply(str)
        expected_histogram4 = {
            "bin_counts": np.array([3]),
            "bin_edges": np.array([-10.0, -10.0]),
        }
        list_data_test.append([df4, expected_histogram4])

        # this data has only one unique value, overflow
        df5 = pd.Series([-(10.0**20)]).apply(str)
        expected_histogram5 = {
            "bin_counts": np.array([1]),
            "bin_edges": np.array([-(10.0**20), -(10.0**20)]),
        }
        list_data_test.append([df5, expected_histogram5])

        for i, (df, expected_histogram) in enumerate(list_data_test):
            profiler = FloatColumn(df.name)
            profiler.update(df)

            profile = profiler.profile
            histogram = profile["histogram"]

            self.assertEqual(
                expected_histogram["bin_counts"].tolist(),
                histogram["bin_counts"].tolist(),
            )
            if i != 4:
                self.assertCountEqual(
                    np.round(expected_histogram["bin_edges"], 12),
                    np.round(histogram["bin_edges"], 12),
                )
            else:  # for overflow, dont use np.round
                self.assertCountEqual(
                    expected_histogram["bin_edges"], histogram["bin_edges"]
                )

    def test_profile_histogram_w_updates(self):
        """
        Checks if histogram properly resets the _profiled histogram after
        merge or update.
        :return:
        """
        list_data_test = []
        # this data has 4 bins, range of 3
        # with equal bin size, each bin has the width of 0.75
        df1 = pd.Series(["1.0", "2.0", "3.0", "4.0"])
        expected_histogram1 = {
            "bin_counts": np.array([1, 1, 1, 1]),
            "bin_edges": np.array([1.0, 1.75, 2.5, 3.25, 4.0]),
        }
        list_data_test.append([df1, expected_histogram1])

        # this data will be the second update of the profile.
        # this results in the combination of the previous data and this data.
        # the range should update to 12 from 3.
        df2 = pd.Series(["1.0", "5.0", "8.0", "13.0"])
        expected_histogram2 = {
            "bin_counts": np.array([4, 1, 1, 1, 0, 1]),
            "bin_edges": np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]),
        }
        list_data_test.append([df2, expected_histogram2])

        profiler = FloatColumn("test")
        for i, (df, expected_histogram) in enumerate(list_data_test):
            profiler.update(df)
            self.assertIsNone(profiler.histogram_selection)

            profile = profiler.profile
            self.assertIsNotNone(profiler.histogram_selection)
            histogram = profile["histogram"]

            self.assertEqual(
                expected_histogram["bin_counts"].tolist(),
                histogram["bin_counts"].tolist(),
            )
            self.assertCountEqual(
                np.round(expected_histogram["bin_edges"], 12),
                np.round(histogram["bin_edges"], 12),
            )

        # apply test to merging profiles
        expected_histogram = {
            "bin_edges": np.array(
                [1.0, 19 / 7, 31 / 7, 43 / 7, 55 / 7, 67 / 7, 79 / 7, 13.0]
            ),
            "bin_counts": np.array([6, 4, 2, 0, 2, 0, 2]),
        }
        merged_profiler = profiler + profiler
        self.assertIsNone(merged_profiler.histogram_selection)

        profile = merged_profiler.profile
        self.assertIsNotNone(merged_profiler.histogram_selection)
        histogram = profile["histogram"]
        self.assertEqual(
            expected_histogram["bin_counts"].tolist(), histogram["bin_counts"].tolist()
        )
        self.assertCountEqual(
            np.round(expected_histogram["bin_edges"], 12),
            np.round(histogram["bin_edges"], 12),
        )

    def test_histogram_with_varying_number_of_bin(self):
        """
        Checks the histogram with large number of bins
        """
        # this data use number of bins less than the max limit
        df1 = pd.Series([1, 2, 3, 4]).apply(str)
        profiler1 = FloatColumn(df1.name)
        profiler1.max_histogram_bin = 50
        profiler1.update(df1)
        num_bins = len(profiler1.profile["histogram"]["bin_counts"])
        self.assertEqual(4, num_bins)

        # this data uses large number of bins, which will be set to
        # the max limit
        df2 = pd.Series(
            [3.195103249264023e18, 9999995.0, 9999999.0, 0.0, -(10**10)]
        ).apply(str)
        profiler2 = FloatColumn(df2.name)
        profiler2.max_histogram_bin = 50
        profiler2.update(df2)
        num_bins = len(profiler2.profile["histogram"]["bin_counts"])
        self.assertEqual(50, num_bins)

        # max number of bin is increased to 10000
        profiler2 = FloatColumn(df2.name)
        profiler2.max_histogram_bin = 10000
        profiler2.update(df2)
        num_bins = len(profiler2.profile["histogram"]["bin_counts"])
        self.assertEqual(10000, num_bins)

    def test_estimate_stats_from_histogram(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler._stored_histogram["histogram"]["bin_counts"] = np.array([1, 2, 1])
        profiler._stored_histogram["histogram"]["bin_edges"] = np.array(
            [1.0, 3.0, 5.0, 7.0]
        )
        expected_mean = (2.0 * 1 + 4.0 * 2 + 6.0 * 1) / 4
        expected_var = (
            1 * (2.0 - expected_mean) ** 2
            + 2 * (4.0 - expected_mean) ** 2
            + 1 * (6.0 - expected_mean) ** 2
        ) / 4
        expected_std = np.sqrt(expected_var)
        est_var = profiler._estimate_stats_from_histogram()
        self.assertEqual(expected_var, est_var)

    def test_total_histogram_bin_variance(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler._stored_histogram["histogram"]["bin_counts"] = np.array([3, 2, 1])
        profiler._stored_histogram["histogram"]["bin_edges"] = np.array(
            [1.0, 3.0, 5.0, 7.0]
        )
        input_array = np.array([1.1, 1.5, 2.3, 3.5, 4.0, 6.5])
        expected_total_var = (
            np.array([1.1, 1.5, 2.3]).var()
            + np.array([3.5, 4.0]).var()
            + np.array([6.5]).var()
        )
        est_total_var = profiler._total_histogram_bin_variance(input_array)
        self.assertEqual(expected_total_var, est_total_var)

    def test_histogram_loss(self):
        # run time is small
        diff_var, avg_diffvar, total_var, avg_totalvar, run_time, avg_runtime = (
            0.3,
            0.2,
            0.1,
            0.05,
            0.0014,
            0.0022,
        )
        expected_loss = 0.1 / 0.2 + 0.05 / 0.05
        est_loss = FloatColumn._histogram_loss(
            diff_var, avg_diffvar, total_var, avg_totalvar, run_time, avg_runtime
        )
        self.assertEqual(expected_loss, est_loss)

        # run time is big
        diff_var, avg_diffvar, total_var, avg_totalvar, run_time, avg_runtime = (
            0.3,
            0.2,
            0.1,
            0.05,
            22,
            14,
        )
        expected_loss = 0.1 / 0.2 + 0.05 / 0.05 + 8 / 14
        est_loss = FloatColumn._histogram_loss(
            diff_var, avg_diffvar, total_var, avg_totalvar, run_time, avg_runtime
        )
        self.assertEqual(expected_loss, est_loss)

    def test_select_method_for_histogram(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        list_method = ["auto", "fd", "doane", "scott", "rice", "sturges", "sqrt"]
        current_exact_var = 0
        # sqrt has the least current loss
        current_est_var = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005])
        current_total_var = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        current_run_time = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        # all methods have the same total loss
        list_total_loss = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        for i, method in enumerate(list_method):
            profiler.histogram_methods[method]["total_loss"] = list_total_loss[i]
        selected_method = profiler._select_method_for_histogram(
            current_exact_var, current_est_var, current_total_var, current_run_time
        )
        self.assertEqual(selected_method, "sqrt")

        # another test
        current_exact_var = 0

        # sqrt has the least current loss
        current_est_var = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.029])
        current_total_var = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        current_run_time = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

        # but sturges has the least total loss
        list_total_loss = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1])
        for i, method in enumerate(list_method):
            profiler.histogram_methods[method]["total_loss"] = list_total_loss[i]
        selected_method = profiler._select_method_for_histogram(
            current_exact_var, current_est_var, current_total_var, current_run_time
        )
        self.assertEqual(selected_method, "sturges")

    def test_histogram_to_array(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler._stored_histogram["histogram"]["bin_counts"] = np.array([3, 2, 1])
        profiler._stored_histogram["histogram"]["bin_edges"] = np.array(
            [1.0, 3.0, 5.0, 7.0]
        )
        array_from_histogram = profiler._histogram_to_array()
        expected_array = [1.0, 1.0, 1.0, 3.0, 3.0, 7.0]
        self.assertEqual(expected_array, array_from_histogram.tolist())

    def test_merge_histogram(self):
        data = pd.Series([], dtype=object)
        profiler = FloatColumn(data.name)
        profiler.update(data)
        profiler._stored_histogram["histogram"]["bin_counts"] = np.array([3, 2])
        profiler._stored_histogram["histogram"]["bin_edges"] = np.array([1.0, 3.0, 5.0])
        input_array = [0.5, 1.0, 2.0, 5.0]

        profiler._merge_histogram(input_array)
        merged_hist = profiler._histogram_for_profile("sqrt")[0]

        expected_bin_counts, expected_bin_edges = [5, 2, 2], [0.5, 2.0, 3.5, 5.0]
        self.assertEqual(expected_bin_counts, merged_hist["bin_counts"].tolist())
        self.assertCountEqual(expected_bin_edges, merged_hist["bin_edges"])

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

        est_quantiles = profile["quantiles"]
        est_Q1 = est_quantiles[249]
        est_Q2 = est_quantiles[499]
        est_Q3 = est_quantiles[749]

        self.assertEqual(999, len(est_quantiles))
        self.assertAlmostEqual(1.000012, est_quantiles[0])
        self.assertEqual(est_Q1, 1.003)
        self.assertEqual(est_Q2, 2.5)
        self.assertEqual(est_Q3, 3.001)
        self.assertAlmostEqual(3.999988, est_quantiles[-1])

    def test_get_median_abs_deviation(self):
        """
        Checks the median absolute deviation of profiled numerical columns.
        :return:
        """
        # with different values
        data = ["1.0", "1.0", "1.0", "1.0", "2.0"]
        df = pd.Series(data)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        profile = profiler.profile

        est_median_abs_dev = profile["median_abs_deviation"]
        self.assertAlmostEqual(0.0, est_median_abs_dev, places=2)

        # with unique values
        data = ["1.0", "1.0", "1.0", "1.0", "1.0"]
        df = pd.Series(data)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        profile = profiler.profile

        est_median_abs_dev = profile["median_abs_deviation"]
        self.assertAlmostEqual(0.0, est_median_abs_dev, places=2)

        # with negative values
        data = ["-1.0", "1.0", "1.0", "1.0", "2.0"]
        df = pd.Series(data)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        profile = profiler.profile

        est_median_abs_dev = profile["median_abs_deviation"]
        self.assertAlmostEqual(0.0, est_median_abs_dev, places=2)

        # multiple edge indices with the counts 0.5
        # in this example, 1.5 and 13.5 both have the counts 0.5
        # then the median absolute deviation should be the average, 7.5
        data = ["-9.0", "-8.0", "4.0", "5.0", "6.0", "7.0", "19.0", "20.0"]
        df = pd.Series(data)
        profiler = FloatColumn(df.name)
        profiler.update(df)
        profile = profiler.profile

        est_median_abs_dev = profile["median_abs_deviation"]
        self.assertAlmostEqual(7.5, est_median_abs_dev, places=2)

    def test_merge_median_abs_deviation(self):
        """
        Checks the median absolute deviation merged from profiles.
        :return:
        """
        # with different values
        data1 = ["1.0", "1.0", "1.0", "2.0"]
        df1 = pd.Series(data1)
        profiler = FloatColumn(df1.name)
        profiler.update(df1)

        data2 = ["0.0", "0.0", "2.0", "3.0", "3.0"]
        df2 = pd.Series(data2)
        profiler.update(df2)
        profile = profiler.profile

        est_median_abs_dev = profile["median_abs_deviation"]
        self.assertAlmostEqual(1.0, est_median_abs_dev, places=2)

        # with unique values
        data1 = ["1.0", "1.0", "1.0", "1.0"]
        df1 = pd.Series(data1)
        profiler = FloatColumn(df1.name)
        profiler.update(df1)

        data2 = ["1.0", "1.0", "1.0", "1.0", "1.0"]
        df2 = pd.Series(data2)
        profiler.update(df2)
        profile = profiler.profile

        est_median_abs_dev = profile["median_abs_deviation"]
        self.assertAlmostEqual(0.0, est_median_abs_dev, places=2)

    def test_data_type_ratio(self):
        data = np.linspace(-5, 5, 4)
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 1.0)

        df = pd.Series(["not a float"])
        profiler.update(df)
        self.assertEqual(profiler.data_type_ratio, 0.8)

    def test_profile(self):
        data = [2.5, 12.5, "not a float", 5, "not a float"]
        df = pd.Series(data).apply(str)

        profiler = FloatColumn(df.name)

        expected_profile = dict(
            min=2.5,
            max=12.5,
            mode=[2.5, 5, 12.5],
            median=5,
            sum=20.0,
            mean=20 / 3.0,
            variance=27 + 1 / 12.0,
            skewness=35 / 13 * np.sqrt(3 / 13),
            kurtosis=np.nan,
            median_abs_deviation=2.5,
            num_negatives=0,
            num_zeros=0,
            stddev=np.sqrt(27 + 1 / 12.0),
            histogram={
                "bin_counts": np.array([1, 1, 0, 1]),
                "bin_edges": np.array([2.5, 5.0, 7.5, 10.0, 12.5]),
            },
            quantiles={
                0: 2.5075,
                1: 5.005,
                2: 12.4925,
            },
            times=defaultdict(
                float,
                {
                    "histogram_and_quantiles": 1.0,
                    "precision": 1.0,
                    "max": 1.0,
                    "min": 1.0,
                    "skewness": 1.0,
                    "kurtosis": 1.0,
                    "sum": 1.0,
                    "variance": 1.0,
                    "num_zeros": 1.0,
                    "num_negatives": 1.0,
                },
            ),
            precision={
                "min": 1,
                "max": 3,
                "mean": 2.0,
                "var": 1.0,
                "std": 1.0,
                "sample_size": 3,
                "margin_of_error": 1.9,
                "confidence_level": 0.999,
            },
        )

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile["times"])
            profiler.update(df)
            profile = profiler.profile
            # Validate mode
            mode = profile.pop("mode")
            expected_mode = expected_profile.pop("mode")
            np.testing.assert_array_almost_equal(expected_mode, mode, decimal=2)
            # pop out the histogram to test separately from the rest of the dict
            # as we need comparison with some precision
            histogram = profile.pop("histogram")
            expected_histogram = expected_profile.pop("histogram")
            quantiles = profile.pop("quantiles")
            expected_quantiles = expected_profile.pop("quantiles")
            median = profile.pop("median")
            expected_median = expected_profile.pop("median")
            skewness = profile.pop("skewness")
            expected_skewness = expected_profile.pop("skewness")
            variance = profile.pop("variance")
            expected_variance = expected_profile.pop("variance")
            median_abs_dev = profile.pop("median_abs_deviation")
            expected_median_abs_dev = expected_profile.pop("median_abs_deviation")

            self.assertDictEqual(expected_profile, profile)
            self.assertDictEqual(expected_profile["precision"], profile["precision"])
            self.assertEqual(
                expected_histogram["bin_counts"].tolist(),
                histogram["bin_counts"].tolist(),
            )
            self.assertCountEqual(
                np.round(expected_histogram["bin_edges"], 12),
                np.round(histogram["bin_edges"], 12),
            )

            self.assertAlmostEqual(expected_quantiles[0], quantiles[249])
            self.assertAlmostEqual(expected_quantiles[1], quantiles[499])
            self.assertAlmostEqual(expected_quantiles[2], quantiles[749])
            self.assertAlmostEqual(expected_skewness, skewness)
            self.assertAlmostEqual(expected_variance, variance)
            self.assertAlmostEqual(expected_median_abs_dev, median_abs_dev)
            self.assertAlmostEqual(expected_median, median, places=2)

            # Validate time in datetime class has expected time after second update
            profiler.update(df)
            expected = defaultdict(
                float,
                {
                    "min": 2.0,
                    "max": 2.0,
                    "sum": 2.0,
                    "variance": 2.0,
                    "precision": 2.0,
                    "histogram_and_quantiles": 2.0,
                    "skewness": 2.0,
                    "kurtosis": 2.0,
                    "num_negatives": 2.0,
                    "num_zeros": 2.0,
                },
            )
            self.assertEqual(expected, profiler.profile["times"])

    def test_report(self):
        """Test report method in FloatColumn class under three (3) scenarios.
        First, test under scenario of disabling the entire
        precision dictionary. Second, test with no options and
        `remove_disabled_flag`=True. Finally, test no options and default
        `remove_disabled_flag`.
        """
        data = [1.1, 2.2, 3.3, 4.4]
        df = pd.Series(data).apply(str)

        # With FloatOptions and remove_disabled_flag == True
        options = FloatOptions()
        options.precision.is_enabled = False

        profiler = FloatColumn(df.name, options)
        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())
        self.assertNotIn("precision", report_keys)

        # w/o FloatOptions and remove_disabled_flag == True
        profiler = FloatColumn(df.name)
        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())
        self.assertIn("precision", report_keys)

        # w/o FloatOptions and remove_disabled_flag default
        profiler = FloatColumn(df.name)
        report = profiler.report()
        report_keys = list(report.keys())
        self.assertIn("precision", report_keys)

    def test_option_precision(self):
        data = [1.1, 2.2, 3.3, 4.4]
        df = pd.Series(data).apply(str)

        # Turn off precision
        options = FloatOptions()
        options.set({"precision.is_enabled": False})
        profiler = FloatColumn(df.name, options=options)
        profiler.update(df)
        self.assertEqual(None, profiler.precision["sample_size"])

        # Turn on precision, check sample_size
        options = FloatOptions()
        options.set({"precision.is_enabled": True})
        profiler = FloatColumn(df.name, options=options)
        profiler.update(df)
        self.assertEqual(4, profiler.precision["sample_size"])

        # Turn on precision, set 0.5 sample_size
        options = FloatOptions()
        options.set({"precision.sample_ratio": 0.5})
        profiler = FloatColumn(df.name, options=options)
        profiler.update(df)
        self.assertEqual(2, profiler.precision["sample_size"])

    def test_option_timing(self):
        data = [2.0, 12.5, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)

        options = FloatOptions()
        options.set({"min.is_enabled": False})

        profiler = FloatColumn(df.name, options=options)

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertEqual(defaultdict(float), profiler.profile["times"])
            profiler.update(df)

            # Validate the time in the datetime class has the expected time.
            profile = profiler.profile

            expected = defaultdict(
                float,
                {
                    "max": 1.0,
                    "sum": 1.0,
                    "variance": 1.0,
                    "precision": 1.0,
                    "skewness": 1.0,
                    "kurtosis": 1.0,
                    "num_negatives": 1.0,
                    "num_zeros": 1.0,
                    "histogram_and_quantiles": 15.0,
                },
            )
            self.assertCountEqual(expected, profile["times"])

            # Validate time in datetime class has expected time after second update
            profiler.update(df)
            expected = defaultdict(
                float,
                {
                    "max": 2.0,
                    "sum": 2.0,
                    "variance": 2.0,
                    "precision": 2.0,
                    "skewness": 2.0,
                    "kurtosis": 2.0,
                    "num_negatives": 2.0,
                    "num_zeros": 2.0,
                    "histogram_and_quantiles": 30.0,
                },
            )
            self.assertCountEqual(expected, profiler.profile["times"])

    def test_profile_merge(self):
        data = [2.0, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)

        data2 = [10.0, "not a float", 15.0, "not a float"]
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        expected_profile = dict(
            min=2.0,
            max=15.0,
            mode=[2, 6, 10, 15],
            sum=33.0,
            mean=8.25,
            variance=30.916666666666668,
            stddev=np.sqrt(30.916),
            skewness=918 * np.sqrt(3 / 371) / 371,
            kurtosis=-16068 / 19663,
            histogram={
                "bin_counts": np.array([1, 1, 1, 1]),
                "bin_edges": np.array([2.0, 5.25, 8.5, 11.75, 15.0]),
            },
        )

        profiler3 = profiler1 + profiler2

        expected_histogram = expected_profile.pop("histogram")
        profile3 = profiler3.profile
        histogram = profile3.pop("histogram")

        expected_mode = expected_profile.pop("mode")
        mode = profile3.pop("mode")
        np.testing.assert_array_almost_equal(expected_mode, mode, decimal=2)

        self.assertTrue(profiler3.bias_correction)
        self.assertAlmostEqual(
            profiler3.stddev, expected_profile.pop("stddev"), places=3
        )
        self.assertAlmostEqual(
            profiler3.variance, expected_profile.pop("variance"), places=3
        )
        self.assertAlmostEqual(
            profiler3.skewness, expected_profile.pop("skewness"), places=3
        )
        self.assertAlmostEqual(
            profiler3.kurtosis, expected_profile.pop("kurtosis"), places=3
        )
        self.assertEqual(profiler3.mean, expected_profile.pop("mean"))
        self.assertEqual(profiler3.histogram_selection, "doane")
        self.assertEqual(profiler3.min, expected_profile.pop("min"))
        self.assertEqual(profiler3.max, expected_profile.pop("max"))
        self.assertEqual(
            histogram["bin_counts"].tolist(), expected_histogram["bin_counts"].tolist()
        )
        self.assertCountEqual(histogram["bin_edges"], expected_histogram["bin_edges"])

    def test_profile_merge_for_zeros_and_negatives(self):
        data = [2.0, 8.5, "not an int", 6.0, -3, 0]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)

        data2 = [0.0, 3.5, "not an int", 125.0, 0, -0.1, -88]
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        expected_profile = dict(num_zeros=3, num_negatives=3)

        profiler3 = profiler1 + profiler2

        self.assertEqual(profiler3.num_zeros, expected_profile.pop("num_zeros"))
        self.assertEqual(profiler3.num_negatives, expected_profile.pop("num_negatives"))

    def test_profile_merge_edge_case(self):
        data = [2.0, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)
        profiler1.match_count = 0

        data2 = [10.0, "not a float", 15.0, "not a float"]
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        profiler3 = profiler1 + profiler2
        self.assertEqual(profiler3.stddev, profiler2.stddev)

        # test merge with empty data
        df1 = pd.Series([], dtype=object)
        profiler1 = FloatColumn("Float")
        profiler1.update(df1)

        df2 = pd.Series([], dtype=object)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        profiler = profiler1 + profiler2
        self.assertTrue(np.isnan(profiler.skewness))
        self.assertTrue(np.isnan(profiler.kurtosis))
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)

        df3 = pd.Series([2.0, 3.0]).apply(str)
        profiler3 = FloatColumn("Float")
        profiler3.update(df3)

        profiler = profiler1 + profiler3
        self.assertTrue(np.isnan(profiler.skewness))
        self.assertTrue(np.isnan(profiler.kurtosis))
        self.assertEqual(profiler.min, 2.0)
        self.assertEqual(profiler.max, 3.0)

        df4 = pd.Series([4.0, 5.0]).apply(str)
        profiler4 = FloatColumn("Float")
        profiler4.update(df4)

        profiler = profiler3 + profiler4
        self.assertEqual(profiler.skewness, 0)
        self.assertAlmostEqual(profiler.kurtosis, -1.2)
        self.assertEqual(profiler.min, 2.0)
        self.assertEqual(profiler.max, 5.0)
        self.assertEqual(profiler.num_zeros, 0)
        self.assertEqual(profiler.num_negatives, 0)

        df5 = pd.Series([0.0, 0.0, -1.1, -1.0]).apply(str)
        profiler5 = FloatColumn("Float")
        profiler5.update(df5)

        profiler = profiler4 + profiler5
        self.assertEqual(profiler.min, -1.1)
        self.assertEqual(profiler.max, 5)
        self.assertEqual(profiler.num_zeros, 2)
        self.assertEqual(profiler.num_negatives, 2)

    def test_custom_bin_count_merge(self):

        options = FloatOptions()
        options.histogram_and_quantiles.bin_count_or_method = 10

        data = [2.0, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float", options)
        profiler1.update(df)

        data2 = [10.0, "not a float", 15.0, "not a float"]
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float", options)
        profiler2.update(df2)

        # no warning should occur
        with warnings.catch_warnings(record=True) as w:
            merge_profile = profiler1 + profiler2
        self.assertListEqual([], w)
        self.assertEqual(10, merge_profile.user_set_histogram_bin)

        # make bin counts different and get warning
        profiler2.user_set_histogram_bin = 120
        with self.assertWarnsRegex(
            UserWarning,
            "User set histogram bin counts did not "
            "match. Choosing the larger bin count.",
        ):
            merged_profile = profiler1 + profiler2
        self.assertEqual(120, merged_profile.user_set_histogram_bin)

    def test_profile_merge_no_bin_overlap(self):

        data = [2.0, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float")
        profiler1.update(df)

        data2 = [10.0, "not a float", 15.0, "not a float"]
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float")
        profiler2.update(df2)

        # set bin names so no overlap
        profiler1.histogram_bin_method_names = ["No overlap 1"]
        profiler2.histogram_bin_method_names = ["No overlap 2"]

        with self.assertRaisesRegex(
            ValueError,
            "Profiles have no overlapping bin methods "
            "and therefore cannot be added together.",
        ):
            profiler1 + profiler2

    def test_profile_merge_with_different_options(self):
        # Creating first profiler with default options
        options = FloatOptions()
        options.max.is_enabled = False
        options.min.is_enabled = False
        options.histogram_and_quantiles.bin_count_or_method = None

        data = [2, 4, 6, 8]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float", options=options)
        profiler1.update(df)

        # Creating second profiler with separate options
        options = FloatOptions()
        options.min.is_enabled = False
        options.precision.is_enabled = False
        options.histogram_and_quantiles.bin_count_or_method = None

        data2 = [10, 15]
        df2 = pd.Series(data2).apply(str)
        profiler2 = FloatColumn("Float", options=options)
        profiler2.update(df2)

        # Asserting warning when adding 2 profilers with different options
        with warnings.catch_warnings(record=True) as w:
            profiler3 = profiler1 + profiler2
            list_of_warning_messages = []
            for warning in w:
                list_of_warning_messages.append(str(warning.message))

            warning1 = (
                "precision is disabled because it is not enabled in " "both profiles."
            )
            warning2 = "max is disabled because it is not enabled in both " "profiles."
            self.assertIn(warning1, list_of_warning_messages)
            self.assertIn(warning2, list_of_warning_messages)

        # Assert that these features are still merged
        profile = profiler3.profile
        self.assertEqual("doane", profiler3.histogram_selection)
        self.assertEqual(21.5, profile["variance"])
        self.assertEqual(45.0, profiler3.sum)

        # Assert that these features are not calculated
        self.assertIsNone(profiler3.max)
        self.assertIsNone(profiler3.min)
        self.assertEqual(None, profiler3.precision["min"])
        self.assertEqual(None, profiler3.precision["max"])

        # Creating profiler with precision to 0.1
        options = FloatOptions()
        options.max.is_enabled = False
        options.min.is_enabled = False
        options.histogram_and_quantiles.method = None

        data = [2, 4, 6, 8]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn("Float", options=options)
        profiler1.update(df)

    def test_float_column_with_wrong_options(self):
        with self.assertRaisesRegex(
            ValueError,
            "FloatColumn parameter 'options' must be of" " type FloatOptions.",
        ):
            profiler = FloatColumn("Float", options="wrong_data_type")

    def test_histogram_option_integration(self):
        # test setting bin methods
        options = FloatOptions()
        options.histogram_and_quantiles.bin_count_or_method = "sturges"
        num_profiler = FloatColumn(name="test", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["sturges"], num_profiler.histogram_bin_method_names)

        options.histogram_and_quantiles.bin_count_or_method = ["sturges", "doane"]
        num_profiler = FloatColumn(name="test2", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["sturges", "doane"], num_profiler.histogram_bin_method_names)

        # test histogram bin count set
        options.histogram_and_quantiles.bin_count_or_method = 100
        num_profiler = FloatColumn(name="test3", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["custom"], num_profiler.histogram_bin_method_names)

        # case when just 1 unique value, should just set bin size to be 1
        num_profiler.update(pd.Series(["1", "1"]))
        self.assertEqual(
            1, len(num_profiler.histogram_methods["custom"]["histogram"]["bin_counts"])
        )

        # case when more than 1 unique value, by virtue of a streaming update
        num_profiler.update(pd.Series(["2"]))
        self.assertEqual(
            100, len(num_profiler._stored_histogram["histogram"]["bin_counts"])
        )

        histogram, _ = num_profiler._histogram_for_profile("custom")
        self.assertEqual(100, len(histogram["bin_counts"]))

    def test_profile_merge_bin_edges_indices(self):
        vals = [
            4948484949555554544949495054485054.0,
            4948484948485749515554495054485054.0,
            4948484948505251545552524952485054.0,
            4948484952485048485551524952485054.0,
            4948484948515550575556535154485054.0,
            4948484950545549485651495054485054.0,
            4948484954565649505449524950485054.0,
            49484849535456545155495054485054.0,
            4948484954515651515451495054485054.0,
            4948484957575651505156554954485054.0,
        ]

        data = pd.Series(vals).astype(str)
        data_1 = data[:5]
        data_2 = data[5:]

        options = FloatOptions()

        options.set({"histogram_and_quantiles.is_enabled": True})

        profile_1 = FloatColumn("Float", options=options)
        profile_2 = FloatColumn("Float", options=options)

        profile_1.update(data_1)
        profile_2.update(data_2)

        profile_1 + profile_2

    def test_invalid_values(self):
        data = pd.Series(["-inf", "inf"])
        profiler = FloatColumn(data.name)

        with self.assertWarnsRegex(
            RuntimeWarning, "Infinite or invalid values found in data."
        ):
            profiler.update(data)
            # Verify values
            self.assertTrue(np.isnan(profiler.sum))
            self.assertTrue(np.isnan(profiler._biased_variance))
            self.assertTrue(np.isnan(profiler._biased_skewness))
            self.assertTrue(np.isnan(profiler._biased_kurtosis))

        # Update the data
        data2 = pd.Series(["-2", "-1", "1", "2", "-inf", "inf"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            profiler.update(data2)
            # Verify values are still NaN
            self.assertTrue(np.isnan(profiler.sum))
            self.assertTrue(np.isnan(profiler._biased_variance))
            self.assertTrue(np.isnan(profiler._biased_skewness))
            self.assertTrue(np.isnan(profiler._biased_kurtosis))
            # Verify warning-related things. In this case, we check
            # to make sure NO warnings were thrown since nothing should
            # even be updated
            self.assertEqual(0, len(w))

    def test_insufficient_counts(self):
        data = pd.Series(["0"])
        profiler = FloatColumn(data.name)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            profiler.update(data)
            var = profiler.variance
            skew = profiler.skewness
            kurt = profiler.kurtosis
            # Verify values are NaN
            self.assertTrue(np.isnan(var))
            self.assertTrue(np.isnan(skew))
            self.assertTrue(np.isnan(kurt))
            # Verify warning was raised properly
            self.assertEqual(3, len(w))
            for i in range(0, len(w)):
                self.assertEqual(w[i].category, RuntimeWarning)
                self.assertTrue(
                    "Insufficient match count to correct bias in" in str(w[i].message)
                )

        # Update the data so that the match count is good
        data2 = pd.Series(["-2", "-1", "1", "2"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            profiler.update(data2)
            var = profiler.variance
            skew = profiler.skewness
            kurt = profiler.kurtosis
            # Verify values are no longer NaN
            self.assertFalse(np.isnan(var))
            self.assertFalse(np.isnan(skew))
            self.assertFalse(np.isnan(kurt))
            # Verify warning-related things. In this case, we check
            # to make sure NO warnings were thrown since we have
            # a sufficient match count.
            self.assertEqual(0, len(w))

    def test_diff(self):
        data = [2.5, 12.5, "not a float", 5, "not a float"]
        df = pd.Series(data).apply(str)
        profiler1 = FloatColumn(df.name)
        profiler1.update(df)
        profile1 = profiler1.profile

        data = [1, 15, 0.5, 0]
        df = pd.Series(data).apply(str)
        profiler2 = FloatColumn(df.name)
        profiler2.update(df)
        profile2 = profiler2.profile

        # Assert the difference report is correct
        diff = profiler1.diff(profiler2)
        expected_diff = {
            "max": -2.5,
            "mean": profile1["mean"] - profile2["mean"],
            "min": 2.5,
            "stddev": profile1["stddev"] - profile2["stddev"],
            "sum": 3.5,
            "variance": profile1["variance"] - profile2["variance"],
            "median": 4.25,
            "mode": [[2.5, 12.5, 5], [], [1, 15, 0.5, 0]],
            "median_absolute_deviation": 2,
            "precision": {
                "min": 1,
                "max": 1,
                "mean": 1.0,
                "var": profile1["precision"]["var"] - profile2["precision"]["var"],
                "std": profile1["precision"]["std"] - profile2["precision"]["std"],
                "sample_size": -1,
                "margin_of_error": profile1["precision"]["margin_of_error"]
                - profiler2["precision"]["margin_of_error"],
            },
            "t-test": {
                "t-statistic": 0.5393164101529813,
                "conservative": {"df": 2.0, "p-value": 0.643676756587475},
                "welch": {"df": 4.999127432888682, "p-value": 0.6128117908944144},
            },
            "psi": 0,
        }
        profile_diff = profiler1.diff(profiler2)
        try:
            json.dumps(profile_diff)
        except TypeError as e:
            self.fail(
                "JSON Serializing issue with the profile diff. "
                "Exception raised: {}".format(str(e))
            )
        self.assertAlmostEqual(
            expected_diff.pop("median"), profile_diff.pop("median"), places=2
        )
        expected_diff_mode = expected_diff.pop("mode")
        diff_mode = profile_diff.pop("mode")
        for i in range(len(expected_diff_mode)):
            np.testing.assert_almost_equal(
                sorted(expected_diff_mode[i]), sorted(diff_mode[i]), 2
            )
        self.assertAlmostEqual(
            expected_diff.pop("median_absolute_deviation"),
            profile_diff.pop("median_absolute_deviation"),
            places=2,
        )
        self.assertDictEqual(expected_diff, profile_diff)

        # Assert type error is properly called
        with self.assertRaises(TypeError) as exc:
            profiler1.diff("Inproper input")
        self.assertEqual(
            str(exc.exception),
            "Unsupported operand type(s) for diff: 'FloatColumn' and" " 'str'",
        )
