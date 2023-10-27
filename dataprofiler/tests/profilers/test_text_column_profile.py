import json
import os
import unittest
import warnings
from collections import defaultdict
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.profilers import TextColumn, profiler_utils
from dataprofiler.profilers.json_decoder import load_column_profile
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profiler_options import TextOptions
from dataprofiler.tests.profilers import utils as test_utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestTextColumnProfiler(unittest.TestCase):
    def setUp(self):
        test_utils.set_seed(seed=0)

    def test_profiled_vocab(self):
        """
        Checks whether the vocab list for the profiler is correct.
        :return:
        """
        df1 = pd.Series(
            [
                "abcd",
                "aa",
                "abcd",
                "aa",
                "b",
                "4",
                "3",
                "2",
                "dfd",
                "2",
            ]
        ).apply(str)
        df2 = pd.Series(
            ["1", "1", "ee", "ff", "ff", "gg", "gg", "abcd", "aa", "b", "ee", "b"]
        ).apply(str)
        df3 = pd.Series(
            [
                "NaN",
                "b",
                "nan",
                "c",
            ]
        ).apply(str)

        text_profiler = TextColumn(df1.name)
        text_profiler.update(df1)

        unique_vocab = dict.fromkeys("".join(df1.tolist())).keys()
        self.assertCountEqual(unique_vocab, text_profiler.vocab)
        self.assertCountEqual(set(text_profiler.vocab), text_profiler.vocab)

        text_profiler.update(df2)
        df = pd.concat([df1, df2])
        unique_vocab = dict.fromkeys("".join(df.tolist())).keys()
        self.assertCountEqual(unique_vocab, text_profiler.vocab)
        self.assertCountEqual(set(text_profiler.vocab), text_profiler.vocab)

        text_profiler.update(df3)
        df = pd.concat([df1, df2, df3])
        unique_vocab = dict.fromkeys("".join(df.tolist())).keys()
        self.assertCountEqual(unique_vocab, text_profiler.vocab)

    def test_profiled_str_numerics(self):
        """
        Checks whether the vocab list for the profiler is correct.
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

        df1 = pd.Series(
            [
                "abcd",
                "aa",
                "abcd",
                "aa",
                "b",
                "4",
                "3",
                "2",
                "dfd",
                "2",
                np.nan,
            ]
        ).apply(str)
        df2 = pd.Series(
            ["1", "1", "ee", "ff", "ff", "gg", "gg", "abcd", "aa", "b", "ee", "b"]
        ).apply(str)
        df3 = pd.Series(
            [
                "NaN",
                "b",
                "nan",
                "c",
                None,
            ]
        ).apply(str)

        text_profiler = TextColumn(df1.name)
        text_profiler.update(df1)

        self.assertEqual(mean(df1.str.len()), text_profiler.mean)
        self.assertAlmostEqual(var(df1.str.len()), text_profiler.variance)
        self.assertAlmostEqual(np.sqrt(var(df1.str.len())), text_profiler.stddev)

        variance = batch_variance(
            mean_a=text_profiler.mean,
            var_a=text_profiler.variance,
            count_a=text_profiler.sample_size,
            mean_b=mean(df2.str.len()),
            var_b=var(df2.str.len()),
            count_b=df2.count(),
        )
        text_profiler.update(df2)
        df = pd.concat([df1, df2])
        self.assertEqual(df.str.len().mean(), text_profiler.mean)
        self.assertAlmostEqual(variance, text_profiler.variance)
        self.assertAlmostEqual(np.sqrt(variance), text_profiler.stddev)

        variance = batch_variance(
            mean_a=text_profiler.mean,
            var_a=text_profiler.variance,
            count_a=text_profiler.match_count,
            mean_b=mean(df3.str.len()),
            var_b=var(df3.str.len()),
            count_b=df3.count(),
        )
        text_profiler.update(df3)

        df = pd.concat([df1, df2, df3])
        self.assertEqual(df.str.len().mean(), text_profiler.mean)
        self.assertAlmostEqual(variance, text_profiler.variance)
        self.assertAlmostEqual(np.sqrt(variance), text_profiler.stddev)

    def test_base_case(self):
        data = pd.Series([], dtype=object)
        profiler = TextColumn(data.name)
        profiler.update(data)
        profiler.update(data)  # intentional to validate no changes if empty

        self.assertEqual(profiler.match_count, 0)
        self.assertEqual(profiler.min, None)
        self.assertEqual(profiler.max, None)
        self.assertEqual(profiler.sum, 0)
        self.assertIsNone(profiler.data_type_ratio)

    def test_data_ratio(self):
        # should always be 1.0 unless empty
        df1 = pd.Series(
            [
                "abcd",
                "aa",
                "abcd",
                "aa",
                "b",
                "4",
                "3",
                "2",
                "dfd",
                "2",
            ]
        ).apply(str)

        profiler = TextColumn(df1.name)
        profiler.update(df1)
        self.assertEqual(profiler.data_type_ratio, 1.0)

        # ensure batch update doesn't alter values
        profiler.update(df1)
        self.assertEqual(profiler.data_type_ratio, 1.0)

    def test_profiled_min(self):
        df = pd.Series(["aaa", "aa", "aaaa", "aaa"]).apply(str)

        profiler = TextColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.min, 2)

        df = pd.Series(["aa", "a"]).apply(str)
        profiler.update(df)
        self.assertEqual(profiler.min, 1)

    def test_profiled_max(self):
        df = pd.Series(["a", "aa", "a", "a"]).apply(str)

        profiler = TextColumn(df.name)
        profiler.update(df)
        self.assertEqual(profiler.max, 2)

        df = pd.Series(["aa", "aaa", "a"]).apply(str)
        profiler.update(df)
        self.assertEqual(profiler.max, 3)

    def test_profile(self):
        df = pd.Series(
            ["abcd", "aa", "abcd", "aa", "b", "4", "3", "2", "dfd", "2"]
        ).apply(str)
        profiler = TextColumn(df.name)
        expected_profile = dict(
            min=1.0,
            max=4.0,
            mode=[1],
            median=1.5,
            sum=20.0,
            mean=20.0 / 10.0,
            variance=14.0 / 9.0,
            skewness=45.0 / (14.0 * np.sqrt(14.0)),
            kurtosis=-1251.0 / 1372.0,
            stddev=np.sqrt(14.0 / 9.0),
            histogram={
                "bin_counts": np.array([5, 0, 2, 0, 1, 2]),
                "bin_edges": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
            },
            quantiles={0: 1.25, 1: 1.5, 2: 3.0},
            median_abs_deviation=0.5,
            vocab=["a", "b", "c", "d", "4", "3", "2", "f"],
            times=defaultdict(
                float,
                {
                    "vocab": 1.0,
                    "max": 1.0,
                    "min": 1.0,
                    "histogram_and_quantiles": 1.0,
                    "sum": 1.0,
                    "variance": 1.0,
                    "skewness": 1.0,
                    "num_negatives": 1.0,
                    "num_zeros": 1.0,
                    "kurtosis": 1.0,
                },
            ),
        )
        time_array = [float(x) for x in range(30, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            profiler.update(df)
            profile = profiler.profile
            expected_histogram = expected_profile.pop("histogram")
            expected_quantiles = expected_profile.pop("quantiles")
            expected_median_abs_dev = expected_profile.pop("median_abs_deviation")
            expected_vocab = expected_profile.pop("vocab")
            quantiles = profile.pop("quantiles")
            histogram = profile.pop("histogram")
            median_abs_dev = profile.pop("median_abs_deviation")
            vocab = profile.pop("vocab")

            # validate mode and median
            expected_mode = expected_profile.pop("mode")
            mode = profile.pop("mode")
            np.testing.assert_array_almost_equal(expected_mode, mode, decimal=2)

            expected_median = expected_profile.pop("median")
            median = profile.pop("median")
            self.assertAlmostEqual(expected_median, median, places=2)

            # key and value populated correctly
            self.assertDictEqual(expected_profile, profile)
            self.assertTrue(
                np.all(expected_histogram["bin_counts"] == histogram["bin_counts"])
            )
            self.assertTrue(
                np.all(expected_histogram["bin_edges"] == histogram["bin_edges"])
            )
            self.assertCountEqual(
                expected_quantiles,
                {0: quantiles[249], 1: quantiles[499], 2: quantiles[749]},
            )
            self.assertAlmostEqual(expected_median_abs_dev, median_abs_dev, places=2)
            self.assertCountEqual(expected_vocab, vocab)

    def test_report(self):
        """Test report method in TextColumn class under three (3) scenarios.
        First, test under scenario of disabling vocab. Second, test with no options and
        `remove_disabled_flag`=True. Finally, test no options and default
        `remove_disabled_flag`.
        """
        data = [2.0, 12.5, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)

        options = TextOptions()  # With TextOptions and remove_disabled_flag == True
        options.vocab.is_enabled = False

        profiler = TextColumn(df.name, options)
        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())
        self.assertNotIn("vocab", report_keys)

        profiler = TextColumn(
            df.name
        )  # w/o TextOptions and remove_disabled_flag == True
        report = profiler.report(remove_disabled_flag=True)
        report_keys = list(report.keys())
        self.assertIn("vocab", report_keys)

        profiler = TextColumn(
            df.name
        )  # w/o TextOptions and remove_disabled_flag default
        report = profiler.report()
        report_keys = list(report.keys())
        self.assertIn("vocab", report_keys)

    def test_option_timing(self):
        data = [2.0, 12.5, "not a float", 6.0, "not a float"]
        df = pd.Series(data).apply(str)

        options = TextOptions()
        options.set({"min.is_enabled": False})

        profiler = TextColumn(df.name, options=options)

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate that the times dictionary is empty
            self.assertCountEqual(defaultdict(float), profiler.profile["times"])
            profiler.update(df)

            # Validate the time in the datetime class has the expected time.
            profile = profiler.profile

            expected = defaultdict(
                float,
                {
                    "max": 1.0,
                    "sum": 1.0,
                    "variance": 1.0,
                    "skewness": 1.0,
                    "kurtosis": 1.0,
                    "histogram_and_quantiles": 15.0,
                    "vocab": 1.0,
                },
            )
            self.assertCountEqual(expected, profile["times"])

            # Validate time in datetime class has expected time after second
            # update
            profiler.update(df)
            expected = defaultdict(
                float,
                {
                    "max": 2.0,
                    "sum": 2.0,
                    "variance": 2.0,
                    "skewness": 2.0,
                    "kurtosis": 2.0,
                    "histogram_and_quantiles": 30.0,
                    "vocab": 2.0,
                },
            )
            self.assertCountEqual(expected, profiler.profile["times"])

    def test_merge_profile(self):
        df = pd.Series(
            ["abcd", "aa", "abcd", "aa", "b", "4", "3", "2", "dfd", "2"]
        ).apply(str)

        df2 = pd.Series(
            ["hello", "my", "name", "is", "Grant", "I", "have", "67", "dogs"]
        ).apply(str)

        expected_vocab = [
            "a",
            "b",
            "c",
            "d",
            "4",
            "3",
            "2",
            "f",
            "h",
            "e",
            "l",
            "o",
            "m",
            "y",
            "n",
            "i",
            "s",
            "G",
            "r",
            "t",
            "I",
            "v",
            "6",
            "7",
            "g",
        ]

        profiler = TextColumn("placeholder_name")
        profiler.update(df)

        profiler2 = TextColumn("placeholder_name")
        profiler2.update(df2)

        profiler3 = profiler + profiler2

        self.assertAlmostEqual(profiler3.mean, 2.578947, 3)
        self.assertEqual(
            profiler3.sample_size, profiler.sample_size + profiler2.sample_size
        )
        self.assertEqual(profiler3.max, profiler2.max)
        self.assertCountEqual(expected_vocab, profiler3.vocab)
        self.assertEqual(49, profiler3.sum)

    def test_merge_timing(self):
        profiler1 = TextColumn("placeholder_name")
        profiler2 = TextColumn("placeholder_name")

        profiler1.times = dict(vocab=2.0)
        profiler2.times = dict(vocab=3.0)

        time_array = [float(i) for i in range(2, 0, -1)]
        with mock.patch("time.time", side_effect=lambda: time_array.pop()):
            profiler3 = profiler1 + profiler2

            # __add__() call adds 1 so expected is 6
            expected_times = defaultdict(float, {"vocab": 6.0})
            self.assertCountEqual(expected_times, profiler3.profile["times"])

    def test_profile_merge_with_different_options(self):
        # Creating first profiler with default options
        options = TextOptions()
        options.max.is_enabled = False
        options.min.is_enabled = False
        options.histogram_and_quantiles.bin_count_or_method = None

        df = pd.Series(
            ["pancake", "banana", "lighthouse", "aa", "b", "4", "3", "2", "dfd", "2"]
        )

        profiler1 = TextColumn("Text", options=options)
        profiler1.update(df)

        # Creating second profiler with separate options
        options = TextOptions()
        options.min.is_enabled = False
        options.max.is_enabled = False
        options.vocab.is_enabled = False
        options.histogram_and_quantiles.bin_count_or_method = None
        df2 = pd.Series(
            ["hello", "my", "name", "is", "Grant", "I", "have", "67", "dogs"]
        )
        profiler2 = TextColumn("Text", options=options)
        profiler2.update(df2)

        # Asserting warning when adding 2 profilers with different options
        with self.assertWarnsRegex(
            RuntimeWarning,
            "vocab is disabled because it is not " "enabled in both profiles.",
        ):
            profiler3 = profiler1 + profiler2

        # Assert that these features are still merged
        profile = profiler3.profile
        self.assertEqual("doane", profiler3.histogram_selection)
        self.assertAlmostEqual(6.20467836, profile["variance"])
        self.assertEqual(62.0, profiler3.sum)

        # Assert that these features are not calculated
        self.assertIsNone(profiler3.max)
        self.assertIsNone(profiler3.min)
        self.assertListEqual([], profiler3.vocab)

    def test_text_column_with_wrong_options(self):
        with self.assertRaisesRegex(
            ValueError, "TextColumn parameter 'options' must be of" " type TextOptions."
        ):
            profiler = TextColumn("Text", options="wrong_data_type")

    def test_custom_bin_count_merge(self):

        options = TextOptions()
        options.histogram_and_quantiles.bin_count_or_method = 10

        data = ["this", "is", "a", "test"]
        df = pd.Series(data).apply(str)
        profiler1 = TextColumn("Float", options)
        profiler1.update(df)

        data2 = ["this", "is", "another", "test"]
        df2 = pd.Series(data2).apply(str)
        profiler2 = TextColumn("Float", options)
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

    def test_histogram_option_integration(self):
        options = TextOptions()
        options.histogram_and_quantiles.bin_count_or_method = "sturges"
        num_profiler = TextColumn(name="test", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["sturges"], num_profiler.histogram_bin_method_names)

        options.histogram_and_quantiles.bin_count_or_method = ["sturges", "doane"]
        num_profiler = TextColumn(name="test2", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["sturges", "doane"], num_profiler.histogram_bin_method_names)

        # test histogram bin count set
        options.histogram_and_quantiles.bin_count_or_method = 100
        num_profiler = TextColumn(name="test3", options=options)
        self.assertIsNone(num_profiler.histogram_selection)
        self.assertEqual(["custom"], num_profiler.histogram_bin_method_names)

        # case when just 1 unique value, should just set bin size to be 1
        num_profiler.update(pd.Series(["1", "1"]))
        self.assertEqual(
            1, len(num_profiler.histogram_methods["custom"]["histogram"]["bin_counts"])
        )

        # case when more than 1 unique value, by virtue of a streaming update
        num_profiler.update(pd.Series(["22"]))
        self.assertEqual(
            100, len(num_profiler._stored_histogram["histogram"]["bin_counts"])
        )

        histogram, _ = num_profiler._histogram_for_profile("custom")
        self.assertEqual(100, len(histogram["bin_counts"]))

    def test_diff(self):
        df = pd.Series(
            ["abcd", "aa", "abcd", "aa", "b", "4", "3", "2", "dfd", "2"]
        ).apply(str)

        df2 = pd.Series(
            ["hello", "my", "name", "is", "Grant", "I", "have", "67", "dogs"]
        ).apply(str)

        profiler1 = TextColumn(df.name)
        profiler1.update(df)
        profile1 = profiler1.profile

        profiler2 = TextColumn(df2.name)
        profiler2.update(df2)
        profile2 = profiler2.profile

        expected_diff = {
            "min": "unchanged",
            "max": -1.0,
            "sum": -9.0,
            "mean": profile1["mean"] - profile2["mean"],
            "median": -2.5,
            "mode": [[1], [], [2, 4]],
            "median_absolute_deviation": -0.5,
            "variance": profile1["variance"] - profile2["variance"],
            "stddev": profile1["stddev"] - profiler2["stddev"],
            "vocab": profiler_utils.find_diff_of_lists_and_sets(
                profile1["vocab"], profile2["vocab"]
            ),
            "t-test": {
                "t-statistic": -1.9339958714826413,
                "conservative": {"deg_of_free": 8.0, "p-value": 0.08916903961929257},
                "welch": {
                    "deg_of_free": 15.761400272034564,
                    "p-value": 0.07127621949432528,
                },
            },
        }

        profile_diff = profiler1.diff(profiler2)
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

    @mock.patch("time.time", return_value=0.0)
    def test_json_encode_after_update(self, time):
        df = pd.Series(
            [
                "abcd",
                "aa",
                "abcd",
                "lito-potamus",
                "b",
                "4",
                ".098",
                "2",
                "dfd",
                "2",
                "12.32",
            ]
        ).apply(str)

        text_options = TextOptions()
        text_options.histogram_and_quantiles.bin_count_or_method = 5
        profiler = TextColumn(df.name, text_options)
        with test_utils.mock_timeit():
            profiler.update(df)

        serialized_dict = json.loads(json.dumps(profiler, cls=ProfileEncoder))

        # popping quantiles and comparing as list below since it is so large
        serialized_quantiles = serialized_dict["data"].pop("quantiles")

        # popping vocab and comparing as set below since order is random
        serialized_vocab = serialized_dict["data"].pop("vocab")

        serialized = json.dumps(serialized_dict)

        expected = json.dumps(
            {
                "class": "TextColumn",
                "data": {
                    "min": 1.0,
                    "max": 12.0,
                    "_top_k_modes": 5,
                    "sum": 38.0,
                    "_biased_variance": 9.33884297520661,
                    "_biased_skewness": 1.8025833203700588,
                    "_biased_kurtosis": 2.7208317017777395,
                    "_median_is_enabled": True,
                    "_median_abs_dev_is_enabled": True,
                    "max_histogram_bin": 100000,
                    "min_histogram_bin": 1000,
                    "histogram_bin_method_names": ["custom"],
                    "histogram_selection": None,
                    "user_set_histogram_bin": 5,
                    "bias_correction": True,
                    "_mode_is_enabled": True,
                    "num_zeros": 0,
                    "num_negatives": 0,
                    "_num_quantiles": 1000,
                    "histogram_methods": {
                        "custom": {
                            "total_loss": 0.0,
                            "current_loss": 0.0,
                            "suggested_bin_count": 5,
                            "histogram": {"bin_counts": None, "bin_edges": None},
                        }
                    },
                    "_stored_histogram": {
                        "total_loss": 7.63,
                        "current_loss": 7.63,
                        "suggested_bin_count": 1000,
                        "histogram": {
                            "bin_counts": [6, 4, 0, 0, 1],
                            "bin_edges": [1.0, 3.2, 5.4, 7.6000000000000005, 9.8, 12.0],
                        },
                    },
                    "_batch_history": [
                        {
                            "match_count": 11,
                            "sample_size": 11,
                            "min": 1.0,
                            "max": 12.0,
                            "sum": 38.0,
                            "biased_variance": 9.33884297520661,
                            "mean": 3.4545454545454546,
                            "biased_skewness": 1.8025833203700588,
                            "biased_kurtosis": 2.7208317017777395,
                        }
                    ],
                    "_NumericStatsMixin__calculations": {
                        "min": "_get_min",
                        "max": "_get_max",
                        "sum": "_get_sum",
                        "variance": "_get_variance",
                        "skewness": "_get_skewness",
                        "kurtosis": "_get_kurtosis",
                        "histogram_and_quantiles": "_get_histogram_and_quantiles",
                    },
                    "name": None,
                    "col_index": np.nan,
                    "sample_size": 11,
                    "metadata": {},
                    "times": {
                        "vocab": 1.0,
                        "min": 1.0,
                        "max": 1.0,
                        "sum": 1.0,
                        "variance": 1.0,
                        "skewness": 1.0,
                        "kurtosis": 1.0,
                        "histogram_and_quantiles": 1.0,
                    },
                    "thread_safe": True,
                    "match_count": 11,
                    "_TextColumn__calculations": {"vocab": "_update_vocab"},
                    "type": "string",
                },
            }
        )

        expected_vocab = profiler.vocab
        expected_quantiles = profiler.quantiles

        self.assertEqual(serialized, expected)
        self.assertSetEqual(set(serialized_vocab), set(expected_vocab))
        self.assertListEqual(serialized_quantiles, expected_quantiles)

    def test_json_decode(self):
        fake_profile_name = None
        expected_profile = TextColumn(fake_profile_name)

        serialized = json.dumps(expected_profile, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_profile)

    def test_json_decode_after_update(self):
        fake_profile_name = "Fake profile name"
        # Actual deserialization

        # Build expected IntColumn
        df_int = pd.Series(
            [
                "abcd",
                "aa",
                "abcd",
                "lito-potamus",
                "b",
                "4",
                ".098",
                "2",
                "dfd",
                "2",
                "12.32",
            ]
        )
        expected_profile = TextColumn(fake_profile_name)

        with test_utils.mock_timeit():
            expected_profile.update(df_int)

        # Validate reporting before deserialization
        expected_profile.report()

        serialized = json.dumps(expected_profile, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        # Validate reporting after deserialization
        deserialized.report()
        test_utils.assert_profiles_equal(deserialized, expected_profile)

        df_str = pd.Series(
            [
                "aa",  # add existing
                "awsome",  # add new
            ]
        )

        # validating update after deserialization
        deserialized.update(df_str)

        assert deserialized.sample_size == 13
        assert set(deserialized.vocab) == {
            ".",
            "-",
            "1",
            "2",
            "3",
            "4",
            "8",
            "9",
            "0",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "i",
            "l",
            "m",
            "o",
            "p",
            "s",
            "t",
            "u",
            "w",
        }
