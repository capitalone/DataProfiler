import json
import os
import unittest
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd

from dataprofiler.profilers import CategoricalColumn
from dataprofiler.profilers.json_decoder import load_column_profile
from dataprofiler.profilers.json_encoder import ProfileEncoder
from dataprofiler.profilers.profile_builder import StructuredColProfiler
from dataprofiler.profilers.profiler_options import CategoricalOptions

from . import utils as test_utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCategoricalColumn(unittest.TestCase):
    @classmethod
    def setUp(self):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        cls.input_file_path = os.path.join(
            test_root_path, "data", "csv/aws_honeypot_marx_geo.csv"
        )
        columns_to_read = ["host", "localeabbr"]
        cls.aws_dataset = pd.read_csv(cls.input_file_path)[columns_to_read]

    def test_correct_categorical_model_string(self):
        dataset = self.aws_dataset["host"].dropna()
        profile = CategoricalColumn(dataset.name)
        profile.update(dataset)
        self.assertEqual(1.0, profile.is_match)
        self.assertEqual(2997, profile.sample_size)
        categories = {
            "groucho-oregon",
            "groucho-us-east",
            "groucho-singapore",
            "groucho-tokyo",
            "groucho-sa",
            "zeppo-norcal",
            "groucho-norcal",
            "groucho-eu",
            "groucho-sydney",
        }
        self.assertCountEqual(categories, profile.categories)

    def test_stop_condition_is_met_initially(self):
        dataset = pd.Series(["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)
        profile = CategoricalColumn("test dataset")
        profile.max_sample_size_to_check_stop_condition = 0
        profile.stop_condition_unique_value_ratio = 0
        profile.update(dataset)

        self.assertTrue(profile._stop_condition_is_met)
        self.assertEqual(profile.categories, [])
        self.assertEqual(profile.unique_ratio, 0.1)
        self.assertEqual(profile.unique_count, 4)
        self.assertFalse(profile.is_match)

    def test_stop_condition_is_met_after_initial_profile(self):
        dataset = pd.Series(["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)
        profile = CategoricalColumn("test dataset")
        profile.max_sample_size_to_check_stop_condition = len(dataset) + 1
        profile.stop_condition_unique_value_ratio = 0
        profile.update(dataset)

        self.assertFalse(profile._stop_condition_is_met)

        dataset.loc[len(dataset.index)] = "Testing past ratio"
        profile.update(dataset)

        self.assertTrue(profile._stop_condition_is_met)
        self.assertEqual([], profile.categories)
        self.assertEqual(5, profile.unique_count)
        self.assertEqual((5 / 81), profile.unique_ratio)
        self.assertFalse(profile.is_match)

        profile.update(dataset)
        self.assertTrue(profile._stop_condition_is_met)
        self.assertEqual([], profile.categories)
        self.assertEqual(5, profile.unique_count)
        self.assertEqual((5 / 81), profile.unique_ratio)
        self.assertEqual(81, profile.sample_size)
        self.assertFalse(profile.is_match)

    def test_timeit_profile(self):
        dataset = self.aws_dataset["host"].dropna()
        profile = CategoricalColumn(dataset.name)

        time_array = [float(x) for x in range(17, 0, -1)]
        with patch("time.time", side_effect=lambda: time_array.pop()):
            # Validate the time in the column class is empty.
            self.assertEqual(defaultdict(float), profile.profile["times"])

            # Validate the time in the column class has the expected time.
            profile.update(dataset)
            expected = defaultdict(float, {"categories": 1.0})
            self.assertEqual(expected, profile.profile["times"])

            # Validate expected time after second update to profile
            profile.update(dataset)
            expected = defaultdict(float, {"categories": 2.0})
            self.assertEqual(expected, profile.profile["times"])

    def test_mixed_categorical_col_integer_string(self):
        dataset = self.aws_dataset["localeabbr"].dropna()
        profile = CategoricalColumn(dataset.name)
        profile.update(dataset)

        categories = {
            "36",
            "OR",
            "IL",
            "41",
            "51",
            "13",
            "21",
            "WA",
            "11",
            "CA",
            "37",
            "TX",
            "10",
            "SPE",
            "34",
            "32",
            "35",
            "23",
            "NM",
            "NV",
            "33",
            "44",
            "22",
            "GR",
            "15",
            "MI",
            "43",
            "FL",
            "TA",
            "KY",
            "SP",
            "SE",
            "AZ",
            "42",
            "NJ",
            "DC",
            "77",
            "50",
            "NGR",
            "31",
            "DIF",
            "61",
            "45",
            "NY",
            "MH",
            "ALT",
            "CH",
            "NSW",
            "MS",
            "81",
            "GP",
            "KU",
            "14",
            "53",
            "64",
            "AP",
            "38",
            "IRK",
            "CL",
            "TXG",
            "LUA",
            "ANT",
            "PA",
            "QC",
            "RS",
            "MO",
            "C",
            "MOW",
            "ENG",
            "ON",
            "CE",
            "TN",
            "PI",
            "VLG",
            "DL",
            "VL",
            "GE",
            "WP",
            "GO",
            "BS",
            "KEM",
            "MA",
            "BEL",
            "LB",
            "CU",
            "EC",
            "PB",
            "RIX",
            "B",
            "RJ",
            "VA",
            "7",
            "SL",
            "BE",
            "47",
            "RM",
            "BIH",
            "SD",
            "OH",
            "PR",
            "M",
            "SN",
            "COR",
            "63",
            "E",
            "BD",
            "VI",
            "SAM",
            "BA",
            "WY",
            "62",
            "4",
            "PER",
            "WKO",
            "KYA",
            "6",
            "MN",
            "SA",
            "8",
            "CO",
            "IS",
            "RIS",
            "FS",
            "IN",
            "LIV",
            "IA",
            "24",
            "VIC",
            "27",
            "16",
            "PK",
            "WB",
            "NH",
            "DAS",
            "CT",
            "CN",
            "BIR",
            "NVS",
            "MG",
            "3",
            "PH",
            "TO",
            "1",
            "HE",
            "VGG",
            "BU",
            "AB",
            "NIZ",
            "92",
            "46",
            "MZ",
            "FR",
        }

        self.assertEqual(2120, profile.sample_size)
        self.assertCountEqual(categories, profile.categories)

    def test_categorical_mapping(self):
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
        )
        df2 = pd.Series(
            [
                "1",
                "null",
                "ee",
                "NaN",
                "ff",
                "nan",
                "gg",
                "None",
                "aa",
                "b",
                "ee",
            ]
        )
        df3 = pd.Series(
            [
                "NaN",
                "b",
                "nan",
                "c",
                None,
            ]
        )

        column_profile = StructuredColProfiler(df1)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        num_null_types = 1
        num_nan_count = 1
        categories = df1.apply(str).unique().tolist()
        self.assertCountEqual(
            categories, cat_profiler.categories + column_profile.null_types
        )
        self.assertEqual(num_null_types, len(column_profile.null_types))
        self.assertEqual(num_nan_count, len(column_profile.null_types_index["nan"]))
        expected = {"abcd": 2, "aa": 2, "b": 1, "4": 1, "3": 1, "2": 2, "dfd": 1}
        self.assertDictEqual(expected, cat_profiler._categories)
        num_null_types = 4
        num_nan_count = 2
        categories = pd.concat([df1, df2]).apply(str).unique().tolist()
        column_profile.update_profile(df2)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        self.assertCountEqual(
            categories, cat_profiler.categories + column_profile.null_types
        )
        self.assertEqual(num_null_types, len(column_profile.null_types))
        self.assertEqual(num_nan_count, len(column_profile.null_types_index["nan"]))
        expected = {
            "abcd": 2,
            "aa": 3,
            "b": 2,
            "4": 1,
            "3": 1,
            "2": 2,
            "dfd": 1,
            "1": 1,
            "ee": 2,
            "ff": 1,
            "gg": 1,
        }
        self.assertDictEqual(expected, cat_profiler._categories)

        num_null_types = 4
        num_nan_count = 3
        categories = pd.concat([df1, df2, df3]).apply(str).unique().tolist()
        column_profile.update_profile(df3)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        self.assertCountEqual(
            categories, cat_profiler.categories + column_profile.null_types
        )
        self.assertEqual(num_null_types, len(column_profile.null_types))
        self.assertEqual(num_nan_count, len(column_profile.null_types_index["nan"]))
        self.assertNotEqual(num_nan_count, len(column_profile.null_types_index["NaN"]))

    def test_true_categorical_report(self):
        df_categorical = pd.Series(
            [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "c",
                "c",
            ]
        )
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        report = profile.profile

        self.assertIsNotNone(report.pop("times", None))
        expected_profile = dict(
            categorical=True,
            statistics=dict(
                [
                    ("unique_count", 3),
                    ("unique_ratio", 0.25),
                    ("categories", ["a", "b", "c"]),
                    ("categorical_count", {"a": 3, "b": 4, "c": 5}),
                    ("gini_impurity", (27 / 144) + (32 / 144) + (35 / 144)),
                    ("unalikeability", 2 * (12 + 15 + 20) / 132),
                ]
            ),
        )

        # We have to pop these values because sometimes the order changes
        self.assertCountEqual(
            expected_profile["statistics"].pop("categories"),
            report["statistics"].pop("categories"),
        )
        self.assertCountEqual(
            expected_profile["statistics"].pop("categorical_count"),
            report["statistics"].pop("categorical_count"),
        )
        self.assertEqual(report, expected_profile)

    def test_false_categorical_report(self):
        df_non_categorical = pd.Series(list(map(str, range(0, 20))))
        profile = CategoricalColumn(df_non_categorical.name)
        profile.update(df_non_categorical)

        report = profile.profile
        self.assertIsNotNone(report.pop("times", None))
        expected_profile = dict(
            categorical=False,
            statistics=dict(
                [
                    ("unique_count", 20),
                    ("unique_ratio", 1),
                ]
            ),
        )
        self.assertEqual(report, expected_profile)

    def test_report(self):
        df_non_categorical = pd.Series(list(map(str, range(0, 20))))
        profile = CategoricalColumn(df_non_categorical.name)
        profile.update(df_non_categorical)

        report1 = profile.profile
        report2 = profile.report(remove_disabled_flag=False)
        report3 = profile.report(remove_disabled_flag=True)
        self.assertDictEqual(report1, report2)
        self.assertDictEqual(report1, report3)

    def test_categorical_merge(self):
        df1 = pd.Series(
            ["abcd", "aa", "abcd", "aa", "b", "4", "3", "2", "dfd", "2", np.nan]
        )
        df2 = pd.Series(
            ["1", "null", "ee", "NaN", "ff", "nan", "gg", "None", "aa", "b", "ee"]
        )

        # Expected is based off insertion order
        expected_categories = [
            "abcd",
            "aa",
            "b",
            "4",
            "3",
            "2",
            "dfd",
            np.nan,
            "1",
            "null",
            "ee",
            "NaN",
            "ff",
            "nan",
            "gg",
            "None",
        ]

        profile = CategoricalColumn("Name")
        profile.update(df1)

        expected_dict = {
            "abcd": 2,
            "aa": 2,
            "b": 1,
            "4": 1,
            "3": 1,
            "2": 2,
            "dfd": 1,
            np.nan: 1,
        }
        self.assertDictEqual(expected_dict, profile._categories)

        profile2 = CategoricalColumn("Name")
        profile2.update(df2)

        # Add profiles
        profile3 = profile + profile2
        self.assertCountEqual(expected_categories, profile3.categories)
        self.assertEqual(
            profile3.sample_size, profile.sample_size + profile2.sample_size
        )
        self.assertEqual(profile3.is_match, False)
        expected_dict = {
            "abcd": 2,
            "aa": 3,
            "b": 2,
            "4": 1,
            "3": 1,
            "2": 2,
            np.nan: 1,
            "dfd": 1,
            "1": 1,
            "ee": 2,
            "ff": 1,
            "gg": 1,
            "NaN": 1,
            "None": 1,
            "nan": 1,
            "null": 1,
        }
        self.assertDictEqual(expected_dict, profile3._categories)

        report = profile3.profile
        self.assertIsNotNone(report.pop("times", None))
        expected_profile = dict(
            categorical=False,
            statistics=dict([("unique_count", 16), ("unique_ratio", 16 / 22)]),
        )
        self.assertEqual(report, expected_profile)

        # Add again
        profile3 = profile + profile3
        self.assertCountEqual(expected_categories, profile3.categories)
        self.assertEqual(profile3.sample_size, 33)

        # Check is_match and unique_ratio if the sample size was small
        self.assertEqual(profile3.is_match, False)
        self.assertEqual(profile3.unique_ratio, 16 / 33)

        report = profile3.profile
        self.assertIsNotNone(report.pop("times", None))
        expected_profile = dict(
            categorical=False,
            statistics=dict(
                [
                    ("unique_count", 16),
                    ("unique_ratio", 16 / 33),
                ]
            ),
        )
        self.assertEqual(report, expected_profile)

        # Check is_match and unique_ratio if the sample size was large
        profile3.sample_size = 1000
        self.assertEqual(profile3.is_match, True)
        self.assertEqual(profile3.unique_ratio, 16 / 1000)

        report = profile3.profile
        self.assertIsNotNone(report.pop("times", None))
        report_categories = report["statistics"].pop("categories")
        report_count = report["statistics"].pop("categorical_count")
        report_gini = report["statistics"].pop("gini_impurity")
        expected_profile = dict(
            categorical=True,
            statistics=dict(
                [
                    ("unique_count", 16),
                    ("unique_ratio", 16 / 1000),
                    ("unalikeability", 32907 / (1000000 - 1000)),
                ]
            ),
        )
        expected_gini = (
            (1 * ((5 / 1000) * (995 / 1000)))
            + (2 * ((4 / 1000) * (996 / 1000)))
            + (1 * ((3 / 1000) * (997 / 1000)))
            + (5 * ((2 / 1000) * (998 / 1000)))
            + (7 * ((1 / 1000) * (999 / 1000)))
        )
        self.assertAlmostEqual(report_gini, expected_gini)
        self.assertEqual(report, expected_profile)
        self.assertCountEqual(
            report_categories,
            [
                "abcd",
                "aa",
                "2",
                np.nan,
                "4",
                "b",
                "3",
                "dfd",
                "ee",
                "ff",
                "nan",
                "None",
                "1",
                "gg",
                "null",
                "NaN",
            ],
        )
        expected_dict = {
            "aa": 5,
            "2": 4,
            "abcd": 4,
            "b": 3,
            np.nan: 2,
            "dfd": 2,
            "3": 2,
            "4": 2,
            "ee": 2,
            "null": 1,
            "ff": 1,
            "NaN": 1,
            "1": 1,
            "nan": 1,
            "gg": 1,
            "None": 1,
        }
        self.assertCountEqual(report_count, expected_dict)

        # Setting up of profile with stop condition not yet met
        profile_w_stop_cond_1 = CategoricalColumn("merge_stop_condition_test")
        profile_w_stop_cond_1.max_sample_size_to_check_stop_condition = 12
        profile_w_stop_cond_1.stop_condition_unique_value_ratio = 0.0002
        profile_w_stop_cond_1.update(df1)

        self.assertFalse(profile_w_stop_cond_1._stop_condition_is_met)

        # Setting up of profile without stop condition met
        profile_w_stop_cond_2 = CategoricalColumn("merge_stop_condition_test")
        profile_w_stop_cond_2.max_sample_size_to_check_stop_condition = 12
        profile_w_stop_cond_2.stop_condition_unique_value_ratio = 0.0001
        profile_w_stop_cond_2.update(df2)

        self.assertFalse(profile_w_stop_cond_1._stop_condition_is_met)

        # Merge profiles w/o condition met
        merged_stop_cond_profile_1 = profile_w_stop_cond_1 + profile_w_stop_cond_2

        # Test whether merge caused stop condition to be hit
        self.assertTrue(merged_stop_cond_profile_1._stop_condition_is_met)
        self.assertEqual([], merged_stop_cond_profile_1.categories)
        self.assertEqual(16, merged_stop_cond_profile_1.unique_count)
        self.assertEqual((16 / 22), merged_stop_cond_profile_1.unique_ratio)
        self.assertEqual(22, merged_stop_cond_profile_1.sample_size)

        # Merge profile w/ and w/o condition met
        merged_stop_cond_profile_2 = merged_stop_cond_profile_1 + profile_w_stop_cond_2

        # Test whether merged profile stays persistently with condition met
        self.assertTrue(merged_stop_cond_profile_2._stop_condition_is_met)
        self.assertEqual([], merged_stop_cond_profile_2.categories)
        self.assertEqual(16, merged_stop_cond_profile_2.unique_count)
        self.assertEqual(
            merged_stop_cond_profile_1.unique_ratio,
            merged_stop_cond_profile_2.unique_ratio,
        )
        self.assertEqual(22, merged_stop_cond_profile_2.sample_size)

        # Merge profile w/ and w/o condition met (ensure operator communitivity)
        merged_stop_cond_profile_3 = profile_w_stop_cond_2 + merged_stop_cond_profile_1
        self.assertTrue(merged_stop_cond_profile_3._stop_condition_is_met)
        self.assertEqual([], merged_stop_cond_profile_3.categories)
        self.assertEqual(16, merged_stop_cond_profile_3.unique_count)
        self.assertEqual(
            merged_stop_cond_profile_1.unique_ratio,
            merged_stop_cond_profile_2.unique_ratio,
        )
        self.assertEqual(22, merged_stop_cond_profile_2.sample_size)

        # Ensure successful merge without stop condition met
        profile_w_stop_cond_1.stop_condition_unique_value_ratio = 0.99
        merge_stop_conditions_not_met = profile_w_stop_cond_1 + profile_w_stop_cond_1
        self.assertFalse(merge_stop_conditions_not_met._stop_condition_is_met)
        self.assertIsNone(merge_stop_conditions_not_met._stopped_at_unique_count)
        self.assertIsNone(merge_stop_conditions_not_met._stopped_at_unique_ratio)
        self.assertEqual(
            0.99, merge_stop_conditions_not_met.stop_condition_unique_value_ratio
        )
        self.assertEqual(
            12, merge_stop_conditions_not_met.max_sample_size_to_check_stop_condition
        )

    def test_gini_impurity(self):
        # Normal test
        df_categorical = pd.Series(["y", "y", "y", "y", "n", "n", "n"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        expected_val = ((4 / 7) * (3 / 7)) + ((4 / 7) * (3 / 7))
        self.assertAlmostEqual(profile.gini_impurity, expected_val)

        # One class only test
        df_categorical = pd.Series(["y", "y", "y", "y", "y", "y", "y"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        expected_val = 0
        self.assertEqual(profile.gini_impurity, expected_val)

        # Empty test
        df_categorical = pd.Series([])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(profile.gini_impurity, None)

    def test_categorical_diff(self):
        # test psi new category in another profile
        df_categorical = pd.Series(["y", "y", "y", "y", "n", "n", "n"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)

        df_categorical = pd.Series(["y", "maybe", "y", "y", "n", "n", "maybe"])
        profile2 = CategoricalColumn(df_categorical.name)
        profile2.update(df_categorical)

        # chi2-statistic = sum((observed-expected)^2/expected for each category in each column)
        # df = categories - 1
        # p-value found through using chi2 CDF
        expected_diff = {
            "categorical": "unchanged",
            "statistics": {
                "unique_count": -1,
                "unique_ratio": -0.14285714285714285,
                "categories": [[], ["y", "n"], ["maybe"]],
                "gini_impurity": -0.16326530612244894,
                "unalikeability": -0.19047619047619047,
                "categorical_count": {"y": 1, "n": 1, "maybe": -2},
                "chi2-test": {
                    "chi2-statistic": 82 / 35,
                    "deg_of_free": 2,
                    "p-value": 0.3099238764710244,
                },
                "psi": 0.0990210257942779,
            },
        }
        actual_diff = profile.diff(profile2)
        self.assertDictEqual(expected_diff, actual_diff)

        # Test with one categorical column matching
        df_not_categorical = pd.Series(
            [
                "THIS",
                "is",
                "not",
                "a",
                "categorical",
                "column",
                "for",
                "testing",
                "purposes",
                "Bada",
                "Bing",
                "Badaboom",
            ]
        )
        profile2 = CategoricalColumn(df_not_categorical.name)
        profile2.update(df_not_categorical)
        expected_diff = {
            "categorical": [True, False],
            "statistics": {"unique_count": -10, "unique_ratio": -0.7142857142857143},
        }
        self.assertDictEqual(expected_diff, profile.diff(profile2))

        # Test diff with psi enabled
        df_categorical = pd.Series(["y", "y", "y", "y", "n", "n", "n", "maybe"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)

        df_categorical = pd.Series(["y", "maybe", "y", "y", "n", "n", "maybe"])
        profile2 = CategoricalColumn(df_categorical.name)
        profile2.update(df_categorical)

        expected_diff = {
            "categorical": "unchanged",
            "statistics": {
                "unique_count": "unchanged",
                "unique_ratio": -0.05357142857142855,
                "chi2-test": {
                    "chi2-statistic": 0.6122448979591839,
                    "deg_of_free": 2,
                    "p-value": 0.7362964551863367,
                },
                "categories": "unchanged",
                "gini_impurity": -0.059311224489795866,
                "unalikeability": -0.08333333333333326,
                "psi": 0.16814961527477595,
                "categorical_count": {"y": 1, "n": 1, "maybe": -1},
            },
        }
        self.assertDictEqual(expected_diff, profile.diff(profile2))

    def test_unalikeability(self):
        df_categorical = pd.Series(["a", "a"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(profile.unalikeability, 0)

        df_categorical = pd.Series(["a", "c", "b"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(profile.unalikeability, 1)

        df_categorical = pd.Series(["a", "a", "a", "b", "b", "b"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(profile.unalikeability, 18 / 30)

        df_categorical = pd.Series(["a", "a", "b", "b", "b", "a", "c", "c", "a", "a"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(profile.unalikeability, 2 * (10 + 15 + 6) / 90)

        df_categorical = pd.Series(["a"])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(0, profile.unalikeability)

        df_categorical = pd.Series([])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        self.assertEqual(None, profile.unalikeability)

    def test_top_k_categories_change(self):
        # Test if top_k_categories is None
        options = CategoricalOptions()
        df_series = pd.Series(["a", "a", "b", "c", "d", "e", "e", "e", "f", "g"])
        profile = CategoricalColumn(df_series.name, options)
        profile.update(df_series)
        self.assertEqual(len(profile.profile["statistics"]["categorical_count"]), 7)

        # Test if top_k_categories is less than the count of categories
        profile._top_k_categories = 6
        self.assertEqual(len(profile.profile["statistics"]["categorical_count"]), 6)

        # Test if top_k_categories is greater than the count of categories
        options.top_k_categories = 6
        df_series = pd.Series(["a", "a", "b", "c", "d"])
        profile = CategoricalColumn(df_series.name, options)
        profile.update(df_series)
        self.assertEqual(len(profile.profile["statistics"]["categorical_count"]), 4)

    def test_categorical_stop_condition_options_set(self):
        # Test if categorical conditions is not set
        options = CategoricalOptions()
        profile = CategoricalColumn("test_unset")
        self.assertIsNone(profile.stop_condition_unique_value_ratio)
        self.assertIsNone(profile.max_sample_size_to_check_stop_condition)

        # Test if categorical conditions is set
        options.stop_condition_unique_value_ratio = 0.20
        options.max_sample_size_to_check_stop_condition = 100
        profile = CategoricalColumn("test_set", options=options)
        self.assertEqual(0.20, profile.stop_condition_unique_value_ratio)
        self.assertEqual(100, profile.max_sample_size_to_check_stop_condition)

    def test_json_encode(self):
        profile = CategoricalColumn("0")

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "CategoricalColumn",
                "data": {
                    "name": "0",
                    "col_index": np.nan,
                    "sample_size": 0,
                    "metadata": dict(),
                    "times": defaultdict(),
                    "thread_safe": True,
                    "_categories": defaultdict(int),
                    "_CategoricalColumn__calculations": dict(),
                    "_top_k_categories": None,
                    "max_sample_size_to_check_stop_condition": None,
                    "stop_condition_unique_value_ratio": None,
                    "_stop_condition_is_met": False,
                    "_stopped_at_unique_ratio": None,
                    "_stopped_at_unique_count": None,
                    "_cms_max_num_heavy_hitters": 5000,
                    "cms_num_hashes": None,
                    "cms_num_buckets": None,
                    "cms": None,
                },
            }
        )

        self.assertEqual(serialized, expected)

    def test_json_encode_after_update(self):
        df_categorical = pd.Series(
            [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "c",
                "c",
            ]
        )
        profile = CategoricalColumn(df_categorical.name)

        with test_utils.mock_timeit():
            profile.update(df_categorical)

        serialized = json.dumps(profile, cls=ProfileEncoder)
        expected = json.dumps(
            {
                "class": "CategoricalColumn",
                "data": {
                    "name": None,
                    "col_index": np.nan,
                    "sample_size": 12,
                    "metadata": {},
                    "times": {"categories": 1.0},
                    "thread_safe": True,
                    "_categories": {"c": 5, "b": 4, "a": 3},
                    "_CategoricalColumn__calculations": {},
                    "_top_k_categories": None,
                    "max_sample_size_to_check_stop_condition": None,
                    "stop_condition_unique_value_ratio": None,
                    "_stop_condition_is_met": False,
                    "_stopped_at_unique_ratio": None,
                    "_stopped_at_unique_count": None,
                    "_cms_max_num_heavy_hitters": 5000,
                    "cms_num_hashes": None,
                    "cms_num_buckets": None,
                    "cms": None,
                },
            }
        )

        self.assertEqual(serialized, expected)

    def test_json_decode(self):
        fake_profile_name = None
        expected_profile = CategoricalColumn(fake_profile_name)

        serialized = json.dumps(expected_profile, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_profile)

    def test_json_decode_after_update(self):
        fake_profile_name = "Fake profile name"
        # Actual deserialization

        # Build expected CategoricalColumn
        df_categorical = pd.Series(
            [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "c",
                "c",
            ]
        )
        expected_profile = CategoricalColumn(fake_profile_name)

        with test_utils.mock_timeit():
            expected_profile.update(df_categorical)

        serialized = json.dumps(expected_profile, cls=ProfileEncoder)
        deserialized = load_column_profile(json.loads(serialized))

        test_utils.assert_profiles_equal(deserialized, expected_profile)

        df_categorical = pd.Series(
            [
                "a",  # add existing
                "d",  # add new
            ]
        )

        # validating update after deserialization
        deserialized.update(df_categorical)

        assert deserialized.sample_size == 14
        assert deserialized.categorical_counts == {"c": 5, "b": 4, "a": 4, "d": 1}

    def test_cms_max_num_heavy_hitters(self):
        df_categorical = pd.Series(["a"] * 5 + ["b"] * 5 + ["c"] * 10)

        options = CategoricalOptions()
        options.cms = True
        options.cms_confidence = 0.95
        options.cms_relative_error = 0.01
        options.cms_max_num_heavy_hitters = 2

        profile = CategoricalColumn("test_name", options)
        profile.update(df_categorical)

        self.assertEqual({"c": 10}, profile._categories)
        self.assertTrue(profile.sample_size >= 10)

    def test_cms_update_hybrid_batch_stream(self):
        dataset = pd.Series(["a"] * 7 + ["b"] * 9 + ["c"] * 14)
        dataset1 = pd.Series(["a"] * 9 + ["b"] * 11 + ["c"] * 9 + ["d"] * 1)

        options = CategoricalOptions()
        options.cms = True
        options.cms_confidence = 0.95
        options.cms_relative_error = 0.01
        options.cms_max_num_heavy_hitters = 3

        profile = CategoricalColumn("test_name", options)
        profile.update(dataset)

        expected_categories = ["c"]
        expected_categories_dict = {"c": 14}

        self.assertEqual(profile.sample_size, len(dataset))
        self.assertEqual(profile._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile.categories)

        profile.update(dataset1)
        expected_categories = ["b", "c"]
        expected_categories_dict = {"b": 20, "c": 23}

        self.assertEqual(profile.sample_size, len(dataset) + len(dataset1))
        self.assertEqual(profile._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile.categories)

    def test_cms_profile_merge_via_add(self):

        dataset = pd.Series(["a"] * 9 + ["b"] * 12 + ["c"] * 9)
        dataset1 = pd.Series(["a"] * 6 + ["b"] * 10 + ["c"] * 14)

        expected_categories = ["b", "c"]
        expected_categories_dict = {"b": 22, "c": 23}
        options = CategoricalOptions()
        options.cms = True
        options.cms_confidence = 0.95
        options.cms_relative_error = 0.01
        options.cms_max_num_heavy_hitters = 3

        profile1 = CategoricalColumn("test_name", options)
        profile1.update(dataset)

        expected_categories = ["b"]
        expected_categories_dict = {"b": 12}

        self.assertEqual(profile1._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile1.categories)

        profile2 = CategoricalColumn("test_name", options)
        profile2.update(dataset1)

        expected_categories = ["b", "c"]
        expected_categories_dict = {"b": 10, "c": 14}

        self.assertEqual(profile2._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile2.categories)

        # Add profiles
        profile3 = profile1 + profile2

        expected_categories = ["b", "c"]
        expected_categories_dict = {"b": 22, "c": 23}

        self.assertEqual(
            profile3.sample_size, profile1.sample_size + profile2.sample_size
        )
        self.assertEqual(profile3._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile3.categories)

    def test_cms_profile_min_max_num_heavy_hitters(self):

        dataset = pd.Series(["a"] * 9 + ["b"] * 12 + ["c"] * 9)
        dataset1 = pd.Series(["a"] * 6 + ["b"] * 10 + ["c"] * 14)

        options = CategoricalOptions()
        options.cms = True
        options.cms_confidence = 0.95
        options.cms_relative_error = 0.01
        options.cms_max_num_heavy_hitters = 3

        profile1 = CategoricalColumn("test_name", options)
        profile1.update(dataset)

        options.cms_max_num_heavy_hitters = 10
        profile2 = CategoricalColumn("test_name", options)
        profile2.update(dataset1)

        # Add profiles
        profile3 = profile1 + profile2

        self.assertEqual(profile3._cms_max_num_heavy_hitters, 3)

    def test_cms_catch_overwriting_with_missing_dict(self):

        dataset = pd.Series(["b"] * 2 + ["c"] * 14)
        dataset1 = pd.Series(["b"] * 5 + ["c"] * 10)

        options = CategoricalOptions()
        options.cms = True
        options.cms_confidence = 0.95
        options.cms_relative_error = 0.01
        options.cms_max_num_heavy_hitters = 3

        profile = CategoricalColumn("test_name", options)
        profile.update(dataset)

        expected_categories = ["c"]
        expected_categories_dict = {"c": 14}

        self.assertEqual(profile.sample_size, len(dataset))
        self.assertEqual(profile._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile.categories)

        profile.update(dataset1)
        expected_categories = ["c"]
        expected_categories_dict = {"c": 24}

        self.assertEqual(profile.sample_size, len(dataset) + len(dataset1))
        self.assertEqual(profile._categories, expected_categories_dict)
        self.assertCountEqual(expected_categories, profile.categories)

    def test_cms_vs_full_mismatch_merge(self):

        dataset = pd.Series(["b"] * 2 + ["c"] * 14)

        options = CategoricalOptions()
        options.cms = True
        options.cms_confidence = 0.95
        options.cms_relative_error = 0.01
        options.cms_max_num_heavy_hitters = 3

        profile_cms = CategoricalColumn("test_name", options)
        profile_cms.update(dataset)
        profile = CategoricalColumn("test_name")
        profile.update(dataset)

        with self.assertRaisesRegex(
            Exception,
            "Unable to add two profiles: One is using count min sketch"
            "and the other is using full.",
        ):
            profile3 = profile_cms + profile


class TestCategoricalSentence(unittest.TestCase):
    def setUp(self):
        test_utils.set_seed(seed=0)
        self.test_sentence = "This is the test sentence "
        self.test_sentence_upper1 = "THIS is the test sentence "
        self.test_sentence_upper2 = "This is the TEST sentence "
        self.test_sentence_upper3 = "THIS IS THE TEST SENTENCE "
        self.test_sentence_long = (
            "This is the test sentence "
            + "this is the test sentence "
            + "this is the test sentence "
            + "this is the test sentence "
        )

    def test_fewer_than_MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL(self):
        """
        Tests whether columns with fewer than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as
        categorical.
        """

        num_sentences = (
            CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL - 1
        )
        cat_sentence_list = [
            self.test_sentence + str(i + 1) for i in range(num_sentences)
        ]

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_greater_than_CATEGORICAL_THRESHOLD_DEFAULT_identify_as_text(self):
        """
        Tests whether columns with a ratio of categorical columns greater than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as text.
        """
        num_unique_values = (
            CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL + 1
        )
        list_unique_values = [
            self.test_sentence + str(i + 1) for i in range(num_unique_values)
        ]
        num_sentences = (
            int(float(1) / CategoricalColumn._CATEGORICAL_THRESHOLD_DEFAULT) - 1
        )
        cat_sentence_list = list_unique_values * num_sentences

        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]

        self.assertEqual(False, cat_profiler.is_match)

    def test_less_than_CATEGORICAL_THRESHOLD_DEFAULT(self):
        """
        Tests whether columns with a ratio of categorical columns less than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as
        categorical.
        """
        num_unique_values = (
            CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL + 1
        )
        list_unique_values = [
            self.test_sentence + str(i + 1) for i in range(num_unique_values)
        ]
        num_sentences = (
            int(float(1) / CategoricalColumn._CATEGORICAL_THRESHOLD_DEFAULT) + 2
        )
        cat_sentence_list = list_unique_values * num_sentences

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_uppercase_less_than_CATEGORICAL_THRESHOLD_DEFAULT(self):
        """
        Tests whether columns with a ratio of categorical columns less than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL and container
        uppercase letters identify as categorical.
        """
        num_unique_values = (
            CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL + 1
        )
        list_unique_values = [
            self.test_sentence + str(i + 1) for i in range(num_unique_values)
        ]
        num_sentences = (
            int(float(1) / CategoricalColumn._CATEGORICAL_THRESHOLD_DEFAULT) + 2
        )
        cat_sentence_list = list_unique_values * num_sentences
        cat_sentence_list[-1] = self.test_sentence_upper1 + str(num_sentences)
        cat_sentence_list[-2] = self.test_sentence_upper2 + str(num_sentences - 1)
        cat_sentence_list[-3] = self.test_sentence_upper3 + str(num_sentences - 2)

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_long_sentences_fewer_than_MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL(
        self,
    ):
        """
        Tests whether columns with the number of unique long sentences fewer
        than MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as
        categorical.
        """
        num_sentences = (
            CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL - 1
        )
        cat_sentence_list = [
            self.test_sentence_long + str(i + 1) for i in range(num_sentences)
        ]

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles["data_stats_profile"]._profiles[
            "category"
        ]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_categorical_column_with_wrong_options(self):
        with self.assertRaisesRegex(
            ValueError,
            "CategoricalColumn parameter 'options' must"
            " be of type CategoricalOptions.",
        ):
            profiler = CategoricalColumn("Categorical", options="wrong_data_type")
