import os
import six
from collections import defaultdict
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from . import utils as test_utils

from dataprofiler.profilers import CategoricalColumn
from dataprofiler.profilers.profile_builder import StructuredColProfiler


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCategoricalColumn(unittest.TestCase):

    @classmethod
    def setUp(self):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        cls.input_file_path = os.path.join(
            test_root_path, 'data', 'csv/aws_honeypot_marx_geo.csv'
        )
        columns_to_read = ['host', 'localeabbr']
        cls.aws_dataset = pd.read_csv(cls.input_file_path)[columns_to_read]

    def test_correct_categorical_model_string(self):
        dataset = self.aws_dataset["host"].dropna()
        profile = CategoricalColumn(dataset.name)
        profile.update(dataset)
        self.assertEqual(1.0, profile.is_match)
        self.assertEqual(2997, profile.sample_size)
        categories = {
            'groucho-oregon', 'groucho-us-east', 'groucho-singapore',
            'groucho-tokyo', 'groucho-sa', 'zeppo-norcal', 'groucho-norcal',
            'groucho-eu', 'groucho-sydney'
        }
        six.assertCountEqual(self, categories, profile.categories)

    def test_timeit_profile(self):
        dataset = self.aws_dataset["host"].dropna()
        profile = CategoricalColumn(dataset.name)

        time_array = [float(x) for x in range(17, 0, -1)]
        with patch('time.time', side_effect=lambda: time_array.pop()):
            # Validate the time in the column class is empty.
            self.assertEqual(defaultdict(float), profile.profile['times'])

            # Validate the time in the column class has the expected time.
            profile.update(dataset)
            expected = defaultdict(float, {'categories': 1.0})
            self.assertEqual(expected, profile.profile['times'])

            # Validate expected time after second update to profile
            profile.update(dataset)
            expected = defaultdict(float, {'categories': 2.0})
            self.assertEqual(expected, profile.profile['times'])

    def test_mixed_categorical_col_integer_string(self):
        dataset = self.aws_dataset["localeabbr"].dropna()
        profile = CategoricalColumn(dataset.name)
        profile.update(dataset)

        categories = {
            '36', 'OR', 'IL', '41', '51', '13', '21', 'WA',
            '11', 'CA', '37', 'TX', '10', 'SPE', '34', '32', '35',
            '23', 'NM', 'NV', '33', '44', '22', 'GR', '15', 'MI',
            '43', 'FL', 'TA', 'KY', 'SP', 'SE', 'AZ', '42', 'NJ',
            'DC', '77', '50', 'NGR', '31', 'DIF', '61', '45',
            'NY', 'MH', 'ALT', 'CH', 'NSW', 'MS', '81', 'GP',
            'KU', '14', '53', '64', 'AP', '38', 'IRK', 'CL',
            'TXG', 'LUA', 'ANT', 'PA', 'QC', 'RS', 'MO', 'C',
            'MOW', 'ENG', 'ON', 'CE', 'TN', 'PI', 'VLG', 'DL',
            'VL', 'GE', 'WP', 'GO', 'BS', 'KEM', 'MA', 'BEL',
            'LB', 'CU', 'EC', 'PB', 'RIX', 'B', 'RJ', 'VA',
            '7', 'SL', 'BE', '47', 'RM', 'BIH', 'SD', 'OH',
            'PR', 'M', 'SN', 'COR', '63', 'E', 'BD', 'VI',
            'SAM', 'BA', 'WY', '62', '4', 'PER', 'WKO', 'KYA',
            '6', 'MN', 'SA', '8', 'CO', 'IS', 'RIS', 'FS',
            'IN', 'LIV', 'IA', '24', 'VIC', '27', '16', 'PK',
            'WB', 'NH', 'DAS', 'CT', 'CN', 'BIR', 'NVS', 'MG',
            '3', 'PH', 'TO', '1', 'HE', 'VGG', 'BU', 'AB',
            'NIZ', '92', '46', 'MZ', 'FR'
        }

        self.assertEqual(2120, profile.sample_size)
        six.assertCountEqual(self, categories, profile.categories)

    def test_categorical_mapping(self):

        df1 = pd.Series([
            "abcd", "aa", "abcd", "aa", "b", "4", "3", "2", "dfd", "2", np.nan,
        ])
        df2 = pd.Series([
            "1", "null", "ee", "NaN", "ff", "nan", "gg", "None", "aa", "b", "ee",
        ])
        df3 = pd.Series([
            "NaN", "b", "nan", "c", None,
        ])

        column_profile = StructuredColProfiler(df1)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]

        num_null_types = 1
        num_nan_count = 1
        categories = df1.apply(str).unique().tolist()
        six.assertCountEqual(
            self,categories,cat_profiler.categories+column_profile.null_types)
        self.assertEqual(num_null_types, len(column_profile.null_types))
        self.assertEqual(
            num_nan_count, len(
                column_profile.null_types_index["nan"]))

        num_null_types = 4
        num_nan_count = 2
        categories = pd.concat([df1, df2]).apply(str).unique().tolist()
        column_profile.update_profile(df2)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        six.assertCountEqual(
            self, categories, cat_profiler.categories+column_profile.null_types)
        self.assertEqual(num_null_types, len(column_profile.null_types))
        self.assertEqual(
            num_nan_count, len(column_profile.null_types_index["nan"]))

        num_null_types = 4
        num_nan_count = 3
        categories = pd.concat([df1, df2, df3]).apply(str).unique().tolist()
        column_profile.update_profile(df3)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        six.assertCountEqual(
            self,categories,cat_profiler.categories+column_profile.null_types)
        self.assertEqual(num_null_types, len(column_profile.null_types))
        self.assertEqual(
            num_nan_count, len(
                column_profile.null_types_index["nan"]))
        self.assertNotEqual(
            num_nan_count, len(
                column_profile.null_types_index["NaN"]))

    def test_true_categorical_report(self):
        df_categorical = pd.Series([
            "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "c",
        ])
        profile = CategoricalColumn(df_categorical.name)
        profile.update(df_categorical)
        report = profile.profile
        six.assertCountEqual(
            self, ['categorical', 'statistics', 'times'], report)
        self.assertTrue(report["categorical"])
        six.assertCountEqual(
            self,
            ['unique_count', 'unique_ratio', 'categories'], report['statistics']
        )
        self.assertEqual(3, report["statistics"]["unique_count"])
        self.assertEqual(0.25, report["statistics"]["unique_ratio"])
        self.assertCountEqual(
            ["a", "b", "c"], report["statistics"]["categories"]
        )

    def test_false_categorical_report(self):
        df_non_categorical = pd.Series(list(map(str, range(0, 20))))
        profile = CategoricalColumn(df_non_categorical.name)
        profile.update(df_non_categorical)

        report = profile.profile
        six.assertCountEqual(
            self, ['categorical', 'statistics', 'times'], report)
        self.assertFalse(report["categorical"])
        six.assertCountEqual(
            self, ['unique_count', 'unique_ratio'], report['statistics']
        )
        self.assertEqual(20, report["statistics"]["unique_count"])
        self.assertEqual(1.0, report["statistics"]["unique_ratio"])

    def test_categorical_merge(self):
        df1 = pd.Series(["abcd", "aa", "abcd", "aa", "b", "4", "3", "2",
                         "dfd", "2", np.nan])
        df2 = pd.Series(["1", "null", "ee", "NaN", "ff", "nan", "gg",
                         "None", "aa", "b", "ee"])

        # Expected is based off insertion order
        expected_categories = ['abcd', 'aa', 'b', '4', '3', '2', 'dfd', np.nan,
                               '1', 'null', 'ee', 'NaN', 'ff', 'nan', 'gg', 'None']

        profile = CategoricalColumn("Name")
        profile.update(df1)

        profile2 = CategoricalColumn("Name")
        profile2.update(df2)
        
        # Add profiles
        profile3 = profile + profile2
        self.assertCountEqual(expected_categories, profile3.categories)
        self.assertEqual(
            profile3.sample_size,
            profile.sample_size +
            profile2.sample_size)
        self.assertEqual(profile3.is_match, False)

        # Add again
        profile3 = profile + profile3
        self.assertCountEqual(expected_categories, profile3.categories)
        self.assertEqual(profile3.sample_size, 33)

        # Check is_match and unique_ratio if the sample size was small
        self.assertEqual(profile3.is_match, False)
        self.assertEqual(profile3.unique_ratio, 16 / 33)

        # Check is_match and unique_ratio if the sample size was large
        profile3.sample_size = 1000
        self.assertEqual(profile3.is_match, True)
        self.assertEqual(profile3.unique_ratio, 16 / 1000)


class TestCategoricalSentence(unittest.TestCase):

    def setUp(self):
        test_utils.set_seed(seed=0)
        self.test_sentence = "This is the test sentence "
        self.test_sentence_upper1 = "THIS is the test sentence "
        self.test_sentence_upper2 = "This is the TEST sentence "
        self.test_sentence_upper3 = "THIS IS THE TEST SENTENCE "
        self.test_sentence_long = "This is the test sentence " +\
                                  "this is the test sentence " +\
                                  "this is the test sentence " +\
                                  "this is the test sentence "

    def test_fewer_than_MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL(self):
        """
        Tests whether columns with fewer than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as
        categorical.
        """

        num_sentences = CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL - 1
        cat_sentence_list = [self.test_sentence +
                             str(i + 1) for i in range(num_sentences)]

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_greater_than_CATEGORICAL_THRESHOLD_DEFAULT_identify_as_text(self):
        """
        Tests whether columns with a ratio of categorical columns greater than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as text.
        """
        num_unique_values = CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL + 1
        list_unique_values = [self.test_sentence +
                              str(i + 1) for i in range(num_unique_values)]
        num_sentences = int(
            float(1) / CategoricalColumn._CATEGORICAL_THRESHOLD_DEFAULT) - 1
        cat_sentence_list = list_unique_values * num_sentences

        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        
        self.assertEqual(False, cat_profiler.is_match)

    def test_less_than_CATEGORICAL_THRESHOLD_DEFAULT(self):
        """
        Tests whether columns with a ratio of categorical columns less than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as
        categorical.
        """
        num_unique_values = CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL + 1
        list_unique_values = [self.test_sentence +
                              str(i + 1) for i in range(num_unique_values)]
        num_sentences = int(
            float(1) / CategoricalColumn._CATEGORICAL_THRESHOLD_DEFAULT) + 2
        cat_sentence_list = list_unique_values * num_sentences

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_uppercase_less_than_CATEGORICAL_THRESHOLD_DEFAULT(self):
        """
        Tests whether columns with a ratio of categorical columns less than
        MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL and container
        uppercase letters identify as categorical.
        """
        num_unique_values = CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL + 1
        list_unique_values = [self.test_sentence +
                              str(i + 1) for i in range(num_unique_values)]
        num_sentences = int(
            float(1) / CategoricalColumn._CATEGORICAL_THRESHOLD_DEFAULT) + 2
        cat_sentence_list = list_unique_values * num_sentences
        cat_sentence_list[-1] = self.test_sentence_upper1 + str(num_sentences)
        cat_sentence_list[-2] = self.test_sentence_upper2 + \
            str(num_sentences - 1)
        cat_sentence_list[-3] = self.test_sentence_upper3 + \
            str(num_sentences - 2)

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_long_sentences_fewer_than_MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL(
            self):
        """
        Tests whether columns with the number of unique long sentences fewer
        than MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL identify as
        categorical.
        """
        num_sentences = CategoricalColumn._MAXIMUM_UNIQUE_VALUES_TO_CLASSIFY_AS_CATEGORICAL - 1
        cat_sentence_list = [self.test_sentence_long +
                             str(i + 1) for i in range(num_sentences)]

        len_unique = len(set(cat_sentence_list))
        cat_sentence_df = pd.Series(cat_sentence_list)
        column_profile = StructuredColProfiler(cat_sentence_df)
        cat_profiler = column_profile.profiles['data_stats_profile']._profiles["category"]
        self.assertEqual(True, cat_profiler.is_match)
        self.assertEqual(len_unique, len(cat_profiler.categories))

    def test_categorical_column_with_wrong_options(self):
        with self.assertRaisesRegex(ValueError,
                                   "CategoricalColumn parameter 'options' must"
                                   " be of type CategoricalOptions."):
            profiler = CategoricalColumn("Categorical", options="wrong_data_type")
