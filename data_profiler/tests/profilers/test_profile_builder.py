from __future__ import print_function


import unittest
from unittest import mock
import random
import six
import os
import re

import numpy as np
import pandas as pd

from . import utils as test_utils

import data_profiler as dp
from data_profiler.profilers.profile_builder import StructuredDataProfile
from data_profiler.profilers.profiler_options import ProfilerOptions, \
    StructuredOptions
from data_profiler.profilers.column_profile_compilers import \
    ColumnPrimitiveTypeProfileCompiler, ColumnStatsProfileCompiler


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestProfiler(unittest.TestCase):

    @classmethod
    def setUp(cls):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):

        cls.input_file_path = os.path.join(
            test_root_path, 'data', 'csv/aws_honeypot_marx_geo.csv'
        )
        cls.aws_dataset = pd.read_csv(cls.input_file_path)
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        cls.trained_schema = dp.Profiler(cls.aws_dataset, len(cls.aws_dataset),
                                         profiler_options=profiler_options)

    @mock.patch('data_profiler.profilers.column_profile_compilers.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('data_profiler.profilers.column_profile_compilers.'
                'ColumnStatsProfileCompiler')
    @mock.patch('data_profiler.profilers.column_profile_compilers.'
                'ColumnDataLabelerCompiler')
    def test_add_profilers(self, *mocks):
        data = pd.DataFrame([1, None, 3, 4, 5, None])
        profile1 = dp.Profiler(data[:2])
        profile2 = dp.Profiler(data[2:])

        # test incorrect type
        with self.assertRaisesRegex(TypeError,
                                    '`Profiler` and `int` are '
                                    'not of the same profiler type.'):
            profile1 + 3

        # test mismatched profiles
        popped_profile = profile2._profile.pop(0)
        with self.assertRaisesRegex(ValueError,
                                    'Profiles do not have the same schema.'):
            profile1 + profile2

        # test mismatched profiles due to options
        profile2._profile[0] = None
        with self.assertRaisesRegex(ValueError,
                                    'The two profilers were not setup with the '
                                    'same options, hence they do not calculate '
                                    'the same profiles and cannot be added '
                                    'together.'):
            profile1 + profile2

        # test success
        profile1._profile = dict(test=1)
        profile2._profile = dict(test=2)
        merged_profile = profile1 + profile2
        self.assertEqual(3, merged_profile._profile['test'])
        self.assertIsNone(merged_profile.encoding)
        self.assertEqual(
            "<class 'pandas.core.frame.DataFrame'>", merged_profile.file_type)
        self.assertEqual(2, merged_profile.null_in_row_count)
        self.assertEqual(6, merged_profile.rows_ingested)
        self.assertEqual(5, len(merged_profile.hashed_row_dict))

        # test success if drawn from multiple files
        profile2.encoding = 'test'
        profile2.file_type = 'test'
        merged_profile = profile1 + profile2
        self.assertEqual('multiple files', merged_profile.encoding)
        self.assertEqual('multiple files', merged_profile.file_type)

    def test_correct_unique_row_ratio_test(self):
        self.assertEqual(2999, len(self.trained_schema.hashed_row_dict))
        self.assertEqual(2999, self.trained_schema.rows_ingested)
        self.assertEqual(1.0, self.trained_schema._get_unique_row_ratio())

    def test_correct_rows_ingested(self):
        self.assertEqual(2999, self.trained_schema.rows_ingested)

    def test_correct_null_row_ratio_test(self):
        self.assertEqual(2999, self.trained_schema.null_in_row_count)
        self.assertEqual(2999, self.trained_schema.rows_ingested)
        self.assertEqual(1.0, self.trained_schema._get_null_row_ratio())

    def test_correct_duplicate_row_count_test(self):
        self.assertEqual(2999, len(self.trained_schema.hashed_row_dict))
        self.assertEqual(2999, self.trained_schema.rows_ingested)
        self.assertEqual(0.0, self.trained_schema._get_duplicate_row_count())

    def test_correct_datatime_schema_test(self):
        profile = self.trained_schema.profile["datetime"]
        col_schema_info = \
            profile.profiles['data_type_profile']._profiles["datetime"]

        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(2, profile.null_count)
        six.assertCountEqual(self, ['nan'], profile.null_types)
        self.assertEqual(['%m/%d/%y %H:%M'], col_schema_info['date_formats'])

    def test_correct_integer_column_detection_src(self):
        profile = self.trained_schema.profile["src"]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]

        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(3, profile.null_count)

    def test_correct_integer_column_detection_int_col(self):
        profile = self.trained_schema.profile["int_col"]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]
        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(0, profile.null_count)

    def test_correct_integer_column_detection_port(self):
        profile = self.trained_schema.profile["srcport"]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]
        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(197, profile.null_count)

    def test_correct_integer_column_detection_destport(self):
        profile = self.trained_schema.profile["destport"]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]
        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(197, profile.null_count)

    def test_report(self):
        report = self.trained_schema.report()
        self.assertListEqual(list(report.keys()), [
                             'global_stats', 'data_stats'])
        self.assertListEqual(
            list(report['global_stats']),
            [
                "samples_used", "column_count", "unique_row_ratio",
                "row_has_null_ratio", "duplicate_row_count", "file_type",
                "encoding", "data_classification", "covariance"
            ]
        )
        flat_report = self.trained_schema.report(report_options={"output_format":"flat"})
        self.assertEqual(test_utils.get_depth(flat_report), 1)
        with mock.patch('data_profiler.profilers.helpers.report_helpers._prepare_report') as pr_mock:
            self.trained_schema.report(report_options={"output_format":'pretty'})
            self.assertEqual(pr_mock.call_count, 2)

    def test_report_quantiles(self):
        report_none = self.trained_schema.report(
            report_options={"num_quantile_groups": None})
        report = self.trained_schema.report()
        self.assertEqual(report_none, report)
        for key, val in report["data_stats"].items():
            if key == "int_col":
                report_quantiles = val["statistics"]["quantiles"]
                break
        self.assertEqual(len(report_quantiles), 3)
        report2 = self.trained_schema.report(
            report_options={"num_quantile_groups": 1000})
        for key, val in report2["data_stats"].items():
            if key == "int_col":
                report2_1000_quant = val["statistics"]["quantiles"]
                break
        self.assertEqual(len(report2_1000_quant), 999)
        self.assertEqual(report_quantiles, {
            0: report2_1000_quant[249],
            1: report2_1000_quant[499],
            2: report2_1000_quant[749],
        })

    def test_profile_key_name_without_space(self):

        def recursive_test_helper(report, prev_key=None):
            for key in report:
                # do not test keys in 'data_stats' as they contain column names
                # neither for 'ave_predictions' and 'data_label_representation'
                # as they contain label names
                # same for 'null_types_index'
                if prev_key not in ['data_stats', 'avg_predictions',
                                    'data_label_representation',
                                    'null_types_index']:
                    # key names should contain only alphanumeric letters or '_'
                    self.assertIsNotNone(re.match('^[a-zA-Z0-9_]+$', str(key)))
                if isinstance(report[key], dict):
                    recursive_test_helper(report[key], key)

        _report = self.trained_schema.report()
        recursive_test_helper(_report)

    def test_data_label_assigned(self):
        # only use 5 samples
        trained_schema = dp.Profiler(self.aws_dataset, samples_per_update=5)
        report = trained_schema.report()
        has_non_null_column = False
        for key in report['data_stats']:
            # only test non-null columns
            if report['data_stats'][key]['data_type'] is not None:
                self.assertIsNotNone(report['data_stats'][key]['data_label'])
                has_non_null_column = True
        if not has_non_null_column:
            self.fail(
                "Dataset tested did not have a non-null column and therefore "
                "could not validate the test.")

    @mock.patch('data_profiler.profilers.profile_builder.StructuredDataProfile')
    def test_duplicate_column_names(self, *mocks):

        # validate works first
        valid_data = pd.DataFrame([[1, 2]], columns=['a', 'b'])
        profile = dp.Profiler(valid_data)
        self.assertIn('a', profile._profile)
        self.assertIn('b', profile._profile)

        # data has duplicate column names
        invalid_data = pd.DataFrame([[1, 2]], columns=['a', 'a'])
        with self.assertRaisesRegex(ValueError,
                                    '`Profiler` does not currently support '
                                    'data which contains columns with duplicate'
                                    ' names.'):
            profile = dp.Profiler(invalid_data)


class TestStructuredDataProfileClass(unittest.TestCase):

    def setUp(self):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        cls.input_file_path = os.path.join(
            test_root_path, 'data', 'csv/aws_honeypot_marx_geo.csv'
        )
        cls.aws_dataset = pd.read_csv(cls.input_file_path)

    def test_base_props(self):
        src_column = self.aws_dataset.src
        src_profile = StructuredDataProfile(
            src_column, sample_size=len(src_column))

        self.assertIsInstance(src_profile.profiles['data_type_profile'],
                              ColumnPrimitiveTypeProfileCompiler)
        self.assertIsInstance(src_profile.profiles['data_stats_profile'],
                              ColumnStatsProfileCompiler)

        data_types = ['int', 'float', 'datetime', 'text']
        six.assertCountEqual(
            self, data_types,
            list(src_profile.profiles['data_type_profile']._profiles.keys())
        )

        stats_types = ['category', 'order']
        six.assertCountEqual(
            self, stats_types,
            list(src_profile.profiles['data_stats_profile']._profiles.keys())
        )

        self.assertEqual(3, src_profile.null_count)
        self.assertEqual(2999, src_profile.sample_size)

        total_nulls = 0
        for _, null_rows in src_profile.null_types_index.items():
            total_nulls += len(null_rows)
        self.assertEqual(3, total_nulls)

        # test updated base props with batch addition
        src_profile.update_profile(src_column)
        src_profile.update_profile(src_column)

        self.assertEqual(3*3, src_profile.null_count)
        self.assertEqual(2999*3, src_profile.sample_size)

    @mock.patch('data_profiler.profilers.column_profile_compilers.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('data_profiler.profilers.column_profile_compilers.'
                'ColumnStatsProfileCompiler')
    @mock.patch('data_profiler.profilers.column_profile_compilers.'
                'ColumnDataLabelerCompiler')
    def test_add_profilers(self, *mocks):
        data = pd.Series([1, None, 3, 4, 5, None])
        profile1 = StructuredDataProfile(data[:2])
        profile2 = StructuredDataProfile(data[2:])

        # test incorrect type
        with self.assertRaisesRegex(TypeError,
                                    '`StructuredDataProfile` and `int` are '
                                    'not of the same profiler type.'):
            profile1 + 3

        # test mismatched names
        profile1.name = 'profile1'
        profile2.name = 'profile2'
        with self.assertRaisesRegex(ValueError,
                                    'Structured profile names are unmatched: '
                                    'profile1 != profile2'):
            profile1 + profile2

        # test mismatched profiles due to options
        profile2.name = 'profile1'
        profile1._profiles = dict(test1=mock.Mock())
        profile2.profiles.pop('data_label_profile')
        with self.assertRaisesRegex(ValueError,
                                    'Structured profilers were not setup with '
                                    'the same options, hence they do not '
                                    'calculate the same profiles and cannot be '
                                    'added together.'):
            profile1 + profile2

        # test success
        profile1.profiles = dict(test=1)
        profile2.profiles = dict(test=2)
        merged_profile = profile1 + profile2
        self.assertEqual(3, merged_profile.profiles['test'])
        self.assertEqual(['4.0', '5.0', '1.0', '3.0'], merged_profile.sample)
        self.assertEqual(6, merged_profile.sample_size)
        self.assertEqual(2, merged_profile.null_count)
        self.assertListEqual(['nan'], merged_profile.null_types)
        self.assertDictEqual({'nan': [1, 5]}, merged_profile.null_types_index)

        # test add with different sampling properties
        profile1._min_sample_size = 10
        profile2._min_sample_size = 100
        profile1._sampling_ratio = 0.5
        profile2._sampling_ratio = 0.3
        profile1._min_true_samples = 11
        profile2._min_true_samples = 1
        merged_profile = profile1 + profile2
        self.assertEqual(100, merged_profile._min_sample_size)
        self.assertEqual(0.5, merged_profile._sampling_ratio)
        self.assertEqual(11, merged_profile._min_true_samples)

    def test_integrated_merge_diff_options(self):
        options = dp.ProfilerOptions()
        options.set({'data_labeler.is_enabled': False})

        data = pd.DataFrame([1, 2, 3, 4])
        profile1 = dp.Profiler(data, profiler_options=options)
        profile2 = dp.Profiler(data)
        with self.assertRaisesRegex(ValueError,
                                    'Structured profilers were not setup with '
                                    'the same options, hence they do not '
                                    'calculate the same profiles and cannot be '
                                    'added together.'):
            profile1 + profile2

    def test_get_base_props_and_clean_null_params(self):
        data = pd.Series([1, None, 3, 4, None, 6],
                         index=['a', 'b', 'c', 'd', 'e', 'f'])

        # validate that if sliced data, still functional
        # previously `iloc` was used at:
        # `df_series = df_series.loc[sorted(true_sample_list)]`
        # which caused errors
        df_series, base_stats = \
            StructuredDataProfile.get_base_props_and_clean_null_params(
                self=None, df_series=data[1:], sample_size=6,
                min_true_samples=0)
        # note data above is a subset `df_series=data[1:]`, 1.0 will not exist
        self.assertTrue(np.issubdtype(np.object_, df_series.dtype))
        self.assertDictEqual(
            {'sample': ['4.0', '6.0', '3.0'], 'sample_size': 5, 'null_count': 2,
             'null_types': dict(nan=['e', 'b'])},
            base_stats)

    def test_update_match_are_abstract(self):
        six.assertCountEqual(
            self,
            {'profile', '_update_helper', 'update'},
            dp.profilers.BaseColumnProfiler.__abstractmethods__
        )

    def test_data_labeler_toggle(self):
        src_column = self.aws_dataset.src
        structured_options = StructuredOptions()
        structured_options.data_labeler.is_enabled = False
        std_profile = StructuredDataProfile(src_column,
                                            sample_size=len(src_column))
        togg_profile = StructuredDataProfile(src_column,
                                            sample_size=len(src_column),
                                            options=structured_options)
        self.assertIn('data_label_profile', std_profile.profiles)
        self.assertNotIn('data_label_profile', togg_profile.profiles)

    def test_null_count(self):
        column = pd.Series([1, float('nan')] * 10)

        # test null_count when subset of full sample size
        random.seed(0)
        profile = StructuredDataProfile(column, sample_size=10)
        self.assertEqual(6, profile.null_count)

        # test null_count when full sample size
        profile = StructuredDataProfile(column, sample_size=len(column))
        self.assertEqual(10, profile.null_count)


class TestProfilerNullValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dict = {
            '1': ['nan', 'null', None, None, ''],
            1: ['nan', 'None', 'null', None, ''],
        }
        test_dataset = pd.DataFrame(data=test_dict)
        cls.trained_schema = dp.Profiler(test_dataset, len(test_dataset))
    def test_correct_rows_ingested(self):
        self.assertEqual(['', 'nan', 'None', 'null'],
                         self.trained_schema.profile['1'].null_types)
        self.assertEqual(
            5, self.trained_schema.profile['1'].null_count)
        self.assertEqual({'': [4], 'nan': [0], 'None': [2, 3], 'null': [
                         1]}, self.trained_schema.profile['1'].null_types_index)
        self.assertEqual(['', 'nan', 'None', 'null'],
                         self.trained_schema.profile[1].null_types)
        self.assertEqual(
            5, self.trained_schema.profile[1].null_count)
        self.assertEqual({'': [4], 'nan': [0], 'None': [1, 3], 'null': [
                         2]}, self.trained_schema.profile[1].null_types_index)


if __name__ == '__main__':
    unittest.main()
