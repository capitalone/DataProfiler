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

import dataprofiler as dp
from dataprofiler.profilers.profile_builder import StructuredDataProfile
from dataprofiler.profilers.profiler_options import ProfilerOptions, \
    StructuredOptions
from dataprofiler.profilers.column_profile_compilers import \
    ColumnPrimitiveTypeProfileCompiler, ColumnStatsProfileCompiler
from dataprofiler.profilers.helpers.report_helpers import _prepare_report

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


    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
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
        self.assertEqual(2, merged_profile.row_has_null_count)
        self.assertEqual(2, merged_profile.row_is_null_count)
        self.assertEqual(6, merged_profile.total_samples)
        self.assertEqual(5, len(merged_profile.hashed_row_dict))

        # test success if drawn from multiple files
        profile2.encoding = 'test'
        profile2.file_type = 'test'
        merged_profile = profile1 + profile2
        self.assertEqual('multiple files', merged_profile.encoding)
        self.assertEqual('multiple files', merged_profile.file_type)

    def test_correct_unique_row_ratio_test(self):
        self.assertEqual(2999, len(self.trained_schema.hashed_row_dict))
        self.assertEqual(2999, self.trained_schema.total_samples)
        self.assertEqual(1.0, self.trained_schema._get_unique_row_ratio())

    def test_correct_rows_ingested(self):
        self.assertEqual(2999, self.trained_schema.total_samples)

    def test_correct_null_row_ratio_test(self):
        self.assertEqual(2999, self.trained_schema.row_has_null_count)
        self.assertEqual(1.0, self.trained_schema._get_row_has_null_ratio())
        self.assertEqual(0, self.trained_schema.row_is_null_count)
        self.assertEqual(0, self.trained_schema._get_row_is_null_ratio())
        self.assertEqual(2999, self.trained_schema.total_samples)

    def test_correct_duplicate_row_count_test(self):
        self.assertEqual(2999, len(self.trained_schema.hashed_row_dict))
        self.assertEqual(2999, self.trained_schema.total_samples)
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
                "samples_used", "column_count", "row_count", 
                "row_has_null_ratio", 'row_is_null_ratio',
                "unique_row_ratio", "duplicate_row_count", "file_type", "encoding"
            ]
        )
        flat_report = self.trained_schema.report(report_options={"output_format":"flat"})
        self.assertEqual(test_utils.get_depth(flat_report), 1)
        with mock.patch('dataprofiler.profilers.helpers.report_helpers._prepare_report') as pr_mock:
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

    def test_report_omit_keys(self):
        omit_keys = [ 'global_stats', 'data_stats' ]
                
        report_omit_keys = self.trained_schema.report(
            report_options={ "omit_keys": omit_keys })
        
        self.assertCountEqual({}, report_omit_keys)


    def test_report_compact(self):
        report = self.trained_schema.report(
            report_options={ "output_format": "pretty" })
        omit_keys = [
            "data_stats.*.statistics.times",
            "data_stats.*.statistics.avg_predictions",
            "data_stats.*.statistics.data_label_representation",
            "data_stats.*.statistics.null_types_index",
            "data_stats.*.statistics.histogram"
        ]

        report = _prepare_report(report, 'pretty', omit_keys)
        
        report_compact = self.trained_schema.report(
            report_options={"output_format": "compact" })

        self.assertEqual(report, report_compact)        
        
        

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

    @mock.patch('dataprofiler.profilers.profile_builder.StructuredDataProfile')
    @mock.patch('dataprofiler.profilers.profile_builder.Profiler._update_row_statistics')
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

    def test_text_data_raises_error(self):
        text_file_path = os.path.join(
            test_root_path, 'data', 'txt/sentence-10x.txt'
        )
        with self.assertRaisesRegex(TypeError, 'Cannot provide TextData object'
                                               ' to Profiler'):
            profile = dp.Profiler(dp.Data(text_file_path))

    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnDataLabelerCompiler')
    def test_sample_size_warning_in_the_profiler(self, *mocks):
        data = pd.DataFrame([1, None, 3, 4, 5, None])
        with self.assertWarnsRegex(UserWarning,
                                   "The data will be profiled with a sample "
                                   "size of 3. All statistics will be based on "
                                   "this subsample and not the whole dataset."):
            profile1 = dp.Profiler(data, samples_per_update=3)


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

    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.column_profile_compilers.'
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
        self.assertCountEqual(['5.0', '4.0', '3.0', '1.0'], merged_profile.sample)
        self.assertEqual(6, merged_profile.sample_size)
        self.assertEqual(2, merged_profile.null_count)
        self.assertListEqual(['nan'], merged_profile.null_types)
        self.assertDictEqual({'nan': {1, 5}}, merged_profile.null_types_index)

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
        self.assertCountEqual({'sample': ['4.0', '6.0', '3.0'],
                               'sample_size': 5, 'null_count': 2,
                               'null_types': dict(nan=['e', 'b'])}, base_stats)            

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

        # test null_count when full sample size
        random.seed(0)
        profile = StructuredDataProfile(column, sample_size=len(column))
        self.assertEqual(10, profile.null_count)


class TestProfilerNullValues(unittest.TestCase):

    def setUp(self):
        test_utils.set_seed(0)

    def test_correct_rows_ingested(self):
        test_dict = {
            '1': ['nan', 'null', None, None, ''],
            1: ['nan', 'None', 'null', None, ''],
        }
        test_dataset = pd.DataFrame(data=test_dict)
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        trained_schema = dp.Profiler(test_dataset, len(test_dataset),
                                     profiler_options=profiler_options)

        self.assertCountEqual(['', 'nan', 'None', 'null'],
                         trained_schema.profile['1'].null_types)
        self.assertEqual(5, trained_schema.profile['1'].null_count)
        self.assertEqual({'': {4}, 'nan': {0}, 'None': {2, 3}, 'null': {
                         1}}, trained_schema.profile['1'].null_types_index)
        self.assertCountEqual(['', 'nan', 'None', 'null'],
                         trained_schema.profile[1].null_types)
        self.assertEqual(5, trained_schema.profile[1].null_count)
        self.assertEqual({'': {4}, 'nan': {0}, 'None': {1, 3}, 'null': {
                         2}}, trained_schema.profile[1].null_types_index)

    def test_correct_null_row_counts(self):
        file_path = os.path.join(test_root_path, 'data', 'csv/empty_rows.txt')
        data = pd.read_csv(file_path)
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        profile = dp.Profiler(data, profiler_options=profiler_options)
        self.assertEqual(2, profile.row_has_null_count)
        self.assertEqual(0.25, profile._get_row_has_null_ratio())
        self.assertEqual(2, profile.row_is_null_count)
        self.assertEqual(0.25, profile._get_row_is_null_ratio())

        file_path = os.path.join(test_root_path, 'data','csv/iris-with-null-rows.csv')
        data = pd.read_csv(file_path)
        profile = dp.Profiler(data, profiler_options=profiler_options)
        self.assertEqual(13, profile.row_has_null_count)
        self.assertEqual(13/24, profile._get_row_has_null_ratio())
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(3/24, profile._get_row_is_null_ratio())


    def test_null_in_file(self):
        filename_null_in_file = os.path.join(
            test_root_path, 'data', 'csv/sparse-first-and-last-column.txt')
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        data = dp.Data(filename_null_in_file)
        profile = dp.Profiler(data, profiler_options=profiler_options)

        report = profile.report(report_options={"output_format":"pretty"})
        
        self.assertEqual(
            report['data_stats']['COUNT']['statistics']['null_types_index'],
            {'': '[2, 3, 4, 5, 7, 8]'}
        )
        
        self.assertEqual(
            report['data_stats'][' NUMBERS']['statistics']['null_types_index'],
            {'': '[5, 6, 8]', ' ': '[2, 4]'}
        )

    def test_correct_total_sample_size_and_counts(self):
        file_path = os.path.join(test_root_path, 'data', 'csv/empty_rows.txt')
        data = pd.read_csv(file_path)
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        # Profile Once
        profile = dp.Profiler(data, profiler_options=profiler_options, samples_per_update=2)
        # Profile Twice
        profile.update_profile(data)
        
        self.assertEqual(16, profile.total_samples)
        self.assertEqual(4, profile._max_col_samples_used)
        self.assertEqual(2, profile.row_has_null_count)
        self.assertEqual(0.5, profile._get_row_has_null_ratio())
        self.assertEqual(2, profile.row_is_null_count)
        self.assertEqual(0.5, profile._get_row_is_null_ratio())
        self.assertEqual(0.4375, profile._get_unique_row_ratio())
        self.assertEqual(9, profile._get_duplicate_row_count())


if __name__ == '__main__':
    unittest.main()
