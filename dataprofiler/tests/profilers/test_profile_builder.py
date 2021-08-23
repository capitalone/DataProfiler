from __future__ import print_function

import unittest
from unittest import mock
from io import BytesIO, StringIO
import random
import six
import os
import re
import logging

import numpy as np
import pandas as pd

from . import utils as test_utils

import dataprofiler as dp
from dataprofiler.profilers.profile_builder import StructuredColProfiler, \
    UnstructuredProfiler, UnstructuredCompiler, StructuredProfiler, Profiler
from dataprofiler.profilers.profiler_options import ProfilerOptions, \
    StructuredOptions, UnstructuredOptions
from dataprofiler.profilers.column_profile_compilers import \
    ColumnPrimitiveTypeProfileCompiler, ColumnStatsProfileCompiler, \
    ColumnDataLabelerCompiler
from dataprofiler import StructuredDataLabeler, UnstructuredDataLabeler

from dataprofiler.profilers.helpers.report_helpers import _prepare_report

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def setup_save_mock_open(mock_open):
    mock_file = BytesIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args: mock_file
    return mock_file


class TestStructuredProfiler(unittest.TestCase):

    @classmethod
    def setUp(cls):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        test_utils.set_seed(seed=0)

        cls.input_file_path = os.path.join(
            test_root_path, 'data', 'csv/aws_honeypot_marx_geo.csv'
        )
        cls.aws_dataset = pd.read_csv(cls.input_file_path)
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        with test_utils.mock_timeit():
            cls.trained_schema = dp.StructuredProfiler(
                cls.aws_dataset, len(cls.aws_dataset), options=profiler_options)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    def test_bad_input_data(self, *mocks):
        allowed_data_types = (r"\(<class 'list'>, " 
                              r"<class 'pandas.core.series.Series'>, " 
                              r"<class 'pandas.core.frame.DataFrame'>\)")
        bad_data_types = [1, {}, np.inf, 'sdfs']
        for data in bad_data_types:
            with self.assertRaisesRegex(TypeError,
                                        r"Data must either be imported using "
                                        r"the data_readers or using one of the "
                                        r"following: " + allowed_data_types):
                StructuredProfiler(data)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_list_data(self, *mocks):
        data = [[1, 1],
                [None, None],
                [3, 3],
                [4, 4],
                [5, 5],
                [None, None],
                [1, 1]]
        with test_utils.mock_timeit():
            profiler = dp.StructuredProfiler(data)

        # test properties
        self.assertEqual("<class 'list'>", profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertEqual(2, profiler.row_has_null_count)
        self.assertEqual(2, profiler.row_is_null_count)
        self.assertEqual(7, profiler.total_samples)
        self.assertEqual(5, len(profiler.hashed_row_dict))
        self.assertListEqual([0, 1], list(profiler._col_name_to_idx.keys()))
        self.assertIsNone(profiler.correlation_matrix)
        self.assertDictEqual({'row_stats': 1}, profiler.times)

        # validates the sample out maintains the same visual data format as the
        # input.
        self.assertListEqual(['5', '1', '1', '3', '4'],
                             profiler.profile[0].sample)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_pandas_series_data(self, *mocks):
        data = pd.Series([1, None, 3, 4, 5, None, 1])
        with test_utils.mock_timeit():
            profiler = dp.StructuredProfiler(data)

        # test properties
        self.assertEqual(
            "<class 'pandas.core.series.Series'>", profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertEqual(2, profiler.row_has_null_count)
        self.assertEqual(2, profiler.row_is_null_count)
        self.assertEqual(7, profiler.total_samples)
        self.assertEqual(5, len(profiler.hashed_row_dict))
        self.assertListEqual([0], list(profiler._col_name_to_idx.keys()))
        self.assertIsNone(profiler.correlation_matrix)
        self.assertDictEqual({'row_stats': 1}, profiler.times)

        # test properties when series has name
        data.name = 'test'
        profiler = dp.StructuredProfiler(data)
        self.assertEqual(
            "<class 'pandas.core.series.Series'>", profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertEqual(2, profiler.row_has_null_count)
        self.assertEqual(2, profiler.row_is_null_count)
        self.assertEqual(7, profiler.total_samples)
        self.assertEqual(5, len(profiler.hashed_row_dict))
        self.assertListEqual(['test'], list(profiler._col_name_to_idx.keys()))
        self.assertIsNone(profiler.correlation_matrix)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._merge_correlation')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_chi2')
    def test_add_profilers(self, *mocks):
        data = pd.DataFrame([1, None, 3, 4, 5, None, 1])
        with test_utils.mock_timeit():
            profile1 = dp.StructuredProfiler(data[:2])
            profile2 = dp.StructuredProfiler(data[2:])

        # test incorrect type
        with self.assertRaisesRegex(TypeError,
                                    '`StructuredProfiler` and `int` are '
                                    'not of the same profiler type.'):
            profile1 + 3

        # test mismatched profiles
        profile2._profile.pop(0)
        profile2._col_name_to_idx.pop(0)
        with self.assertRaisesRegex(ValueError,
                                    "Cannot merge empty profiles."):
            profile1 + profile2

        # test mismatched profiles due to options
        profile2._profile.append(None)
        profile2._col_name_to_idx[0] = [0]
        with self.assertRaisesRegex(ValueError,
                                    'The two profilers were not setup with the '
                                    'same options, hence they do not calculate '
                                    'the same profiles and cannot be added '
                                    'together.'):
            profile1 + profile2

        # test success
        profile1._profile = [1]
        profile1._col_name_to_idx = {"test": [0]}
        profile2._profile = [2]
        profile2._col_name_to_idx = {"test": [0]}
        merged_profile = profile1 + profile2
        self.assertEqual(3, merged_profile._profile[
            merged_profile._col_name_to_idx["test"][0]])
        self.assertIsNone(merged_profile.encoding)
        self.assertEqual(
            "<class 'pandas.core.frame.DataFrame'>", merged_profile.file_type)
        self.assertEqual(2, merged_profile.row_has_null_count)
        self.assertEqual(2, merged_profile.row_is_null_count)
        self.assertEqual(7, merged_profile.total_samples)
        self.assertEqual(5, len(merged_profile.hashed_row_dict))
        self.assertDictEqual({'row_stats': 2}, merged_profile.times)

        # test success if drawn from multiple files
        profile2.encoding = 'test'
        profile2.file_type = 'test'
        merged_profile = profile1 + profile2
        self.assertEqual('multiple files', merged_profile.encoding)
        self.assertEqual('multiple files', merged_profile.file_type)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._get_correlation')
    def test_stream_profilers(self, *mocks):
        mocks[0].return_value = None
        data = pd.DataFrame([
            ['test1', 1.0],
            ['test2', None],
            ['test1', 1.0],
            [None, None],
            [None, 5.0],
            [None, 5.0],
            [None, None],
            ['test3', 7.0]])

        # check prior to update
        with test_utils.mock_timeit():
            profiler = dp.StructuredProfiler(data[:3])
        self.assertEqual(1, profiler.row_has_null_count)
        self.assertEqual(0, profiler.row_is_null_count)
        self.assertEqual(3, profiler.total_samples)
        self.assertEqual(2, len(profiler.hashed_row_dict))
        self.assertIsNone(profiler.correlation_matrix)
        self.assertDictEqual({'row_stats': 1}, profiler.times)

        # check after update
        with test_utils.mock_timeit():
            profiler.update_profile(data[3:])

        self.assertIsNone(profiler.encoding)
        self.assertEqual(
            "<class 'pandas.core.frame.DataFrame'>", profiler.file_type)
        self.assertEqual(5, profiler.row_has_null_count)
        self.assertEqual(2, profiler.row_is_null_count)
        self.assertEqual(8, profiler.total_samples)
        self.assertEqual(5, len(profiler.hashed_row_dict))
        self.assertIsNone(profiler.correlation_matrix)
        self.assertDictEqual({'row_stats': 2}, profiler.times)

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

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    def test_correlation(self, *mock):
        # Use the following formula to obtain the pairwise correlation
        # sum((x - np.mean(x))*(y-np.mean(y))) /
        # np.sqrt(sum((x - np.mean(x)**2)))/np.sqrt(sum((y - np.mean(y)**2)))
        profile_options = dp.ProfilerOptions()
        profile_options.set({"correlation.is_enabled": True})

        # data with a sole numeric column
        data = pd.DataFrame([1.0, 8.0, 1.0, -2.0, 5.0])
        with test_utils.mock_timeit():
            profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([[1.0]])
        np.testing.assert_array_equal(expected_corr_mat,
                                      profiler.correlation_matrix)
        self.assertDictEqual({'row_stats': 1, 'correlation': 1}, profiler.times)

        # data with one column with non-numeric calues
        data = pd.DataFrame([1.0, None, 1.0, None, 5.0])
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([[1]])
        np.testing.assert_array_equal(expected_corr_mat,
                                      profiler.correlation_matrix)

        # data with two columns, but one is numerical
        data = pd.DataFrame([
            ['test1', 1.0],
            ['test2', None],
            ['test1', 1.0],
            [None, None]])
        profiler = dp.StructuredProfiler(data, options=profile_options)
        # Even the correlation with itself is NaN because the variance is zero
        expected_corr_mat = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        np.testing.assert_array_equal(expected_corr_mat,
                                      profiler.correlation_matrix)

        # data with multiple numerical columns
        data = pd.DataFrame({'a': [3, 2, 1, 7, 5, 9, 4, 10, 7, 2],
                             'b': [10, 11, 1, 4, 2, 5, 6, 3, 9, 8],
                             'c': [1, 5, 3, 5, 7, 2, 6, 8, 1, 2]})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([
            [1.0, -0.26559388521279237, 0.26594894270403086],
            [-0.26559388521279237, 1.0, -0.49072329],
            [0.26594894270403086, -0.49072329, 1.0]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # data with multiple numerical columns, with nan values
        data = pd.DataFrame({'a': [np.nan, np.nan, 1, 7, 5, 9, 4, 10, 7, 2],
                             'b': [10, 11, np.nan, 4, 2, 5, 6, 3, 9, 8],
                             'c': [1, 5, 3, 5, 7, 2, 6, 8, np.nan, np.nan]})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([
            [1, -0.28527657, 0.18626508],
            [-0.28527657, 1, -0.52996792],
            [0.18626508, -0.52996792, 1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # data with multiple numerical columns, with nan values in only one
        # column
        data = pd.DataFrame({'a': [np.nan, np.nan, 1, 7, 5, 9, 4, 10, 7, 2],
                             'b': [10, 11, 1, 4, 2, 5, 6, 3, 9, 8],
                             'c': [1, 5, 3, 5, 7, 2, 6, 8, 1, 2]})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([
            [1, 0.03673504, 0.22844891],
            [0.03673504, 1, -0.49072329],
            [0.22844891, -0.49072329, 1]])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # data with only one numerical columns without nan values
        data = pd.DataFrame({'a': [3, 2, 1, 7, 5, 9, 4, 10, 7, 2]})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([[1]])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # data with no numeric columns
        data = pd.DataFrame({'a': ['hi', 'hi2', 'hi3'],
                             'b': ['test1', 'test2', 'test3']})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # data with only one numeric column
        # data with no numeric columns
        data = pd.DataFrame({'a': ['hi', 'hi2', 'hi3'],
                             'b': ['test1', 'test2', 'test3'],
                             'c': [1, 2, 3]})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        expected_corr_mat = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # Data with null rows
        data = pd.DataFrame({'a': [None, 2, 1, np.nan, 5, np.nan, 4, 10, 7, np.nan],
                             'b': [np.nan, 11, 1, 'nan', 2, np.nan, 6, 3, 9, np.nan],
                             'c': [np.nan, 5, 3, np.nan, 7, np.nan, 6, 8, 1, None]})
        profiler = dp.StructuredProfiler(data, options=profile_options)

        # correlation between [2, 1, 5, 4, 10, 7],
        #                     [11, 1, 2, 6, 3, 9],
        #                     [5, 3, 7, 6, 8, 1]
        expected_corr_mat = np.array([
            [1, -0.06987956, 0.32423975],
            [-0.06987956, 1, -0.3613099],
            [0.32423975, -0.3613099, 1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # Data with null rows and some imputed values
        data = pd.DataFrame({'a': [None, np.nan, 1, 7, 5, 9, 4, 10, np.nan, 2],
                             'b': [10, 11, 1, 4, 2, 5, np.nan, 3, np.nan, 8],
                             'c': [1, 5, 3, 5, np.nan, 2, 6, 8, np.nan, 2]})
        profiler = dp.StructuredProfiler(data, options=profile_options)
        # correlation between [*38/7*, *38/7*, 1, 7, 5, 9, 4, 10, 2],
        #                     [10, 11, 1, 4, 2, 5, *11/2*, 3, 8],
        #                     [1, 5, 3, 5, *4*, 2, 6, 8, 2]
        expected_corr_mat = np.array([
            [1, -0.03283837,  0.40038038],
            [-0.03283837, 1, -0.30346637],
            [0.40038038, -0.30346637, 1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    def test_merge_correlation(self, *mocks):
        # Use the following formular to obtain the pairwise correlation
        # sum((x - np.mean(x))*(y-np.mean(y))) /
        # np.sqrt(sum((x - np.mean(x)**2)))/np.sqrt(sum((y - np.mean(y)**2)))
        profile_options = dp.ProfilerOptions()
        profile_options.set({"correlation.is_enabled": True})

        # merge between two existing correlations
        data = pd.DataFrame({'a': [3, 2, 1, 7, 5, 9, 4, 10, 7, 2],
                             'b': [10, 11, 1, 4, 2, 5, 6, 3, 9, 8],
                             'c': [1, 5, 3, 5, 7, 2, 6, 8, 1, 2]})
        data1 = data[:5]
        data2 = data[5:]

        with test_utils.mock_timeit():
            profile1 = dp.StructuredProfiler(data1, options=profile_options)
            profile2 = dp.StructuredProfiler(data2, options=profile_options)
        merged_profile = profile1 + profile2

        expected_corr_mat = np.array([
            [1.0, -0.26559388521279237, 0.26594894270403086],
            [-0.26559388521279237, 1.0, -0.49072329],
            [0.26594894270403086, -0.49072329, 1.0]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             merged_profile.correlation_matrix)
        self.assertDictEqual({'row_stats': 2, 'correlation': 2},
                             merged_profile.times)

        # merge between an existing corr and None correlation (without data)
        with test_utils.mock_timeit():
            profile1 = dp.StructuredProfiler(None, options=profile_options)
            profile2 = dp.StructuredProfiler(data, options=profile_options)
        # TODO: remove the mock below when merge profile is update
        with mock.patch('dataprofiler.profilers.profile_builder.'
                        'StructuredProfiler._add_error_checks'):
            merged_profile = profile1 + profile2

        expected_corr_mat = np.array([
            [1.0, -0.26559388521279237, 0.26594894270403086],
            [-0.26559388521279237, 1.0, -0.49072329],
            [0.26594894270403086, -0.4907239, 1.0]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             merged_profile.correlation_matrix)
        self.assertDictEqual({'row_stats': 1, 'correlation': 1},
                             merged_profile.times)

        # Merge between existing data and empty data that still has samples
        data = pd.DataFrame({'a': [1, 2, 4, np.nan, None, np.nan],
                             'b': [5, 7, 1, np.nan, np.nan, 'nan']})
        data1 = data[:3]
        data2 = data[3:]

        profile1 = dp.StructuredProfiler(data1, options=profile_options)
        expected_corr_mat = np.array([
            [1, -0.78571429],
            [-0.78571429,  1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profile1.correlation_matrix)
        profile2 = dp.StructuredProfiler(data2, options=profile_options)
        merged_profile = profile1 + profile2
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             merged_profile.correlation_matrix)

    def test_correlation_update(self):
        profile_options = dp.ProfilerOptions()
        profile_options.set({"correlation.is_enabled": True})

        # Test with all numeric columns
        data = pd.DataFrame({'a': [3, 2, 1, 7, 5, 9, 4, 10, 7, 2],
                             'b': [10, 11, 1, 4, 2, 5, 6, 3, 9, 8],
                             'c': [1, 5, 3, 5, 7, 2, 6, 8, 1, 2]})
        data1 = data[:5]
        data2 = data[5:]

        with test_utils.mock_timeit():
            profiler = dp.StructuredProfiler(data1, options=profile_options)
            profiler.update_profile(data2)

        expected_corr_mat = np.array([
            [1.0, -0.26559388521279237, 0.26594894270403086],
            [-0.26559388521279237, 1.0, -0.4907239],
            [0.26594894270403086, -0.4907239, 1.0]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)
        self.assertDictEqual({'row_stats': 2, 'correlation': 2}, profiler.times)

        # Test when there's a non-numeric column
        data = pd.DataFrame({'a': [3, 2, 1, 7, 5, 9, 4, 10, 7, 2],
                             'b': [10, 11, 1, 4, 2, 5, 6, 3, 9, 8],
                             'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']})
        data1 = data[:5]
        data2 = data[5:]

        profiler = dp.StructuredProfiler(data1, options=profile_options)
        profiler.update_profile(data2)

        expected_corr_mat = np.array([
            [1.0, -0.26559388521279237, np.nan],
            [-0.26559388521279237, 1.0, np.nan],
            [np.nan, np.nan, np.nan]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # Data with multiple numerical and non-numeric columns, with nan values in only one column
        # NaNs imputed to (9+4+10)/3
        data = pd.DataFrame({'a': [7, 2, 1, 7, 5, 9, 4, 10, np.nan, np.nan],
                             'b': [10, 11, 1, 4, 2, 5, 6, 3, 9, 8],
                             'c': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                             'd': [1, 5, 3, 5, 7, 2, 6, 8, 1, 2]})
        data1 = data[:5]
        data2 = data[5:]

        profiler = dp.StructuredProfiler(data1, options=profile_options)
        profiler.update_profile(data2)
        expected_corr_mat = np.array([
            [ 1,  0.04721482, np.nan, -0.09383408],
            [ 0.04721482,  1, np.nan,-0.49072329],
            [np.nan, np.nan, np.nan, np.nan],
            [-0.09383408, -0.49072329, np.nan, 1]]
        )
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # Data with null rows, all null rows are dropped
        data = pd.DataFrame({'a': [np.nan, 2, 1, None, 5, np.nan, 4, 10, 7, 'NaN'],
                             'b': [np.nan, 11, 1, np.nan, 2, np.nan, 6, 3, 9, np.nan],
                             'c': [np.nan, 5, 3, np.nan, 7, None, 6, 8, 1, np.nan]})
        data1 = data[:5]
        data2 = data[5:]
        profiler = dp.StructuredProfiler(data1, options=profile_options)
        profiler.update_profile(data2)
        # correlation between [2, 1, 5, 4, 10, 7],
        #                     [11, 1, 2, 6, 3, 9],
        #                     [5, 3, 7, 6, 8, 1]
        expected_corr_mat = np.array([
            [1, -0.06987956, 0.32423975],
            [-0.06987956, 1, -0.3613099],
            [0.32423975, -0.3613099, 1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

        # Data with null rows and some imputed values
        data = pd.DataFrame({'a': [None, np.nan, 1, 7, 5, 9, 4, 10, 'nan', 2],
                             'b': [10, 11, 1, 4, 2, 5, 'NaN', 3, None, 8],
                             'c': [1, 5, 3, 5, np.nan, 2, 6, 8, None, 2]})
        data1 = data[:5]
        data2 = data[5:]
        profiler = dp.StructuredProfiler(data1, options=profile_options)
        profiler.update_profile(data2)
        # correlation between [*13/3*, *13/3*, 1, 7, 5]
        #                     [10, 11, 1, 4, 2]
        #                     [1, 5, 3, 5, *7/2*]
        # then updated with correlation (9th row dropped) between
        #                     [9, 4, 10, 2],
        #                     [5, *16/3*, 3, 8],
        #                     [2, 6, 8, 2]
        expected_corr_mat = np.array([
            [1, -0.16079606,  0.43658332],
            [-0.16079606, 1, -0.2801748],
            [0.43658332, -0.2801748, 1]
        ])
        np.testing.assert_array_almost_equal(expected_corr_mat,
                                             profiler.correlation_matrix)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    def test_chi2(self, *mocks):
        # Empty
        data = pd.DataFrame([])
        profiler = dp.StructuredProfiler(data)
        self.assertIsNone(profiler.chi2_matrix)

        # Single column
        data = pd.DataFrame({'a': ["y", "y", "n", "n", "y"]})
        profiler = dp.StructuredProfiler(data)
        expected_mat = np.array([1])
        self.assertEqual(expected_mat, profiler.chi2_matrix)

        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "maybe", "y", "y", "n", "n", "maybe"],
                             'c': ["n", "maybe", "n", "n", "n", "y", "y"]})

        profiler = dp.StructuredProfiler(data)
        expected_mat = np.array([
            [1, 0.309924, 0.404638],
            [0.309924, 1, 0.548812],
            [0.404638, 0.548812, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

        # All different categories
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["a", "maybe", "a", "a", "b", "b", "maybe"],
                             'c': ["d", "d", "g", "g", "g", "t", "t"]})

        profiler = dp.StructuredProfiler(data)
        expected_mat = np.array([
            [1, 0.007295, 0.007295],
            [0.007295, 1, 0.015609],
            [0.007295, 0.015609, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

        # Identical columns
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "y", "y", "y", "n", "n", "n"],
                             'c': ["y", "y", "y", "y", "n", "n", "n"]})

        profiler = dp.StructuredProfiler(data)
        expected_mat = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    def test_merge_chi2(self, *mocks):
        # Merge empty data
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "maybe", "y", "y", "n", "n", "maybe"],
                             'c': ["n", "maybe", "n", "n", "n", "y", "y"]})
        profiler1 = dp.StructuredProfiler(None)
        profiler2 = dp.StructuredProfiler(data)
        with mock.patch('dataprofiler.profilers.profile_builder.'
                        'StructuredProfiler._add_error_checks'):
            profiler3 = profiler1 + profiler2
        expected_mat = np.array([
            [1, 0.309924, 0.404638],
            [0.309924, 1, 0.548812],
            [0.404638, 0.548812, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler3.chi2_matrix)

        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "maybe", "y", "y", "n", "n", "maybe"],
                             'c': ["n", "maybe", "n", "n", "n", "y", "y"]})

        data1 = data[:4]
        data2 = data[4:]
        profiler1 = dp.StructuredProfiler(data1)
        profiler2 = dp.StructuredProfiler(data2)
        profiler3 = profiler1 + profiler2
        expected_mat = np.array([
            [1, 0.309924, 0.404638],
            [0.309924, 1, 0.548812],
            [0.404638, 0.548812, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler3.chi2_matrix)

        # All different categories
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["a", "maybe", "a", "a", "b", "b", "maybe"],
                             'c': ["d", "d", "g", "g", "g", "t", "t"]})
        data1 = data[:4]
        data2 = data[4:]
        profiler1 = dp.StructuredProfiler(data1)
        profiler2 = dp.StructuredProfiler(data2)
        profiler3 = profiler1 + profiler2
        expected_mat = np.array([
            [1, 0.007295, 0.007295],
            [0.007295, 1, 0.015609],
            [0.007295, 0.015609, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler3.chi2_matrix)

        # Identical columns
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "y", "y", "y", "n", "n", "n"],
                             'c': ["y", "y", "y", "y", "n", "n", "n"]})
        data1 = data[:4]
        data2 = data[4:]
        profiler1 = dp.StructuredProfiler(data1)
        profiler2 = dp.StructuredProfiler(data2)
        profiler3 = profiler1 + profiler2
        expected_mat = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler3.chi2_matrix)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
                spec=StructuredDataLabeler)
    def test_update_chi2(self, *mocks):
        # Update with empty data
        data1 = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                              'b': ["y", "maybe", "y", "y", "n", "n", "maybe"],
                              'c': ["n", "maybe", "n", "n", "n", "y", "y"]})
        data2 = pd.DataFrame({'a': [],
                              'b': [],
                              'c': []})
        profiler = dp.StructuredProfiler(data1)
        profiler.update_profile(data2)
        expected_mat = np.array([
            [1, 0.309924, 0.404638],
            [0.309924, 1, 0.548812],
            [0.404638, 0.548812, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "maybe", "y", "y", "n", "n", "maybe"],
                             'c': ["n", "maybe", "n", "n", "n", "y", "y"]})
        data1 = data[:4]
        data2 = data[4:]
        profiler = dp.StructuredProfiler(data1)
        profiler.update_profile(data2)
        expected_mat = np.array([
            [1, 0.309924, 0.404638],
            [0.309924, 1, 0.548812],
            [0.404638, 0.548812, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

        # All different categories
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["a", "maybe", "a", "a", "b", "b", "maybe"],
                             'c': ["d", "d", "g", "g", "g", "t", "t"]})

        data1 = data[:4]
        data2 = data[4:]
        profiler = dp.StructuredProfiler(data1)
        profiler.update_profile(data2)
        expected_mat = np.array([
            [1, 0.007295, 0.007295],
            [0.007295, 1, 0.015609],
            [0.007295, 0.015609, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

        # Identical columns
        data = pd.DataFrame({'a': ["y", "y", "y", "y", "n", "n", "n"],
                             'b': ["y", "y", "y", "y", "n", "n", "n"],
                             'c': ["y", "y", "y", "y", "n", "n", "n"]})
        data1 = data[:4]
        data2 = data[4:]
        profiler = dp.StructuredProfiler(data1)
        profiler.update_profile(data2)
        expected_mat = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        np.testing.assert_array_almost_equal(expected_mat,
                                             profiler.chi2_matrix)

    def test_correct_datatime_schema_test(self):
        profile_idx = self.trained_schema._col_name_to_idx["datetime"][0]
        profile = self.trained_schema.profile[profile_idx]
        col_schema_info = \
            profile.profiles['data_type_profile']._profiles["datetime"]

        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(2, profile.null_count)
        six.assertCountEqual(self, ['nan'], profile.null_types)
        self.assertEqual(['%m/%d/%y %H:%M'], col_schema_info['date_formats'])

    def test_correct_integer_column_detection_src(self):
        profile_idx = self.trained_schema._col_name_to_idx["src"][0]
        profile = self.trained_schema.profile[profile_idx]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]

        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(3, profile.null_count)

    def test_correct_integer_column_detection_int_col(self):
        profile_idx = self.trained_schema._col_name_to_idx["int_col"][0]
        profile = self.trained_schema.profile[profile_idx]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]
        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(0, profile.null_count)

    def test_correct_integer_column_detection_port(self):
        profile_idx = self.trained_schema._col_name_to_idx["srcport"][0]
        profile = self.trained_schema.profile[profile_idx]
        col_schema_info = profile.profiles['data_type_profile']._profiles["int"]
        self.assertEqual(2999, profile.sample_size)
        self.assertEqual(col_schema_info.sample_size,
                         col_schema_info.match_count)
        self.assertEqual(197, profile.null_count)

    def test_correct_integer_column_detection_destport(self):
        profile_idx = self.trained_schema._col_name_to_idx["destport"][0]
        profile = self.trained_schema.profile[profile_idx]
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
                "unique_row_ratio", "duplicate_row_count", "file_type",
                "encoding", "correlation_matrix", "chi2_matrix", "profile_schema", "times"
            ]
        )
        flat_report = self.trained_schema.report(
            report_options={"output_format": "flat"})
        self.assertEqual(test_utils.get_depth(flat_report), 1)
        with mock.patch('dataprofiler.profilers.helpers.report_helpers'
                        '._prepare_report') as pr_mock:
            self.trained_schema.report(
                report_options={"output_format": 'pretty'})
            # Once for global_stats, once for each of 16 columns
            self.assertEqual(pr_mock.call_count, 17)

    def test_report_schema_and_data_stats_match_order(self):
        data = pd.DataFrame([[1, 2, 3, 4, 5, 6],
                             [10, 20, 30, 40, 50, 60]],
                            columns=["a", "b", "a", "b", "c", "d"])
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        profiler = dp.StructuredProfiler(data=data, options=profiler_options)

        report = profiler.report()
        schema = report["global_stats"]["profile_schema"]
        data_stats = report["data_stats"]

        expected_schema = {"a": [0, 2], "b": [1, 3], "c": [4], "d": [5]}
        self.assertDictEqual(expected_schema, schema)

        # Check that the column order in the report matches the column order
        # In the schema (and in the data)
        for name in schema:
            for idx in schema[name]:
                # Use min of column to validate column order amongst duplicates
                col_min = data.iloc[0, idx]
                self.assertEqual(name, data_stats[idx]["column_name"])
                self.assertEqual(col_min, data_stats[idx]["statistics"]["min"])

    def test_pretty_report_doesnt_cast_schema(self):
        report = self.trained_schema.report(
            report_options={"output_format": "pretty"})
        # Want to ensure the values of this dict are of type list[int]
        # Since pretty "prettifies" lists into strings with ... to shorten
        expected_schema = {"datetime": [0], "host": [1], "src": [2],
                           "proto": [3], "type": [4], "srcport": [5],
                           "destport": [6], "srcip": [7], "locale": [8],
                           "localeabbr": [9], "postalcode": [10],
                           "latitude": [11], "longitude": [12], "owner": [13],
                           "comment": [14], "int_col": [15]}
        self.assertDictEqual(expected_schema,
                             report["global_stats"]["profile_schema"])

    def test_omit_keys_with_duplicate_cols(self):
        data = pd.DataFrame([[1, 2, 3, 4, 5, 6],
                             [10, 20, 30, 40, 50, 60]],
                            columns=["a", "b", "a", "b", "c", "d"])
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        profiler = dp.StructuredProfiler(data=data, options=profiler_options)
        report = profiler.report(report_options={
            "omit_keys": ["data_stats.a.statistics.min",
                          "data_stats.d.statistics.max",
                          "data_stats.*.statistics.null_types_index"]})
        # Correctness of schema asserted in prior test
        schema = report["global_stats"]["profile_schema"]
        data_stats = report["data_stats"]

        for idx in range(len(report["data_stats"])):
            # Assert that min is absent from a's data_stats and not the others
            if idx in schema["a"]:
                self.assertNotIn("min", data_stats[idx]["statistics"])
            else:
                self.assertIn("min", report["data_stats"][idx]["statistics"])

            # Assert that max is absent from d's data_stats and not the others
            if idx in schema["d"]:
                self.assertNotIn("max", report["data_stats"][idx]["statistics"])
            else:
                self.assertIn("max", report["data_stats"][idx]["statistics"])

            # Assert that null_types_index not present in any
            self.assertNotIn("null_types_index",
                             report["data_stats"][idx]["statistics"])

    def test_omit_cols_preserves_schema(self):
        data = pd.DataFrame([[1, 2, 3, 4, 5, 6],
                             [10, 20, 30, 40, 50, 60]],
                            columns=["a", "b", "a", "b", "c", "d"])
        omit_cols = ["a", "d"]
        omit_idxs = [0, 2, 5]
        omit_keys = [f"data_stats.{col}" for col in omit_cols]
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        profiler = dp.StructuredProfiler(data=data, options=profiler_options)
        report = profiler.report(report_options={"omit_keys": omit_keys})

        for idx in range(len(report["data_stats"])):
            if idx in omit_idxs:
                self.assertIsNone(report["data_stats"][idx])
            else:
                self.assertIsNotNone(report["data_stats"][idx])

        # This will keep the data_stats key but remove all columns
        report = profiler.report(report_options={"omit_keys": ["data_stats.*"]})

        for col_report in report["data_stats"]:
            self.assertIsNone(col_report)

    def test_report_quantiles(self):
        report_none = self.trained_schema.report(
            report_options={"num_quantile_groups": None})
        report = self.trained_schema.report()
        self.assertEqual(report_none, report)
        for col in report["data_stats"]:
            if col["column_name"] == "int_col":
                report_quantiles = col["statistics"]["quantiles"]
                break
        self.assertEqual(len(report_quantiles), 3)
        report2 = self.trained_schema.report(
            report_options={"num_quantile_groups": 1000})
        for col in report2["data_stats"]:
            if col["column_name"] == "int_col":
                report2_1000_quant = col["statistics"]["quantiles"]
                break
        self.assertEqual(len(report2_1000_quant), 999)
        self.assertEqual(report_quantiles, {
            0: report2_1000_quant[249],
            1: report2_1000_quant[499],
            2: report2_1000_quant[749],
        })

    def test_report_omit_keys(self):
        # Omit both report keys manually
        no_report_keys = self.trained_schema.report(
            report_options={"omit_keys": ['global_stats', 'data_stats']})
        self.assertCountEqual({}, no_report_keys)

        # Omit just data_stats
        no_data_stats = self.trained_schema.report(
            report_options={"omit_keys": ['data_stats']})
        self.assertCountEqual({"global_stats"}, no_data_stats)

        # Omit a global stat
        no_samples_used = self.trained_schema.report(
            report_options={"omit_keys": ['global_stats.samples_used']})
        self.assertNotIn("samples_used", no_samples_used["global_stats"])

        # Omit all keys
        nothing = self.trained_schema.report(
            report_options={"omit_keys": ['*']})
        self.assertCountEqual({}, nothing)

        # Omit every data_stats column
        empty_data_stats_cols = self.trained_schema.report(
            report_options={"omit_keys": ['global_stats', 'data_stats.*']})
        # data_stats key still present, but all columns are None
        self.assertCountEqual({"data_stats"}, empty_data_stats_cols)
        self.assertTrue(all([rep is None
                             for rep in empty_data_stats_cols["data_stats"]]))

        # Omit specific data_stats column
        no_datetime = self.trained_schema.report(
            report_options={"omit_keys": ['data_stats.datetime']})
        self.assertNotIn("datetime", no_datetime["data_stats"])

        # Omit a statistic from each column
        no_sum = self.trained_schema.report(
            report_options={"omit_keys": ['data_stats.*.statistics.sum']})
        self.assertTrue(all(["sum" not in rep["statistics"]
                             for rep in no_sum["data_stats"]]))

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
            report_options={"output_format": "compact"})

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
                                    'null_types_index', 'categorical_count']:
                    # key names should contain only alphanumeric letters or '_'
                    self.assertIsNotNone(re.match('^[a-zA-Z0-9_]+$', str(key)))
                if isinstance(report[key], dict):
                    recursive_test_helper(report[key], key)

        _report = self.trained_schema.report()
        recursive_test_helper(_report)

    def test_data_label_assigned(self):
        # only use 5 samples
        trained_schema = dp.StructuredProfiler(self.aws_dataset, samples_per_update=5)
        report = trained_schema.report()
        has_non_null_column = False
        for i in range(len(report['data_stats'])):
            # only test non-null columns
            if report['data_stats'][i]['data_type'] is not None:
                self.assertIsNotNone(report['data_stats'][i]['data_label'])
                has_non_null_column = True
        if not has_non_null_column:
            self.fail(
                "Dataset tested did not have a non-null column and therefore "
                "could not validate the test.")

    def test_text_data_raises_error(self):
        text_file_path = os.path.join(
            test_root_path, 'data', 'txt/sentence-10x.txt'
        )
        with self.assertRaisesRegex(TypeError, 'Cannot provide TextData object'
                                               ' to StructuredProfiler'):
            profiler = dp.StructuredProfiler(dp.Data(text_file_path))

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_chi2')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch('dataprofiler.profilers.profile_builder.StructuredProfiler.'
                '_update_row_statistics')
    @mock.patch('dataprofiler.profilers.profile_builder.StructuredColProfiler')
    def test_sample_size_warning_in_the_profiler(self, *mocks):
        # structure data profile mock
        sdp_mock = mock.Mock()
        sdp_mock.clean_data_and_get_base_stats.return_value = (None, None)
        mocks[0].return_value = sdp_mock

        data = pd.DataFrame([1, None, 3, 4, 5, None])
        with self.assertWarnsRegex(UserWarning,
                                   "The data will be profiled with a sample "
                                   "size of 3. All statistics will be based on "
                                   "this subsample and not the whole dataset."):
            profile1 = dp.StructuredProfiler(data, samples_per_update=3)

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_min_col_samples_used(self, *mocks):
        # No cols sampled since no cols to sample
        empty_df = pd.DataFrame([])
        empty_profile = dp.StructuredProfiler(empty_df)
        self.assertEqual(0, empty_profile._min_col_samples_used)

        # Every column fully sampled
        full_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        full_profile = dp.StructuredProfiler(full_df)
        self.assertEqual(3, full_profile._min_col_samples_used)

        # First col sampled only twice, so that is min
        sparse_df = pd.DataFrame([[1, None, None],
                                  [1, 1, None],
                                  [1, None, 1]])
        sparse_profile = dp.StructuredProfiler(sparse_df, min_true_samples=2,
                                               samples_per_update=1)
        self.assertEqual(2, sparse_profile._min_col_samples_used)

    @mock.patch('dataprofiler.profilers.profile_builder.StructuredProfiler.'
                '_update_profile_from_chunk')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    def test_min_true_samples(self, *mocks):
        empty_df = pd.DataFrame([])

        # Test invalid input
        msg = "`min_true_samples` must be an integer or `None`."
        with self.assertRaisesRegex(ValueError, msg):
            profile = dp.StructuredProfiler(empty_df, min_true_samples="Bloop")

        # Test invalid input given to update_profile
        profile = dp.StructuredProfiler(empty_df)
        with self.assertRaisesRegex(ValueError, msg):
            profile.update_profile(empty_df, min_true_samples="Bloop")

        # Test None input (equivalent to zero)
        profile = dp.StructuredProfiler(empty_df, min_true_samples=None)
        self.assertEqual(None, profile._min_true_samples)
        
        # Test valid input
        profile = dp.StructuredProfiler(empty_df, min_true_samples=10)
        self.assertEqual(10, profile._min_true_samples)

    def test_save_and_load(self):
        datapth = "dataprofiler/tests/data/"
        test_files = ["csv/guns.csv", "csv/iris.csv"]

        for test_file in test_files:
            # Create Data and StructuredProfiler objects
            data = dp.Data(os.path.join(datapth, test_file))
            options = ProfilerOptions()
            options.set({"correlation.is_enabled": True})
            save_profile = dp.StructuredProfiler(data)

            # store the expected data_labeler
            data_labeler = save_profile.options.data_labeler.data_labeler_object

            # Save and Load profile with Mock IO
            with mock.patch('builtins.open') as m:
                mock_file = setup_save_mock_open(m)
                save_profile.save()
                mock_file.seek(0)
                with mock.patch('dataprofiler.profilers.profile_builder.'
                                'DataLabeler', return_value=data_labeler):
                    load_profile = dp.StructuredProfiler.load("mock.pkl")

                # validate loaded profile has same data labeler class
                self.assertIsInstance(
                    load_profile.options.data_labeler.data_labeler_object,
                    data_labeler.__class__)

                # only checks first columns
                # get first column
                first_column_profile = load_profile.profile[0]
                self.assertIsInstance(
                    first_column_profile.profiles['data_label_profile']
                        ._profiles['data_labeler'].data_labeler,
                    data_labeler.__class__)

            # Check that reports are equivalent
            save_report = test_utils.clean_report(save_profile.report())
            load_report = test_utils.clean_report(load_profile.report())
            np.testing.assert_equal(save_report, load_report)

    def test_save_and_load_no_labeler(self):
        # Create Data and UnstructuredProfiler objects
        data = pd.DataFrame([1, 2, 3], columns=["a"])

        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})

        save_profile = dp.StructuredProfiler(data, options=profile_options)

        # Save and Load profile with Mock IO
        with mock.patch('builtins.open') as m:
            mock_file = setup_save_mock_open(m)
            save_profile.save()

            mock_file.seek(0)
            with mock.patch('dataprofiler.profilers.profile_builder.'
                            'DataLabeler'):
                load_profile = dp.StructuredProfiler.load("mock.pkl")

        # Check that reports are equivalent
        save_report = test_utils.clean_report(save_profile.report())
        load_report = test_utils.clean_report(load_profile.report())
        self.assertDictEqual(save_report, load_report)

        # validate both are still usable after
        save_profile.update_profile(pd.DataFrame({"a": [4, 5]}))
        load_profile.update_profile(pd.DataFrame({"a": [4, 5]}))

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_string_index_doesnt_cause_error(self, *mocks):
        dp.StructuredProfiler(pd.DataFrame([[1, 2, 3]], index=["hello"]))

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_dict_in_data_no_error(self, *mocks):
        # validates that _update_row_statistics does not error when trying to
        # hash a dict.
        profiler = dp.StructuredProfiler(pd.DataFrame([[{'test': 1}], [None]]))
        self.assertEqual(1, profiler.row_is_null_count)
        self.assertEqual(2, profiler.total_samples)

    def test_duplicate_columns(self):
        data = pd.DataFrame([[1, 2, 3, 4, 5, 6],
                             [10, 20, 30, 40, 50, 60]],
                            columns=["a", "b", "a", "b", "c", "d"])
        profiler = dp.StructuredProfiler(data)

        # Ensure columns are correctly allocated to profiles in list
        expected_mapping = {"a": [0, 2], "b": [1, 3], "c": [4], "d": [5]}
        self.assertDictEqual(expected_mapping, profiler._col_name_to_idx)
        for col in profiler._col_name_to_idx:
            for idx in profiler._col_name_to_idx[col]:
                # Make sure every index that a column name maps to represents
                # A profile for that named column
                self.assertEqual(col, profiler._profile[idx].name)

        # Check a few stats to ensure calculation with data occurred
        # Initialization ensures column ids and profile ids are identical
        for col_idx in range(len(profiler._profile)):
            col_min = data.iloc[0, col_idx]
            col_max = data.iloc[1, col_idx]
            col_sum = col_min + col_max
            self.assertEqual(col_min, profiler._profile[col_idx].
                             profile["statistics"]["min"])
            self.assertEqual(col_max, profiler._profile[col_idx].
                             profile["statistics"]["max"])
            self.assertEqual(col_sum, profiler._profile[col_idx].
                             profile["statistics"]["sum"])

        # Check that update works as expected
        new_data = pd.DataFrame([[100, 200, 300, 400, 500, 600]],
                                columns=["a", "b", "a", "b", "c", "d"])
        profiler.update_profile(new_data)
        self.assertDictEqual(expected_mapping, profiler._col_name_to_idx)
        for col in profiler._col_name_to_idx:
            for idx in profiler._col_name_to_idx[col]:
                # Make sure every index that a column name maps to represents
                # A profile for that named column
                self.assertEqual(col, profiler._profile[idx].name)

        for col_idx in range(len(profiler._profile)):
            col_min = data.iloc[0, col_idx]
            col_max = new_data.iloc[0, col_idx]
            col_sum = col_min + col_max + data.iloc[1, col_idx]
            self.assertEqual(col_min, profiler._profile[col_idx].
                             profile["statistics"]["min"])
            self.assertEqual(col_max, profiler._profile[col_idx].
                             profile["statistics"]["max"])
            self.assertEqual(col_sum, profiler._profile[col_idx].
                             profile["statistics"]["sum"])

    def test_unique_col_permutation(self, *mocks):
        data = pd.DataFrame([[1, 2, 3, 4],
                             [5, 6, 7, 8]],
                            columns=["a", "b", "c", "d"])
        perm_data = pd.DataFrame([[4, 3, 2, 1],
                                  [8, 7, 6, 5]],
                                 columns=["d", "c", "b", "a"])

        # Test via add
        first_profiler = dp.StructuredProfiler(data)
        perm_profiler = dp.StructuredProfiler(perm_data)
        profiler = first_profiler + perm_profiler

        for col_idx in range(len(profiler._profile)):
            col_min = data.iloc[0, col_idx]
            col_max = data.iloc[1, col_idx]
            # Sum is doubled since it was updated with the same vals
            col_sum = 2 * (col_min + col_max)
            self.assertEqual(col_min, profiler._profile[col_idx].
                             profile["statistics"]["min"])
            self.assertEqual(col_max, profiler._profile[col_idx].
                             profile["statistics"]["max"])
            self.assertEqual(col_sum, profiler._profile[col_idx].
                             profile["statistics"]["sum"])

        # Test via update
        profiler = dp.StructuredProfiler(data)
        profiler.update_profile(perm_data)

        for col_idx in range(len(profiler._profile)):
            col_min = data.iloc[0, col_idx]
            col_max = data.iloc[1, col_idx]
            # Sum is doubled since it was updated with the same vals
            col_sum = 2 * (col_min + col_max)
            self.assertEqual(col_min, profiler._profile[col_idx].
                             profile["statistics"]["min"])
            self.assertEqual(col_max, profiler._profile[col_idx].
                             profile["statistics"]["max"])
            self.assertEqual(col_sum, profiler._profile[col_idx].
                             profile["statistics"]["sum"])

    def test_get_and_validate_schema_mapping(self):
        unique_schema_1 = {"a": [0], "b": [1], "c": [2]}
        unique_schema_2 = {"a": [2], "b": [0], "c": [1]}
        unique_schema_3 = {"a": [0], "b": [1], "d": [2]}

        msg = "Columns do not match, cannot update or merge profiles."
        with self.assertRaisesRegex(ValueError, msg):
            dp.StructuredProfiler._get_and_validate_schema_mapping(
                unique_schema_1,unique_schema_3)

        expected_schema = {0: 0, 1: 1, 2: 2}
        actual_schema = dp.StructuredProfiler. \
            _get_and_validate_schema_mapping(unique_schema_1, {})
        self.assertDictEqual(actual_schema, expected_schema)

        expected_schema = {0: 2, 1: 0, 2: 1}
        actual_schema = dp.StructuredProfiler. \
            _get_and_validate_schema_mapping(unique_schema_1, unique_schema_2)
        self.assertDictEqual(actual_schema, expected_schema)

        dupe_schema_1 = {"a": [0], "b": [1, 2], "c": [3, 4, 5]}
        dupe_schema_2 = {"a": [0], "b": [1, 3], "c": [2, 4, 5]}
        dupe_schema_3 = {"a": [0, 1], "b": [2, 3, 4], "c": [5]}

        four_col_schema = {"a": [0], "b": [1, 2], "c": [3, 4, 5], "d": [6]}

        msg = ("Different number of columns detected for "
               "'a', cannot update or merge profiles.")
        with self.assertRaisesRegex(ValueError, msg):
            dp.StructuredProfiler._get_and_validate_schema_mapping(
                dupe_schema_1, dupe_schema_3)

        msg = ("Different column indices under "
               "duplicate name 'b', cannot update "
               "or merge unless schema is identical.")
        with self.assertRaisesRegex(ValueError, msg):
            dp.StructuredProfiler._get_and_validate_schema_mapping(
                dupe_schema_1, dupe_schema_2)

        msg = "Attempted to merge profiles with different numbers of columns"
        with self.assertRaisesRegex(ValueError, msg):
            dp.StructuredProfiler._get_and_validate_schema_mapping(
                dupe_schema_1, four_col_schema)

        expected_schema = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        actual_schema = dp.StructuredProfiler. \
            _get_and_validate_schema_mapping(dupe_schema_1, dupe_schema_1)
        self.assertDictEqual(actual_schema, expected_schema)

    @mock.patch("dataprofiler.profilers.data_labeler_column_profile."
                "DataLabelerColumn.update")
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnPrimitiveTypeProfileCompiler.diff")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnStatsProfileCompiler.diff")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnDataLabelerCompiler.diff")
    def test_diff(self, *mocks):
        # Data labeler compiler diff
        mocks[0].return_value = {
            'statistics': {
                'avg_predictions': {
                    'a': 'unchanged'
                },
                'label_representation': {
                    'a': 'unchanged'
                }
            },
            'data_label': [[], ['a'], []]
        }
        # stats compiler diff
        mocks[1].return_value = {
            'order': ['ascending', 'descending'], 
            'categorical': 'unchanged', 
            'statistics': {
                'all_compiler_stats': 'unchanged'
            }
        }
        # primitive stats compiler diff
        mocks[2].return_value = {
            'data_type_representation': {
                'all_data_types': 'unchanged'
            },
             'data_type': 'unchanged',
             'statistics': {
                 'numerical_statistics_here': "unchanged"
             }
        }

        data1 = pd.DataFrame([[1, 2], [5, 6]], columns=["a", "b"])
        data2 = pd.DataFrame([[4, 3], [8, 7], [None, None], [9, 10]],
                             columns=["a", "b"])

        options = dp.ProfilerOptions()
        options.structured_options.correlation.is_enabled = True
        profile1 = dp.StructuredProfiler(data1, options=options)
        options2 = dp.ProfilerOptions()
        options2.structured_options.correlation.is_enabled = True
        profile2 = dp.StructuredProfiler(data2, options=options2)

        expected_diff = {
            'global_stats': {
                'samples_used': -2, 
                'column_count': 'unchanged', 
                'row_count': -2, 
                'row_has_null_ratio': -0.25, 
                'row_is_null_ratio': -0.25, 
                'unique_row_ratio': 'unchanged', 
                'duplicate_row_count': -0.25, 
                'file_type': 'unchanged', 
                'encoding': 'unchanged', 
                'correlation_matrix': 
                    np.array([[1.11022302e-16, 3.13803955e-02],
                              [3.13803955e-02, 0.00000000e+00]],
                             dtype=np.float),
                'chi2_matrix':
                    np.array([[ 0.        , -0.04475479],
                              [-0.04475479,  0.        ]],
                             dtype=np.float),
                'profile_schema':
                    [{}, {'a': 'unchanged', 'b': 'unchanged'}, {}]},
            'data_stats': [
                {
                    'column_name': 'a', 
                     'data_type': 'unchanged', 
                     'data_label': [[], ['a'], []], 
                     'categorical': 'unchanged', 
                     'order': ['ascending', 'descending'], 
                     'statistics': {
                         'numerical_statistics_here': 
                             'unchanged', 
                         'all_compiler_stats': 
                             'unchanged', 
                         'avg_predictions': {'a': 'unchanged'}, 
                         'label_representation': {'a': 'unchanged'}, 
                         'sample_size': -2, 
                         'null_count': -1, 
                         'null_types': [[], [], ['nan']], 
                         'null_types_index': [{}, {}, {'nan': {2}}], 
                         'data_type_representation': {
                             'all_data_types': 'unchanged'
                         }
                     }
                },
                {
                     'column_name': 'b', 
                     'data_type': 'unchanged', 
                     'data_label': [[], ['a'], []], 
                     'categorical': 'unchanged', 
                     'order': ['ascending', 'descending'], 
                     'statistics': {
                         'numerical_statistics_here': 'unchanged', 
                         'all_compiler_stats': 'unchanged', 
                         'avg_predictions': {'a': 'unchanged'}, 
                         'label_representation': {'a': 'unchanged'}, 
                         'sample_size': -2, 
                         'null_count': -1, 
                         'null_types': [[], [], ['nan']], 
                         'null_types_index': [{}, {}, {'nan': {2}}], 
                         'data_type_representation': {
                             'all_data_types': 'unchanged'
                         }
                     }
                 }
            ]
        }

        diff = profile1.diff(profile2)
        expected_corr_mat = expected_diff["global_stats"].pop("correlation_matrix")
        diff_corr_mat = diff["global_stats"].pop("correlation_matrix")
        expected_chi2_mat = expected_diff["global_stats"].pop("chi2_matrix")
        diff_chi2_mat = diff["global_stats"].pop("chi2_matrix")

        np.testing.assert_array_almost_equal(expected_corr_mat, diff_corr_mat)
        np.testing.assert_array_almost_equal(expected_chi2_mat, diff_chi2_mat)
        self.assertDictEqual(expected_diff, diff)
    
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch("dataprofiler.profilers.data_labeler_column_profile."
                "DataLabelerColumn.update")
    def test_diff_type_checking(self, *mocks):
        data = pd.DataFrame([[1, 2], [5, 6]],
                            columns=["a", "b"])
        profile = dp.StructuredProfiler(data)
        with self.assertRaisesRegex(TypeError, 
                                    '`StructuredProfiler` and `str` are not of '
                                    'the same profiler type.'):
            profile.diff("ERROR")
        
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch("dataprofiler.profilers.data_labeler_column_profile."
                "DataLabelerColumn.update")
    def test_diff_with_different_schema(self, *mocks):
        
        data1 = pd.DataFrame([[1, 2], [5, 6]],
                             columns=["G", "b"])
        data2 = pd.DataFrame([[4, 3, 1], [8, 7, 3], [None, None, 1], [9, 1, 10]],
                             columns=["a", "b", "c"])

        # Test via add
        profile1 = dp.StructuredProfiler(data1)
        profile2 = dp.StructuredProfiler(data2)
        
        expected_diff = {
            'global_stats': {
                'file_type': 'unchanged', 
                'encoding': 'unchanged', 
                'samples_used': -2, 
                'column_count': -1, 
                'row_count': -2, 
                'row_has_null_ratio': -0.25, 
                'row_is_null_ratio': 'unchanged', 
                'unique_row_ratio': 'unchanged', 
                'duplicate_row_count': 'unchanged', 
                'correlation_matrix': None,
                'chi2_matrix': None,
                'profile_schema': [{'G': [0]}, 
                                   {'b': 'unchanged'}, 
                                   {'a': [0], 'c': [2]}]}, 
            'data_stats': []
        }

        self.assertDictEqual(expected_diff, profile1.diff(profile2))

    @mock.patch("dataprofiler.profilers.data_labeler_column_profile."
                "DataLabelerColumn.update")
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnPrimitiveTypeProfileCompiler.diff")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnStatsProfileCompiler.diff")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnDataLabelerCompiler.diff")
    @mock.patch("sys.stderr", new_callable=StringIO)
    def test_logs(self, mock_stderr, *mocks):
        options = StructuredOptions()
        options.multiprocess.is_enabled = False

        # Capture logs of level INFO and above
        with self.assertLogs('DataProfiler.profilers.profile_builder',
                             level='INFO') as logs:
            StructuredProfiler(pd.DataFrame([[0, 1], [2, 3]]), options=options)

        # Logs to update user on nulls and statistics
        self.assertEqual(['INFO:DataProfiler.profilers.profile_builder:'
                          'Finding the Null values in the columns... ',
                          'INFO:DataProfiler.profilers.profile_builder:'
                          'Calculating the statistics... '],
                         logs.output)

        # Ensure tqdm printed progress bar
        self.assertIn('#' * 10, mock_stderr.getvalue())

        # Clear stderr
        mock_stderr.seek(0)
        mock_stderr.truncate(0)

        # Now tqdm shouldn't be printed
        dp.set_verbosity(logging.WARNING)

        StructuredProfiler(pd.DataFrame([[0, 1], [2, 3]]))

        # Ensure no progress bar printed
        self.assertNotIn('#' * 10, mock_stderr.getvalue())

    def test_unique_row_ratio_empty_profiler(self):
        profiler = StructuredProfiler(pd.DataFrame([]))
        self.assertEqual(0, profiler._get_unique_row_ratio())


class TestStructuredColProfilerClass(unittest.TestCase):

    def setUp(self):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        test_utils.set_seed(seed=0)
        cls.input_file_path = os.path.join(
            test_root_path, 'data', 'csv/aws_honeypot_marx_geo.csv'
        )
        cls.aws_dataset = pd.read_csv(cls.input_file_path)

    def test_base_props(self):
        src_column = self.aws_dataset.src
        src_profile = StructuredColProfiler(
            src_column, sample_size=len(src_column))

        self.assertIsInstance(src_profile.profiles['data_type_profile'],
                              ColumnPrimitiveTypeProfileCompiler)
        self.assertIsInstance(src_profile.profiles['data_stats_profile'],
                              ColumnStatsProfileCompiler)
        self.assertIsInstance(src_profile.profiles['data_label_profile'],
                              ColumnDataLabelerCompiler)

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
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_add_profilers(self, *mocks):
        data = pd.Series([1, None, 3, 4, 5, None])
        profile1 = StructuredColProfiler(data[:2])
        profile2 = StructuredColProfiler(data[2:])

        # test incorrect type
        with self.assertRaisesRegex(TypeError,
                                    '`StructuredColProfiler` and `int` are '
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
        profile1 = dp.StructuredProfiler(data, options=options)
        profile2 = dp.StructuredProfiler(data)
        with self.assertRaisesRegex(ValueError,
                                    'Structured profilers were not setup with '
                                    'the same options, hence they do not '
                                    'calculate the same profiles and cannot be '
                                    'added together.'):
            profile1 + profile2

    def test_clean_data_and_get_base_stats(self, *mocks):
        data = pd.Series([1, None, 3, 4, None, 6],
                         index=['a', 'b', 'c', 'd', 'e', 'f'])

        # validate that if sliced data, still functional
        # previously `iloc` was used at:
        # `df_series = df_series.loc[sorted(true_sample_list)]`
        # which caused errors

        #Tests with default null values set
        profiler = mock.Mock(spec=StructuredColProfiler)
        null_values = {
            "": 0,
            "nan": re.IGNORECASE,
            "none": re.IGNORECASE,
            "null": re.IGNORECASE,
            "  *": 0,
            "--*": 0,
            "__*": 0,
        }

        test_utils.set_seed(seed=0)
        df_series, base_stats = \
            StructuredColProfiler.clean_data_and_get_base_stats(
                df_series=data[1:], sample_size=6, null_values=null_values,
                min_true_samples=0)
        # note data above is a subset `df_series=data[1:]`, 1.0 will not exist
        self.assertTrue(np.issubdtype(np.object_, df_series.dtype))
        self.assertDictEqual({'sample': ['4.0', '6.0', '3.0'],
                              'sample_size': 5, 'null_count': 2,
                              'null_types': dict(nan=['e', 'b']),
                              'min_id': None, 'max_id': None}, base_stats)

        # Tests with some other null values set
        null_values = {
            "1.0": 0,
            "3.0": 0
        }
        df_series, base_stats = \
            StructuredColProfiler.clean_data_and_get_base_stats(
                df_series=data, sample_size=6, null_values=null_values,
                min_true_samples=0)
        self.assertDictEqual({'sample': ["nan", '6.0', '4.0', "nan"],
                              'sample_size': 6, 'null_count': 2,
                              'null_types': {'1.0': ['a'], '3.0': ['c']},
                              'min_id': None, 'max_id': None}, base_stats)

        # Tests with no null values set
        null_values = {}
        df_series, base_stats = \
            StructuredColProfiler.clean_data_and_get_base_stats(
                df_series=data, sample_size=6, null_values=null_values,
                min_true_samples=0)
        self.assertDictEqual({'sample': ["3.0", "4.0", '6.0', "nan", "1.0"],
                              'sample_size': 6, 'null_count': 0,
                              'null_types': {},
                              'min_id': None, 'max_id': None}, base_stats)

    def test_column_names(self):
        data = [['a', 1], ['b', 2], ['c', 3]]
        df = pd.DataFrame(data, columns=['letter', 'number'])
        profile1 = StructuredColProfiler(df['letter'])
        profile2 = StructuredColProfiler(df['number'])
        self.assertEqual(profile1.name, 'letter')
        self.assertEqual(profile2.name, 'number')

        df_series = pd.Series([1, 2, 3, 4, 5])
        profile = StructuredColProfiler(df_series)
        self.assertEqual(profile.name, df_series.name)

        # Ensure issue raised
        profile = StructuredColProfiler(df['letter'])
        with self.assertRaises(ValueError) as context:
            profile.update_profile(df['number'])
        self.assertTrue(
            'Column names have changed, col number does not match prior name letter',
            context
        )

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
        std_profile = StructuredColProfiler(src_column,
                                            sample_size=len(src_column))
        togg_profile = StructuredColProfiler(src_column,
                                             sample_size=len(src_column),
                                             options=structured_options)
        self.assertIn('data_label_profile', std_profile.profiles)
        self.assertNotIn('data_label_profile', togg_profile.profiles)

    def test_null_count(self):
        column = pd.Series([1, float('nan')] * 10)

        # test null_count when full sample size
        random.seed(0)
        profile = StructuredColProfiler(column, sample_size=len(column))
        self.assertEqual(10, profile.null_count)

    def test_generating_report_ensure_no_error(self):
        file_path = os.path.join(test_root_path, 'data', 'csv/diamonds.csv')
        data = pd.read_csv(file_path)
        profile = dp.StructuredProfiler(data[:1000])
        readable_report = profile.report(
            report_options={"output_format": "compact"})

    def test_get_sample_size(self):
        data = pd.DataFrame([0] * int(50e3))

        # test data size < min_sample_size = 5000 by default
        profiler = dp.StructuredProfiler(pd.DataFrame([]))
        profiler._min_sample_size = 5000
        profiler._sampling_ratio = 0.2
        sample_size = profiler._get_sample_size(data[:1000])
        self.assertEqual(1000, sample_size)

        # test data size * 0.20 < min_sample_size < data size
        sample_size = profiler._get_sample_size(data[:10000])
        self.assertEqual(5000, sample_size)

        # test min_sample_size > data size * 0.20
        sample_size = profiler._get_sample_size(data)
        self.assertEqual(10000, sample_size)

        # test min_sample_size > data size * 0.10
        profiler._sampling_ratio = 0.5
        sample_size = profiler._get_sample_size(data)
        self.assertEqual(25000, sample_size)

    @mock.patch('dataprofiler.profilers.profile_builder.StructuredProfiler.'
                '_update_profile_from_chunk')
    def test_sample_size_passed_to_profile(self, *mocks):

        update_mock = mocks[0]

        # data setup
        data = pd.DataFrame([0] * int(50e3))

        # option setup
        profiler_options = ProfilerOptions()
        profiler_options.structured_options.multiprocess.is_enabled = False
        profiler_options.set({'data_labeler.is_enabled': False})

        # test data size < min_sample_size = 5000 by default
        profiler = dp.StructuredProfiler(data[:1000], options=profiler_options)
        profiler._min_sample_size = 5000
        profiler._sampling_ratio = 0.2
        self.assertEqual(1000, update_mock.call_args[0][1])

        # test data size * 0.20 < min_sample_size < data size
        profiler = dp.StructuredProfiler(data[:10000], options=profiler_options)
        profiler._min_sample_size = 5000
        profiler._sampling_ratio = 0.2
        self.assertEqual(5000, update_mock.call_args[0][1])

        # test min_sample_size > data size * 0.20
        profiler = dp.StructuredProfiler(data, options=profiler_options)
        profiler._min_sample_size = 5000
        profiler._sampling_ratio = 0.2
        self.assertEqual(10000, update_mock.call_args[0][1])

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    def test_index_overlap_for_update_profile(self, *mocks):
        data = pd.Series([0, None, 1, 2, None])
        profile = StructuredColProfiler(data)
        self.assertEqual(0, profile._min_id)
        self.assertEqual(4, profile._max_id)
        self.assertDictEqual(profile.null_types_index, {'nan': {1, 4}})
        profile.update_profile(data)
        # Now all indices will be shifted by max_id + 1 (5)
        # So the 2 None will move from indices 1, 4 to 6, 9
        self.assertEqual(0, profile._min_id)
        self.assertEqual(9, profile._max_id)
        self.assertDictEqual(profile.null_types_index, {'nan': {1, 4, 6, 9}})

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    def test_index_overlap_for_merge(self, *mocks):
        data = pd.Series([0, None, 1, 2, None])
        profile1 = StructuredColProfiler(data)
        profile2 = StructuredColProfiler(data)

        # Ensure merged profile included shifted indices
        profile3 = profile1 + profile2
        self.assertEqual(0, profile3._min_id)
        self.assertEqual(9, profile3._max_id)
        self.assertDictEqual(profile3.null_types_index, {'nan': {1, 4, 6, 9}})

        # Ensure original profiles not overwritten
        self.assertEqual(0, profile1._min_id)
        self.assertEqual(4, profile1._max_id)
        self.assertDictEqual(profile1.null_types_index, {'nan': {1, 4}})
        self.assertEqual(0, profile2._min_id)
        self.assertEqual(4, profile2._max_id)
        self.assertDictEqual(profile2.null_types_index, {'nan': {1, 4}})

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    def test_min_max_id_properly_update(self, *mocks):
        data = pd.Series([1, None, 3, 4, 5, None, 1])
        profile1 = StructuredColProfiler(data[:2])
        profile2 = StructuredColProfiler(data[2:])

        # Base initialization
        self.assertEqual(0, profile1._min_id)
        self.assertEqual(1, profile1._max_id)
        self.assertEqual(2, profile2._min_id)
        self.assertEqual(6, profile2._max_id)

        # Needs to work with merge
        profile3 = profile1 + profile2
        self.assertEqual(0, profile3._min_id)
        self.assertEqual(6, profile3._max_id)

        # Needs to work with update_profile
        profile = StructuredColProfiler(data[:2])
        profile.update_profile(data[2:])
        self.assertEqual(0, profile._min_id)
        self.assertEqual(6, profile._max_id)
        
    @mock.patch('dataprofiler.profilers.data_labeler_column_profile.DataLabeler')
    @mock.patch("dataprofiler.profilers.data_labeler_column_profile."
                "DataLabelerColumn.update")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnPrimitiveTypeProfileCompiler.diff")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnStatsProfileCompiler.diff")
    @mock.patch("dataprofiler.profilers.column_profile_compilers."
                "ColumnDataLabelerCompiler.diff")
    def test_diff(self, *mocks):
        # Data labeler compiler diff
        mocks[0].return_value = {
            'statistics': {
                'avg_predictions': {
                    'a': 'unchanged'
                },
                'label_representation': {
                    'a': 'unchanged'
                }
            },
            'data_label': [[], ['a'], []]
        }
        # stats compiler diff
        mocks[1].return_value = {
            'order': ['ascending', 'descending'],
            'categorical': 'unchanged',
            'statistics': {
                'all_compiler_stats': 'unchanged'
            }
        }
        # primitive stats compiler diff
        mocks[2].return_value = {
            'data_type_representation': {
                'all_data_types': 'unchanged'
            },
            'data_type': 'unchanged',
            'statistics': {
                'numerical_statistics_here': "unchanged"
            }
        }
        
        data = pd.Series([1, None, 3, 4, 5, None, 1])
        data2 = pd.Series(["hello", "goodby", 125, 0])
        data.name = "TEST"
        data2.name = "TEST"

        profile1 = StructuredColProfiler(data)
        profile2 = StructuredColProfiler(data2)
        
        expected_diff = {
            'column_name': 'TEST', 
            'data_type': 'unchanged', 
            'data_label': [[], ['a'], []], 
            'categorical': 'unchanged', 
            'order': ['ascending', 'descending'], 
            'statistics': {
                'numerical_statistics_here': 'unchanged', 
                'all_compiler_stats': 'unchanged', 
                'avg_predictions': {'a': 'unchanged'}, 
                'label_representation': {'a': 'unchanged'}, 
                'sample_size': 3, 
                'null_count': 2, 
                'null_types': [['nan'], [], []], 
                'null_types_index': [{'nan': {1, 5}}, {}, {}], 
                'data_type_representation': {
                    'all_data_types': 'unchanged'
                }
            }
        }

        self.assertDictEqual(expected_diff, dict(profile1.diff(profile2)))

@mock.patch('dataprofiler.profilers.profile_builder.UnstructuredCompiler',
            spec=UnstructuredCompiler)
@mock.patch('dataprofiler.profilers.profile_builder.DataLabeler',
            spec=UnstructuredDataLabeler)
class TestUnstructuredProfiler(unittest.TestCase):

    @classmethod
    def setUp(cls):
        test_utils.set_seed(seed=0)

    def test_base(self, *mocks):
        # ensure can make an empty profiler
        profiler = UnstructuredProfiler(None)
        self.assertIsNone(profiler.encoding)
        self.assertIsNone(profiler.file_type)
        self.assertIsNone(profiler._profile)
        self.assertIsNone(profiler._samples_per_update)
        self.assertEqual(0, profiler._min_true_samples)
        self.assertEqual(0, profiler.total_samples)
        self.assertEqual(0, profiler._empty_line_count)
        self.assertEqual(0, profiler.memory_size)
        self.assertEqual(0.2, profiler._sampling_ratio)
        self.assertEqual(5000, profiler._min_sample_size)
        self.assertEqual([], profiler.sample)
        self.assertIsInstance(profiler.options, UnstructuredOptions)
        self.assertDictEqual({}, profiler.times)

        # can set samples_per_update and min_true_samples
        profiler = UnstructuredProfiler(None, samples_per_update=10,
                                        min_true_samples=5)
        self.assertEqual(profiler._samples_per_update, 10)
        self.assertEqual(profiler._min_true_samples, 5)

        # can properties update correctly for data
        data = pd.Series(['this', 'is my', '\n\r', 'test'])
        profiler = UnstructuredProfiler(data)
        self.assertEqual(4, profiler.total_samples)
        self.assertCountEqual(['this', 'is my', 'test'], profiler.sample)
        self.assertEqual(1, profiler._empty_line_count)
        self.assertEqual(15 / 1024 ** 2, profiler.memory_size)
        self.assertEqual("<class 'pandas.core.series.Series'>",
                         profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertIsInstance(profiler._profile, UnstructuredCompiler)
        self.assertIn('clean_and_base_stats', profiler.times)

        # can properties update correctly for data loaded from file
        data = pd.Series(['this', 'is my', '\n\r', 'test'])
        mock_data_reader = mock.Mock(spec=dp.data_readers.csv_data.CSVData)
        mock_data_reader.data = data
        mock_data_reader.data_type = 'csv'
        mock_data_reader.file_encoding = 'utf-8'
        mock_data_reader.input_file_path = 'fake/path/file.csv'

        profiler = UnstructuredProfiler(mock_data_reader)
        self.assertEqual(4, profiler.total_samples)
        self.assertCountEqual(['this', 'is my', 'test'], profiler.sample)
        self.assertEqual(1, profiler._empty_line_count)
        self.assertEqual(15 / 1024 ** 2, profiler.memory_size)
        self.assertEqual("csv", profiler.file_type)
        self.assertEqual("utf-8", profiler.encoding)
        self.assertIsInstance(profiler._profile, UnstructuredCompiler)

    def test_bad_input_data(self, *mocks):
        allowed_data_types = (r"\(<class 'str'>, "
                              r"<class 'list'>, "
                              r"<class 'pandas.core.series.Series'>, "
                              r"<class 'pandas.core.frame.DataFrame'>\)")
        bad_data_types = [1, {}, np.inf]
        for data in bad_data_types:
            with self.assertRaisesRegex(TypeError,
                                        r"Data must either be imported using "
                                        r"the data_readers or using one of the "
                                        r"following: " + allowed_data_types):
                UnstructuredProfiler(data)

    def test_str_input_data(self, *mocks):
        data = 'this is my\n\rtest'
        profiler = UnstructuredProfiler(data)
        self.assertEqual(1, profiler.total_samples)
        self.assertEqual(0, profiler._empty_line_count)
        self.assertEqual(16 / 1024 ** 2, profiler.memory_size)
        self.assertEqual("<class 'str'>", profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertIsInstance(profiler._profile, UnstructuredCompiler)

    def test_list_input_data(self, *mocks):
        data = ['this', 'is my', '\n\r', 'test']
        profiler = UnstructuredProfiler(data)
        self.assertEqual(4, profiler.total_samples)
        self.assertEqual(1, profiler._empty_line_count)
        self.assertEqual(15 / 1024 ** 2, profiler.memory_size)
        self.assertEqual("<class 'list'>", profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertIsInstance(profiler._profile, UnstructuredCompiler)

    def test_dataframe_input_data(self, *mocks):
        data = pd.DataFrame(['this', 'is my', '\n\r', 'test'])
        profiler = UnstructuredProfiler(data)
        self.assertEqual(4, profiler.total_samples)
        self.assertEqual(1, profiler._empty_line_count)
        self.assertEqual(15 / 1024 ** 2, profiler.memory_size)
        self.assertEqual("<class 'pandas.core.frame.DataFrame'>", profiler.file_type)
        self.assertIsNone(profiler.encoding)
        self.assertIsInstance(profiler._profile, UnstructuredCompiler)

    def test_merge_profiles(self, *mocks):
        # can properties update correctly for data
        data1 = pd.Series(['this', 'is my', '\n\r', 'test'])
        data2 = pd.Series(['here\n', '\t    ', ' ', ' is', '\n\r', 'more data'])

        # create profilers
        with test_utils.mock_timeit():
            profiler1 = UnstructuredProfiler(data1)
            profiler2 = UnstructuredProfiler(data2)

        self.assertDictEqual({'clean_and_base_stats': 1}, profiler1.times)
        self.assertDictEqual({'clean_and_base_stats': 1}, profiler2.times)

        # mock out _profile
        profiler1._profile = 1
        profiler2._profile = 2

        # merge profilers
        with test_utils.mock_timeit():
            merged_profile = profiler1 + profiler2
        self.assertEqual(10, merged_profile.total_samples)
        self.assertEqual(4, merged_profile._empty_line_count)
        self.assertEqual(40 / 1024 ** 2, merged_profile.memory_size)
        # note how sample doesn't include whitespace lines
        self.assertCountEqual(['this', ' is', 'here\n', 'more data', 'is my'],
                              merged_profile.sample)
        self.assertEqual(3, merged_profile._profile)
        self.assertDictEqual({'clean_and_base_stats': 2}, merged_profile.times)

    @mock.patch('dataprofiler.profilers.profile_builder.UnstructuredCompiler.diff')
    def test_diff(self, *mocks):

        # Set up compiler diff
        mocks[2].side_effect = [UnstructuredCompiler(), UnstructuredCompiler()]
        mocks[0].return_value = {
            'statistics': {
                'all_vocab_and_word_stats': [['A', 'B'], ['C'], ['D']]
            }, 
            'data_label': {
                'entity_counts': {
                    'word_and_char_level_stats': {
                        'LABEL': 'unchanged'
                    }
                }, 
                'entity_percentages': {
                    'word_and_char_level_stats': {
                        'LABEL': 'unchanged'
                    }
                }
            }
        }

        data1 = pd.Series(['this', 'is my', '\n\r', 'test'])
        data2 = pd.Series(['here\n', '\t    ', ' ', ' is', '\n\r', 'more data'])
        profiler1 = UnstructuredProfiler(data1)
        profiler2 = UnstructuredProfiler(data2)

        expected_diff = {
            'global_stats': {
                'samples_used': -2, 
                'empty_line_count': -2, 
                'file_type': 'unchanged', 
                'encoding': 'unchanged', 
                'memory_size': -10/1024**2
            }, 
            'data_stats': {
                'statistics': {
                    'all_vocab_and_word_stats': [['A', 'B'], ['C'], ['D']]}, 
                'data_label': {
                    'entity_counts': {
                        'word_and_char_level_stats': 
                            {'LABEL': 'unchanged'}
                    }, 
                    'entity_percentages': {
                        'word_and_char_level_stats': {
                            'LABEL': 'unchanged'
                        }
                    }
                }
            }
        }
        self.assertDictEqual(expected_diff, profiler1.diff(profiler2))

    def test_get_sample_size(self, *mocks):
        data = pd.DataFrame([0] * int(50e3))

        # test data size < min_sample_size = 5000 by default
        profiler = UnstructuredProfiler(None)
        profiler._min_sample_size = 5000
        profiler._sampling_ratio = 0.2
        sample_size = profiler._get_sample_size(data[:1000])
        self.assertEqual(1000, sample_size)

        # test data size * 0.20 < min_sample_size < data size
        sample_size = profiler._get_sample_size(data[:10000])
        self.assertEqual(5000, sample_size)

        # test min_sample_size > data size * 0.20
        sample_size = profiler._get_sample_size(data)
        self.assertEqual(10000, sample_size)

        # test min_sample_size > data size * 0.5
        profiler._sampling_ratio = 0.5
        sample_size = profiler._get_sample_size(data)
        self.assertEqual(25000, sample_size)

    def test_clean_data_and_get_base_stats(self, *mocks):
        data = pd.Series(['here\n', '\t    ', 'a', ' is', '\n\r', 'more data'])

        # needed bc _clean_data_and_get_base_stats is not static
        # for timeit which wraps this func and uses the class
        profiler = mock.Mock(spec=UnstructuredProfiler)
        profiler.times = {'clean_and_base_stats': 0}

        # case when min_true_samples not set and subset of data
        df_series, base_stats = \
            UnstructuredProfiler._clean_data_and_get_base_stats(
                profiler, data=data, sample_size=3)

        # note: bc the sample size is 3, only a subset of the data was sampled
        self.assertTrue(np.issubdtype(np.object_, df_series.dtype))
        self.assertDictEqual(
            {
                'sample': ['more data'],  # bc of subset sampled
                'sample_size': 3,
                'empty_line_count': 2,
                'memory_size': 25 / 1024 ** 2
            },
            base_stats)

        # case when min_true_samples set and subset of data
        df_series, base_stats = \
            UnstructuredProfiler._clean_data_and_get_base_stats(
                profiler, data=data, sample_size=3, min_true_samples=2)

        # note: bc the sample size is 3, only a subset of the data was sampled
        self.assertTrue(np.issubdtype(np.object_, df_series.dtype))
        self.assertDictEqual(
            {
                'sample': ['more data', 'here\n', 'a', ' is'],
                'sample_size': 6,
                'empty_line_count': 2,
                'memory_size': 25 / 1024 ** 2
            },
            base_stats)

    def test_update_profile(self, *mocks):
        # can properties update correctly for data
        data1 = pd.Series(['this', 'is my', '\n\r', 'test'])
        data2 = pd.Series(['here\n', '\t    ', ' ', ' is', '\n\r', 'more data'])

        # profiler with first dataset
        with test_utils.mock_timeit():
            profiler = UnstructuredProfiler(data1)
        self.assertEqual(4, profiler.total_samples)
        self.assertEqual(1, profiler._empty_line_count)
        self.assertEqual(15 / 1024 ** 2, profiler.memory_size)
        # note how sample doesn't include whitespace lines
        self.assertCountEqual(['this', 'is my', 'test'], profiler.sample)
        self.assertDictEqual({'clean_and_base_stats': 1}, profiler.times)

        # update with second dataset
        with test_utils.mock_timeit():
            profiler.update_profile(data2)
        self.assertEqual(10, profiler.total_samples)
        self.assertEqual(4, profiler._empty_line_count)
        self.assertEqual(40 / 1024 ** 2, profiler.memory_size)
        # note how sample doesn't include whitespace lines
        self.assertCountEqual(['here\n', ' is', 'more data'], profiler.sample)
        self.assertDictEqual({'clean_and_base_stats': 2}, profiler.times)

    @mock.patch('dataprofiler.profilers.profile_builder.UnstructuredProfiler.'
                '_update_profile_from_chunk')
    def test_min_true_samples(self, *mocks):
        empty_df = pd.DataFrame([])

        # Test invalid input
        msg = "`min_true_samples` must be an integer or `None`."
        with self.assertRaisesRegex(ValueError, msg):
            profile = dp.UnstructuredProfiler(empty_df, 
                                              min_true_samples="Bloop")

        # Test invalid input given to update_profile
        profile = dp.UnstructuredProfiler(empty_df)
        with self.assertRaisesRegex(ValueError, msg):
            profile.update_profile(empty_df, min_true_samples="Bloop")

        # Test None input (equivalent to zero)
        profile = dp.UnstructuredProfiler(empty_df, min_true_samples=None)
        self.assertEqual(None, profile._min_true_samples)

        # Test valid input
        profile = dp.UnstructuredProfiler(empty_df, min_true_samples=10)
        self.assertEqual(10, profile._min_true_samples)


class TestUnstructuredProfilerWData(unittest.TestCase):

    @classmethod
    def setUp(cls):
        test_utils.set_seed(seed=0)

    @classmethod
    def setUpClass(cls):
        test_utils.set_seed(0)
        cls.maxDiff = None
        cls.input_data = [
            'edited 9 hours ago',
            '6. Do not duplicate code.',
            '\t',
            'Just want to caution against following this too rigidly.',
            '\t',
            '   ',
            'When you try to DRY them up into a single generic abstraction, '
            'you have inadvertently coupled those two business rules together.',
            '   ',
            '   ',
            'Removing duplication that repeats the handling of the exact same '
            'business rule is also usually a win.',
            '',
            'Duplicate words: business, win, code',
            '\n\r',
            'Reply',
            'Share',
            'Report',
        ]
        cls.dataset = pd.DataFrame(cls.input_data)

        # turn off data labeler because if model changes, results also change
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})

        with test_utils.mock_timeit():
            cls.profiler = UnstructuredProfiler(
                cls.dataset, len(cls.dataset), options=profiler_options)
            cls.profiler2 = UnstructuredProfiler(
                pd.DataFrame(['extra', '\n', 'test\n', 'data .',
                              'For merging.']),
                options=profiler_options
            )
        cls.report = cls.profiler.report()

    def test_sample(self):
        self.maxDiff = None
        self.assertCountEqual(
            ['Report',
             'Reply',
             'Removing duplication that repeats the handling of the exact same '
             'business rule is also usually a win.',
             'edited 9 hours ago',
             'Just want to caution against following this too rigidly.'],
            self.profiler.sample
        )

    def test_total_samples(self):
        self.assertEqual(16, self.profiler.total_samples)

    def test_empty_line_count(self):
        self.assertEqual(7, self.profiler._empty_line_count)

    def test_get_memory_size(self):
        self.assertEqual(393 / 1024 ** 2, self.profiler.memory_size)

    def test_text_profiler_results(self):
        # vocab order doesn't matter
        expected_vocab = ['x', 'i', 'y', 's', '9', ',', 'u', 'b', 'f', 'Y', 'J',
                          'v', 'r', 'o', 'a', '6', 'n', 'h', ' ', 'g', 'R', 't',
                          'W', '.', 'm', 'c', 'l', 'e', 'p', 'w', 'S', 'd', 'D',
                          ':']
        self.assertCountEqual(
            expected_vocab,
            self.report['data_stats']['statistics'].pop('vocab'))

        # assuming if words are correct, rest of TextProfiler is merged properly
        # vocab order doesn't matter, case insensitive, remove stop words
        expected_word_count = {
            'edited': 1, '9': 1, 'hours': 1, 'ago': 1, '6': 1, 'Do': 1,
            'not': 1, 'duplicate': 1, 'code': 2, 'Just': 1, 'want': 1,
            'to': 2, 'caution': 1, 'against': 1, 'following': 1, 'this': 1,
            'too': 1, 'rigidly': 1, 'When': 1, 'you': 2, 'try': 1, 'DRY': 1,
            'them': 1, 'up': 1, 'into': 1, 'a': 2, 'single': 1, 'generic': 1,
            'abstraction': 1, 'have': 1, 'inadvertently': 1, 'coupled': 1,
            'those': 1, 'two': 1, 'business': 3, 'rules': 1, 'together': 1,
            'Removing': 1, 'duplication': 1, 'that': 1, 'repeats': 1, 'the': 2,
            'handling': 1, 'of': 1, 'exact': 1, 'same': 1, 'rule': 1, 'is': 1,
            'also': 1, 'usually': 1, 'win': 2, 'Duplicate': 1, 'words': 1,
            'Reply': 1, 'Share': 1, 'Report': 1}

        # adapt to the stop words (brittle test)
        stop_words = \
            self.profiler._profile._profiles['text']._stop_words
        for key in list(expected_word_count.keys()):
            if key.lower() in stop_words:
                expected_word_count.pop(key)

        expected_words = expected_word_count.keys()
        self.assertCountEqual(
            expected_words,
            self.report['data_stats']['statistics'].pop('words'))

        # test for vocab_count
        expected_vocab_count = {' ': 55, ',': 3, '.': 5, '6': 1, '9': 1,
                                ':': 1, 'D': 3, 'J': 1, 'R': 4, 'S': 1,
                                'W': 1, 'Y': 1, 'a': 22, 'b': 4, 'c': 10,
                                'd': 11, 'e': 33, 'f': 2, 'g': 9, 'h': 12,
                                'i': 24, 'l': 16, 'm': 3, 'n': 21, 'o': 27,
                                'p': 8, 'r': 13, 's': 23, 't': 31, 'u': 17,
                                'v': 3, 'w': 6, 'x': 1, 'y': 7}

        # expected after the popping: times, vocab, words
        expected_report = {
            'global_stats': {
                'samples_used': 16,
                'empty_line_count': 7,
                'memory_size': 393 / 1024 ** 2,
                'file_type': "<class 'pandas.core.frame.DataFrame'>",
                'encoding': None,
                'times': {'clean_and_base_stats': 1}
            },
            'data_stats': {
                'data_label': {},
                'statistics': {
                    'word_count': expected_word_count,
                    'vocab_count': expected_vocab_count,
                    'times': {'words': 1, 'vocab': 1},
                }
            }
        }
        self.assertDictEqual(expected_report, self.report)

    def test_add_profilers(self):
        merged_profiler = self.profiler + self.profiler2
        report = merged_profiler.report()

        self.assertEqual(21, merged_profiler.total_samples)
        self.assertEqual(8, merged_profiler._empty_line_count)
        self.assertEqual(422 / 1024 ** 2, merged_profiler.memory_size)
        self.assertCountEqual(
            ['test\n',
             'extra',
             'Reply',
             'edited 9 hours ago',
             'Removing duplication that repeats the handling of the exact same '
             'business rule is also usually a win.'],
            merged_profiler.sample
        )

        # assuming if words are correct, rest of TextProfiler is merged properly
        # vocab order doesn't matter, case insensitive, remove stop words
        expected_word_count = {
            'edited': 1, '9': 1, 'hours': 1, 'ago': 1, '6': 1, 'Do': 1,
            'not': 1, 'duplicate': 1, 'code': 2, 'Just': 1, 'want': 1,
            'to': 2, 'caution': 1, 'against': 1, 'following': 1, 'this': 1,
            'too': 1, 'rigidly': 1, 'When': 1, 'you': 2, 'try': 1, 'DRY': 1,
            'them': 1, 'up': 1, 'into': 1, 'a': 2, 'single': 1, 'generic': 1,
            'abstraction': 1, 'have': 1, 'inadvertently': 1, 'coupled': 1,
            'those': 1, 'two': 1, 'business': 3, 'rules': 1, 'together': 1,
            'Removing': 1, 'duplication': 1, 'that': 1, 'repeats': 1, 'the': 2,
            'handling': 1, 'of': 1, 'exact': 1, 'same': 1, 'rule': 1, 'is': 1,
            'also': 1, 'usually': 1, 'win': 2, 'Duplicate': 1, 'words': 1,
            'Reply': 1, 'Share': 1, 'Report': 1, 'extra': 1, 'test': 1,
            'data': 1, 'merging': 1}

        # adapt to the stop words (brittle test)
        stop_words = \
            merged_profiler._profile._profiles['text']._stop_words
        for key in list(expected_word_count.keys()):
            if key.lower() in stop_words:
                expected_word_count.pop(key)
        
        expected_words = expected_word_count.keys()
        self.assertCountEqual(
            expected_words,
            report['data_stats']['statistics']['words'])
        self.assertDictEqual(
            expected_word_count,
            report['data_stats']['statistics']['word_count'])

    def test_update_profile(self):
        # turn off data labeler because if model changes, results also change
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})

        # update profiler and get report
        update_profiler = UnstructuredProfiler(self.dataset,
                                               options=profiler_options)
        update_profiler.update_profile(pd.DataFrame(['extra', '\n', 'test\n',
                                                     'data .', 'For merging.']))
        report = update_profiler.report()

        # tests
        self.assertEqual(21, update_profiler.total_samples)
        self.assertEqual(8, update_profiler._empty_line_count)
        self.assertEqual(422 / 1024 ** 2, update_profiler.memory_size)

        # Note: different from merge because sample is from last update only
        self.assertCountEqual(
            ['test\n', 'extra', 'For merging.', 'data .'],
            update_profiler.sample
        )

        # assuming if words are correct, rest of TextProfiler is merged properly
        # vocab order doesn't matter, case insensitive, remove stop words
        expected_word_count = {
            'edited': 1, '9': 1, 'hours': 1, 'ago': 1, '6': 1, 'Do': 1,
            'not': 1, 'duplicate': 1, 'code': 2, 'Just': 1, 'want': 1,
            'to': 2, 'caution': 1, 'against': 1, 'following': 1, 'this': 1,
            'too': 1, 'rigidly': 1, 'When': 1, 'you': 2, 'try': 1, 'DRY': 1,
            'them': 1, 'up': 1, 'into': 1, 'a': 2, 'single': 1, 'generic': 1,
            'abstraction': 1, 'have': 1, 'inadvertently': 1, 'coupled': 1,
            'those': 1, 'two': 1, 'business': 3, 'rules': 1, 'together': 1,
            'Removing': 1, 'duplication': 1, 'that': 1, 'repeats': 1, 'the': 2,
            'handling': 1, 'of': 1, 'exact': 1, 'same': 1, 'rule': 1, 'is': 1,
            'also': 1, 'usually': 1, 'win': 2, 'Duplicate': 1, 'words': 1,
            'Reply': 1, 'Share': 1, 'Report': 1, 'extra': 1, 'test': 1,
            'data': 1, 'merging': 1}

        # adapt to the stop words (brittle test)
        stop_words = \
            update_profiler._profile._profiles['text']._stop_words
        for key in list(expected_word_count.keys()):
            if key.lower() in stop_words:
                expected_word_count.pop(key)

        expected_words = expected_word_count.keys()
        self.assertCountEqual(
            expected_words,
            report['data_stats']['statistics']['words'])
        self.assertDictEqual(
            expected_word_count,
            report['data_stats']['statistics']['word_count'])

    def test_save_and_load(self):
        data_folder = "dataprofiler/tests/data/"
        test_files = ["txt/code.txt", "txt/sentence-10x.txt"]

        for test_file in test_files:
            # Create Data and StructuredProfiler objects
            data = dp.Data(os.path.join(data_folder, test_file))
            save_profile = UnstructuredProfiler(data)

            # If profile _empty_line_count = 0, it won't test if the variable is
            # saved correctly since that is also the default value. Ensure
            # not the default
            save_profile._empty_line_count = 1

            # store the expected data_labeler
            data_labeler = save_profile.options.data_labeler.data_labeler_object

            # Save and Load profile with Mock IO
            with mock.patch('builtins.open') as m:
                mock_file = setup_save_mock_open(m)
                save_profile.save()

                # make sure data_labeler unchanged
                self.assertIs(
                    data_labeler,
                    save_profile.options.data_labeler.data_labeler_object)
                self.assertIs(
                    data_labeler,
                    save_profile._profile._profiles['data_labeler'].data_labeler)

                mock_file.seek(0)
                with mock.patch('dataprofiler.profilers.profile_builder.'
                                'DataLabeler', return_value=data_labeler):
                    load_profile = UnstructuredProfiler.load("mock.pkl")

            # validate loaded profile has same data labeler class
            self.assertIsInstance(
                load_profile.options.data_labeler.data_labeler_object,
                data_labeler.__class__)
            self.assertIsInstance(
                load_profile.profile._profiles['data_labeler'].data_labeler,
                data_labeler.__class__)

            # Check that reports are equivalent
            save_report = save_profile.report()
            load_report = load_profile.report()
            self.assertDictEqual(save_report, load_report)

            # Check that sample was properly saved and loaded
            save_sample = save_profile.sample
            load_sample = load_profile.sample
            self.assertEqual(save_sample, load_sample)

            # validate both are still usable after
            save_profile.update_profile(pd.DataFrame(['test', 'test2']))
            load_profile.update_profile(pd.DataFrame(['test', 'test2']))

    def test_save_and_load_no_labeler(self):

        # Create Data and UnstructuredProfiler objects
        data = 'this is my test data: 123-456-7890'

        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})

        save_profile = dp.UnstructuredProfiler(data, options=profile_options)

        # Save and Load profile with Mock IO
        with mock.patch('builtins.open') as m:
            mock_file = setup_save_mock_open(m)
            save_profile.save()

            mock_file.seek(0)
            with mock.patch('dataprofiler.profilers.profile_builder.'
                            'DataLabeler'):
                load_profile = dp.UnstructuredProfiler.load("mock.pkl")

        # Check that reports are equivalent
        save_report = save_profile.report()
        load_report = load_profile.report()
        self.assertDictEqual(save_report, load_report)

        # Check that sample was properly saved and loaded
        save_sample = save_profile.sample
        load_sample = load_profile.sample
        self.assertEqual(save_sample, load_sample)

        # validate both are still usable after
        save_profile.update_profile(pd.DataFrame(['test', 'test2']))
        load_profile.update_profile(pd.DataFrame(['test', 'test2']))

    def test_options_ingested_correctly(self):
        self.assertIsInstance(self.profiler.options, UnstructuredOptions)
        self.assertIsInstance(self.profiler2.options, UnstructuredOptions)
        self.assertFalse(self.profiler.options.data_labeler.is_enabled)
        self.assertFalse(self.profiler2.options.data_labeler.is_enabled)


class TestStructuredProfilerNullValues(unittest.TestCase):

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
        trained_schema = dp.StructuredProfiler(test_dataset, len(test_dataset),
                                               options=profiler_options)
        ts_profile = trained_schema.profile
        ts_mapping = trained_schema._col_name_to_idx

        self.assertCountEqual(['', 'nan', 'None', 'null'],
                              ts_profile[ts_mapping['1'][0]].null_types)
        self.assertEqual(5, ts_profile[ts_mapping['1'][0]].null_count)
        self.assertEqual({'': {4}, 'nan': {0}, 'None': {2, 3}, 'null': {1}},
                         ts_profile[ts_mapping['1'][0]].null_types_index)
        self.assertCountEqual(['', 'nan', 'None', 'null'],
                              ts_profile[ts_mapping[1][0]].null_types)
        self.assertEqual(5, ts_profile[ts_mapping[1][0]].null_count)
        self.assertEqual({'': {4}, 'nan': {0}, 'None': {1, 3}, 'null': {2}},
                         ts_profile[ts_mapping[1][0]].null_types_index)

    def test_correct_null_row_counts(self):
        file_path = os.path.join(test_root_path, 'data', 'csv/empty_rows.txt')
        data = pd.read_csv(file_path)
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})
        profile = dp.StructuredProfiler(data, options=profiler_options)
        self.assertEqual(2, profile.row_has_null_count)
        self.assertEqual(0.25, profile._get_row_has_null_ratio())
        self.assertEqual(2, profile.row_is_null_count)
        self.assertEqual(0.25, profile._get_row_is_null_ratio())

        file_path = os.path.join(test_root_path, 'data','csv/iris-with-null-rows.csv')
        data = pd.read_csv(file_path)
        profile = dp.StructuredProfiler(data, options=profiler_options)
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
        profile = dp.StructuredProfiler(data, options=profiler_options)

        report = profile.report(report_options={"output_format": "pretty"})
        count_idx = report["global_stats"]["profile_schema"]["COUNT"][0]
        numbers_idx = report["global_stats"]["profile_schema"][" NUMBERS"][0]

        self.assertEqual(
            report['data_stats'][count_idx]['statistics']['null_types_index'],
            {'': '[2, 3, 4, 5, 7, 8]'}
        )

        self.assertEqual(
            report['data_stats'][numbers_idx]['statistics']['null_types_index'],
            {'': '[5, 6, 8]', ' ': '[2, 4]'}
        )

    def test_correct_total_sample_size_and_counts_and_mutability(self):
        data = [['test1', 1.0],
                ['test2', 2.0],
                ['test3', 3.0],
                [None, None],
                ['test5', 5.0],
                ['test6', 6.0],
                [None, None],
                ['test7', 7.0]]
        data = pd.DataFrame(data, columns=['NAME', 'VALUE'])
        profiler_options = ProfilerOptions()
        profiler_options.set({'data_labeler.is_enabled': False})

        col_one_len = len(data['NAME'])
        col_two_len = len(data['VALUE'])

        # Test reloading data, ensuring immutable
        for i in range(2):

            # Profile Once
            data.index = pd.RangeIndex(0, 8)
            profile = dp.StructuredProfiler(data, options=profiler_options,
                                            samples_per_update=2)

            # Profile Twice
            data.index = pd.RangeIndex(8, 16)
            profile.update_profile(data)

            # rows sampled are [5, 6], [13, 14] (0 index)
            self.assertEqual(16, profile.total_samples)
            self.assertEqual(4, profile._max_col_samples_used)
            self.assertEqual(2, profile.row_has_null_count)
            self.assertEqual(0.5, profile._get_row_has_null_ratio())
            self.assertEqual(2, profile.row_is_null_count)
            self.assertEqual(0.5, profile._get_row_is_null_ratio())
            self.assertEqual(0.4375, profile._get_unique_row_ratio())
            self.assertEqual(9, profile._get_duplicate_row_count())

        self.assertEqual(col_one_len, len(data['NAME']))
        self.assertEqual(col_two_len, len(data['VALUE']))

    def test_null_calculation_with_differently_sampled_cols(self):
        opts = ProfilerOptions()
        opts.structured_options.multiprocess.is_enabled = False
        data = pd.DataFrame({"full": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                             "sparse": [1, None, 3, None, 5, None, 7, None, 9]})
        profile = dp.StructuredProfiler(data, samples_per_update=5, min_true_samples=5,
                                        options=opts)
        # Rows 2, 4, 5, 6, 7 are sampled in first column
        # Therefore only those rows should be considered for null calculations
        # The only null in those rows in second column in that subset are 5, 7
        # Therefore only 2 rows have null according to row_has_null_count
        self.assertEqual(0, profile.row_is_null_count)
        self.assertEqual(2, profile.row_has_null_count)
        # Accordingly, make sure ratio of null rows accounts for the fact that
        # Only 5 total rows were sampled (5 in col 1, 9 in col 2)
        self.assertEqual(0, profile._get_row_is_null_ratio())
        self.assertEqual(0.4, profile._get_row_has_null_ratio())

        data2 = pd.DataFrame(
            {"sparse": [1, None, 3, None, 5, None, 7, None],
             "sparser": [1, None, None, None, None, None, None, 8]})
        profile2 = dp.StructuredProfiler(data2, samples_per_update=2, min_true_samples=2,
                                         options=opts)
        # Rows are sampled as follows: [6, 5], [1, 4], [2, 3], [0, 7]
        # First column gets min true samples from ids 1, 4, 5, 6
        # Second column gets completely sampled (has a null in 1, 4, 5, 6)
        # rows 1 and 5 are completely null, 4 and 6 only null in col 2
        self.assertEqual(2, profile2.row_is_null_count)
        self.assertEqual(4, profile2.row_has_null_count)
        # Only 4 total rows sampled, ratio accordingly
        self.assertEqual(0.5, profile2._get_row_is_null_ratio())
        self.assertEqual(1, profile2._get_row_has_null_ratio())

    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnPrimitiveTypeProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnStatsProfileCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'ColumnDataLabelerCompiler')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    @mock.patch('dataprofiler.profilers.profile_builder.'
                'StructuredProfiler._update_correlation')
    def test_null_row_stats_correct_after_updates(self, *mocks):
        data1 = pd.DataFrame([[1, None],
                             [1, 1],
                             [None, None],
                             [None, 1]])
        data2 = pd.DataFrame([[None, None],
                             [1, None],
                             [None, None],
                             [None, 1]])
        opts = ProfilerOptions()
        opts.structured_options.multiprocess.is_enabled = False

        # When setting min true samples/samples per update
        profile = dp.StructuredProfiler(data1, min_true_samples=2,
                                        samples_per_update=2, options=opts)
        self.assertEqual(3, profile.row_has_null_count)
        self.assertEqual(1, profile.row_is_null_count)
        self.assertEqual(0.75, profile._get_row_has_null_ratio())
        self.assertEqual(0.25, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3}, profile._profile[0].null_types_index['nan'])
        self.assertSetEqual({0, 2}, profile._profile[1].null_types_index['nan'])

        profile.update_profile(data2, min_true_samples=2, sample_size=2)
        self.assertEqual(7, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(0.875, profile._get_row_has_null_ratio())
        self.assertEqual(0.375, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3, 4, 6, 7},
                            profile._profile[0].null_types_index['nan'])
        self.assertSetEqual({0, 2, 4, 5, 6},
                            profile._profile[1].null_types_index['nan'])

        # When not setting min true samples/samples per update
        opts = ProfilerOptions()
        opts.structured_options.multiprocess.is_enabled = False
        profile = dp.StructuredProfiler(data1, options=opts)
        self.assertEqual(3, profile.row_has_null_count)
        self.assertEqual(1, profile.row_is_null_count)
        self.assertEqual(0.75, profile._get_row_has_null_ratio())
        self.assertEqual(0.25, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3}, profile._profile[0].null_types_index['nan'])
        self.assertSetEqual({0, 2}, profile._profile[1].null_types_index['nan'])

        profile.update_profile(data2)
        self.assertEqual(7, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(0.875, profile._get_row_has_null_ratio())
        self.assertEqual(0.375, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3, 4, 6, 7},
                            profile._profile[0].null_types_index['nan'])
        self.assertSetEqual({0, 2, 4, 5, 6},
                            profile._profile[1].null_types_index['nan'])

        # Test that update with emtpy data doesn't change stats
        profile.update_profile(pd.DataFrame([]))
        self.assertEqual(7, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(0.875, profile._get_row_has_null_ratio())
        self.assertEqual(0.375, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3, 4, 6, 7},
                            profile._profile[0].null_types_index['nan'])
        self.assertSetEqual({0, 2, 4, 5, 6},
                            profile._profile[1].null_types_index['nan'])

        # Test one row update
        profile.update_profile(pd.DataFrame([[1, None]]))
        self.assertEqual(8, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(8/9, profile._get_row_has_null_ratio())
        self.assertEqual(3/9, profile._get_row_is_null_ratio())
        self.assertEqual(1, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3, 4, 6, 7},
                            profile._profile[0].null_types_index['nan'])
        self.assertSetEqual({0, 2, 4, 5, 6},
                            profile._profile[1].null_types_index['nan'])
        # Weird pandas behavior makes this None since this column will be
        # recognized as object, not float64
        self.assertSetEqual({8}, profile._profile[1].null_types_index['None'])


class TestProfilerFactoryClass(unittest.TestCase):

    def test_profiler_factory_class_bad_input(self):
        with self.assertRaisesRegex(ValueError, "Must specify 'profiler_type' "
                                                "to be 'structured' or "
                                                "'unstructured'."):
            Profiler(pd.DataFrame([]), profiler_type="whoops")

        with self.assertRaisesRegex(ValueError, "Data must either be imported "
                                                "using the data_readers, "
                                                "pd.Series, or pd.DataFrame."):
            Profiler({'test': 1})

    @mock.patch('dataprofiler.profilers.profile_builder.StructuredProfiler',
                spec=StructuredProfiler)
    @mock.patch('dataprofiler.profilers.profile_builder.UnstructuredProfiler',
                spec=UnstructuredProfiler)
    def test_profiler_factory_class_creates_correct_profiler(self, *mocks):
        """
        Ensure Profiler factory class either respects user input or makes
        reasonable inference in the absence of user specificity.
        """
        # User specifies via profiler_type
        data_df = pd.DataFrame(['test'])
        self.assertIsInstance(Profiler(data_df, profiler_type="structured"),
                              StructuredProfiler)
        self.assertIsInstance(Profiler(data_df, profiler_type="unstructured"),
                              UnstructuredProfiler)

        # User gives data that has .is_structured == True
        data_csv_df = dp.Data(data=data_df, data_type="csv")
        self.assertIsInstance(Profiler(data_csv_df), StructuredProfiler)

        # User gives data that has .is_structured == False
        data_csv_rec = dp.Data(data=data_df, data_type="csv",
                               options={"data_format": "records"})
        self.assertIsInstance(Profiler(data_csv_rec), UnstructuredProfiler)

        # user gives structured: list, pd.Series, pd.DataFrame
        data_series = pd.Series(['test'])
        data_list = ['test']
        self.assertIsInstance(Profiler(data_list), StructuredProfiler)
        self.assertIsInstance(Profiler(data_series), StructuredProfiler)
        self.assertIsInstance(Profiler(data_df), StructuredProfiler)

        # user gives unstructured: str
        data_str = 'test'
        self.assertIsInstance(Profiler(data_str), UnstructuredProfiler)

    def test_save_and_load_structured(self):
        datapth = "dataprofiler/tests/data/"
        test_files = ["csv/guns.csv", "csv/iris.csv"]

        for test_file in test_files:
            # Create Data and StructuredProfiler objects
            data = dp.Data(os.path.join(datapth, test_file))
            options = ProfilerOptions()
            options.set({"correlation.is_enabled": True})
            save_profile = dp.StructuredProfiler(data)

            # store the expected data_labeler
            data_labeler = save_profile.options.data_labeler.data_labeler_object

            # Save and Load profile with Mock IO
            with mock.patch('builtins.open') as m:
                mock_file = setup_save_mock_open(m)
                save_profile.save()
                mock_file.seek(0)
                with mock.patch('dataprofiler.profilers.profile_builder.'
                                'DataLabeler', return_value=data_labeler):
                    load_profile = dp.Profiler.load("mock.pkl")

            # validate loaded profile has same data labeler class
            self.assertIsInstance(
                load_profile.options.data_labeler.data_labeler_object,
                data_labeler.__class__)

            # only checks first columns
            # get first column
            first_column_profile = load_profile.profile[0]
            self.assertIsInstance(
                first_column_profile.profiles['data_label_profile']
                    ._profiles['data_labeler'].data_labeler,
                data_labeler.__class__)

            # Check that reports are equivalent
            save_report = test_utils.clean_report(save_profile.report())
            load_report = test_utils.clean_report(load_profile.report())
            np.testing.assert_equal(save_report, load_report)

            # validate both are still usable after
            save_profile.update_profile(data.data.iloc[:2])
            load_profile.update_profile(data.data.iloc[:2])

    def test_save_and_load_unstructured(self):
        data_folder = "dataprofiler/tests/data/"
        test_files = ["txt/code.txt", "txt/sentence-10x.txt"]

        for test_file in test_files:
            # Create Data and StructuredProfiler objects
            data = dp.Data(os.path.join(data_folder, test_file))
            save_profile = UnstructuredProfiler(data)

            # If profile _empty_line_count = 0, it won't test if the variable is
            # saved correctly since that is also the default value. Ensure
            # not the default
            save_profile._empty_line_count = 1

            # store the expected data_labeler
            data_labeler = save_profile.options.data_labeler.data_labeler_object

            # Save and Load profile with Mock IO
            with mock.patch('builtins.open') as m:
                mock_file = setup_save_mock_open(m)
                save_profile.save()

                # make sure data_labeler unchanged
                self.assertIs(
                    data_labeler,
                    save_profile.options.data_labeler.data_labeler_object)
                self.assertIs(
                    data_labeler,
                    save_profile._profile._profiles[
                        'data_labeler'].data_labeler)

                mock_file.seek(0)
                with mock.patch('dataprofiler.profilers.profile_builder.'
                                'DataLabeler', return_value=data_labeler):
                    load_profile = Profiler.load("mock.pkl")

            # validate loaded profile has same data labeler class
            self.assertIsInstance(
                load_profile.options.data_labeler.data_labeler_object,
                data_labeler.__class__)
            self.assertIsInstance(
                load_profile.profile._profiles['data_labeler'].data_labeler,
                data_labeler.__class__)

            # Check that reports are equivalent
            save_report = save_profile.report()
            load_report = load_profile.report()
            self.assertDictEqual(save_report, load_report)

            # validate both are still usable after
            save_profile.update_profile(pd.DataFrame(['test', 'test2']))
            load_profile.update_profile(pd.DataFrame(['test', 'test2']))

    @mock.patch('dataprofiler.profilers.profile_builder.StructuredProfiler.'
                '_update_profile_from_chunk')
    @mock.patch('dataprofiler.profilers.profile_builder.DataLabeler')
    def test_min_true_samples(self, *mocks):
        empty_df = pd.DataFrame([])

        # Test invalid input
        msg = "`min_true_samples` must be an integer or `None`."
        with self.assertRaisesRegex(ValueError, msg):
            profile = dp.Profiler(empty_df, min_true_samples="Bloop")

        # Test invalid input given to update_profile
        profile = dp.Profiler(empty_df)
        with self.assertRaisesRegex(ValueError, msg):
            profile.update_profile(empty_df, min_true_samples="Bloop")

        # Test None input (equivalent to zero)
        profile = dp.Profiler(empty_df, min_true_samples=None)
        self.assertEqual(None, profile._min_true_samples)

        # Test valid input
        profile = dp.Profiler(empty_df, min_true_samples=10)
        self.assertEqual(10, profile._min_true_samples)

if __name__ == '__main__':
    unittest.main()
