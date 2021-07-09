from __future__ import print_function

import six
import unittest
from unittest import mock
import pandas as pd

from dataprofiler.profilers import column_profile_compilers as \
    col_pro_compilers
from dataprofiler.profilers.profiler_options import BaseOption, StructuredOptions


class TestBaseProfileCompilerClass(unittest.TestCase):

    def test_cannot_instantiate(self):
        """showing we normally can't instantiate an abstract class"""
        with self.assertRaises(TypeError) as e:
            col_pro_compilers.BaseCompiler()
        self.assertEqual(
            "Can't instantiate abstract class BaseCompiler with "
            "abstract methods profile",
            str(e.exception)
        )

    @mock.patch.multiple(
        col_pro_compilers.BaseCompiler, __abstractmethods__=set(),
        _profilers=[mock.Mock()], _option_class=mock.Mock(spec=BaseOption))
    @mock.patch.multiple(
        col_pro_compilers.ColumnStatsProfileCompiler, _profilers=[mock.Mock()])
    def test_add_profilers(self):
        compiler1 = col_pro_compilers.BaseCompiler(mock.Mock())
        compiler2 = col_pro_compilers.BaseCompiler(mock.Mock())

        # test incorrect type
        with self.assertRaisesRegex(TypeError,
                                    '`BaseCompiler` and `int` are '
                                    'not of the same profile compiler type.'):
            compiler1 + 3

        compiler3 = col_pro_compilers.ColumnStatsProfileCompiler(mock.Mock())
        compiler3._profiles = [mock.Mock()]
        with self.assertRaisesRegex(TypeError,
                                    '`BaseCompiler` and '
                                    '`ColumnStatsProfileCompiler` are '
                                    'not of the same profile compiler type.'):
            compiler1 + compiler3

        # test mismatched names
        compiler1.name = 'compiler1'
        compiler2.name = 'compiler2'
        with self.assertRaisesRegex(ValueError,
                                    'Column profile names are unmatched: '
                                    'compiler1 != compiler2'):
            compiler1 + compiler2

        # test mismatched profiles due to options
        compiler2.name = 'compiler1'
        compiler1._profiles = dict(test1=mock.Mock())
        compiler2._profiles = dict(test2=mock.Mock())
        with self.assertRaisesRegex(ValueError,
                                    'Column profilers were not setup with the '
                                    'same options, hence they do not calculate '
                                    'the same profiles and cannot be added '
                                    'together.'):
            compiler1 + compiler2

        # test success
        compiler1._profiles = dict(test=1)
        compiler2._profiles = dict(test=2)
        merged_compiler = compiler1 + compiler2
        self.assertEqual(3, merged_compiler._profiles['test'])
        self.assertEqual('compiler1', merged_compiler.name)

    
    def test_diff_primitive_compilers(self):
        # Test different data types
        data1 = pd.Series(['-2', '-1', '1', '2'])
        data2 = pd.Series(["YO YO YO", "HELLO"])
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)

        expected_diff = {
            'data_type_representation': {
                'datetime': 'unchanged',
                'int': 1.0,
                'float': 1.0,
                'text': 'unchanged'
            },
            'data_type': ['int', 'text']
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))
        
        
        # Test different data types with datetime specifically
        data1 = pd.Series(['-2', '-1', '1', '2'])
        data2 = pd.Series(["01/12/1967", "11/9/2024"])
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)

        expected_diff = {
            'data_type_representation': {
                'datetime': -1.0,
                'int': 1.0,
                'float': 1.0,
                'text': 'unchanged'
            },
            'data_type': ['int', 'datetime']
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        
        # Test same data types
        data1 = pd.Series(['-2', '15', '1', '2'])
        data2 = pd.Series(['5', '-1'])

        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)
        expected_diff = {
            'data_type_representation': {
                'datetime': 'unchanged',
                'int': 'unchanged',
                'float': 'unchanged',
                'text': 'unchanged'
            },
             'data_type': 'unchanged',
             'statistics': {
                     'min': -1.0,
                     'max': 10.0,
                     'sum': 12.0,
                     'mean': 2.0,
                     'variance': 38.666666666666664,
                     'stddev': 3.285085839971525
                 }
        }

        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))
        
        # Test different compilers
        data1 = pd.Series(['-2', '-1', '1', '2'])
        data2 = pd.Series(['5', '15'])

        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1)
        compiler2 = col_pro_compilers.ColumnStatsProfileCompiler(data2)
        # Assert type error is properly called
        with self.assertRaises(TypeError) as exc:
            compiler1.diff(compiler2)
        self.assertEqual(str(exc.exception),
                         "`ColumnPrimitiveTypeProfileCompiler` and "
                         "`ColumnStatsProfileCompiler` are not of the same "
                         "profile compiler type.")
        
    def test_disabling_columns_during_primitive_diff(self):
        
        data1 = pd.Series(['-2', '-1', '1', '2'])
        data2 = pd.Series(['5', '15'])
        options = StructuredOptions()

        # Test disabled column in one compiler
        options.int.is_enabled = False
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1,
                                                                         options)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)
        expected_diff = {
            'data_type_representation': {
                'datetime': 'unchanged', 
                'float': 'unchanged', 
                'text': 'unchanged', 
                'int': [None, 1.0]}, 
            'data_type': ['float', 'int']
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))
        
        # Test disabled column in both compilers
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2,
                                                                         options)
        expected_diff = {
            'data_type_representation': {
                'datetime': 'unchanged', 
                'float': 'unchanged', 
                'text': 'unchanged'
            },
            'data_type': "unchanged",
            'statistics': {
                'min': -7.0, 
                'max': -13.0, 
                'sum': -20.0,
                'mean': -10.0, 
                'variance': -46.666666666666664,
                'stddev': data1.astype(int).std() - data2.astype(int).std(),
                'precision': {
                    'min': 'unchanged', 
                    'max': -1, 
                    'mean': -0.5,
                    'var': -0.5, 
                    'std': -0.71, 
                    'sample_size': 2,
                    'margin_of_error': -1.6}
            }
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))
        
        # Test disabling all columns in one compiler
        options.float.is_enabled = False
        options.text.is_enabled = False
        options.datetime.is_enabled = False
        compiler1 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data1,
                                                                         options)
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2)
        expected_diff = {
            'data_type_representation': {
                'datetime': [None, 0.0], 
                'int': [None, 1.0], 
                'float': [None, 1.0], 
                'text': [None, 1.0]
            }, 
            'data_type': [None, 'int']
        }
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))

        # Test disabling all columns in all compilers
        compiler2 = col_pro_compilers.ColumnPrimitiveTypeProfileCompiler(data2,
                                                                         options)
        expected_diff = {}
        self.assertDictEqual(expected_diff, compiler1.diff(compiler2))


    @mock.patch.multiple(
        col_pro_compilers.BaseCompiler, __abstractmethods__=set())
    def test_no_profilers_error(self):
        with self.assertRaises(NotImplementedError) as e:
            col_pro_compilers.BaseCompiler()
        self.assertEqual("Must add profilers.", str(e.exception))

    @mock.patch.multiple(
        col_pro_compilers.BaseCompiler, __abstractmethods__=set(),
        _profilers='mock')
    def test_no_options_error(self):
        with self.assertRaisesRegex(NotImplementedError,
                                    "Must set the expected OptionClass."):
            col_pro_compilers.BaseCompiler()

    def test_update_match_are_abstract(self):
        six.assertCountEqual(
            self,
            {'profile'},
            col_pro_compilers.BaseCompiler.__abstractmethods__
        )


class TestUnstructuredCompiler(unittest.TestCase):

    @mock.patch('dataprofiler.profilers.unstructured_labeler_profile.'
                'DataLabeler')
    @mock.patch('dataprofiler.profilers.unstructured_labeler_profile.'
                'CharPostprocessor')
    def test_base(self, *mocks):
        import pandas as pd
        from collections import defaultdict
        df_series = pd.Series(['test', 'hi my name is John Doe. 123-432-1234'])

        time_array = [float(i) for i in range(100, 0, -1)]
        with mock.patch('time.time', side_effect=lambda: time_array.pop()):
            compiler = col_pro_compilers.UnstructuredCompiler(df_series)

        expected_dict = {
            'data_label': {
                'entity_counts': {
                    'postprocess_char_level': defaultdict(int),
                    'true_char_level': defaultdict(int),
                    'word_level': defaultdict(int)},
                'entity_percentages': {
                    'postprocess_char_level': defaultdict(int),
                    'true_char_level': defaultdict(int),
                    'word_level': defaultdict(int)},
                'times': {'data_labeler_predict': 1.0}},
            'statistics': {
                'times': {'vocab': 1.0, 'words': 1.0},
                'vocab_count': {' ': 6, '-': 2, '.': 1, '1': 2, '2': 3,
                                '3': 3, '4': 2, 'D': 1, 'J': 1, 'a': 1,
                                'e': 3, 'h': 2, 'i': 2, 'm': 2, 'n': 2,
                                'o': 2, 's': 2, 't': 2, 'y': 1},
                'vocab': [' ', '-', '.', '1', '2', '3', '4', 'D', 'J', 'a', 'e',
                          'h', 'i', 'm', 'n', 'o', 's', 't', 'y'],
                'word_count': {'123': 1, '1234': 1, '432': 1, 'Doe': 1,
                               'John': 1, 'hi': 1, 'name': 1, 'test': 1},
                'words': ['test', 'hi', 'name', 'John', 'Doe', '123', '432',
                          '1234']}}

        output_profile = compiler.profile

        # because vocab uses a set, it will be random order every time, hence
        # we need to sort to check exact match between profiles
        if ('statistics' in output_profile
                and 'vocab' in output_profile['statistics']):
            output_profile['statistics']['vocab'] = \
                sorted(output_profile['statistics']['vocab'])

        self.maxDiff = None
        self.assertDictEqual(expected_dict, output_profile)


if __name__ == '__main__':
    unittest.main()
