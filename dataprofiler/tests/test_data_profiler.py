from __future__ import print_function

import os
import unittest
from unittest import mock

import dataprofiler

from . import test_utils

from dataprofiler import Data, Profiler, ProfilerOptions


# This is taken from: https://github.com/rlworkgroup/dowel/pull/36/files
# undo when cpython#4800 is merged.
unittest.case._AssertWarnsContext.__enter__ = test_utils.patched_assert_warns


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestDataProfiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(MODULE_PATH, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.parquet'),
                 type='csv'),
        ]

    def test_set_seed(self):
        import dataprofiler as dp
        self.assertEqual(dp.settings._seed, None)

        dp.set_seed(5)
        self.assertEqual(dp.settings._seed, 5)

        with self.assertRaisesRegex(ValueError, "Seed should be a non-negative integer."):
            dp.set_seed(-5)

        with self.assertRaisesRegex(ValueError, "Seed should be a non-negative integer."):
            dp.set_seed(5.2)

    def test_data_import(self):
        for file in self.input_file_names:
            data = Data(file['path'])
            self.assertEqual(data.data_type, file['type'])

    def test_data_profiling(self):
        for file in self.input_file_names:
            data = Data(file['path'])
            profile = Profiler(data)
            self.assertIsNotNone(profile.profile)
            self.assertIsNotNone(profile.report())
            
    def test_data_profiling_diff(self):
        file = self.input_file_names[0]
        data = Data(file['path'])

        options = ProfilerOptions()
        profile = Profiler(data.data[:50], options=options)
        profile2 = Profiler(data.data[50:], options=options)

        expected_diff = \
            {'global_stats': {'samples_used': -50, 'column_count': 'unchanged',
                              'row_count': -50,
                              'row_has_null_ratio': 'unchanged',
                              'row_is_null_ratio': 'unchanged',
                              'unique_row_ratio': 'unchanged',
                              'duplicate_row_count': 'unchanged',
                              'file_type': 'unchanged', 'encoding': 'unchanged',
                              'profile_schema': [{}, {'Id': 'unchanged',
                                                      'SepalLengthCm': 'unchanged',
                                                      'SepalWidthCm': 'unchanged',
                                                      'PetalLengthCm': 'unchanged',
                                                      'PetalWidthCm': 'unchanged',
                                                      'Species': 'unchanged'},
                                                 {}]}, 'data_stats': [
                {'column_name': 'Id', 'data_type': 'unchanged',
                 'data_label': 'unchanged', 'categorical': 'unchanged',
                 'order': 'unchanged',
                 'statistics': {'min': -50.0, 'max': -100.0, 'sum': -8775.0,
                                'mean': -75.0, 'variance': -629.1666666666667,
                                'stddev': -14.434112238768769,
                                'unique_count': -50,
                                'unique_ratio': 'unchanged',
                                'avg_predictions': {'UNKNOWN': 'unchanged',
                                                    'ADDRESS': 'unchanged',
                                                    'BAN': 'unchanged',
                                                    'CREDIT_CARD': 'unchanged',
                                                    'DATE': 'unchanged',
                                                    'TIME': 'unchanged',
                                                    'DATETIME': 'unchanged',
                                                    'DRIVERS_LICENSE': 'unchanged',
                                                    'EMAIL_ADDRESS': 'unchanged',
                                                    'UUID': 'unchanged',
                                                    'HASH_OR_KEY': 'unchanged',
                                                    'IPV4': 'unchanged',
                                                    'IPV6': 'unchanged',
                                                    'MAC_ADDRESS': 'unchanged',
                                                    'PERSON': 'unchanged',
                                                    'PHONE_NUMBER': 'unchanged',
                                                    'SSN': -0.03,
                                                    'URL': 'unchanged',
                                                    'US_STATE': 'unchanged',
                                                    'INTEGER': 0.0249999999999998,
                                                    'FLOAT': 'unchanged',
                                                    'QUANTITY': 'unchanged',
                                                    'ORDINAL': 0.005},
                                'label_representation': {'UNKNOWN': 'unchanged',
                                                         'ADDRESS': 'unchanged',
                                                         'BAN': 'unchanged',
                                                         'CREDIT_CARD': 'unchanged',
                                                         'DATE': 'unchanged',
                                                         'TIME': 'unchanged',
                                                         'DATETIME': 'unchanged',
                                                         'DRIVERS_LICENSE': 'unchanged',
                                                         'EMAIL_ADDRESS': 'unchanged',
                                                         'UUID': 'unchanged',
                                                         'HASH_OR_KEY': 'unchanged',
                                                         'IPV4': 'unchanged',
                                                         'IPV6': 'unchanged',
                                                         'MAC_ADDRESS': 'unchanged',
                                                         'PERSON': 'unchanged',
                                                         'PHONE_NUMBER': 'unchanged',
                                                         'SSN': -0.02,
                                                         'URL': 'unchanged',
                                                         'US_STATE': 'unchanged',
                                                         'INTEGER': 0.010000000000000009,
                                                         'FLOAT': 'unchanged',
                                                         'QUANTITY': 'unchanged',
                                                         'ORDINAL': 0.01},
                                'sample_size': -50, 'null_count': 'unchanged',
                                'null_types_index': 'unchanged',
                                'data_type_representation': {
                                    'float': 'unchanged', 'int': 'unchanged',
                                    'datetime': 'unchanged',
                                    'text': 'unchanged'}}},
                {'column_name': 'SepalLengthCm', 'data_type': 'unchanged',
                 'data_label': 'unchanged', 'categorical': 'unchanged',
                 'order': 'unchanged',
                 'statistics': {'min': -0.6000000000000005,
                                'max': -2.1000000000000005,
                                'sum': -375.90000000000003,
                                'mean': -1.2560000000000002,
                                'variance': -0.3151005153576583,
                                'stddev': -0.3103447528615157,
                                'precision': {'min': 'unchanged',
                                              'max': 'unchanged',
                                              'mean': -0.09999999999999987,
                                              'var': 0.05700000000000001,
                                              'std': 0.08000000000000002,
                                              'sample_size': -50,
                                              'margin_of_error': 0.07500000000000001},
                                'unique_count': -13,
                                'unique_ratio': 0.019999999999999962,
                                'avg_predictions': {'UNKNOWN': 'unchanged',
                                                    'ADDRESS': 'unchanged',
                                                    'BAN': 'unchanged',
                                                    'CREDIT_CARD': 'unchanged',
                                                    'DATE': 'unchanged',
                                                    'TIME': 'unchanged',
                                                    'DATETIME': 'unchanged',
                                                    'DRIVERS_LICENSE': 'unchanged',
                                                    'EMAIL_ADDRESS': 'unchanged',
                                                    'UUID': 'unchanged',
                                                    'HASH_OR_KEY': 'unchanged',
                                                    'IPV4': 'unchanged',
                                                    'IPV6': 'unchanged',
                                                    'MAC_ADDRESS': 'unchanged',
                                                    'PERSON': 'unchanged',
                                                    'PHONE_NUMBER': 'unchanged',
                                                    'SSN': 'unchanged',
                                                    'URL': 'unchanged',
                                                    'US_STATE': 'unchanged',
                                                    'INTEGER': 'unchanged',
                                                    'FLOAT': 0.006666666666666668,
                                                    'QUANTITY': 'unchanged',
                                                    'ORDINAL': -0.00666666666666671},
                                'label_representation': {'UNKNOWN': 'unchanged',
                                                         'ADDRESS': 'unchanged',
                                                         'BAN': 'unchanged',
                                                         'CREDIT_CARD': 'unchanged',
                                                         'DATE': 'unchanged',
                                                         'TIME': 'unchanged',
                                                         'DATETIME': 'unchanged',
                                                         'DRIVERS_LICENSE': 'unchanged',
                                                         'EMAIL_ADDRESS': 'unchanged',
                                                         'UUID': 'unchanged',
                                                         'HASH_OR_KEY': 'unchanged',
                                                         'IPV4': 'unchanged',
                                                         'IPV6': 'unchanged',
                                                         'MAC_ADDRESS': 'unchanged',
                                                         'PERSON': 'unchanged',
                                                         'PHONE_NUMBER': 'unchanged',
                                                         'SSN': 'unchanged',
                                                         'URL': 'unchanged',
                                                         'US_STATE': 'unchanged',
                                                         'INTEGER': 'unchanged',
                                                         'FLOAT': 0.01,
                                                         'QUANTITY': 'unchanged',
                                                         'ORDINAL': -0.010000000000000009},
                                'sample_size': -50, 'null_count': 'unchanged',
                                'null_types_index': 'unchanged',
                                'data_type_representation': {
                                    'float': 'unchanged', 'int': 0.07,
                                    'datetime': 'unchanged',
                                    'text': 'unchanged'}}},
                {'column_name': 'SepalWidthCm', 'data_type': 'unchanged',
                 'data_label': 'unchanged', 'categorical': [False, True],
                 'order': 'unchanged', 'statistics': {'min': 0.2999999999999998,
                                                      'max': 0.6000000000000005,
                                                      'sum': -116.29999999999993,
                                                      'mean': 0.5460000000000007,
                                                      'variance': 0.03445635951350237,
                                                      'stddev': 0.04827339145999626,
                                                      'precision': {
                                                          'min': 'unchanged',
                                                          'max': 'unchanged',
                                                          'mean': 0.09999999999999987,
                                                          'var': -0.05000000000000002,
                                                          'std': -0.06,
                                                          'sample_size': -50,
                                                          'margin_of_error': 0.03},
                                                      'unique_count': 'unchanged',
                                                      'unique_ratio': 0.16,
                                                      'avg_predictions': {
                                                          'UNKNOWN': 'unchanged',
                                                          'ADDRESS': 'unchanged',
                                                          'BAN': 'unchanged',
                                                          'CREDIT_CARD': 'unchanged',
                                                          'DATE': 'unchanged',
                                                          'TIME': 'unchanged',
                                                          'DATETIME': 'unchanged',
                                                          'DRIVERS_LICENSE': 'unchanged',
                                                          'EMAIL_ADDRESS': 'unchanged',
                                                          'UUID': 'unchanged',
                                                          'HASH_OR_KEY': 'unchanged',
                                                          'IPV4': 'unchanged',
                                                          'IPV6': 'unchanged',
                                                          'MAC_ADDRESS': 'unchanged',
                                                          'PERSON': 'unchanged',
                                                          'PHONE_NUMBER': 'unchanged',
                                                          'SSN': 'unchanged',
                                                          'URL': 'unchanged',
                                                          'US_STATE': 'unchanged',
                                                          'INTEGER': 'unchanged',
                                                          'FLOAT': 0.010000000000000002,
                                                          'QUANTITY': 'unchanged',
                                                          'ORDINAL': -0.010000000000000009},
                                                      'label_representation': {
                                                          'UNKNOWN': 'unchanged',
                                                          'ADDRESS': 'unchanged',
                                                          'BAN': 'unchanged',
                                                          'CREDIT_CARD': 'unchanged',
                                                          'DATE': 'unchanged',
                                                          'TIME': 'unchanged',
                                                          'DATETIME': 'unchanged',
                                                          'DRIVERS_LICENSE': 'unchanged',
                                                          'EMAIL_ADDRESS': 'unchanged',
                                                          'UUID': 'unchanged',
                                                          'HASH_OR_KEY': 'unchanged',
                                                          'IPV4': 'unchanged',
                                                          'IPV6': 'unchanged',
                                                          'MAC_ADDRESS': 'unchanged',
                                                          'PERSON': 'unchanged',
                                                          'PHONE_NUMBER': 'unchanged',
                                                          'SSN': 'unchanged',
                                                          'URL': 'unchanged',
                                                          'US_STATE': 'unchanged',
                                                          'INTEGER': 'unchanged',
                                                          'FLOAT': 'unchanged',
                                                          'QUANTITY': 'unchanged',
                                                          'ORDINAL': 'unchanged'},
                                                      'sample_size': -50,
                                                      'null_count': 'unchanged',
                                                      'null_types_index': 'unchanged',
                                                      'data_type_representation': {
                                                          'float': 'unchanged',
                                                          'int': -0.06999999999999998,
                                                          'datetime': 'unchanged',
                                                          'text': 'unchanged'}}},
                {'column_name': 'PetalLengthCm', 'data_type': 'unchanged',
                 'data_label': 'unchanged', 'categorical': [True, False],
                 'order': 'unchanged',
                 'statistics': {'min': -2.0, 'max': -5.0, 'sum': -417.4,
                                'mean': -3.4419999999999997,
                                'variance': -0.6514736755308186,
                                'stddev': -0.6520673032064448,
                                'precision': {'min': 'unchanged',
                                              'max': 'unchanged',
                                              'mean': 0.10000000000000009,
                                              'var': -0.09, 'std': -0.19,
                                              'sample_size': -50,
                                              'margin_of_error': -0.044},
                                'unique_count': -25,
                                'unique_ratio': -0.16000000000000003,
                                'avg_predictions': {'UNKNOWN': 'unchanged',
                                                    'ADDRESS': 'unchanged',
                                                    'BAN': 'unchanged',
                                                    'CREDIT_CARD': 'unchanged',
                                                    'DATE': 'unchanged',
                                                    'TIME': 'unchanged',
                                                    'DATETIME': 'unchanged',
                                                    'DRIVERS_LICENSE': 'unchanged',
                                                    'EMAIL_ADDRESS': 'unchanged',
                                                    'UUID': 'unchanged',
                                                    'HASH_OR_KEY': 'unchanged',
                                                    'IPV4': 'unchanged',
                                                    'IPV6': 'unchanged',
                                                    'MAC_ADDRESS': 'unchanged',
                                                    'PERSON': 'unchanged',
                                                    'PHONE_NUMBER': 'unchanged',
                                                    'SSN': 'unchanged',
                                                    'URL': 'unchanged',
                                                    'US_STATE': 'unchanged',
                                                    'INTEGER': 'unchanged',
                                                    'FLOAT': 0.01,
                                                    'QUANTITY': 'unchanged',
                                                    'ORDINAL': -0.010000000000000009},
                                'label_representation': {'UNKNOWN': 'unchanged',
                                                         'ADDRESS': 'unchanged',
                                                         'BAN': 'unchanged',
                                                         'CREDIT_CARD': 'unchanged',
                                                         'DATE': 'unchanged',
                                                         'TIME': 'unchanged',
                                                         'DATETIME': 'unchanged',
                                                         'DRIVERS_LICENSE': 'unchanged',
                                                         'EMAIL_ADDRESS': 'unchanged',
                                                         'UUID': 'unchanged',
                                                         'HASH_OR_KEY': 'unchanged',
                                                         'IPV4': 'unchanged',
                                                         'IPV6': 'unchanged',
                                                         'MAC_ADDRESS': 'unchanged',
                                                         'PERSON': 'unchanged',
                                                         'PHONE_NUMBER': 'unchanged',
                                                         'SSN': 'unchanged',
                                                         'URL': 'unchanged',
                                                         'US_STATE': 'unchanged',
                                                         'INTEGER': 'unchanged',
                                                         'FLOAT': 0.01,
                                                         'QUANTITY': 'unchanged',
                                                         'ORDINAL': -0.010000000000000009},
                                'sample_size': -50, 'null_count': 'unchanged',
                                'null_types_index': 'unchanged',
                                'data_type_representation': {
                                    'float': 'unchanged',
                                    'int': -0.09999999999999999,
                                    'datetime': 'unchanged',
                                    'text': 'unchanged'}}},
                {'column_name': 'PetalWidthCm', 'data_type': 'unchanged',
                 'data_label': [['FLOAT'], [], ['ORDINAL']],
                 'categorical': 'unchanged', 'order': 'unchanged',
                 'statistics': {'min': -0.9, 'max': -1.9,
                                'sum': -155.40000000000006,
                                'mean': -1.4320000000000006,
                                'variance': -0.16893440527726244,
                                'stddev': -0.31755900190460573,
                                'precision': {'min': 'unchanged', 'max': -1,
                                              'mean': -0.8999999999999999,
                                              'var': -0.11, 'std': -0.34,
                                              'sample_size': -50,
                                              'margin_of_error': -0.11},
                                'unique_count': -10,
                                'unique_ratio': -0.04000000000000001,
                                'categories': [
                                    ['0.2', '0.4', '0.3', '0.1', '0.6', '0.5'],
                                    [],
                                    ['1.3', '1.5', '1.8', '1.4', '2.3', '1.0',
                                     '2.0', '2.1', '1.9', '1.2', '1.6', '1.1',
                                     '2.5', '2.2', '2.4', '1.7']],
                                'gini_impurity': -0.28680000000000017,
                                'unalikeability': -0.28318284889713463,
                                'categorical_count': {'0.2': [28, None],
                                                      '0.4': [7, None],
                                                      '0.3': [7, None],
                                                      '0.1': [6, None],
                                                      '0.6': [1, None],
                                                      '0.5': [1, None],
                                                      '1.3': [None, 13],
                                                      '1.5': [None, 12],
                                                      '1.8': [None, 12],
                                                      '1.4': [None, 8],
                                                      '2.3': [None, 8],
                                                      '1.0': [None, 7],
                                                      '2.0': [None, 6],
                                                      '2.1': [None, 6],
                                                      '1.9': [None, 5],
                                                      '1.2': [None, 5],
                                                      '1.6': [None, 4],
                                                      '1.1': [None, 3],
                                                      '2.5': [None, 3],
                                                      '2.2': [None, 3],
                                                      '2.4': [None, 3],
                                                      '1.7': [None, 2]},
                                'avg_predictions': {'UNKNOWN': 'unchanged',
                                                    'ADDRESS': 'unchanged',
                                                    'BAN': 'unchanged',
                                                    'CREDIT_CARD': 'unchanged',
                                                    'DATE': 'unchanged',
                                                    'TIME': 'unchanged',
                                                    'DATETIME': 'unchanged',
                                                    'DRIVERS_LICENSE': 'unchanged',
                                                    'EMAIL_ADDRESS': 'unchanged',
                                                    'UUID': 'unchanged',
                                                    'HASH_OR_KEY': 'unchanged',
                                                    'IPV4': 'unchanged',
                                                    'IPV6': 'unchanged',
                                                    'MAC_ADDRESS': 'unchanged',
                                                    'PERSON': 'unchanged',
                                                    'PHONE_NUMBER': 'unchanged',
                                                    'SSN': 'unchanged',
                                                    'URL': 'unchanged',
                                                    'US_STATE': 'unchanged',
                                                    'INTEGER': 'unchanged',
                                                    'FLOAT': 0.44333333333333347,
                                                    'QUANTITY': 'unchanged',
                                                    'ORDINAL': -0.44333333333333347},
                                'label_representation': {'UNKNOWN': 'unchanged',
                                                         'ADDRESS': 'unchanged',
                                                         'BAN': 'unchanged',
                                                         'CREDIT_CARD': 'unchanged',
                                                         'DATE': 'unchanged',
                                                         'TIME': 'unchanged',
                                                         'DATETIME': 'unchanged',
                                                         'DRIVERS_LICENSE': 'unchanged',
                                                         'EMAIL_ADDRESS': 'unchanged',
                                                         'UUID': 'unchanged',
                                                         'HASH_OR_KEY': 'unchanged',
                                                         'IPV4': 'unchanged',
                                                         'IPV6': 'unchanged',
                                                         'MAC_ADDRESS': 'unchanged',
                                                         'PERSON': 'unchanged',
                                                         'PHONE_NUMBER': 'unchanged',
                                                         'SSN': 'unchanged',
                                                         'URL': 'unchanged',
                                                         'US_STATE': 'unchanged',
                                                         'INTEGER': 'unchanged',
                                                         'FLOAT': 0.64,
                                                         'QUANTITY': 'unchanged',
                                                         'ORDINAL': -0.6399999999999999},
                                'sample_size': -50, 'null_count': 'unchanged',
                                'null_types_index': 'unchanged',
                                'data_type_representation': {
                                    'float': 'unchanged', 'int': -0.13,
                                    'datetime': 'unchanged',
                                    'text': 'unchanged'}}},
                {'column_name': 'Species', 'data_type': 'unchanged',
                 'data_label': 'unchanged', 'categorical': 'unchanged',
                 'order': ['constant value', 'ascending'],
                 'statistics': {'min': -3.0, 'max': -4.0, 'sum': -900.0,
                                'mean': -3.5, 'variance': -0.25252525252525254,
                                'stddev': -0.502518907629606, 'vocab': [['t'],
                                                                        ['r',
                                                                         'i',
                                                                         'e',
                                                                         '-',
                                                                         's',
                                                                         'I',
                                                                         'o',
                                                                         'a'],
                                                                        ['c',
                                                                         'n',
                                                                         'l',
                                                                         'g',
                                                                         'v']],
                                'unique_count': -1, 'unique_ratio': 'unchanged',
                                'categories': [['Iris-setosa'], [],
                                               ['Iris-virginica',
                                                'Iris-versicolor']],
                                'gini_impurity': -0.5,
                                'unalikeability': -0.5050505050505051,
                                'categorical_count': {'Iris-setosa': [50, None],
                                                      'Iris-virginica': [None,
                                                                         50],
                                                      'Iris-versicolor': [None,
                                                                          50]},
                                'avg_predictions': {'UNKNOWN': 'unchanged',
                                                    'ADDRESS': 'unchanged',
                                                    'BAN': 'unchanged',
                                                    'CREDIT_CARD': 'unchanged',
                                                    'DATE': 'unchanged',
                                                    'TIME': 'unchanged',
                                                    'DATETIME': 'unchanged',
                                                    'DRIVERS_LICENSE': 'unchanged',
                                                    'EMAIL_ADDRESS': 'unchanged',
                                                    'UUID': 'unchanged',
                                                    'HASH_OR_KEY': 'unchanged',
                                                    'IPV4': 'unchanged',
                                                    'IPV6': 'unchanged',
                                                    'MAC_ADDRESS': 'unchanged',
                                                    'PERSON': 'unchanged',
                                                    'PHONE_NUMBER': 'unchanged',
                                                    'SSN': 'unchanged',
                                                    'URL': 'unchanged',
                                                    'US_STATE': 'unchanged',
                                                    'INTEGER': 'unchanged',
                                                    'FLOAT': 'unchanged',
                                                    'QUANTITY': 'unchanged',
                                                    'ORDINAL': 'unchanged'},
                                'label_representation': {'UNKNOWN': 'unchanged',
                                                         'ADDRESS': 'unchanged',
                                                         'BAN': 'unchanged',
                                                         'CREDIT_CARD': 'unchanged',
                                                         'DATE': 'unchanged',
                                                         'TIME': 'unchanged',
                                                         'DATETIME': 'unchanged',
                                                         'DRIVERS_LICENSE': 'unchanged',
                                                         'EMAIL_ADDRESS': 'unchanged',
                                                         'UUID': 'unchanged',
                                                         'HASH_OR_KEY': 'unchanged',
                                                         'IPV4': 'unchanged',
                                                         'IPV6': 'unchanged',
                                                         'MAC_ADDRESS': 'unchanged',
                                                         'PERSON': 'unchanged',
                                                         'PHONE_NUMBER': 'unchanged',
                                                         'SSN': 'unchanged',
                                                         'URL': 'unchanged',
                                                         'US_STATE': 'unchanged',
                                                         'INTEGER': 'unchanged',
                                                         'FLOAT': 'unchanged',
                                                         'QUANTITY': 'unchanged',
                                                         'ORDINAL': 'unchanged'},
                                'sample_size': -50, 'null_count': 'unchanged',
                                'null_types_index': 'unchanged',
                                'data_type_representation': {
                                    'float': 'unchanged', 'int': 'unchanged',
                                    'datetime': 'unchanged',
                                    'text': 'unchanged'}}}]}

        self.assertDictEqual(expected_diff, profile.diff(profile2))


    def test_no_snappy(self):
        import sys
        import importlib
        import types
        orig_import = __import__
        # necessary for any wrapper around the library to test if snappy caught
        # as an issue

        def reload_data_profiler():
            """Recursively reload modules."""
            sys_modules = sys.modules.copy()
            for module_name, module in sys_modules.items():
                # Only reload top level of the dataprofiler
                if ('dataprofiler' in module_name and
                        len(module_name.split('.')) < 3):
                    if isinstance(module, types.ModuleType):
                        importlib.reload(module)

        def import_mock(name, *args):
            if name == 'snappy':
                raise ImportError('test')
            return orig_import(name, *args)

        with mock.patch('builtins.__import__', side_effect=import_mock):
            with self.assertWarns(ImportWarning) as w:
                import dataprofiler
                reload_data_profiler()

        self.assertEqual(
            str(w.warning),
            'Snappy must be installed to use parquet/avro datasets.'
            '\n\n'
            'For macOS use Homebrew:\n'
            '\t`brew install snappy`'
            '\n\n'
            'For linux use apt-get:\n`'
            '\tsudo apt-get -y install libsnappy-dev`\n',
        )

    def test_no_tensorflow(self):
        import sys
        import pandas
        orig_import = __import__
        # necessary for any wrapper around the library to test if snappy caught
        # as an issue

        def import_mock(name, *args):
            if name == 'tensorflow':
                raise ImportError('test')
            return orig_import(name, *args)

        with mock.patch('builtins.__import__', side_effect=import_mock):

            with self.assertWarnsRegex(RuntimeWarning,
                                       "Partial Profiler Failure"):
                modules_with_tf = [
                    'dataprofiler.labelers.character_level_cnn_model',
                ]
                for module in modules_with_tf:
                    if module in sys.modules:
                        del sys.modules[module]
                df = pandas.DataFrame([[1, 2.0], [1, 2.2], [-1, 3]])
                profile = Profiler(df)


if __name__ == '__main__':
    unittest.main()
