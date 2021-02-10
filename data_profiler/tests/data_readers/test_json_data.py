from __future__ import print_function
from __future__ import absolute_import

import os
import unittest

from data_profiler.data_readers.data import Data
from data_profiler.data_readers import json_data


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestNestedJSON(unittest.TestCase):

    def test_flat_to_nested_json(self):
        dic = {
            'a.b': 'ab',
            'a.c': 'ac',
            'a.d.f': 'adf',
            'b': 'b'
        }

        converted_dic = json_data.JSONData._convert_flat_to_nested_cols(dic)
        self.assertTrue(converted_dic == {
            'a': {'b': 'ab', 'c': 'ac', 'd': {'f': 'adf'}},
            'b': 'b'
        })


class TestJSONDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_file_path = None
        cls.output_file_path = None
        cls.ss = None

        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'json/iris-utf-8.json'), encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'json/iris-utf-16.json'), encoding='utf-16'),
            dict(path=os.path.join(test_dir, "json/honeypot"), encoding='utf-8'),
            dict(path=os.path.join(test_dir, "json/honeypot_intentially_mislabeled_file.csv"), encoding='utf-8'),
            dict(path=os.path.join(test_dir, "json/honeypot_intentially_mislabeled_file.parquet"), encoding='utf-8')
        ]

    def test_json_file_identification(self):
        """
        Determine if the json file can be automatically identified
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"])
            self.assertEqual(input_data_obj.data_type, 'json')

    def test_specifying_data_type(self):
        """
        Determine if the json file can be loaded with manual data_type setting
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"], data_type='json')
            self.assertEqual(input_data_obj.data_type, 'json')

    def test_reload_data(self):
        """
        Determine if the json file can be reloaded
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"])
            input_data_obj.reload(input_file["path"])
            self.assertEqual(input_data_obj.data_type, 'json')

    def test_json_from_string(self):
        """
        Determine if the json file can be loaded with manual data_type setting
        """
        passing_json_strings = [
            '[]',
            '{}',
            '[1, 2, 3]',
            '[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 1, "b":2}]',
        ]
        failing_json_strings = [
            dict(value='[1,[1]]',
                 error='Only JSON which represents structured data is '
                       'supported for this data type (i.e. list-dicts).'),
            dict(value='[{"a": 1}, 2, [3]]',
                 error='Only JSON which represents structured data is '
                       'supported for this data type (i.e. list-dicts).'),
            dict(value='{',
                 error='No JSON data could be read from these data.')
        ]
        for json_string in passing_json_strings:
            # in memory data must specify the data_type category
            input_data_obj = Data(data=json_string, data_type='json')
            self.assertEqual(input_data_obj.data_type, 'json')

        for json_string in failing_json_strings:
            # in memory data must specify the data_type category
            with self.assertRaises(ValueError) as assert_raised:
                Data(data=json_string['value'], data_type='json')
            self.assertEqual(
                str(assert_raised.exception),
                json_string['error']
            )

    def test_data_formats(self):
        """
        Determine if the json file data_formats can be used
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"])
            for data_format in list(input_data_obj._data_formats.keys()):
                input_data_obj.data_format = data_format
                self.assertEqual(input_data_obj.data_format, data_format)
                data = input_data_obj.data
                if data_format == "dataframe":
                    import pandas as pd
                    self.assertIsInstance(data, pd.DataFrame)
                elif data_format in ["records", "json"]:
                    self.assertIsInstance(data, list)
                    self.assertIsInstance(data[0], str)


if __name__ == '__main__':
    unittest.main()
