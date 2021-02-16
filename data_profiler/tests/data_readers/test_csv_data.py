import os
import unittest

import pandas as pd

from data_profiler.data_readers.data import Data, CSVData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCSVDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/diamonds.csv'), count=1000, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris.csv'), count=150, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris-utf-8.csv'), count=150, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris-utf-16.csv'), count=150, delimiter=',', encoding='utf-16'),
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.parquet'), 
                 count=150, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.txt'), 
                 count=150, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.json'), 
                 count=150, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/guns.csv'), count=1316, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/wisconsin_cancer_train.csv'), 
                 count=25, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/aws_honeypot_marx_geo.csv'), 
                 count=25, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num.csv'), count=5, delimiter=None, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col.txt'), count=5, delimiter=None, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_long.txt'), count=753, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-last-column.txt'), count=5, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_sparse.txt'), count=20, delimiter=',', encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_super_sparse.txt'), 
                 count=20, delimiter=',', encoding='utf-8'),
        ]
        cls.output_file_path = None

    def test_auto_file_identification(self):
        """
        Determine if the csv file can be automatically identified
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'csv')
            self.assertEqual(input_data_obj.delimiter, input_file['delimiter'])

    def test_specifying_data_type(self):
        """
        Determine if the csv file can be loaded with manual data_type setting
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"], data_type='csv')
            self.assertEqual(input_data_obj.data_type, 'csv')
            self.assertEqual(input_data_obj.delimiter, input_file['delimiter'])

    def test_data_formats(self):
        """
        Test the data format options.
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            self.assertIsInstance(input_data_obj.data, pd.DataFrame)

            input_data_obj.data_format = "records"
            self.assertIsInstance(input_data_obj.data, list)

            with self.assertRaises(ValueError) as exc:
                input_data_obj.data_format = "NON_EXISTENT"
            self.assertEqual(
                str(exc.exception),
                "The data format must be one of the following: " +
                "['dataframe', 'records']"
            )

    def test_reload_data(self):
        """
        Determine if the csv file can be reloaded
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            input_data_obj.reload(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'csv')
            self.assertEqual(input_data_obj.delimiter, input_file['delimiter'])

    def test_data_formats(self):
        """
        Determine if the csv file data_formats can be used
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
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

    def test_header_check_files(self):
        """
        Determine if files with no header are properly determined.
        """
        test_dir = os.path.join(test_root_path, 'data')

        # File w/o header, set to None
        path = os.path.join(test_dir, 'csv/iris_no_header.csv')
        options = dict()
        CSVData.is_match(path, options)
        self.assertIsNone(options.get("header"))

        # File with header, set to 0 for auto determine
        path = os.path.join(test_dir, 'csv/iris.csv')
        options = dict()
        CSVData.is_match(path, options)
        self.assertEqual(0, options.get("header"))

if __name__ == '__main__':
    unittest.main()
