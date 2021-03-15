import os
import unittest

import pandas as pd

from dataprofiler.data_readers.data import Data, CSVData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCSVDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/diamonds.csv'),
                 count=1000, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris.csv'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris_no_header.csv'),
                 count=150, delimiter=',', has_header=[None],
                 num_columns=6, encoding='utf-8'),            
            dict(path=os.path.join(test_dir, 'csv/iris-utf-8.csv'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris-utf-16.csv'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-16'),
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.parquet'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.txt'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris_intentionally_mislabled_file.json'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/guns.csv'),
                 count=1316, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/wisconsin_cancer_train.csv'),
                 count=25, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/aws_honeypot_marx_geo.csv'),
                 count=25, delimiter=',', has_header=[0],
                 num_columns=16, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num.csv'),
                 count=5, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num-negative.csv'),
                 count=5, delimiter=None, has_header=[None],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col.txt'),
                 count=5, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col-empty.txt'),
                 count=5, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_long.txt'),
                 count=753, delimiter=',', has_header=[None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-last-column.txt'),
                 count=5, delimiter=',', has_header=[0],
                 num_columns=2, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-column.txt'),
                 count=5, delimiter=',', has_header=[0],
                 num_columns=2, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column.txt'),
                 count=5, delimiter=',', has_header=[0],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-no-header.txt'),
                 count=5, delimiter=',', has_header=[None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_sparse.txt'),
                 count=20, delimiter=',', has_header=[None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_super_sparse.txt'),
                 count=20, delimiter=',', has_header=[None],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-columns-test.csv'),
                 count=20, delimiter=',', has_header=[0],
                 num_columns=36, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sentence-4x.txt'),
                 count=4, delimiter='.', has_header=[0, None],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/quote-test.txt'),
                 count=10, delimiter=' ', has_header=[0, None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/multiple-col-delimiter-last.txt'),
                 count=6, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col-without-space.txt'),
                 count=6, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-two-headers.txt'),
                 count=6, delimiter=',', has_header=[1],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-interchange-none.txt'),
                 count=6, delimiter=',', has_header=[0],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/blogposts.csv'),
                 count=25, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-standard-header.csv'),
                 count=11, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-standard-header-quotes.csv'),
                 count=11, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-standard-header-data-quotes.csv'),
                 count=11, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-skip-header.csv'),
                 count=11, delimiter=',', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-skip-header-author.csv'),
                count=7, delimiter=',', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-skip-partial-header.csv'),
                 count=7, delimiter=',', has_header=[None, 1],
                 num_columns=4, encoding='utf-8'),
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
            self.assertEqual(len(input_data_obj.data.columns),
                             input_file['num_columns'])

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

    def test_allowed_data_formats(self):
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

    def test_set_header(self):
        test_dir = os.path.join(test_root_path, 'data')
        filename = 'csv/sparse-first-and-last-column-two-headers.txt'
        filename = os.path.join(test_dir, filename)

        # set bad header setting
        options = dict(header=-2)
        with self.assertRaisesRegex(ValueError,
                                    '`header` must be one of following: auto, '
                                    'none for no header, or a non-negative '
                                    'integer for the row that represents the '
                                    'header \(0 based index\)'):
            csv_data = CSVData(filename, options=options)
            first_value = csv_data.data.loc[0][0]

        # set bad header setting
        options = dict(header='abcdef')
        with self.assertRaisesRegex(ValueError,
                                    '`header` must be one of following: auto, '
                                    'none for no header, or a non-negative '
                                    'integer for the row that represents the '
                                    'header \(0 based index\)'):
            csv_data = CSVData(filename, options=options)
            first_value = csv_data.data.loc[0][0]

        # set header auto
        options = dict(header='auto')
        csv_data = CSVData(filename, options=options)
        first_value = csv_data.data.loc[0][0]
        self.assertEqual(1, csv_data.header)
        self.assertEqual('1', first_value)

        # set header None (no header)
        options = dict(header=None)
        csv_data = CSVData(filename, options=options)
        first_value = csv_data.data.loc[0][0]
        self.assertIsNone(csv_data.header)  # should be None
        self.assertEqual('COUNT', first_value)

        # set header 0
        options = dict(header=0)
        csv_data = CSVData(filename, options=options)
        first_value = csv_data.data.loc[0][0]
        self.assertEqual(0, csv_data.header)
        self.assertEqual('CONTAR', first_value)

        # set header 1
        options = dict(header=1)
        csv_data = CSVData(filename, options=options)
        first_value = csv_data.data.loc[0][0]
        self.assertEqual(1, csv_data.header)
        self.assertEqual('1', first_value)

    def test_header_check_files(self):
        """
        Determine if files with no header are properly determined.
        """
        from itertools import islice
        from dataprofiler.data_readers import data_utils

        # add some more files to the list to test the header detection
        # these files have some first lines which are not the header
        test_dir = os.path.join(test_root_path, 'data')
        file_with_header_and_authors = [
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-header-and-author.txt'),
                 count=6, delimiter=',', has_header=[1],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-header-and-author-description.txt'),
                 count=6, delimiter=',', has_header=[3],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-empty-first-row.txt'),
                 count=11, delimiter=',', has_header=[1],
                 num_columns=3, encoding='utf-8'),
        ]

        input_file_names = self.input_file_names[:]
        input_file_names += file_with_header_and_authors
        for input_file in input_file_names:
            file_encoding = data_utils.detect_file_encoding(input_file['path'])
            with open(input_file['path'], encoding=file_encoding) as csvfile:
                data_as_str = ''.join(list(islice(csvfile, 5)))
            header_line = CSVData._guess_header_row(data_as_str, input_file['delimiter'])
            self.assertIn(header_line, input_file['has_header'])


if __name__ == '__main__':
    unittest.main()
