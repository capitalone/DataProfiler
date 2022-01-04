from dataprofiler.data_readers.data_utils import is_stream_buffer
import os
import unittest
from io import StringIO, BytesIO, TextIOWrapper

import pandas as pd

from dataprofiler.data_readers.data import Data, CSVData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCSVDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        test_dir = os.path.join(test_root_path, 'data')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'csv/diamonds.csv'),
                 count=53940, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris.csv'),
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/iris_no_header.csv'),
                 count=20, delimiter=',', has_header=[None],
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
                 count=100798, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/wisconsin_cancer_train.csv'),
                 count=497, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/aws_honeypot_marx_geo.csv'),
                 count=2999, delimiter=',', has_header=[0],
                 num_columns=16, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num.csv'),
                 count=5, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num-spaces.csv'),
                 count=5, delimiter=',', has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/small-num-negative.csv'),
                 count=5, delimiter=None, has_header=[None],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col.txt'),
                 count=6, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col-empty.txt'),
                 count=33, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_long.txt'),
                 count=753, delimiter=',', has_header=[None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-last-column.txt'),
                 count=9, delimiter=',', has_header=[0],
                 num_columns=2, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-column.txt'),
                 count=9, delimiter=',', has_header=[0],
                 num_columns=2, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column.txt'),
                 count=9, delimiter=',', has_header=[0],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-no-header.txt'),
                 count=9, delimiter=',', has_header=[None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_sparse.txt'),
                 count=20, delimiter=',', has_header=[None],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/log_data_super_sparse.txt'),
                 count=20, delimiter=',', has_header=[None],
                 num_columns=6, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-columns-test.csv'),
                 count=25, delimiter=',', has_header=[0],
                 num_columns=36, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sentence-4x.txt'),
                 count=4, delimiter='.', has_header=[0, None],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/quote-test.txt'),
                 count=8, delimiter=' ', has_header=[0, None],
                 num_columns=3, encoding='utf-8'),            
            dict(path=os.path.join(test_dir, 'csv/quote-test-singlequote.txt'),
                 count=8, delimiter=' ', has_header=[0, None],
                 num_columns=3, encoding='utf-8'),            
            dict(path=os.path.join(test_dir, 'csv/multiple-col-delimiter-last.txt'),
                 count=5, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/names-col-without-space.txt'),
                 count=10, delimiter=None, has_header=[0],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-two-headers.txt'),
                 count=9, delimiter=',', has_header=[1],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-interchange-none.txt'),
                 count=9, delimiter=',', has_header=[0],
                 num_columns=3, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/blogposts.csv'),
                 count=25, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-standard-header.csv'),
                 count=10, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-standard-header-quotes.csv'),
                 count=10, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-standard-header-data-quotes.csv'),
                 count=10, delimiter=',', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-skip-header.csv'),
                 count=10, delimiter=',', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-skip-header-author.csv'),
                 count=5, delimiter=',', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/all-strings-skip-partial-header.csv'),
                 count=6, delimiter=',', has_header=[None, 1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/num-negative-title.csv'),
                 count=5, delimiter=None, has_header=[None],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/num-negative-title-large.csv'),
                 count=5, delimiter=None, has_header=[None],
                 num_columns=1, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/daily-activity-sheet-@.csv'),
                 count=30, delimiter='@', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/daily-activity-sheet-int-description.csv'),
                 count=30, delimiter=',', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/daily-activity-sheet-@-singlequote.csv'),
                 count=30, delimiter='@', has_header=[1],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/daily-activity-sheet-tab.csv'),
                 count=30, delimiter='\t', has_header=[0],
                 num_columns=4, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/preferred-check-small-num.csv'),
                 count=5, delimiter=',', has_header=[None],
                 num_columns=2, encoding='utf-8'),
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-empty-first-row.txt'),
                 count=9, delimiter=',', has_header=[1],
                 num_columns=3, encoding='utf-8'),            
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-header-and-author.txt'),
                 count=9, delimiter=',', has_header=[1],
                 num_columns=3, encoding='utf-8'),            
            dict(path=os.path.join(test_dir, 'csv/sparse-first-and-last-column-header-and-author-description.txt'),
                 count=9, delimiter=',', has_header=[3],
                 num_columns=3, encoding='utf-8'),
           dict(path=os.path.join(test_dir, 'csv/flight_delays.csv'),
                 count=20, delimiter=',', has_header=[0],
                 num_columns=8, encoding='utf-8'),
        ]

        cls.buffer_list = []
        for input_file in cls.input_file_names:
            # add StringIO
            buffer_info = input_file.copy()
            with open(input_file['path'], 'r', encoding=input_file['encoding']) as fp:
                buffer_info['path'] = StringIO(fp.read())
            cls.buffer_list.append(buffer_info)
            
            # add BytesIO
            buffer_info = input_file.copy()
            with open(input_file['path'], 'rb') as fp:
                buffer_info['path'] = BytesIO(fp.read())
            cls.buffer_list.append(buffer_info)

        cls.file_or_buf_list = cls.input_file_names + cls.buffer_list

        cls.output_file_path = None
    
    @classmethod
    def setUp(cls):
        for buffer in cls.buffer_list:
            buffer['path'].seek(0)

    def test_is_match(self):
        """
        Determine if the csv file can be automatically identified from
        byte stream or stringio stream or file path
        """
        for input_file in self.file_or_buf_list:
            self.assertTrue(CSVData.is_match(input_file['path']))

    def test_auto_file_identification(self):
        """
        Determine if the csv file can be automatically identified
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            try:
                self.assertEqual(input_data_obj.delimiter, input_file['delimiter'],
                                 input_file['path'])
                self.assertEqual(len(input_data_obj.data.columns),
                                 input_file['num_columns'],
                                 input_file['path'])
            except AttributeError as e:
                raise AttributeError(repr(e)+': '+input_file['path'].split("/")[-1])

    def test_specifying_data_type(self):
        """
        Determine if the csv file can be loaded with manual data_type setting
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"], data_type='csv')
            self.assertEqual(input_data_obj.data_type, 'csv',
                             input_file["path"])
            self.assertEqual(input_data_obj.delimiter,
                             input_file['delimiter'],
                             input_file["path"])

    def test_data_formats(self):
        """
        Test the data format options.
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'csv')
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
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            input_data_obj.reload(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'csv',
                             input_file['path'])
            self.assertEqual(input_data_obj.delimiter, input_file['delimiter'],
                             input_file['path'])
            self.assertEqual(input_file['path'], input_data_obj.input_file_path)

    def test_allowed_data_formats(self):
        """
        Determine if the csv file data_formats can be used
        """
        for input_file in self.file_or_buf_list:
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

        # add some more files to the list to test the header detection
        # these files have some first lines which are not the header
        for input_file in self.input_file_names:
            with open(input_file['path'],
                      encoding=input_file['encoding']) as csvfile:
                data_as_str = ''.join(list(islice(csvfile, 5)))
            header_line = CSVData._guess_header_row(data_as_str,
                                                    input_file['delimiter'])
            self.assertIn(header_line, input_file['has_header'],
                          input_file['path'])

        for input_buf in self.buffer_list:
            # BytesIO is wrapped so that it is fed into guess header row 
            # the same way it would internally
            buffer = input_buf['path']
            if isinstance(input_buf['path'], BytesIO):
                buffer = TextIOWrapper(
                    input_buf['path'], encoding=input_buf['encoding'])

            data_as_str = ''.join(list(islice(buffer, 5)))
            header_line = CSVData._guess_header_row(
                data_as_str, input_buf['delimiter'])
            self.assertIn(header_line, input_buf['has_header'],
                          input_buf['path'])

            # since BytesIO was wrapped, it now has to be detached
            if isinstance(buffer, TextIOWrapper):
                buffer.detach()

    def test_options(self):

        def _test_options(option, valid, invalid, expected_error):
            # Test Valid
            for value in valid:
                CSVData(options={option: value})
            
            # Test Invalid
            for value in invalid:
                with self.assertRaisesRegex(ValueError, expected_error):
                    CSVData(options={option: value})

        _test_options(
            "header", valid=["auto", None, 0, 1],
            invalid=["error", CSVData(), -1],
            expected_error='`header` must be one of following: auto, ')
        
        _test_options(
            "delimiter", valid=[',', '\t', '', None],
            invalid=[CSVData(), 1],
            expected_error="'delimiter' must be a string or None")
        
        _test_options(
            "data_format", valid=['dataframe', 'records'],
            invalid=["error", CSVData(), 1, None],
            expected_error="'data_format' must be one of the following: ")
        
        _test_options(
            "selected_columns", valid=[['hello', 'world'], ["test"], []],
            invalid=["error", CSVData(), 1, None],
            expected_error="'selected_columns' must be a list")
        
        _test_options(
            "selected_columns", valid=[], invalid=[[0, 1, 2, 3]],
            expected_error="'selected_columns' must be a list of strings")

        _test_options(
            "record_samples_per_line", valid=[1, 10],
            invalid=[[-1, int, '', None, dict()]],
            expected_error="'record_samples_per_line' must be an int more than "
                           "0")

        # test edge case for header being set
        file = self.input_file_names[0]
        filepath = file['path']
        expected_header_value = file['has_header'][0]
        options = {'header': 'auto', 'delimiter': ','}  # default values
        data = CSVData(options=options)
        self.assertEqual('auto', data.header)
        self.assertFalse(data._checked_header)

        data = CSVData(filepath, options=options)
        retrieve_data = data.data
        self.assertEqual(expected_header_value, data.header)
        self.assertTrue(data._checked_header)

    def test_len_data(self):
        """
        Validate that length called on CSVData is appropriately determining the
        length value.
        """

        for input_file in self.file_or_buf_list:
            data = Data(input_file["path"])
            self.assertEqual(input_file['count'],
                             len(data),
                             msg=input_file['path'])
            self.assertEqual(input_file['count'],
                             data.length,
                             msg=input_file['path'])

    def test_is_structured(self):
        # Default construction
        data = CSVData()
        self.assertTrue(data.is_structured)

        # With option specifying dataframe as data_format
        data = CSVData(options={"data_format": "dataframe"})
        self.assertTrue(data.is_structured)

        # With option specifying records as data_format
        data = CSVData(options={"data_format": "records"})
        self.assertFalse(data.is_structured)


if __name__ == '__main__':
    unittest.main()
