import unittest
from unittest import mock

import pandas as pd

from dataprofiler.data_readers.data import Data


class TestDataReadingWriting(unittest.TestCase):

    def test_read_data(self):
        """Ensure error logs for trying to save empty data."""
        data_types = ['csv', 'json', 'parquet', 'text']
        none_types = [pd.DataFrame(), pd.DataFrame(), "", ""]
        for data_type, none_type in zip(data_types, none_types):
            Data(data=none_type, data_type=data_type)

class TestDataReadFromURL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        http_url = 'https://raw.githubusercontent.com/capitalone/DataProfiler/main/dataprofiler/tests/data/'
        cls.input_url_names = [
            dict(path=http_url+'csv/diamonds.csv',
                 count=53940, delimiter=',', has_header=[0],
                 num_columns=10, encoding='utf-8', data_type='csv'),
            dict(path=http_url+'csv/iris-utf-16.csv',
                 count=150, delimiter=',', has_header=[0],
                 num_columns=6, encoding='utf-16', data_type='csv'),
            dict(path=http_url+'avro/users.avro', count=4,  data_type='avro'),
            dict(path=http_url+'avro/userdata1.avro', count=1000,  data_type='avro'),
            dict(path=http_url+'json/iris-utf-8.json', encoding='utf-8', count=150, data_type='json'),
            dict(path=http_url+'json/iris-utf-16.json', encoding='utf-16', count=150, data_type='json'),
            dict(path=http_url+'parquet/iris.parq', count=150, data_type='parquet'),
            dict(path=http_url+'parquet/titanic.parq', count=1316, data_type='parquet')
        ]

    def test_read_from_url(self):
        """Ensure error logs for trying to save empty data."""

        for url in self.input_url_names:
            data_obj = Data(url['path'])
            self.assertEqual(data_obj.data_type, url['data_type'])

    @mock.patch('requests.get')
    def test_read_url_overflow(self, mock_request_get):
        # assumed chunk size
        c_size = 8192
        max_allows_file_size = 1024 ** 3 # 1GB
        def infinite_loop(*args, **kwargs):
            max_file_list = ([b'test']
                             * (int(max_allows_file_size) // c_size + 1))
            for value in max_file_list:
                yield value
        # mock the iter_content to return up to 1GB + so error raises
        mock_request_get.return_value.__enter__.return_value\
            .iter_content.side_effect = infinite_loop

        with self.assertRaisesRegex(ValueError, 'The downloaded file from the url may not be larger than 1GB'):
            data_obj = Data('https://test.com')

            

if __name__ == '__main__':
    unittest.main()
