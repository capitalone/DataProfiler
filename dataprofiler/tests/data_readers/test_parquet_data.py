import os
import unittest
from io import BytesIO

from dataprofiler.data_readers.data import Data
from dataprofiler.data_readers.parquet_data import ParquetData


test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestParquetDataClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        test_dir = os.path.join(test_root_path, 'data', 'parquet')
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, 'iris.parq'), count=150),
            dict(path=os.path.join(test_dir, 'titanic.parq'), count=1316),
            dict(path=os.path.join(test_dir, 'gzip-nation.impala.parquet'), count=25),
            dict(path=os.path.join(test_dir, 'nation.dict.parquet'), count=25),
            dict(path=os.path.join(test_dir, 'nation.plain.parquet'), count=25),
            dict(path=os.path.join(test_dir, 'snappy-nation.impala.parquet'), count=25),
            dict(path=os.path.join(test_dir, 'nation.plain.intentionally_mislabled_file.csv'), count=25),
            dict(path=os.path.join(test_dir, 'nation.plain.intentionally_mislabled_file.txt'), count=25),
            dict(path=os.path.join(test_dir, 'nation.plain.intentionally_mislabled_file.json'), count=25),
            dict(path=os.path.join(test_dir, 'brotli_compressed_intentionally_mislabeled_parquet_file.csv'), count=2999),
            dict(path=os.path.join(test_dir, 'gzip_compressed_intentionally_mislabeled_parquet_file.csv'), count=2999),
            dict(path=os.path.join(test_dir, 'snappy_compressed_intentionally_mislabeled_parquet_file.csv'), count=2999),

        ]

        cls.buffer_list = []
        for input_file in cls.input_file_names:
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
        Determine if the parquet file can be automatically identified from
        byte stream or file path
        """
        for input_file in self.file_or_buf_list:
            self.assertTrue(ParquetData.is_match(input_file['path']))

    def test_auto_file_identification(self):
        """
        Determine if the parquet file can be automatically identified
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'parquet')

    def test_specifying_data_type(self):
        """
        Determine if the parquet file can be loaded with manual data_type setting
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"], data_type='parquet')
            self.assertEqual(input_data_obj.data_type, 'parquet')

    def test_reload_data(self):
        """
        Determine if the parquet file can be reloaded
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file['path'])
            input_data_obj.reload(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'parquet')
            self.assertEqual(input_file['path'], input_data_obj.input_file_path)

    def test_data_formats(self):
        """
        Determine if the parquet file data_formats can be used
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

    def test_mixed_string_col(self):
        """
        Determine if parquet can handle mixed string column types.
        """

        test_file = os.path.join(test_root_path, 'data', 'parquet',
                                 'mixed_string_data_col.parquet')
        parq_data = ParquetData(test_file)

        # assert str and not bytes
        self.assertIsInstance(parq_data.data['col2'][1], str)
        self.assertIsInstance(parq_data.data['col2'][3], str)

        # assert no 'b"data"' encapsulated, just 'data'
        self.assertNotIn('b"', parq_data.data['col2'][1])
        self.assertNotIn("b'", parq_data.data['col2'][1])
        self.assertNotIn('b"', parq_data.data['col2'][3])
        self.assertNotIn("b'", parq_data.data['col2'][3])

    def test_mixed_non_string_col(self):
        """
        Determine if parquet can handle mixed non-string column types.
        """

        test_file = os.path.join(test_root_path, 'data', 'parquet',
                                 'mixed_datetime_data_col.parquet')
        parq_data = ParquetData(test_file)

        # assert str and not bytes
        self.assertIsInstance(parq_data.data['col2'][1], str)
        self.assertIsInstance(parq_data.data['col2'][3], str)

        # assert no 'b"data"' encapsulated, just 'data'
        self.assertNotIn('b"', parq_data.data['col2'][1])
        self.assertNotIn("b'", parq_data.data['col2'][1])
        self.assertNotIn('b"', parq_data.data['col2'][3])
        self.assertNotIn("b'", parq_data.data['col2'][3])

    def test_len_data(self):
        """
        Validate that length called on ParquetData is appropriately determining
        the length value.
        """

        for input_file in self.file_or_buf_list:
            data = Data(input_file["path"])
            self.assertEqual(input_file['count'],
                             len(data),
                             msg=input_file['path'])
            self.assertEqual(input_file['count'],
                             data.length,
                             msg=input_file['path'])

    def test_file_encoding(self):
        """Tests to ensure file_encoding set to None"""
        for input_file in self.file_or_buf_list:
            data = ParquetData(input_file["path"])
            self.assertIsNone(data.file_encoding)

    def test_is_structured(self):
        # Default construction
        data = ParquetData()
        self.assertTrue(data.is_structured)

        # With option specifying dataframe as data_format
        data = ParquetData(options={"data_format": "dataframe"})
        self.assertTrue(data.is_structured)

        # With option specifying records as data_format
        data = ParquetData(options={"data_format": "records"})
        self.assertFalse(data.is_structured)

        # With option specifying json as data_format
        data = ParquetData(options={"data_format": "json"})
        self.assertFalse(data.is_structured)
