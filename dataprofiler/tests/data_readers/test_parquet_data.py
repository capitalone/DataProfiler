import os
import unittest

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
        cls.output_file_path = None

    def test_auto_file_identification(self):
        """
        Determine if the parquet file can be automatically identified
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'parquet')

    def test_specifying_data_type(self):
        """
        Determine if the parquet file can be loaded with manual data_type setting
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file["path"], data_type='parquet')
            self.assertEqual(input_data_obj.data_type, 'parquet')

    def test_reload_data(self):
        """
        Determine if the parquet file can be reloaded
        """
        for input_file in self.input_file_names:
            input_data_obj = Data(input_file['path'])
            input_data_obj.reload(input_file['path'])
            self.assertEqual(input_data_obj.data_type, 'parquet')

    def test_data_formats(self):
        """
        Determine if the parquet file data_formats can be used
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

        for input_file in self.input_file_names:
            data = Data(input_file["path"])
            self.assertEqual(input_file['count'],
                             len(data),
                             msg=input_file['path'])
            self.assertEqual(input_file['count'],
                             data.length,
                             msg=input_file['path'])
