import os
import unittest
from io import BytesIO

from dataprofiler.data_readers.avro_data import AVROData
from dataprofiler.data_readers.data import Data

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestAVRODataClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_file_path = None
        cls.output_file_path = None
        cls.ss = None

        test_dir = os.path.join(test_root_path, "data")
        cls.input_file_names = [
            dict(path=os.path.join(test_dir, "avro/users.avro"), count=4),
            dict(path=os.path.join(test_dir, "avro/userdata1.avro"), count=1000),
            dict(
                path=os.path.join(
                    test_dir, "avro/userdata1_intentionally_mislabled_file.parquet"
                ),
                count=1000,
            ),
            dict(
                path=os.path.join(
                    test_dir, "avro/userdata1_intentionally_mislabled_file.csv"
                ),
                count=1000,
            ),
            dict(
                path=os.path.join(
                    test_dir, "avro/userdata1_intentionally_mislabled_file.json"
                ),
                count=1000,
            ),
            dict(
                path=os.path.join(
                    test_dir, "avro/userdata1_intentionally_mislabled_file.txt"
                ),
                count=1000,
            ),
            dict(
                path=os.path.join(
                    test_dir,
                    "avro/deflate_compressed_intentionally_mislabeled_file.csv",
                ),
                count=4,
            ),
            dict(
                path=os.path.join(
                    test_dir, "avro/snappy_compressed_intentionally_mislabeled_file.csv"
                ),
                count=4,
            ),
        ]

        cls.buffer_list = []
        for input_file in cls.input_file_names:
            # add BytesIO
            buffer_info = input_file.copy()
            with open(input_file["path"], "rb") as fp:
                buffer_info["path"] = BytesIO(fp.read())
            cls.buffer_list.append(buffer_info)

        cls.file_or_buf_list = cls.input_file_names + cls.buffer_list

    @classmethod
    def setUp(cls):
        for buffer in cls.buffer_list:
            buffer["path"].seek(0)

    def test_is_match(self):
        """
        Determine if the avro file can be automatically identified from
        byte stream or file path
        """
        for input_file in self.file_or_buf_list:
            self.assertTrue(AVROData.is_match(input_file["path"]))

    def test_avro_file_identification(self):
        """
        Determine if the avro file can be automatically identified
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"])
            self.assertEqual(input_data_obj.data_type, "avro")

    def test_specifying_data_type(self):
        """
        Determine if the avro file can be loaded with manual data_type setting
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"], data_type="avro")
            self.assertEqual(input_data_obj.data_type, "avro")

    def test_reload_data(self):
        """
        Determine if the avro file can be reloaded
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"])
            input_data_obj.reload(input_file["path"])
            self.assertEqual(input_data_obj.data_type, "avro")
            self.assertEqual(input_file["path"], input_data_obj.input_file_path)

    def test_data_formats(self):
        """
        Determine if the avro file data_formats can be used
        """
        for input_file in self.file_or_buf_list:
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

    def test_nested_keys(self):
        """
        Determine if the avro file data_formats can be used
        """
        dict = [
            {"name": 1, "favorite_number": 1},
            {"favorite_color": 1, "address": {"streetaddress": 1, "city": 1}},
        ]
        nested_keys = AVROData._get_nested_keys_from_dicts(dict)
        self.assertIsNotNone(nested_keys)
        schema_avro = {
            "namespace": "avro_namespace",
            "name": "avro_filename",
            "type": "record",
            "fields": [
                {"name": "name", "type": ["string", "null"]},
                {"name": "favorite_number", "type": ["string", "null"]},
                {"name": "favorite_color", "type": ["string", "null"]},
                {
                    "name": "address",
                    "type": [
                        {
                            "namespace": "avro_namespace",
                            "name": "address",
                            "type": "record",
                            "fields": [
                                {"name": "streetaddress", "type": ["string", "null"]},
                                {"name": "city", "type": ["string", "null"]},
                            ],
                        },
                        "null",
                    ],
                },
            ],
        }
        schema_avro = AVROData._get_schema_avro(nested_keys, schema_avro)
        self.assertIsNotNone(schema_avro)

    def test_len_data(self):
        """
        Validate that length called on JSONData is appropriately determining the
        length value.
        """

        for input_file in self.file_or_buf_list:
            data = Data(input_file["path"])
            self.assertEqual(input_file["count"], len(data), msg=input_file["path"])
            self.assertEqual(input_file["count"], data.length, msg=input_file["path"])

    def test_file_encoding(self):
        """Tests to ensure file_encoding set to None"""
        for input_file in self.file_or_buf_list:
            data = AVROData(input_file["path"])
            self.assertIsNone(data.file_encoding)

    def test_is_structured(self):
        # Default construction
        data = AVROData()
        self.assertTrue(data.is_structured)

        # With option specifying dataframe as data_format
        data = AVROData(options={"data_format": "dataframe"})
        self.assertTrue(data.is_structured)

        # With option specifying flattened_dataframe as data_format
        data = AVROData(options={"data_format": "flattened_dataframe"})
        self.assertTrue(data.is_structured)

        # With option specifying records as data_format
        data = AVROData(options={"data_format": "records"})
        self.assertFalse(data.is_structured)

        # With option specifying json as data_format
        data = AVROData(options={"data_format": "json"})
        self.assertFalse(data.is_structured)


if __name__ == "__main__":
    unittest.main()
