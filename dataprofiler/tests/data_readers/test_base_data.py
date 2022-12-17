import locale
import os
import unittest
from io import BytesIO, StringIO

from dataprofiler.data_readers.base_data import BaseData

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestBaseDataClass(unittest.TestCase):

    buffer_list = []

    @classmethod
    def setUpClass(cls) -> None:
        test_dir = os.path.join(test_root_path, "data")
        cls.input_file_names = [
            dict(
                path=os.path.join(test_dir, "csv/diamonds.csv"),
                encoding="utf-8",
                data_type="csv",
            ),
            dict(
                path=os.path.join(test_dir, "avro/users.avro"),
                encoding=None,
                data_type="avro",
            ),
            dict(
                path=os.path.join(test_dir, "json/iris-utf-16.json"),
                encoding="utf-16",
                data_type="json",
            ),
            dict(
                path=os.path.join(test_dir, "json/iris-utf-32.json"),
                encoding="utf-32",
                data_type="json",
            ),
            dict(
                path=os.path.join(test_dir, "parquet/iris.parq"),
                encoding=None,
                data_type="parquet",
            ),
            dict(
                path=os.path.join(test_dir, "txt/code.txt"),
                encoding="utf-8",
                data_type="text",
            ),
            dict(
                path=os.path.join(test_dir, "txt/empty.txt"),
                encoding="utf-8",
                data_type="text",
            ),
        ]

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

    def test_can_apply_data_functions(self):
        class FakeDataClass:
            # matches the `data_type` value in BaseData for validating priority
            data_type = "FakeData"
            options = {"not_empty": "data"}

            def func1(self):
                return "success"

        # initialize the data class
        data = BaseData(input_file_path="", data=FakeDataClass(), options={})

        # if the function exists in BaseData fail the test because the results
        # may become inaccurate.
        self.assertFalse(hasattr(BaseData, "func1"))

        with self.assertRaisesRegex(
            AttributeError,
            "Neither 'BaseData' nor 'FakeDataClass' " "objects have attribute 'test'",
        ):
            data.test

        # validate it will take BaseData attribute over the data attribute
        self.assertFalse(data.options)

        # validate will auto call the data function if it doesn't exist in
        # BaseData
        self.assertEqual("success", data.func1())

    def test_file_encoding(self):
        """
        Determine if the file encoding can be automatically identified
        """
        for input_file in self.file_or_buf_list:
            # do not test StringIO, avro, parquet
            if isinstance(input_file["path"], StringIO) or input_file["data_type"] in [
                "avro",
                "parquet",
            ]:
                continue

            data = BaseData(input_file_path=input_file["path"], data=None, options={})
            self.assertEqual(
                input_file["encoding"].lower(),
                data.file_encoding.lower(),
                input_file["path"],
            )

        # test when data is specified without input_file_object
        file_encoding = locale.getpreferredencoding(False)
        if file_encoding.lower() in ["ascii", "ansi_x3.4-1968"]:
            file_encoding = "utf-8"
        data = BaseData(input_file_path=None, data=[], options={})
        self.assertEqual(file_encoding, data.file_encoding)
