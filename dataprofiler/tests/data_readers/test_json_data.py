from __future__ import absolute_import, print_function

import os
import unittest
from io import BytesIO, StringIO

from dataprofiler.data_readers import json_data
from dataprofiler.data_readers.data import Data, JSONData

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestNestedJSON(unittest.TestCase):
    def test_flat_to_nested_json(self):
        dic = {"a.b": "ab", "a.c": "ac", "a.d.f": "adf", "b": "b", 1: 3}

        converted_dic = json_data.JSONData._convert_flat_to_nested_cols(dic)
        self.assertTrue(
            converted_dic
            == {"a": {"b": "ab", "c": "ac", "d": {"f": "adf"}}, "b": "b", 1: 3}
        )


class TestJSONDataClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_file_path = None
        cls.output_file_path = None
        cls.ss = None

        test_dir = os.path.join(test_root_path, "data")
        cls.input_file_names = [
            dict(
                path=os.path.join(test_dir, "json/iris-utf-8.json"),
                encoding="utf-8",
                count=150,
            ),
            dict(
                path=os.path.join(test_dir, "json/iris-utf-16.json"),
                encoding="utf-16",
                count=150,
            ),
            dict(
                path=os.path.join(test_dir, "json/honeypot"), encoding="utf-8", count=14
            ),
            dict(
                path=os.path.join(
                    test_dir, "json/honeypot_intentially_mislabeled_file.csv"
                ),
                encoding="utf-8",
                count=14,
            ),
            dict(
                path=os.path.join(
                    test_dir, "json/honeypot_intentially_mislabeled_file.parquet"
                ),
                encoding="utf-8",
                count=14,
            ),
            dict(
                path=os.path.join(test_dir, "json/simple-list.json"),
                encoding="utf-8",
                count=3,
            ),
        ]

        cls.buffer_list = []
        for input_file in cls.input_file_names:
            # add StringIO
            buffer_info = input_file.copy()
            with open(input_file["path"], "r", encoding=input_file["encoding"]) as fp:
                buffer_info["path"] = StringIO(fp.read())
            cls.buffer_list.append(buffer_info)

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
        Determine if the json file can be automatically identified from
        byte stream or stringio stream or filepath
        """
        for input_file in self.file_or_buf_list:
            self.assertTrue(JSONData.is_match(input_file["path"]))

    def test_json_file_identification(self):
        """
        Determine if the json file can be automatically identified
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"])
            self.assertEqual(input_data_obj.data_type, "json")

    def test_specifying_data_type(self):
        """
        Determine if the json file can be loaded with manual data_type setting
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"], data_type="json")
            self.assertEqual(input_data_obj.data_type, "json")

    def test_reload_data(self):
        """
        Determine if the json file can be reloaded
        """
        for input_file in self.file_or_buf_list:
            input_data_obj = Data(input_file["path"])
            input_data_obj.reload(input_file["path"])
            self.assertEqual(input_data_obj.data_type, "json")
            self.assertEqual(input_file["path"], input_data_obj.input_file_path)

    def test_json_from_string(self):
        """
        Determine if the json file can be loaded with manual data_type setting
        """
        passing_json_strings = [
            "[]",
            "{}",
            "[1, 2, 3]",
            '[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 1, "b":2}]',
        ]
        failing_json_strings = [
            dict(
                value="[1,[1]]",
                error="Only JSON which represents structured data is "
                "supported for this data type (i.e. list-dicts).",
            ),
            dict(
                value='[{"a": 1}, 2, [3]]',
                error="Only JSON which represents structured data is "
                "supported for this data type (i.e. list-dicts).",
            ),
            dict(value="{", error="No JSON data could be read from these data."),
        ]
        for json_string in passing_json_strings:
            # in memory data must specify the data_type category
            input_data_obj = Data(data=json_string, data_type="json")
            self.assertEqual(input_data_obj.data_type, "json")

        for json_string in failing_json_strings:
            # in memory data must specify the data_type category
            with self.assertRaises(ValueError) as assert_raised:
                Data(data=json_string["value"], data_type="json").data
            self.assertEqual(str(assert_raised.exception), json_string["error"])

    def test_data_formats(self):
        """
        Determine if the json file data_formats can be used
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

    def test_len_data(self):
        """
        Validate that length called on JSONData is appropriately determining the
        length value.
        """

        for input_file in self.file_or_buf_list:
            data = Data(input_file["path"])
            self.assertEqual(input_file["count"], len(data), msg=input_file["path"])
            self.assertEqual(input_file["count"], data.length, msg=input_file["path"])

    def test_flattened_dataframe_format_with_no_payload(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/simple.json")

        simple = Data(
            input_file_name,
            options={"data_format": "flattened_dataframe", "payload_keys": "data"},
        )

        self.assertEqual(3, len(simple.data_and_metadata.columns))
        self.assertEqual(2, len(simple.data.columns))
        self.assertEqual(1, len(simple.metadata.columns))

        simple = Data(
            input_file_name,
            options={
                "data_format": "flattened_dataframe",
                "payload_keys": "no_data_key_test",
            },
        )

        self.assertEqual(3, len(simple.data_and_metadata.columns))
        self.assertEqual(3, len(simple.data.columns))
        with self.assertWarnsRegex(UserWarning, "No metadata was detected."):
            self.assertIsNone(simple.metadata)

    def test_key_separator_in_flattened_dataframe_format(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/simple.json")

        simple = Data(input_file_name, options={"key_separator": "~~~"})
        expected_columns = [
            "data~~~list_of_things~~~id",
            "data~~~list_of_things~~~tags",
        ]

        self.assertListEqual(expected_columns, list(simple.data.columns))

    def test_complex_nested_json_in_flattened_dataframe_format(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/complex_nested.json")

        complex = Data(input_file_name)
        self.assertEqual(8, len(complex.data.columns))
        self.assertEqual(
            "Depression", complex.data["payload.Lion.medical_condition"][0]
        )

        self.assertEqual(11, len(complex.data_and_metadata.columns))
        self.assertEqual("Frodo", complex.data_and_metadata["meta.creator"][0])

        self.assertEqual(3, len(complex.metadata.columns))
        self.assertEqual("Frodo", complex.data_and_metadata["meta.creator"][0])

    def test_list_of_dictionaries_in_flattened_dataframe_format(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/iris-utf-8.json")

        simple = Data(input_file_name)
        self.assertEqual(6, len(simple.data.columns))
        self.assertEqual(150, len(simple.data))

    def test_flattened_dataframe_format(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/math.json")

        math = Data(input_file_name)
        self.assertIn(
            "meta.view.columns.cachedContents.largest", math.data_and_metadata.columns
        )
        self.assertEqual(
            math.metadata["meta.view.columns.cachedContents.largest"][9], "102188"
        )
        self.assertIn("data.22", math.data.columns)
        self.assertEqual(math.data["data.22"][167], "77.9")

    def test_payload_key(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/hits.json")

        hits = Data(input_file_name, options={"payload_keys": "hits"})
        self.assertIn("hits._highlightResult.story_url.value", hits.data.columns)
        self.assertNotIn("hits._highlightResult.story_url.value", hits.metadata.columns)
        self.assertNotIn("processingTimeMS", hits.data.columns)
        self.assertIn("processingTimeMS", hits.metadata.columns)

        self.assertIn("processingTimeMS", hits.data_and_metadata.columns)
        self.assertIn(
            "hits._highlightResult.story_url.value", hits.data_and_metadata.columns
        )

    def test_find_data(self):
        JSONDataObject = json_data.JSONData()

        data = {
            "Top_level": [
                {
                    "mid_level_one": {
                        "third_level_one": "badabing",
                        "third_level_two": "badaboom",
                        "third_level_three": "hello",
                    }
                },
                {"mid_level_two": "hello1"},
                {"mid_level_three": "hello2"},
            ]
        }
        expected_list = [
            {"Top_level.mid_level_one.third_level_one": "badabing"},
            {"Top_level.mid_level_one.third_level_two": "badaboom"},
            {"Top_level.mid_level_one.third_level_three": "hello"},
            {"Top_level.mid_level_two": "hello1"},
            {"Top_level.mid_level_three": "hello2"},
        ]

        self.assertListEqual(expected_list, JSONDataObject._find_data(data))

    def test_flattened_dataframe_format_with_dual_payload(self):
        test_dir = os.path.join(test_root_path, "data")
        input_file_name = os.path.join(test_dir, "json/dual_payloads.json")

        dual_payload = Data(
            input_file_name, options={"data_format": "flattened_dataframe"}
        )
        # Make sure the larger payload is selected
        self.assertIn("payload.bigger_list_of_things.id", dual_payload.data.columns)
        self.assertEqual(2, len(dual_payload.data.columns))

    def test_is_structured(self):
        # Default construction
        data = JSONData()
        self.assertTrue(data.is_structured)

        # With option specifying dataframe as data_format
        data = JSONData(options={"data_format": "dataframe"})
        self.assertTrue(data.is_structured)

        # With option specifying flattened_dataframe as data_format
        data = JSONData(options={"data_format": "flattened_dataframe"})
        self.assertTrue(data.is_structured)

        # With option specifying records as data_format
        data = JSONData(options={"data_format": "records"})
        self.assertFalse(data.is_structured)

        # With option specifying json as data_format
        data = JSONData(options={"data_format": "json"})
        self.assertFalse(data.is_structured)


if __name__ == "__main__":
    unittest.main()
