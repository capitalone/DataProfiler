"""Contains class to save and load json data."""
import json
import re
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from six import StringIO

from .._typing import JSONType
from . import data_utils
from .base_data import BaseData
from .filepath_or_buffer import FileOrBufferHandler
from .structured_mixins import SpreadSheetDataMixin


class JSONData(SpreadSheetDataMixin, BaseData):
    """SpreadsheetData class to save and load spreadsheet data."""

    data_type: str = "json"

    def __init__(
        self,
        input_file_path: Optional[str] = None,
        data: Optional[Union[str, pd.DataFrame]] = None,
        options: Optional[Dict] = None,
    ):
        """
        Initialize Data class for loading datasets of type JSON.

        Can be specified by passing in memory data or via a file path.
        Options pertaining the JSON may also be specified using the
        options dict parameter.
        Possible Options::

            options = dict(
                data_format= type: str, choices: "dataframe", "records", "json",
                 "flattened_dataframe"
                selected_keys= type: list(str)
                payload_keys= type: Union[str, list(str)]
            )


        data_format: user selected format in which to return data
        can only be of specified types
        selected_keys: keys being selected from the entire dataset
        payload_keys: list of dictionary keys that determine the payload

        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        options = self._check_and_return_options(options)
        BaseData.__init__(self, input_file_path, data, options)
        SpreadSheetDataMixin.__init__(self, input_file_path, data, options)

        # 'Private' properties
        #  _data_formats: dict containing data_formats (key) and function
        #                 calls (values) which take self._data and convert it
        #                 into the desired data_format for output.
        #  _selected_data_format: user selected format in which to return data
        #                         can only be of types in _data_formats
        #  _selected_keys: keys being selected from the entire dataset
        #  _payload_keys: (list of) dictionary key(s) that determines the payload

        self._data_formats["records"] = self._get_data_as_records
        self._data_formats["json"] = self._get_data_as_json
        self._data_formats[
            "flattened_dataframe"
        ] = self._get_data_as_flattened_dataframe
        self._selected_data_format: str = options.get(
            "data_format", "flattened_dataframe"
        )
        self._payload_keys: List[str] = options.get("payload_keys", ["data", "payload"])
        if not isinstance(self._payload_keys, list):
            self._payload_keys = [self._payload_keys]
        self._key_separator: str = options.get("key_separator", ".")
        self._selected_keys: Optional[List[str]] = options.get("selected_keys", list())
        self._metadata: Optional[pd.DataFrame] = None
        if data is not None:
            self._load_data(data)

    @property
    def selected_keys(self) -> Optional[List[str]]:
        """Return selected keys."""
        return self._selected_keys

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Return a data frame that contains the metadata."""
        if self._metadata is None or self._metadata.empty:
            warnings.warn("No metadata was detected.")
        return self._metadata

    @property
    def data_and_metadata(self) -> Optional[pd.DataFrame]:
        """Return a data frame that joins the data and the metadata."""
        data = self.data
        if self._metadata is not None and not self._metadata.empty:
            data = [self._metadata, data]
            data = pd.concat(data, axis=1)
        return data

    @property
    def is_structured(self):
        """Determine compatibility with StructuredProfiler."""
        return self.data_format in ["dataframe", "flattened_dataframe"]

    def _find_data(self, json_data, path=""):
        """
        Find all the col headers/data in Json and return them as list.

        :param json_data: the json data or a subset of the json data
        :type json_data: Union[list, dict, str]
        :param path: path of keys to the json_data
        :type path: str
        :return: list of dicts of {column headers: values}
        """
        if path != "" and path[-len(self._key_separator) :] != self._key_separator:
            path = path + self._key_separator

        list_of_dict = []
        if isinstance(json_data, dict):
            for key in json_data:
                if isinstance(json_data[key], dict) or isinstance(json_data[key], list):
                    list_of_dict = list_of_dict + self._find_data(
                        json_data[key], path + key
                    )
                else:
                    list_of_dict = list_of_dict + [{path + key: json_data[key]}]
        elif isinstance(json_data, list):
            if all(isinstance(x, dict) for x in json_data):
                for key in json_data:
                    list_of_dict = list_of_dict + self._find_data(key, path)
            else:
                list_of_dict = list_of_dict + [
                    {path[: -len(self._key_separator)]: json_data}
                ]
        else:
            list_of_dict = list_of_dict + [
                {path[: -len(self._key_separator)]: json_data}
            ]
        return list_of_dict

    def _coalesce_dicts(self, list_of_dicts):
        """
        Merge all the dictionaries into as few dictionaries as possible.

        :param list_of_dicts: the list of dictionaries with one item in each dict
        to be coalesced
        :type list_of_dicts: list(dict)
        :return: Coalesced list of dictionaries
        """
        coalesced_list_of_dicts = [{}]
        for item in list_of_dicts:
            found = False
            for dict_items in coalesced_list_of_dicts:
                if list(item.keys())[0] not in dict_items:
                    dict_items.update(item)
                    found = True
                    break
            if not found:
                coalesced_list_of_dicts.append(item)
        return coalesced_list_of_dicts

    def _get_data_as_flattened_dataframe(self, json_lines):
        """
        Load the data when in a JSON format from a data stream.

        (Nested lists of dictionaries and a key value for a payload.)

        :param json_lines: json format list of dicts or dict
        :type json_lines: Union[dict, list(dict)]
        :return: Pandas.DataFrame from a flattened json
        """
        if isinstance(json_lines, pd.DataFrame):
            return json_lines
        payload_data = None
        if isinstance(json_lines, dict):
            # Glean Payload Data
            found_payload_key = None
            payloads = {}
            for payload_key in self._payload_keys:
                if payload_key in json_lines.keys():
                    payload_data = json_lines[payload_key]
                    if isinstance(payload_data, dict):
                        payload_data = self._find_data(payload_data)
                        payload_data = self._coalesce_dicts(payload_data)
                    payload_data, original_df_dtypes = data_utils.json_to_dataframe(
                        json_lines=payload_data,
                        selected_columns=self.selected_keys,
                        read_in_string=False,
                    )
                    for column in payload_data.columns:
                        payload_data.rename(
                            columns={
                                column: payload_key + self._key_separator + str(column)
                            },
                            inplace=True,
                        )
                    payloads[payload_key] = payload_data

            max_payload_length = 0
            for payload in payloads:
                if len(payloads[payload]) > max_payload_length:
                    payload_data = payloads[payload]
                    max_payload_length = len(payloads[payload])
                    found_payload_key = payload

            # Get the non-payload data
            flattened_json = []
            for key in json_lines:
                if key != found_payload_key:
                    flattened_json = flattened_json + self._find_data(
                        json_lines[key], path=key
                    )

            # Coalesce the data together
            json_lines = self._coalesce_dicts(flattened_json)

        data, original_df_dtypes = data_utils.json_to_dataframe(
            json_lines=json_lines,
            selected_columns=self.selected_keys,
            read_in_string=False,
        )
        self._original_df_dtypes = original_df_dtypes

        if payload_data is not None:
            self._metadata = data
            data = payload_data

        return data

    def _load_data_from_str(self, data_as_str: str) -> JSONType:
        """
        Load the data from a string.

        :param data_as_str: data in string format.
        :type data_as_str: str
        :return: JSONType
        """
        data: JSONType
        try:
            data = json.loads(data_as_str)
        except json.JSONDecodeError:
            data_generator = data_utils.data_generator(data_as_str.splitlines())
            data = data_utils.read_json(
                data_generator=data_generator,
                selected_columns=self.selected_keys,
                read_in_string=False,
            )
        return data

    def _load_data_from_file(self, input_file_path: str) -> JSONType:
        """
        Load the data from a file.

        :param input_file_path: file path to file being loaded.
        :type input_file_path: str
        :return: JSONType
        """
        with FileOrBufferHandler(
            input_file_path, "r", encoding=self.file_encoding
        ) as input_file:
            data: JSONType
            try:
                data = json.load(input_file)
            except (json.JSONDecodeError, UnicodeDecodeError):
                input_file.seek(0)
                data = data_utils.read_json(
                    data_generator=input_file,
                    selected_columns=self.selected_keys,
                    read_in_string=False,
                )
            return data

    def _get_data_as_records(self, data: Union[pd.DataFrame, Dict, List]) -> List[str]:
        """
        Extract the data as a record format.

        :param data: raw data
        :type data: Union[pd.DataFrame, Dict, List])
        :return: dataframe in record format
        """
        data = self._get_data_as_df(data)
        data = data.to_dict(orient="records", into=OrderedDict)
        for i, sample in enumerate(data):
            sample = self._convert_flat_to_nested_cols(sample)
            data[i] = json.dumps(
                self._convert_flat_to_nested_cols(sample), ensure_ascii=False
            )
        return super(JSONData, self)._get_data_as_records(data)

    def _get_data_as_json(self, data: Union[pd.DataFrame, Dict, List]) -> List[str]:
        """
        Extract the data as a json format.

        :param data: raw data
        :type data: Union[pd.DataFrame, Dict, List])
        :return: dataframe in json format
        """
        data = self._get_data_as_df(data)
        data = data.to_json(orient="records")
        char_per_line = min(len(data), self.SAMPLES_PER_LINE_DEFAULT)
        return list(map("".join, zip(*[iter(data)] * char_per_line)))

    def _get_data_as_df(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """
        Extract the data as pandas formats it.

        :param data: raw data
        :type data: Union[pd.DataFrame, Dict, List])
        :return: pandas dataframe
        """
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            data = [data]
        data, original_df_dtypes = data_utils.json_to_dataframe(
            json_lines=data, selected_columns=self.selected_keys, read_in_string=False
        )
        self._original_df_dtypes = original_df_dtypes
        return data

    @classmethod
    def _convert_flat_to_nested_cols(cls, dic: Dict, separator: str = ".") -> Dict:
        """
        Convert a flat dict to nested dict.

        Ex:

        dict = {
            'a.b': 'ab',
            'a.c': 'ac',
            'a.d.f': 'adf'
        }

        will be converted into:

        {'a': {'b': 'ab', 'c': 'ac', 'd': {'f': 'adf'}}}

        :param dic: dictionary to be nested
        :type dic: dict
        :param separator: separator of the nested keys
        :type separator: str
        :return:
        """
        for key in list(dic.keys()):
            if not isinstance(key, str):
                continue
            if separator in key:
                new_key, nested_key = key.split(separator, 1)
                new_value = dic.get(new_key, {})
                new_value = {} if new_value in [None, np.nan, "nan"] else new_value
                new_value[nested_key] = dic[key]
                dic.pop(key, None)
                new_value = cls._convert_flat_to_nested_cols(new_value, separator)
                dic[new_key] = new_value
        return dic

    @classmethod
    def is_match(
        cls, file_path: Union[str, StringIO], options: Optional[Dict] = None
    ) -> bool:
        """
        Test whether first 1000 lines of file has valid JSON format or not.

        At least 60 percent of the lines in the first 1000
        lines have to be valid json.

        :param file_path: path to the file to be examined
        :type file_path: str
        :param options: json read options
        :type options: dict
        :return: is file a json file or not
        :rtype: bool
        """
        valid_json_line_count = 0
        total_line_count = 0

        if options is None:
            options = dict()

        file_encoding = None
        if not isinstance(file_path, StringIO):
            file_encoding = data_utils.detect_file_encoding(file_path=file_path)

        with FileOrBufferHandler(file_path, "r", encoding=file_encoding) as data_file:
            try:
                json.load(data_file)
                return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                data_file.seek(0)
            json_identifier_re = re.compile(r"(:|\[)")
            for k in range(1000):
                total_line_count += 1
                try:
                    raw_line = data_file.readline()
                    if not raw_line:
                        break
                    if (
                        json_identifier_re.search(raw_line) is not None
                    ):  # Ensure can be JSON
                        json.loads(raw_line)  # Check load
                        valid_json_line_count += 1
                except UnicodeDecodeError:
                    return False
                except ValueError:
                    continue

        ratio_of_valid_json_line = float(valid_json_line_count) / total_line_count

        if ratio_of_valid_json_line >= 0.5:
            return True
        else:
            return False

    def reload(
        self,
        input_file_path: Optional[str] = None,
        data: Optional[Union[str, pd.DataFrame]] = None,
        options: Optional[Dict] = None,
    ) -> None:
        """
        Reload the data class with a new dataset.

        This erases all existing data/options and replaces it
        with the input data/options.

        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        self._selected_keys = None
        super(JSONData, self).reload(input_file_path, data, options)
        self.__init__(self.input_file_path, data, options)  # type: ignore
