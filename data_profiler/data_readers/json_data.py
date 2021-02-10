from collections import OrderedDict
import json

import numpy as np

from . import data_utils
from .base_data import BaseData
from .structured_mixins import SpreadSheetDataMixin


class JSONData(SpreadSheetDataMixin, BaseData):
    """
    SpreadsheetData class to save and load spreadsheet data
    """

    data_type = 'json'

    def __init__(self, input_file_path=None, data=None, options=None):
        """
        Data class for loading datasets of type JSON. Can be specified by
        passing in memory data or via a file path. Options pertaining the JSON
        may also be specified using the options dict parameter.
        Possible Options::
        
            options = dict(
                data_format= type: str, choices: "dataframe", "records", "json"
                selected_keys= type: list(str)
            )

        
        data_format: user selected format in which to return data
        can only be of specified types
        selected_keys: keys being selected from the entire dataset

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
        self._data_formats["records"] = self._get_data_as_records
        self._data_formats["json"] = self._get_data_as_json
        self._selected_data_format = options.get("data_format", "dataframe")
        self._selected_keys = options.get("selected_keys", list())

        if data is not None:
            self._load_data(data)

    @property
    def selected_keys(self):
        return self._selected_keys

    def _load_data_from_json_format(self, json_lines):
        """
        Loads the data when in a JSON format such as
        `json_lines = [dict(a=1), dict(a=2)]`
        
        :param json_lines: json format list of dicts or dict
        :return:
        """
        if isinstance(json_lines, dict):
            json_lines = [json_lines]
        data, original_df_dtypes = data_utils.json_to_dataframe(
            json_lines=json_lines,
            selected_columns=self.selected_keys,
            read_in_string=False
        )
        return data, original_df_dtypes

    def _load_data_from_str(self, data_as_str):
        """
        Loads the data from a string.
        
        :param data_as_str: data in string format.
        :type data_as_str: str
        :return:
        """
        try:
            json_lines = json.loads(data_as_str)
            data, original_df_dtypes = \
                self._load_data_from_json_format(json_lines)
        except json.JSONDecodeError:
            data_generator = data_utils.data_generator(data_as_str.splitlines())
            data, original_df_dtypes = data_utils.read_json_df(
                data_generator=data_generator,
                selected_columns=self.selected_keys,
                read_in_string=False
            )
        self._original_df_dtypes = original_df_dtypes
        return data

    def _load_data_from_file(self, input_file_path):
        """
        Loads the data from a file.
        
        :param input_file_path: file path to file being loaded.
        :type input_file_path: str
        :return:
        """
        self._file_encoding = data_utils.detect_file_encoding(input_file_path)
        with open(input_file_path, encoding=self.file_encoding) as input_file:
            try:
                json_lines = json.load(input_file)
                data, original_df_dtypes = \
                    self._load_data_from_json_format(json_lines)
            except (json.JSONDecodeError, UnicodeDecodeError):
                input_file.seek(0)
                data_generator = data_utils.generator_on_file(input_file)
                data, original_df_dtypes = data_utils.read_json_df(
                    data_generator=data_generator,
                    selected_columns=self.selected_keys,
                    read_in_string=False
                )
            self._original_df_dtypes = original_df_dtypes
            return data

    def _get_data_as_records(self, data):
        data = data.to_dict(orient="records", into=OrderedDict)
        for i, sample in enumerate(data):
            data[i] = json.dumps(
                self._convert_flat_to_nested_cols(sample), ensure_ascii=False
            )
        return super(JSONData, self)._get_data_as_records(data)

    def _get_data_as_json(self, data):
        data = data.to_json(orient="records")
        char_per_line = min(len(data), self.SAMPLES_PER_LINE_DEFAULT)
        return list(map(''.join, zip(*[iter(data)] * char_per_line)))

    @classmethod
    def _convert_flat_to_nested_cols(cls, dic, separator='.'):
        """
        Converts a flat dict to nested dic. Example

        dic = {
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
            if separator in key:
                new_key, nested_key = key.split(separator, 1)
                new_value = dic.get(new_key, {})
                new_value = {} if new_value in [None, np.nan, 'nan'] else new_value
                new_value[nested_key] = dic[key]
                dic.pop(key, None)
                new_value = cls._convert_flat_to_nested_cols(
                    new_value, separator
                )
                dic[new_key] = new_value
        return dic

    @classmethod
    def is_match(cls, file_path, options=None):
        """
        Test the first 1000 lines of a given file to check if the file has valid
        JSON format or not. At least 60 percent of the lines in the first 1000
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

        file_encoding = data_utils.detect_file_encoding(file_path=file_path)
        with open(file_path, 'r', encoding=file_encoding) as data_file:
            try:
                json.load(data_file)
                return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                data_file.seek(0)

            for k in range(1000):
                total_line_count += 1
                try:
                    raw_line = data_file.readline()
                    if not raw_line:
                        break                        
                    if raw_line.find(":") >= 0: # Ensure can be JSON
                        json.loads(raw_line) # Check load
                        valid_json_line_count += 1
                except UnicodeDecodeError:
                    return False
                except ValueError:
                    continue
            
        ratio_of_valid_json_line = float(
            valid_json_line_count) / total_line_count
        
        if ratio_of_valid_json_line >= 0.5:
            return True
        else:
            return False

    def reload(self, input_file_path=None, data=None, options=None):
        """
        Reload the data class with a new dataset. This erases all existing
        data/options and replaces it with the input data/options.
        
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
        self.__init__(input_file_path, data, options)
