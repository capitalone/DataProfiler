import pyarrow.parquet as pq

from . import data_utils
from .base_data import BaseData
from .structured_mixins import SpreadSheetDataMixin


class ParquetData(SpreadSheetDataMixin, BaseData):
    """
    SpreadsheetData class to save and load spreadsheet data
    """

    data_type = 'parquet'

    def __init__(self, input_file_path=None, data=None, options=None):
        """
        Data class for loading datasets of type PARQUET. Can be specified by
        passing in memory data or via a file path. Options pertaining the
        PARQUET may also be specified using the options dict parameter.
        Possible Options::
        
            options = dict(
                data_format= type: str, choices: "dataframe", "records", "json"
                selected_columns= type: list(str)
                header= type: any
            )
        
        data_format: user selected format in which to return data
        can only be of specified types
        selected_columns: columns being selected from the entire dataset

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
        #  _selected_columns: columns being selected from the entire dataset
        self._data_formats["records"] = self._get_data_as_records
        self._data_formats["json"] = self._get_data_as_json
        self._selected_data_format = options.get("data_format", "dataframe")
        self._selected_columns = options.get("selected_columns", list())

        if data is not None:
            self._load_data(data)

    @property
    def selected_columns(self):
        return self._selected_columns

    def _load_data_from_str(self, data_as_str):
        data_generator = data_utils.data_generator(data_as_str.splitlines())
        data, original_df_dtypes = data_utils.read_json_df(
            data_generator=data_generator,
            read_in_string=True
        )
        self._original_df_dtypes = original_df_dtypes
        return data

    def _load_data_from_file(self, input_file_path):
        data, original_df_dtypes = data_utils.read_parquet_df(
            input_file_path,
            self.selected_columns,
            read_in_string=True
        )
        self._original_df_dtypes = original_df_dtypes
        return data

    def _get_data_as_records(self, data):
        # split into row samples separate by `\n`
        data = data.to_json(orient="records", lines=True)
        data = data.splitlines()
        return super(ParquetData, self)._get_data_as_records(data)

    def _get_data_as_json(self, data):
        data = data.to_json(orient="records")
        chars_per_line = min(len(data), self.SAMPLES_PER_LINE_DEFAULT)
        return list(map(''.join, zip(*[iter(data)] * chars_per_line)))

    @classmethod
    def is_match(cls, file_path, options=None):
        """
        Test the given file to check if the file has valid
        Parquet format or not.
        
        :param file_path: path to the file to be examined
        :type file_path: str
        :param options: parquet read options
        :type options: dict
        :return: is file a parquet file or not
        :rtype: bool
        """
        try:
            pfile = pq.ParquetFile(file_path)
            is_valid_parquet = True
        except:
            is_valid_parquet = False

        return is_valid_parquet

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
        super(ParquetData, self).reload(input_file_path, data, options)
        self.__init__(input_file_path, data, options)
