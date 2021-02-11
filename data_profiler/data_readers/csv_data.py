import re
import csv
from itertools import islice
from six import StringIO

import numpy as np

from . import data_utils
from .base_data import BaseData
from .json_data import JSONData
from .parquet_data import ParquetData
from .avro_data import AVROData
from .structured_mixins import SpreadSheetDataMixin


class CSVData(SpreadSheetDataMixin, BaseData):
    """
    SpreadsheetData class to save and load spreadsheet data
    """

    data_type = 'csv'

    def __init__(self, input_file_path=None, data=None, options=None):
        """
        Data class for loading datasets of type CSV. Can be specified by passing
        in memory data or via a file path. Options pertaining the CSV may also
        be specified using the options dict parameter.
        Possible Options::
        
            options = dict(
                delimiter= type: str
                data_format= type: str, choices: "dataframe", "records"
                selected_columns= type: list(str)
                header= type: any
            )
        
        delimiter: delimiter used to decipher the csv input file
        data_format: user selected format in which to return data
        can only be of specified types
        selected_columns: columns being selected from the entire dataset
        header: any information pertaining to the file header.

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
        #  _delimiter: delimiter used to decipher the csv input file
        #  _selected_columns: columns being selected from the entire dataset
        #  _header: any information pertaining to the file header.
        self._data_formats["records"] = self._get_data_as_records
        self._selected_data_format = options.get("data_format", "dataframe")
        self._delimiter = options.get("delimiter", None)
        self._selected_columns = options.get("selected_columns", list())
        self._header = options.get("header", 0)  # 0 for pandas to auto get
        self._checked_header = "header" in options
        self._default_delimiter = ','
        
        if data is not None:
            self._load_data(data)
            if not self._delimiter:
                self._delimiter = self._default_delimiter

    @property
    def selected_columns(self):
        return self._selected_columns

    @property
    def delimiter(self):
        return self._delimiter

    @property
    def header(self):
        return self._header

    @staticmethod
    def _determine_delimiter_of_str(data_as_str):
        """
        Automatically checks for what delimiter exists in a text document.
        
        :param data_as_str: Single string containing rows (lines seperated by "\n")
        :type data_as_str: str
        :return: Delimiter, if none can be found None is returned
        :rtype: str or None
        """
        sniffer = csv.Sniffer()
        sniffer.preferred = [',', '\t', ';'] # removes ' ', ';' from preferred
        try:
            dialect = sniffer.sniff(data_as_str)
        except csv.Error as exc:
            return None
        return dialect.delimiter

    @staticmethod
    def _determine_has_header(data_as_str):
        """Automatically checks if the string has a header."""
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(data_as_str)
        except csv.Error:
            has_header = 0
        return 0 if has_header else None

    def _load_data_from_str(self, data_as_str):
        """Loads the data into memory from the str."""
        if not self._delimiter:
            self._delimiter = self._determine_delimiter_of_str(data_as_str)
        data_buffered = StringIO(data_as_str)
        self._determine_has_header(data_as_str)
        return data_utils.read_csv_df(
            data_buffered,
            self.delimiter, self.header, self.selected_columns,
            read_in_string=True
        )

    def _load_data_from_file(self, input_file_path):
        """Loads the data into memory from the file."""
        self._file_encoding = data_utils.detect_file_encoding(input_file_path)
        if not self._delimiter or not self._checked_header:
            with open(input_file_path, encoding=self.file_encoding) as csvfile:
                num_lines = 5
                data_as_str = ''.join(islice(csvfile, num_lines))
            if not self._delimiter:
                self._delimiter = self._determine_delimiter_of_str(data_as_str)
            if not self._header:
                self._header = self._determine_has_header(data_as_str)
                self._checked_header = True
        return data_utils.read_csv_df(
            input_file_path,
            self.delimiter, self.header, self.selected_columns,
            read_in_string=True,
            encoding=self.file_encoding
        )

    def _get_data_as_records(self, data):
        sep = self.delimiter if self.delimiter else self._default_delimiter 
        data = data.to_csv(sep=sep, index=False)
        data = data.splitlines()
        return super(CSVData, self)._get_data_as_records(data)

    @classmethod
    def is_match(cls, file_path, options=None):
        """
        Test the first 1000 lines of a given file to check if the file has valid
        delimited format or not.
        
        :param file_path: path to the file to be examined
        :type file_path: str
        :param options: delimiter read options dict(delimiter=",")
        :type options: dict
        :return: is file a csv file or not
        :rtype: bool
        """
        
        if JSONData.is_match(file_path) or ParquetData.is_match(file_path)\
                or AVROData.is_match(file_path):
            return False
        
        file_encoding = data_utils.detect_file_encoding(file_path=file_path)
        delimiter = options.get("delimiter", None)
        
        header = options.get("header", None)
        if not delimiter or not header:
            
            data_as_str = None
            with open(file_path, encoding=file_encoding) as csvfile:
                num_lines = 5
                data_as_str = ''.join(islice(csvfile, num_lines))
                
            # Checks if delimiter is a space; If so, returns false
            if not delimiter:
                delimiter = cls._determine_delimiter_of_str(data_as_str)
            
            if header is None:
                options.update(header=cls._determine_has_header(data_as_str))
        
        max_line_count = 1000
        min_line_count = 2
        line_count = 0
        delimiter_count = dict()

        # ignore delimiters inside of quotes
        base_regex = "(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"        
        delimiter_regex = re.compile(re.escape(str(delimiter)) + base_regex)
        
        # Count the possible delimiters 
        with open(file_path, encoding=file_encoding) as f:
            for line in f:
                line_count += 1
                
                # Find the location(s) where each delimiter was detected
                delimiter_locs = [i.start() for i in re.finditer(delimiter_regex, line)]
                if delimiter:
                    # Count all locations that aren't the last entry
                    # This can cause errors, if you could have a sparse end column
                    count = 0
                    for loc in delimiter_locs:
                        # Reason: len(line) - 2; -1 to start count 0 & -1 for last loc
                        if loc != len(line)-2:
                            count += 1
                else:
                    # If no delimiter, see if spaces are regular intervals
                    count = line.count(" ")
                    
                if count in delimiter_count:
                    delimiter_count[count] = delimiter_count[count] + 1
                else:
                    delimiter_count[count] = 1

                if line_count >= max_line_count:
                    break
                    
        if line_count <= min_line_count:
            return False
        
        delimiter_count_values = np.array(list(delimiter_count.values()))
        count_percent = delimiter_count_values / np.sum(delimiter_count_values)
        if not count_percent.size:
            return False
        
        max_count_index = count_percent.argmax()
        max_count_value = list(delimiter_count.keys())[max_count_index]
        max_count_percent = count_percent[max_count_index]
        
        # Infered the file was a CSV
        if ((max_count_value > 0 or delimiter is None) \
            and max_count_percent >= 0.8):
            options.update(delimiter=delimiter)
            return True
        
        # Assume not a CSV
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
        if not options:
            options = dict()
        options.update(delimiter=self.delimiter)
        super(CSVData, self).reload(input_file_path, data, options)
        self.__init__(input_file_path, data, options)
