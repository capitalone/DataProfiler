import re
import csv
import re
from itertools import islice
from six import StringIO

import random
import dateutil
from collections import Counter

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
        self._quotechar = options.get("quotechar", None)
        self._selected_columns = options.get("selected_columns", list())
        self._header = options.get("header", 'auto')
        self._checked_header = "header" in options
        self._default_delimiter = ','
        self._default_quotechar = '"'

        if data is not None:
            self._load_data(data)
            if not self._delimiter:
                self._delimiter = self._default_delimiter
            if not self._quotechar:
                self._quotechar = self._default_quotechar

    @property
    def selected_columns(self):
        return self._selected_columns

    @property
    def delimiter(self):
        return self._delimiter

    @property
    def quotechar(self):
        return self._quotechar

    @property
    def header(self):
        return self._header

    def _check_and_return_options(self, options):
        """
        Ensures options are valid inputs to the data reader.

        :param options: dictionary of options for the csv reader to validate
        :type options: dict
        :return: None
        """
        options = super()._check_and_return_options(options)
        
        if 'header' in options.keys():
            value = options["header"]
            if value != 'auto' and value is not None \
                and not (isinstance(value, int) and value > -1):
                raise ValueError('`header` must be one of following: auto, '
                                 'none for no header, or a non-negative '
                                 'integer for the row that represents the '
                                 'header (0 based index)')
        if 'delimiter' in options.keys():
            value = options["delimiter"]
            if value is not None and not isinstance(value, str):
                raise ValueError("'delimiter' must be a string or None")
        if 'data_format' in options.keys():
            value = options["data_format"]
            if value not in ["dataframe", "records"]:
                raise ValueError("'data_format' must be one of the following: "
                                 "'dataframe' or 'records' ") 
        if 'selected_columns' in options.keys():
            value = options["selected_columns"]
            if not isinstance(value, list):
                raise ValueError("'selected_columns' must be a list")
            for sc in value:
                if not isinstance(sc, str):
                    raise ValueError("'selected_columns' must be a list of strings")
        return options

    
    @staticmethod
    def _guess_delimiter_and_quotechar(data_as_str, quotechar=None,
                                       preferred=[',', '\t'], omitted=['"', "'"]):
        """
        Automatically checks for what delimiter exists in a text document.

        :param data_as_str: Single string containing rows (lines separated by "\n")
        :type data_as_str: str
        :param preferred: Delimiters that will be weighted and selected if possible
        :type preferred: list
        :param omitted: Delimiters that will be omitted from selection
        :type omitted: list
        :param quotechar: Character used for quotes
        :type quotechar: str
        :return: delimiter, if none can be found None is returned
        :rtype: str or None
        :return: quotechar, if none can be found None is returned
        :rtype: str or None
        """

        # Detect vocabulary (and count)
        vocab = Counter(data_as_str)
        if '\n' in vocab: vocab.pop('\n')
        for char in omitted+[quotechar]:
            if char in vocab:
                vocab.pop(char)
                
        # Sort vocabulary by count
        ordered_vocab = []
        sorted_keys = sorted(vocab, key=vocab.get, reverse=True)
        for c in sorted_keys:
            if c not in preferred:
                ordered_vocab.append(c)

        # Attempt to identify the quote character
        if not quotechar:            
            sniffer = csv.Sniffer()
            sniffer.preferred = preferred
            try:
                # NOTE: Pull the first element, the quote character
                quotechar = sniffer._guess_quote_and_delimiter(
                    data_as_str, ordered_vocab[:20])[0]
            except csv.Error as exc:
                quotechar = None
            if not quotechar or len(quotechar) == 0:
                quotechar = '"'
                
        # Evaluate vocab, reviewing rows and columns
        delimiter = None
        validated_proposed_delimiters = {}
        for proposed_delim in preferred+ordered_vocab[:20]:

            col_types = {}
            max_col_count = 0
            prior_col_count = None

            valid_delim_flag = False
            incorrect_delimiter_flag = False
            cell_type_safe_flag = True
            
            proposed_delim_type = data_utils.detect_cell_type(proposed_delim)
        
            # If large dataset, select first 25 rows then random sampling
            proposed_dataset = data_as_str.split('\n')
            if len(proposed_dataset) > 25:
                sample_count = min(int(0.1*(len(proposed_dataset)-25)), 1000)
                proposed_dataset = proposed_dataset[:25] \
                    + random.choices(proposed_dataset[:25], k=sample_count)

            # Reverse to start at bottom where likely the most columns
            # Fewer columns are okay, as long as earlier in file
            for row_idx in range(len(proposed_dataset)-1, -1, -1):

                row = proposed_dataset[row_idx]
                
                # Skip - extra split from "\n" with no data 
                if len(row)<=1 and row_idx==len(proposed_dataset)-1:
                    continue
                
                delimiter_regex = data_utils.get_delimiter_regex(proposed_delim, quotechar)
                proposed_cells = re.split(delimiter_regex, row)

                # Keep track of largest number of col's
                if prior_col_count is None:
                    prior_col_count = len(proposed_cells)
                if max_col_count == 0:
                    max_col_count = len(proposed_cells)

                # Ensure rows have same number of cols, if more than one col
                if len(proposed_cells) > prior_col_count: 
                    incorrect_delimiter_flag = True
                    break
                
                # Ensure there's more than one cell, if there's a delim
                if len(proposed_cells) > 1:
                    valid_delim_flag = True

                prior_col_count = len(proposed_cells)

                prior_cell_type = None # Checks for int/alpha values and delims     
                for col_id in range(len(proposed_cells)):
                    
                    proposed_cell = proposed_cells[col_id]
                    cell_type = data_utils.detect_cell_type(proposed_cell)
                    col_types[col_id] = cell_type

                    # Handle if alpha character are seperator                     
                    # NOTE: delimiter needs two ajoining cells to flag
                    if cell_type in ['str', 'none'] \
                       and prior_cell_type in ['str', 'none']:
                        if proposed_delim.isalpha():
                            cell_type_safe_flag = False
                            break
                        if proposed_delim == ' ' and 2 >= len(proposed_cells):
                            cell_type_safe_flag = False
                            break
                            
                    # Handle if integer characters are seperators
                    # NOTE: delimiter need one adjoining cell to flag
                    if proposed_delim_type == 'int' and cell_type=='int':
                        cell_type_safe_flag = False
                        break

                    prior_cell_type = cell_type
                
            if not incorrect_delimiter_flag and cell_type_safe_flag and valid_delim_flag:
                validated_proposed_delimiters[proposed_delim] = max_col_count
                if max_col_count \
                   and max_col_count > validated_proposed_delimiters[proposed_delim]:
                    delimiter = proposed_delim

        # Use preferred delimiters with highest count, if possible
        largest_delim_count = 0
        for proposed_delim in validated_proposed_delimiters.keys():
            weighted_delim_count = validated_proposed_delimiters[proposed_delim]
            if proposed_delim in preferred:
                weighted_delim_count = 5*validated_proposed_delimiters[proposed_delim]
            if weighted_delim_count > largest_delim_count:
                delimiter = proposed_delim
                largest_delim_count = weighted_delim_count

        return delimiter, quotechar
            

    @staticmethod
    def _guess_header_row(data_as_str,
                          suggested_delimiter=None, suggested_quotechar=None,
                          diff_thresh=0.1, none_thresh=0.5, str_thresh=0.9):
        """
        This function attempts to select the best row for which a header would be valid.
        
        :param data_as_str: Single string containing rows (lines seperated by "\n")
        :type data_as_str: str
        :param suggested_delimiter: Delimiter suggested to use when trying to find the header
        :type suggested_delimiter: str
        :param suggested_delimiter: quotechar suggested to use when trying to find the header
        :type suggested_delimiter: str
        :param diff_threshold: Max percent difference in cell types between rows allowed 
        :type diff_threshold: float    
        :param none_thresh: Max percent difference number of none values allowed
        :type none_thresh: float    
        :param str_thresh: Min percent of strings (omitting none) in row to be a header
        :type str_thresh: float
        
        :return: index for row estimated to be the last valid header
        :type: int
        """
        
        # Catch base cases
        if not data_as_str or len(data_as_str) == 0:
            return None
    
        # Ensure no "None" delimiter, delimiter is required for evaluating single columns
        delimiter = suggested_delimiter
        if not delimiter:
            delimiter = ','
        quotechar = suggested_quotechar
        if not quotechar or len(quotechar) == 0:
            quotechar = '"'

        # Determine type for every cell
        header_check_list = []
        only_string_flag = True # Requires additional checks
        for row in data_as_str.split('\n'):
            
            delimiter_regex = data_utils.get_delimiter_regex(delimiter, quotechar)
            row_list = re.split(delimiter_regex, row)
            
            header_check_list.append([])

            for i in range(len(row_list)):
                cell = row_list[i].strip()
                cell_type = data_utils.detect_cell_type(cell)

                if cell_type not in ['str', 'none']:
                    only_string_flag = False
                header_check_list[-1].append(cell_type)
                
        # Flags differences in types between each row (true/false)
        potential_header = header_check_list[0]
        differences = []
        for i in range(0, len(header_check_list)):
            differences.append([])
        
            # Identify if the row has any data
            len_not_none = len(header_check_list[i]) - header_check_list[i].count("none")
            len_pot_head = len(potential_header) - potential_header.count("none")
        
            # If row has more elements or has no data, mark as "skip", no difference
            if len_not_none > len_pot_head or len_not_none == 0:
                differences[i] = [False] * len(header_check_list[i])
            else:
                for j in range(len(header_check_list[i])):
                    diff_flag = False
                    if j >= len(potential_header) or \
                            header_check_list[i][j] != potential_header[j]:
                        diff_flag = True
                    differences[i].append(diff_flag)
        
            # If there is data in the row, set new max potential header to current row
            if len_not_none > 0:
                potential_header = header_check_list[i]
                
        # Predicts the last row that could be the header, given the criteria
        prior_len = 0
        row_classic_header_ends = None
        change_flag = False
        for i in range(0, len(differences)):

            # Skip if there's nothing in the given row
            if len(header_check_list[i]) == 0:
                continue
            
            # Determine ratio of none in row, must be BELOW threshold
            none = float(header_check_list[i].count("none")) / float(len(header_check_list[i]))
        
            # Determine percent of differences between prior row, must be BELOW threshold
            diff = float(differences[i].count(True)) / float(len(differences[i]))
        
            # Determine percent of string, uppercase string or none in row,
            # must be ABOVE threshold
            rstr = float((header_check_list[i].count("str") 
                          + header_check_list[i].count("upstr")
                          + header_check_list[i].count("none")))
            rstr /= float(len(header_check_list[i]))
            
            # Determines if the number of elements in the row is increasing or decreasing
            len_increase = False
            len_not_none = len(header_check_list[i]) - header_check_list[i].count("none")
            if len_not_none >= prior_len and len_not_none > 0:
                prior_len = len_not_none
                len_increase = True
                
            # Returns the last row that could reasonably be the header
            if (rstr > str_thresh and none < none_thresh and diff < diff_thresh):
                if len_increase and not change_flag:
                    row_classic_header_ends = i

            # If difference occurs & data in row, mark as change
            if diff > 0 and len_not_none > 0:
                change_flag = True
                
        # If change in statistics never occurs, return no header
        if not change_flag:
            row_classic_header_ends = None


        # Attempt to resolve case where only strings in every cell
        if only_string_flag:
            col_stats = {}
            rows = data_as_str.split('\n')
            for i in range(0, len(rows)):
                
                delimiter_regex = data_utils.get_delimiter_regex(delimiter, quotechar)
                cells = re.split(delimiter_regex, rows[i])
                
                for j in range(0, len(cells)):

                    # Determine number of words in cell
                    word_count = 0
                    if len(cells[j].strip()) > 0:
                        words = cells[j].strip().split(' ')
                        word_count = len(words)
                    
                    # First row, set base
                    if j not in col_stats:
                        col_stats[j] = {"max":word_count, "min":word_count}

                    # Identify min / max for a column
                    if word_count > col_stats[j]['max']:
                        col_stats[j]['max'] = word_count
                    if word_count < col_stats[j]['min']:
                        col_stats[j]['min'] = word_count

                    # First index with value
                    if 'first_index_with_value' not in col_stats[j] and word_count > 0:
                        col_stats[j]['first_index_with_value'] = i

            # Identify columns with variance
            variance = [False] * len(col_stats.keys())
            last_row_with_first_col_value = 0
            last_row_with_first_col_value_count = 0
            for i in col_stats.keys():
                col = col_stats[i]
                
                # Determines if there's variance in the column
                if (col['max'] - col['min']) > 1:
                    variance[i] = True

                # First last row, keeps a count of new col first in row
                if 'first_index_with_value' in col:
                    if col['first_index_with_value'] > last_row_with_first_col_value:
                        last_row_with_first_col_value = col['first_index_with_value']
                        last_row_with_first_col_value_count = 1
                    elif col['first_index_with_value'] == last_row_with_first_col_value:
                        last_row_with_first_col_value_count += 1

            # Ensures there is at least some variance
            if variance.count(True) > 0: 
                
                # Ensures most first lines are the same row
                if last_row_with_first_col_value_count > (len(variance) // 2):
                    row_classic_header_ends = last_row_with_first_col_value

        return row_classic_header_ends

    def _load_data_from_str(self, data_as_str):
        """Loads the data into memory from the str."""
        delimiter, quotechar = None, None
        if not self._delimiter or not self._quotechar:
            delimiter, quotechar = self._guess_delimiter_and_quotechar(data_as_str)
        if not self._delimiter:
            self._delimiter = delimiter
        if not self._quotechar:
            self._quotechar = quotechar

        data_buffered = StringIO(data_as_str)
        if self._header == 'auto':
            self._header = self._guess_header_row(
                data_as_str, self._delimiter, self._quotechar)
        return data_utils.read_csv_df(
            data_buffered,
            self.delimiter, self.header, self.selected_columns,
            read_in_string=True
        )

    def _load_data_from_file(self, input_file_path):
        """
        Loads the data into memory from the file.
        """
        
        self._file_encoding = data_utils.detect_file_encoding(input_file_path)
        data_as_str = data_utils.load_as_str_from_file(input_file_path,
                                                       self._file_encoding)

        if not self._delimiter or not self._checked_header:
            delimiter, quotechar = None, None
            if not self._delimiter or not self._quotechar:
                delimiter, quotechar = self._guess_delimiter_and_quotechar(data_as_str)
            if not self._delimiter:
                self._delimiter = delimiter
            if not self._quotechar:
                self._quotechar = quotechar
                
            if self._header == 'auto':
                self._header = self._guess_header_row(
                    data_as_str, self._delimiter, self._quotechar)
                self._checked_header = True

        # if there is only one delimiter at the end of each row,
        # set delimiter to None
        if self._delimiter:
            if len(data_as_str) > 0:
                num_lines_read = 0
                count_delimiter_last = 0
                for line in data_as_str.split('\n'):
                    if len(line) > 0:
                        if line.count(self._delimiter) == 1 \
                           and line.strip()[-1] == self._delimiter:
                            count_delimiter_last += 1
                        num_lines_read += 1
                if count_delimiter_last == num_lines_read:
                    self._delimiter = None

        
        return data_utils.read_csv_df(
            input_file_path,
            self.delimiter, self.header, self.selected_columns,
            read_in_string=True,
            encoding=self.file_encoding
        )

    def _get_data_as_records(self, data):
        sep = self.delimiter if self.delimiter else self._default_delimiter
        quote = self.quotechar if self.quotechar else self._default_quotechar
        data = data.to_csv(sep=sep, quotechar=quote, index=False)
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

        if JSONData.is_match(file_path) or ParquetData.is_match(file_path) \
                or AVROData.is_match(file_path):
            return False

        file_encoding = data_utils.detect_file_encoding(file_path=file_path)
        delimiter = options.get("delimiter", None)
        quotechar = options.get("quotechar", None)
        header = options.get("header", 'auto')

        data_as_str = data_utils.load_as_str_from_file(file_path, file_encoding)
        
        if not delimiter or header == 'auto':
            
            # Checks if delimiter is a space; If so, returns false
            quotetmp = None
            if not delimiter:
                delimiter, quotetmp = cls._guess_delimiter_and_quotechar(data_as_str)
            if quotetmp:
                quotechar = quotetmp
            if header == 'auto':
                options.update(header=cls._guess_header_row(
                    data_as_str, delimiter, quotechar))
                
        max_line_count = 1000
        min_line_count = 3
        line_count = 0
        empty_line_count = 0
        delimiter_count = dict()
        
        delimiter_regex = data_utils.get_delimiter_regex(delimiter, quotechar)
        space_regex = data_utils.get_delimiter_regex(" ", quotechar)

        # Count the possible delimiters
        for line in data_as_str.split('\n'):

            line_count += 1
            count = 0

            # Must have content in line, >1 due to the \n character
            if len(line) <= 1:
                empty_line_count += 1
                continue

            # Find the location(s) where each delimiter was detected
            if delimiter:
                count = len([i.start() for i in re.finditer(delimiter_regex, line)])
            else:                    
                # If no delimiter, see if spaces are regular intervals
                count = len([i.start() for i in re.finditer(space_regex, line)])

            # Track the delimiter count per file
            if count not in delimiter_count:
                delimiter_count[count] = 0
            delimiter_count[count] += 1

            if line_count >= max_line_count:
                break

        if line_count - empty_line_count <= min_line_count:
            return False

        # ================================================================
        # Section calculates the most common number of delimiters per line
        # The delimiters per line must be fairly consistent to be a CSV
        # ================================================================

        # Min percentage of consistent delimtier counts per line, per file
        # Dynamically determined and the percentage increases with file length
        # 4 lines need 3 to be consistent, 1000 lines need 992 to be consistent
        max_deviation_count = 2 ** (len(str(line_count)) - 1)
        active_line_count = line_count - empty_line_count
        min_consistency_percent = (
                (active_line_count - max_deviation_count) / active_line_count
        )

        delimiter_count_values = np.array(list(delimiter_count.values()))
        count_percent = delimiter_count_values / np.sum(delimiter_count_values)

        if not count_percent.size:
            return False

        max_count_index = count_percent.argmax()
        max_count_value = list(delimiter_count.keys())[max_count_index]
        max_count_percent = count_percent[max_count_index]

        # Infered the file was a CSV
        if ((max_count_value > 0 or delimiter is None)
                and (max_count_percent >= min_consistency_percent)):
            options.update(delimiter=delimiter, quotechar=quotechar)
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
        options.update(
            header=self.header, delimiter=self.delimiter, quotechar=self.quotechar
        )
        super(CSVData, self).reload(input_file_path, data, options)
        self.__init__(input_file_path, data, options)
