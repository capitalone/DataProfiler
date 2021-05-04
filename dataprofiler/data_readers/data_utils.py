from builtins import next
import re
import json
from io import open
from collections import OrderedDict
import dateutil

import pandas as pd
import pyarrow.parquet as pq
from chardet.universaldetector import UniversalDetector


def data_generator(data_list):
    """
    Takes a list and returns a generator on the list.
    
    :param data_list: list of strings
    :type data_list: list
    :return: item from the list
    :rtype: generator
    """
    for item in data_list:
        yield item


def generator_on_file(file_object):
    """
    Takes a file and returns a generator that returns lines
    
    :param file_path: path to the file
    :type file_path: path
    :return: Line from file
    :rtype: generator
    """
    while True:
        line = file_object.readline()
        if not line:
            break
        yield line

    file_object.close()


def convert_int_to_string(x):
    """
    Converts the given input to string. In particular, it is int,
    it converts it ensuring there is no . or 00 in the converted string.
    In addition, if the input is np.nan, the output will be 'nan' which is
    what we need to handle data properly.

    :param x:
    :type x: Union[int, float, str, numpy.nan]
    :return:
    :rtype: str
    """
    try:
        # TODO: Check if float is necessary
        return str(int(float(x)))
    except:
        return str(x)


def unicode_to_str(data, ignore_dicts=False):
    """
    Convert data to string representation if it is a unicode string.
    
    :param data: input data
    :type data: str
    :param ignore_dicts: if set, ignore the dictionary type processing
    :type ignore_dicts: boolean
    :return: string representation of data
    :rtype: str
    """
    if isinstance(data, str):
        return data.encode('utf-8').decode()

    # if data is a list of values
    if isinstance(data, list):
        return [unicode_to_str(item, ignore_dicts=True) for item in data]

    # if data is a dictionary
    if isinstance(data, dict) and not ignore_dicts:
        return {unicode_to_str(key, ignore_dicts=True):
                    unicode_to_str(value, ignore_dicts=True)
                for key, value in data.items()
                }

    return data


def json_to_dataframe(json_lines, selected_columns=None, read_in_string=False):
    """
    This function takes a list of json objects and returns the dataframe
    representing the json list.
    
    :param json_lines: list of json objects
    :type json_lines: list(dict)
    :param selected_columns: a list of keys to be processed
    :type selected_columns: list(str)
    :param read_in_string: if True, all the values in dataframe will be 
        converted to string
    :type read_in_string: bool
    :return: dataframe converted from json list and list of dtypes for each
        column
    :rtype: tuple(pd.DataFrame, pd.Series(dtypes))
    """
    if len(json_lines) == 0:
        return pd.DataFrame(), None

    first_item_type = type(json_lines[0])
    if not all(map(lambda x: isinstance(x, first_item_type), json_lines)):
        raise ValueError(
            'Only JSON which represents structured data is supported for this '
            'data type (i.e. list-dicts).'
        )
    elif first_item_type == dict:
        df = pd.json_normalize(json_lines)
    else:
        df = pd.DataFrame(json_lines)
    original_df_dtypes = df.dtypes

    df[df.columns] = df[df.columns].astype(str)

    # filter some columns to be processed if specified by users
    if selected_columns:
        df = df[selected_columns]
    return df, original_df_dtypes


def read_json_df(data_generator, selected_columns=None, read_in_string=False):
    """
    This function returns an iterator that returns a chunk of data
    as dataframe in each call. The source of input to this function is either a
    file or a list of JSON structured strings. If the file path is given as
    input, the file is expected to have one JSON structures in each line. The
    lines that are not valid json will be ignored. Therefore, a file with
    pretty printed JSON objects will not be considered valid JSON. If the
    input is a data list, it is expected to be a list of strings where each
    string is a valid JSON object. if the individual object is not valid
    JSON, it will be ignored.
    
    NOTE: both data_list and file_path cannot be passed at the same time.
    
    :param data_generator: The generator you want to read.
    :type data_generator: generator
    :param selected_columns: a list of keys to be processed
    :type selected_columns: list(str)
    :param read_in_string: if True, all the values in dataframe will be
        converted to string
    :type read_in_string: bool
    :return: returns an iterator that returns a chunk of file as dataframe in
        each call as well as original dtypes of the dataframe columns.
    :rtype: typle(Iterator(pd.DataFrame), pd.Series(dtypes)
    """

    lines = list()
    k = 0
    while True:
        try:
            raw_line = next(data_generator)
        except StopIteration:
            raw_line = None

        if not raw_line:
            break
        try:
            obj = unicode_to_str(
                json.loads(raw_line,
                           object_hook=unicode_to_str,
                           object_pairs_hook=OrderedDict),
                ignore_dicts=True
            )
            lines.append(obj)
        except ValueError:
            pass
            # To ignore malformatted lines.
        k += 1
    if not lines and k:
        raise ValueError('No JSON data could be read from these data.')
    return json_to_dataframe(lines, selected_columns, read_in_string)


def read_json(data_generator, selected_columns=None, read_in_string=False):
    """
    This function returns the lines of a json. The source of input to this 
    function is either a file or a list of JSON structured strings. If the 
    file path is given as input, the file is expected to have one JSON 
    structures in each line. The lines that are not valid json will be ignored. 
    Therefore, a file with pretty printed JSON objects will not be considered 
    valid JSON. If the input is a data list, it is expected to be a list of 
    strings where each string is a valid JSON object. if the individual object 
    is not valid JSON, it will be ignored.

    NOTE: both data_list and file_path cannot be passed at the same time.

    :param data_generator: The generator you want to read.
    :type data_generator: generator
    :param selected_columns: a list of keys to be processed
    :type selected_columns: list(str)
    :param read_in_string: if True, all the values in dataframe will be
        converted to string
    :type read_in_string: bool
    :return: returns the lines of a json file
    :rtype: list(dict)
    """

    lines = list()
    k = 0
    while True:
        try:
            raw_line = next(data_generator)
        except StopIteration:
            raw_line = None

        if not raw_line:
            break
        try:
            obj = unicode_to_str(
                json.loads(raw_line,
                           object_hook=unicode_to_str,
                           object_pairs_hook=OrderedDict),
                ignore_dicts=True
            )
            lines.append(obj)
        except ValueError:
            pass
            # To ignore malformatted lines.
        k += 1
    if not lines and k:
        raise ValueError('No JSON data could be read from these data.')
    return lines


def read_csv_df(file_path, delimiter, header, selected_columns=[],
                read_in_string=False, encoding='utf-8'):
    """
    Reads a CSV file in chunks and returns a dataframe in the form of iterator.
    
    :param file_path: path to the CSV file.
    :type file_path: str
    :param delimiter: character used to separate csv values.
    :type delimiter: str
    :param header: the header row in the csv file.
    :type header: int
    :param selected_columns: a list of columns to be processed
    :type selected_columns: list(str)
    :param read_in_string: if True, all the values in dataframe will be
        converted to string
    :type read_in_string: bool
    :return: Iterator
    :rtype: pd.DataFrame
    """
    args = {
        'delimiter': delimiter,
        'header': header,
        'iterator': True,
        'dtype': 'object',
        'keep_default_na': False,
        'encoding': encoding
    }

    # If a header can be identified, don't skip blanks
    if header is not None:
        args.update({
            'skip_blank_lines': False
        })

    if read_in_string:
        args['dtype'] = str

    if len(selected_columns) > 0:
        args['usecols'] = selected_columns
    fo = pd.read_csv(file_path, **args)
    data = fo.read()
    fo.close()
    return data


def read_parquet_df(file_path, selected_columns=None, read_in_string=False):
    """
    Returns an iterator that returns one row group each time.
    
    :param file_path: path to the Parquet file.
    :type file_path: str
    :return:
    :rtype: Iterator(pd.DataFrame)
    """

    parquet_file = pq.ParquetFile(file_path)
    data = pd.DataFrame()
    for i in range(parquet_file.num_row_groups):

        data_row_df = parquet_file.read_row_group(i).to_pandas()

        # Convert all the unicode columns to utf-8
        types = data_row_df.apply(lambda x: pd.api.types.infer_dtype(
                                                x.values, skipna=True))

        mixed_and_unicode_cols = types[types == 'unicode'] \
            .index.union(types[types == 'mixed'].index)

        for col in mixed_and_unicode_cols:
            data_row_df[col] = data_row_df[col].apply(
                lambda x: x.encode('utf-8').strip() if isinstance(x, str) else x)
            data_row_df[col] = data_row_df[col].apply(
                lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x)

        if selected_columns:
            data_row_df = data_row_df[selected_columns]

        data = pd.concat([data, data_row_df])

    original_df_dtypes = data.dtypes
    if read_in_string:
        data = data.astype(str)

    return data, original_df_dtypes


def read_text_as_list_of_strs(file_path):
    """
    Returns a list of strings relative to the chunk size. Each line is 1 chunk.
    
    :param file_path: path to the file
    :type file_path: str
    :return:
    :rtype: list(str)
    """
    with open(file_path, encoding="utf-8") as input_file:
        data = list(input_file)
    return data


def detect_file_encoding(file_path, buffer_size=1024, max_lines=20):
    """
    Determines the encoding of files within the initial `max_lines` of length
    `buffer_size`.
    
    :param file_path: path to the file
    :type file_path: str
    :param buffer_size: buffer length for each line being read
    :type buffer_size: int
    :param max_lines: number of lines to read from file of length buffer_size
    :type max_lines: int
    :return: encoding type
    :rtype: str
    """
    detector = UniversalDetector()
    line_count = 0
    with open(file_path, 'rb') as input_file:
        chunk = input_file.read(buffer_size)
        while chunk and line_count < max_lines:
            detector.feed(chunk)
            chunk = input_file.read(buffer_size)
            line_count += 1
    detector.close()
    encoding = detector.result["encoding"]

    # Typical file representation is utf-8 instead of ascii, treat as such.
    if not encoding or encoding.lower() in ['ascii', 'windows-1254']:
        encoding = 'utf-8'

    # Check if encoding can be used to decode without throwing an error
    def _decode_is_valid(encoding):
        try: 
            with open(file_path, encoding=encoding) as input_file:
                input_file.read(1024*1024)
                return True
        except: return False

    if not _decode_is_valid(encoding):
        try:
            from charset_normalizer import CharsetNormalizerMatches as CnM
            
            # Try with small sample 
            with open(file_path, 'rb') as input_file:
                raw_data = input_file.read(10000)
                result = CnM.from_bytes(raw_data, steps=5, 
                                        chunk_size=512, threshold=0.2,
                                        cp_isolation=None, cp_exclusion=None,
                                        preemptive_behaviour=True, explain=False)
                result = result.best().first()
            if result: encoding = result.encoding

            # Try again with full sample
            if not _decode_is_valid(encoding): 
                with open(file_path, 'rb') as input_file:
                    raw_data = input_file.read(max_lines*buffer_size)
                    result = CnM.from_bytes(raw_data, steps=max_lines, 
                                            chunk_size=buffer_size, threshold=0.2,
                                            cp_isolation=None, cp_exclusion=None,
                                            preemptive_behaviour=True, explain=False)
                    result = result.best().first()
                if result: encoding = result.encoding

        except:
            print("Install charset_normalizer for improved file encoding detection")

    # If no encoding is still found, default to utf-8
    if not encoding:
        encoding = 'utf-8'
    return encoding.lower()


def detect_cell_type(cell):
    """
    Detects the cell type (int, float, etc)

    :param cell: String designated for data type detection
    :type cell: str
    """
    
    cell_type = 'str'
    if len(cell) == 0:
        cell_type = 'none'
    else:

        try:
            if dateutil.parser.parse(cell, fuzzy=False):
                cell_type = 'date'
        except (ValueError, OverflowError, TypeError):
            pass

        try:
            f_cell = float(cell)
            cell_type = 'float'
            if f_cell.is_integer():
                cell_type = 'int'
        except ValueError:
            pass

        if cell.isupper():
            cell_type = 'upstr'
            
    return cell_type


def get_delimiter_regex(delimiter=",", quotechar=","):
    """
    Builds regex for delimiter checks
    
    :param delimiter: Delimiter to be added to regex
    :type delimiter: str
    :param quotechar: Quotechar to be added to regex
    :type delimiter: str
    """
    
    if delimiter is None:
        return ""

    if quotechar is None:
        quotechar = '"'

    delimiter_regex = re.escape(str(delimiter))
    quotechar_escape= re.escape(quotechar)
    quotechar_regex = "(?=" 
    quotechar_regex +=  "(?:"
    quotechar_regex +=    "[^"+quotechar_escape+"]*"
    quotechar_regex +=    quotechar_escape
    quotechar_regex +=    "[^"+quotechar_escape+"]*"
    quotechar_regex +=    quotechar_escape
    quotechar_regex +=   ")*"
    quotechar_regex +=   "[^"+quotechar_escape+"]*"
    quotechar_regex += "$)"

    return re.compile(delimiter_regex + quotechar_regex)


def find_nth_loc(string=None, search_query=None, n=0):
    """
    Searches the string via the search_query and
    returns the nth index in which the query occurs.
    If there are less than 'n' the last loc is returned
    
    :param string: Input string, to be searched
    :type string: str
    :param search_query: char(s) to find nth occurance of
    :type search_query: str
    :param n: The number of occurances to iterate through
    :type n: int
    
    :return idx: Index of the nth or last occurance of the search_query
    :rtype idx: int
    :return id_count: Number of identifications prior to idx
    :rtype id_count: int
    """

    # Return base case, if there's no string, query or n
    if not string or not search_query or 0 >= n:
        return -1, 0

    # Find index of nth occurrence of search_query 
    idx, pre_idx = -1, -1
    id_count = 0
    for i in range(0, n):

        idx = string.find(search_query, idx+1)
        
        # Exit for loop once string ends
        if idx == -1:

            if pre_idx > 0:
                idx = pre_idx+1 # rest to last location

            # If final discovery is not the search query
            if string[-len(search_query):] != search_query:
                idx = len(string)                
            break
        
        # Keep track of identifications
        if idx != pre_idx:
            id_count += 1
            pre_idx = idx
        
    return idx, id_count


def load_as_str_from_file(file_path, file_encoding, max_lines=10,
                          max_bytes=65536, chunk_size_bytes=1024):
    """
    Loads data from a csv file up to a specific line OR byte_size.

    :param file_path: Path to file to load data from
    :type file_path: str
    :param file_encoding: File encoding
    :type file_encoding: str
    :param max_lines: Maximum number of lines to load from file
    :type max_lines: int
    :param max_bytes: Maximum number of bytes to load from file
    :type max_bytes: int
    :param chunk_size_bytes: Chunk size to load every data load
    :type chunk_size_bytes: int
    
    :return data_as_str: Data as string
    :type data_as_str: str
    """

    data_as_str = ""
    total_occurances = 0
    with open(file_path, encoding=file_encoding) as csvfile:

        sample_size_bytes = min(max_bytes, chunk_size_bytes)
        
        # Read the file until the appropriate number of occurances                    
        for byte_count in range(0, max_bytes, sample_size_bytes):
            
            sample_lines = csvfile.read(sample_size_bytes)
            if len(sample_lines) == 0:                
                break # No more bytes in file

            # Number of lines remaining to be added to data_as_str
            remaining_lines = max_lines - total_occurances
            
            # Return either the last index of sample_lines OR
            # the index of the newline char that matches remaining_lines
            loc, occurance = find_nth_loc(sample_lines,
                                          search_query='\n',
                                          n=remaining_lines)

            # Add sample_lines to data_as_str no more than max_lines
            data_as_str += sample_lines[:loc]
            total_occurances += occurance
            if total_occurances >= max_lines:
                break
            
    return data_as_str
