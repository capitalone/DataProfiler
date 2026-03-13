"""Contains functions for data readers."""
import json
import logging
import os
import random
import re
import urllib
from collections import OrderedDict
from io import BytesIO, StringIO, TextIOWrapper
from itertools import islice
from math import floor, log, log1p
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
)

import boto3
import botocore
import dateutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from chardet.universaldetector import UniversalDetector
from typing_extensions import TypeGuard

from .. import dp_logging, rng_utils
from .._typing import JSONType, Url
from .filepath_or_buffer import FileOrBufferHandler, is_stream_buffer  # NOQA

logger = dp_logging.get_child_logger(__name__)


def data_generator(data_list: List[str]) -> Generator[str, None, None]:
    """
    Take a list and return a generator on the list.

    :param data_list: list of strings
    :type data_list: list
    :return: item from the list
    :rtype: generator
    """
    yield from data_list


def generator_on_file(
    file_object: Union[StringIO, BytesIO]
) -> Generator[Union[str, bytes], None, None]:
    """
    Take a file and return a generator that returns lines.

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


def convert_int_to_string(x: int) -> str:
    """
    Convert the given input to string.

    In particular, it is int, it converts it ensuring there is no . or 00.
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
    except Exception:
        return str(x)


def unicode_to_str(data: JSONType, ignore_dicts: bool = False) -> JSONType:
    """
    Convert data to string representation if it is a unicode string.

    :param data: input data
    :type data: JSONType
    :param ignore_dicts: if set, ignore the dictionary type processing
    :type ignore_dicts: boolean
    :return: string representation of data
    :rtype: str
    """
    if isinstance(data, str):
        return data.encode("utf-8").decode()

    # if data is a list of values
    if isinstance(data, list):
        return [unicode_to_str(item, ignore_dicts=True) for item in data]

    # if data is a dictionary
    if isinstance(data, dict) and not ignore_dicts:
        return {
            unicode_to_str(key, ignore_dicts=True): unicode_to_str(
                value, ignore_dicts=True
            )
            for key, value in data.items()
        }

    return data


def json_to_dataframe(
    json_lines: List[JSONType],
    selected_columns: Optional[List[str]] = None,
    read_in_string: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Take list of json objects and return dataframe representing json list.

    :param json_lines: list of json objects
    :type json_lines: list(JSONType)
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
            "Only JSON which represents structured data is supported for this "
            "data type (i.e. list-dicts)."
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


def read_json_df(
    data_generator: Generator,
    selected_columns: Optional[List[str]] = None,
    read_in_string: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return an iterator that returns a chunk of data as dataframe in each call.

    The source of input to this function is either a
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
    :rtype: tuple(pd.DataFrame, pd.Series(dtypes))
    """
    lines: List[JSONType] = list()
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
                json.loads(
                    raw_line, object_hook=unicode_to_str, object_pairs_hook=OrderedDict
                ),
                ignore_dicts=True,
            )
            lines.append(obj)
        except ValueError:
            pass
            # To ignore malformatted lines.
        k += 1
    if not lines and k:
        raise ValueError("No JSON data could be read from these data.")
    return json_to_dataframe(lines, selected_columns, read_in_string)


def read_json(
    data_generator: Iterator,
    selected_columns: Optional[List[str]] = None,
    read_in_string: bool = False,
) -> List[JSONType]:
    """
    Return the lines of a json.

    The source of input to this function is either a file or
    a list of JSON structured strings.
    If the file path is given as input, the file is expected to have one JSON
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
    lines: List[JSONType] = list()
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
                json.loads(
                    raw_line, object_hook=unicode_to_str, object_pairs_hook=OrderedDict
                ),
                ignore_dicts=True,
            )
            lines.append(obj)
        except ValueError:
            pass
            # To ignore malformatted lines.
        k += 1
    if not lines and k:
        raise ValueError("No JSON data could be read from these data.")
    return lines


def reservoir(file: TextIOWrapper, sample_nrows: int) -> list:
    """
    Implement the mathematical logic of Reservoir sampling.

    :param file: wrapper of the opened csv file
    :type file: TextIOWrapper
    :param sample_nrows: number of rows to sample
    :type sample_nrows: int

    :raises: ValueError()

    :return: sampled values
    :rtype: list
    """
    # Copyright 2021 Oscar Benjamin
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    # https://gist.github.com/oscarbenjamin/4c1b977181f34414a425f68589e895d1

    iterator = iter(file)
    values = list(islice(iterator, sample_nrows))

    irange = range(len(values))
    indices = dict(zip(irange, irange))

    kinv = 1 / sample_nrows
    W = 1.0
    rng = rng_utils.get_random_number_generator()

    while True:
        W *= rng.random() ** kinv
        # random() < 1.0 but random() ** kinv might not be
        # W == 1.0 implies "infinite" skips
        if W == 1.0:
            break
        # skip is geometrically distributed with parameter W
        skip = floor(log(rng.random()) / log1p(-W))
        try:
            newval = next(islice(iterator, skip, skip + 1))
        except StopIteration:
            break
        # Append new, replace old with dummy, and keep track of order
        remove_index = rng.integers(0, sample_nrows)
        values[indices[remove_index]] = str(None)
        indices[remove_index] = len(values)
        values.append(newval)

    values = [values[indices[i]] for i in irange]
    return values


def rsample(file_path: TextIOWrapper, sample_nrows: int, args: dict) -> StringIO:
    """
    Implement Reservoir Sampling to sample n rows out of a total of M rows.

    :param file_path: path of the csv file to be read in
    :type file_path: TextIOWrapper
    :param sample_nrows: number of rows being sampled
    :type sample_nrows: int
    :param args: options to read the csv file
    :type args: dict
    """
    header = args["header"]
    result = []

    if header is not None:
        result = [[next(file_path) for i in range(header + 1)][-1]]
        args["header"] = 0

    result += reservoir(file_path, sample_nrows)

    fo = StringIO("".join([i if (i[-1] == "\n") else i + "\n" for i in result]))
    return fo


def read_csv_df(
    file_path: Union[str, BytesIO, TextIOWrapper],
    delimiter: Optional[str],
    header: Optional[int],
    sample_nrows: Optional[int] = None,
    selected_columns: List[str] = [],
    read_in_string: bool = False,
    encoding: Optional[str] = "utf-8",
) -> pd.DataFrame:
    """
    Read a CSV file in chunks and return dataframe in form of iterator.

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
    args: Dict[str, Any] = {
        "delimiter": delimiter,
        "header": header,
        "iterator": True,
        "dtype": "object",
        "keep_default_na": False,
        "encoding": encoding,
    }

    # If a header can be identified, don't skip blanks
    if header is not None:
        args.update({"skip_blank_lines": False})

    if read_in_string:
        args["dtype"] = str

    if len(selected_columns) > 0:
        args["usecols"] = selected_columns

    # account for py3.6 requirement for pandas, can remove if >= py3.7
    is_buf_wrapped = False
    is_file_open = False
    if isinstance(file_path, BytesIO):
        # a BytesIO stream has to be wrapped in order to properly be detached
        # in 3.6 this avoids read_csv wrapping the stream and closing too early
        file_path = TextIOWrapper(file_path, encoding=encoding)
        is_buf_wrapped = True
    elif isinstance(file_path, str):
        file_path = open(file_path, encoding=encoding)
        is_file_open = True

    file_data = file_path
    if sample_nrows:
        file_data = rsample(file_path, sample_nrows, args)
    fo = pd.read_csv(file_data, **args)
    data = fo.read()

    # if the buffer was wrapped, detach it before returning
    if is_buf_wrapped:
        file_path = cast(TextIOWrapper, file_path)
        file_path.detach()
    elif is_file_open:
        file_path.close()
    fo.close()

    return data


def convert_unicode_col_to_utf8(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all unicode columns in input dataframe to utf-8.

    :param input_df: input dataframe
    :type input_df: pd.DataFrame
    :return: corrected dataframe
    :rtype: pd.DataFrame
    """
    # Convert all the unicode columns to utf-8
    input_column_types = input_df.apply(
        lambda x: pd.api.types.infer_dtype(x.values, skipna=True)
    )

    mixed_and_unicode_cols = input_column_types[
        input_column_types == "unicode"
    ].index.union(input_column_types[input_column_types == "mixed"].index)

    for iter_column in mixed_and_unicode_cols:
        # Encode sting to bytes
        input_df[iter_column] = input_df[iter_column].apply(
            lambda x: x.encode("utf-8").strip() if isinstance(x, str) else x
        )

        # Decode bytes back to string
        input_df[iter_column] = input_df[iter_column].apply(
            lambda x: x.decode("utf-8").strip() if isinstance(x, bytes) else x
        )

    return input_df


def sample_parquet(
    file_path: str,
    sample_nrows: int,
    selected_columns: Optional[List[str]] = None,
    read_in_string: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Read parquet file, sample specified number of rows from it and return a data frame.

    :param file_path: path to the Parquet file.
    :type file_path: str
    :param sample_nrows: number of rows being sampled
    :type sample_nrows: int
    :param selected_columns: columns need to be read
    :type selected_columns: list
    :param read_in_string: return as string type
    :type read_in_string: bool
    :return:
    :rtype: Iterator(pd.DataFrame)
    """
    # read parquet file into table
    if selected_columns:
        parquet_table = pq.read_table(file_path, columns=selected_columns)
    else:
        parquet_table = pq.read_table(file_path)

    # sample
    n_rows = parquet_table.num_rows
    if n_rows > sample_nrows:
        sample_index = np.array([False] * n_rows)
        sample_index[random.sample(range(n_rows), sample_nrows)] = True
    else:
        sample_index = np.array([True] * n_rows)
    sample_df = parquet_table.filter(sample_index).to_pandas()

    # Convert all the unicode columns to utf-8
    sample_df = convert_unicode_col_to_utf8(sample_df)

    original_df_dtypes = sample_df.dtypes
    if read_in_string:
        sample_df = sample_df.astype(str)

    return sample_df, original_df_dtypes


def read_parquet_df(
    file_path: str,
    sample_nrows: Optional[int] = None,
    selected_columns: Optional[List[str]] = None,
    read_in_string: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return an iterator that returns one row group each time.

    :param file_path: path to the Parquet file.
    :type file_path: str
    :param sample_nrows: number of rows being sampled
    :type sample_nrows: int
    :param selected_columns: columns need to be read
    :type selected_columns: list
    :param read_in_string: return as string type
    :type read_in_string: bool
    :return:
    :rtype: Iterator(pd.DataFrame)
    """
    if sample_nrows is None:
        parquet_file = pq.ParquetFile(file_path)
        data = pd.DataFrame()
        for i in range(parquet_file.num_row_groups):

            data_row_df = parquet_file.read_row_group(i).to_pandas()

            # Convert all the unicode columns to utf-8
            data_row_df = convert_unicode_col_to_utf8(data_row_df)

            if selected_columns:
                data_row_df = data_row_df[selected_columns]

            data = pd.concat([data, data_row_df])

        original_df_dtypes = data.dtypes
        if read_in_string:
            data = data.astype(str)
        return data, original_df_dtypes
    else:
        data, original_df_dtypes = sample_parquet(
            file_path,
            sample_nrows,
            selected_columns=selected_columns,
            read_in_string=read_in_string,
        )
        return data, original_df_dtypes


def read_text_as_list_of_strs(
    file_path: str, encoding: Optional[str] = None
) -> List[str]:
    """
    Return list of strings relative to the chunk size.

    Each line is 1 chunk.

    :param file_path: path to the file
    :type file_path: str
    :return:
    :rtype: list(str)
    """
    if encoding is None:
        encoding = detect_file_encoding(file_path)
    with FileOrBufferHandler(file_path, encoding=encoding) as input_file:
        data = list(input_file)
    return data


def detect_file_encoding(
    file_path: str, buffer_size: int = 1024, max_lines: int = 20
) -> str:
    """
    Determine encoding of files within initial `max_lines` of length `buffer_size`.

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
    with FileOrBufferHandler(file_path, "rb") as input_file:
        chunk = input_file.read(buffer_size)
        while chunk and line_count < max_lines:
            detector.feed(chunk)
            chunk = input_file.read(buffer_size)
            line_count += 1
    detector.close()
    encoding = detector.result["encoding"]

    # Typical file representation is utf-8 instead of ascii, treat as such.
    if not encoding or encoding.lower() in ["ascii", "windows-1254"]:
        encoding = "utf-8"

    # Check if encoding can be used to decode without throwing an error
    def _decode_is_valid(encoding):
        try:
            with FileOrBufferHandler(file_path, encoding=encoding) as input_file:
                input_file.read(1024 * 1024)
                return True
        except Exception:
            return False

    if not _decode_is_valid(encoding):
        try:
            from charset_normalizer import from_bytes

            # Try with small sample
            with FileOrBufferHandler(file_path, "rb") as input_file:
                raw_data = input_file.read(10000)
                results = from_bytes(
                    raw_data,
                    steps=5,
                    chunk_size=512,
                    threshold=0.2,
                    cp_isolation=None,
                    cp_exclusion=None,
                    preemptive_behaviour=True,
                    explain=False,
                )
                best_result = results.best()
            if best_result:
                encoding = best_result.encoding

            # Try again with full sample
            if not _decode_is_valid(encoding):
                with FileOrBufferHandler(file_path, "rb") as input_file:
                    raw_data = input_file.read(max_lines * buffer_size)
                    results = from_bytes(
                        raw_data,
                        steps=max_lines,
                        chunk_size=buffer_size,
                        threshold=0.2,
                        cp_isolation=None,
                        cp_exclusion=None,
                        preemptive_behaviour=True,
                        explain=False,
                    )
                    best_result = results.best()
                if best_result:
                    encoding = best_result.encoding

        except Exception:
            logger.info(
                "Install charset_normalizer for improved file " "encoding detection"
            )

    # If no encoding is still found, default to utf-8
    if not encoding:
        encoding = "utf-8"
    return encoding.lower()


def detect_cell_type(cell: str) -> str:
    """
    Detect the cell type (int, float, etc).

    :param cell: String designated for data type detection
    :type cell: str
    """
    cell_type = "str"
    if len(cell) == 0:
        cell_type = "none"
    else:

        try:
            # need to ingore type bc https://github.com/python/mypy/issues/8878
            if dateutil.parser.parse(cell, fuzzy=False):  # type:ignore
                cell_type = "date"
        except (ValueError, OverflowError, TypeError):
            pass

        try:
            f_cell = float(cell)
            cell_type = "float"
            if f_cell.is_integer():
                cell_type = "int"
        except ValueError:
            pass

        if cell.isupper():
            cell_type = "upstr"

    return cell_type


def get_delimiter_regex(delimiter: str = ",", quotechar: str = ",") -> Pattern[str]:
    """
    Build regex for delimiter checks.

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
    quotechar_escape = re.escape(quotechar)
    quotechar_regex = "(?="
    quotechar_regex += "(?:"
    quotechar_regex += "[^" + quotechar_escape + "]*"
    quotechar_regex += quotechar_escape
    quotechar_regex += "[^" + quotechar_escape + "]*"
    quotechar_regex += quotechar_escape
    quotechar_regex += ")*"
    quotechar_regex += "[^" + quotechar_escape + "]*"
    quotechar_regex += "$)"

    return re.compile(delimiter_regex + quotechar_regex)


def find_nth_loc(
    string: Optional[str] = None,
    search_query: Optional[str] = None,
    n: int = 0,
    ignore_consecutive: bool = True,
) -> Tuple[int, int]:
    """
    Search string via search_query and return nth index in which query occurs.

    If there are less than 'n' the last loc is returned

    :param string: Input string, to be searched
    :type string: str
    :param search_query: char(s) to find nth occurrence of
    :type search_query: str
    :param n: The number of occurrences to iterate through
    :type n: int
    :param ignore_consecutive: Ignore consecutive matches in the search query.
    :type ignore_consecutive: bool

    :return idx: Index of the nth or last occurrence of the search_query
    :rtype idx: int
    :return id_count: Number of identifications prior to idx
    :rtype id_count: int
    """
    # Return base case, if there's no string, query or n
    if not string or not search_query or 0 >= n:
        return -1, 0

    # create the search pattern
    pattern = re.escape(search_query)
    if ignore_consecutive:
        pattern += "+"
    r_iter = re.finditer(pattern, string)

    # Find index of nth occurrence of search_query
    idx = id_count = -1
    for id_count, match in enumerate(r_iter):
        idx = match.start()
        if id_count + 1 == n:
            break

    # enumerate starts at 0 and so does the init
    id_count += 1

    if id_count != n:
        idx = len(string)

    return idx, id_count


def load_as_str_from_file(
    file_path: str,
    file_encoding: Optional[str] = None,
    max_lines: int = 10,
    max_bytes: int = 65536,
    chunk_size_bytes: int = 1024,
) -> str:
    """
    Load data from a csv file up to a specific line OR byte_size.

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

    :return: Data as string
    :rtype: str
    """
    data_as_str = ""
    total_occurrences = 0
    with FileOrBufferHandler(file_path, encoding=file_encoding) as csvfile:

        sample_size_bytes = min(max_bytes, chunk_size_bytes)

        # Read the file until the appropriate number of occurrences
        for byte_count in range(0, max_bytes, sample_size_bytes):

            sample_lines = csvfile.read(sample_size_bytes)
            if len(sample_lines) == 0:
                break  # No more bytes in file

            # Number of lines remaining to be added to data_as_str
            remaining_lines = max_lines - total_occurrences

            # Return either the last index of sample_lines OR
            # the index of the newline char that matches remaining_lines
            search_query_value: Union[str, bytes] = "\n"
            if isinstance(sample_lines, bytes):
                search_query_value = b"\n"

            start_loc = 0
            len_sample_lines = len(sample_lines)
            while start_loc < len_sample_lines - 1 and total_occurrences < max_lines:
                loc, occurrence = find_nth_loc(
                    sample_lines[start_loc:],
                    search_query=cast(str, search_query_value),
                    # TODO: make sure find_nth_loc() works with search_query as bytes
                    n=remaining_lines,
                )

                # Add sample_lines to data_as_str no more than max_lines
                if isinstance(sample_lines[start_loc:loc], bytes):
                    data_as_str += sample_lines[:loc].decode(file_encoding)
                else:
                    data_as_str += sample_lines[start_loc:loc]

                total_occurrences += occurrence
                start_loc = loc
            if total_occurrences >= max_lines:
                break

    return data_as_str


def is_valid_url(url_as_string: Any) -> TypeGuard[Url]:
    """
    Determine whether a given string is a valid URL.

    :param url_as_string: string to be tested if URL
    :type url_as_string: str
    :return: true if string is a valid URL
    :rtype: boolean
    """
    if not isinstance(url_as_string, str):
        return False

    result = urllib.parse.urlparse(url_as_string)
    # this is the minimum characteristics needed for a valid URL
    return all([result.scheme, result.netloc])


def url_to_bytes(url_as_string: Url, options: Dict) -> BytesIO:
    """
    Read in URL and converts it to a byte stream.

    :param url_as_string: string to read as URL
    :type url_as_string: str
    :param options: options for the url
    :type options: dict
    :return: BytesIO stream of data downloaded from URL
    :rtype: BytesIO stream
    """
    stream = BytesIO()

    verify_ssl = True
    if "verify_ssl" in options:
        verify_ssl = options["verify_ssl"]

    try:
        with requests.get(url_as_string, stream=True, verify=verify_ssl) as url:
            url = cast(requests.Response, url)
            url.raise_for_status()
            if (
                "Content-length" in url.headers
                and int(url.headers["Content-length"]) >= 1024**3
            ):
                raise ValueError(
                    "The downloaded file from the url may not be " "larger than 1GB"
                )

            total_bytes = 0
            c_size = 8192

            for chunk in url.iter_content(chunk_size=c_size):
                stream.write(chunk)
                total_bytes += c_size

                if total_bytes > 1024**3:
                    raise ValueError(
                        "The downloaded file from the url may not " "be larger than 1GB"
                    )
    except requests.exceptions.SSLError as e:
        raise RuntimeError(
            "The URL given has an untrusted SSL certificate. Although highly "
            "discouraged, you can proceed with reading the data by setting "
            "'verify_ssl' to False in options "
            "(i.e. options=dict(verify_ssl=False))."
        ) from e

    stream.seek(0)
    return stream


class S3Helper:
    """
    A utility class for working with Amazon S3.

    This class provides methods to check if a path is an S3 URI
        and to create an S3 client.
    """

    @staticmethod
    def is_s3_uri(path: str, logger: logging.Logger) -> bool:
        """
        Check if the given path is an S3 URI.

        This function checks for common S3 URI prefixes "s3://" and "s3a://".

        Args:
            path (str): The path to check for an S3 URI.
            logger (logging.Logger): The logger instance for logging.

        Returns:
            bool: True if the path is an S3 URI, False otherwise.
        """
        # Define the S3 URI prefixes to check
        s3_uri_prefixes = ["s3://", "s3a://"]
        path = path.strip()
        # Check if the path starts with any of the specified prefixes
        is_s3 = any(path.startswith(prefix) for prefix in s3_uri_prefixes)
        if not is_s3:
            logger.debug(f"'{path}' is not a valid S3 URI")

        return is_s3

    @staticmethod
    def _create_boto3_client(
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
        region_name: Optional[str],
    ) -> boto3.client:
        return boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )

    @staticmethod
    def create_s3_client(
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> boto3.client:
        """
        Create and return an S3 client.

        Args:
            aws_access_key_id (str): The AWS access key ID.
            aws_secret_access_key (str): The AWS secret access key.
            aws_session_token (str): The AWS session token
                (optional, typically used for temporary credentials).
            region_name (str): The AWS region name (default is 'us-east-1').

        Returns:
            boto3.client: A S3 client instance.
        """
        # Check if credentials are not provided
        # and use environment variables as fallback
        if aws_access_key_id is None:
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        if aws_secret_access_key is None:
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if aws_session_token is None:
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

        # Check if region is not provided and use environment variable as fallback
        if region_name is None:
            region_name = os.environ.get("AWS_REGION", "us-east-1")

        # Check if IAM roles for service accounts are available
        try:
            s3 = S3Helper._create_boto3_client(
                aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
            )
        except botocore.exceptions.NoCredentialsError:
            # IAM roles are not available, so fall back to provided credentials
            if aws_access_key_id is None or aws_secret_access_key is None:
                raise ValueError(
                    "AWS access key ID and secret access key are required."
                )
            s3 = S3Helper._create_boto3_client(
                aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
            )

        return s3

    @staticmethod
    def get_s3_uri(s3_uri: str, s3_client: boto3.client) -> BytesIO:
        """
        Download an object from an S3 URI and return its content as BytesIO.

        Args:
            s3_uri (str): The S3 URI specifying the location of the object to download.
            s3_client (boto3.client): An initialized AWS S3 client
                for accessing the S3 service.

        Returns:
            BytesIO: A BytesIO object containing the content of
                the downloaded S3 object.
        """
        # Parse the S3 URI
        parsed_uri = urllib.parse.urlsplit(s3_uri)
        bucket_name = parsed_uri.netloc
        file_key = parsed_uri.path.lstrip("/")
        # Download the S3 object
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

        # Return the object's content as BytesIO
        return BytesIO(response["Body"].read())
