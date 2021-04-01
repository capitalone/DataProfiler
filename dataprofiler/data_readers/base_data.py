import sys
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

logger = logging.getLogger('DataProfiler.data')


class BaseData(object):
    """
    Abstract class for data loading and saving
    """

    data_type = None
    info = None

    def __init__(self, input_file_path, data, options):
        """
        Base class for loading a dataset. Options can be specified and maybe
        more specific to the subclasses.
        
        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        # Public properties
        self.input_file_path = input_file_path
        self.options = options

        # 'Private' properties
        #  _data_formats: dict containing data_formats (key) and function
        #                 calls (values) which take self._data and convert it
        #                 into the desired data_format for output.
        #  _selected_data_format: user selected format in which to return data
        #                         can only be of types in _data_formats
        #  _data: data being stored im memory
        #  _batch_info: when iterating through batches, information about the
        #               iteration/permutation are necessary to be held to keep
        #               constant across function calls.
        #  _tmp_file_name: randomly set variables for file name usable by system
        self._data_formats = OrderedDict()
        self._selected_data_format = None
        self._data = data
        self._batch_info = dict(perm=list(), iter=0)
        self._tmp_file_name = None
        self._file_encoding = None

    @property
    def data(self):
        if self._data is None:
            self._load_data()

        allowed_data_formats = list(self._data_formats.keys())
        if not self._selected_data_format:
            return self._data
        elif self._selected_data_format in allowed_data_formats:
            return self._data_formats[self._selected_data_format](self._data)
        else:
            raise ValueError(
                "The data format must be one of the following: {}".format(
                    str(allowed_data_formats)
                )
            )

    @property
    def data_format(self):
        return self._selected_data_format

    @data_format.setter
    def data_format(self, value):
        allowed_data_formats = list(self._data_formats.keys())
        if value.lower() not in allowed_data_formats:
            raise ValueError(
                "The data format must be one of the following: {}".format(
                    str(allowed_data_formats)
                )
            )
        self._selected_data_format = value.lower()

    @property
    def file_encoding(self):
        if not self._file_encoding:
            return sys.getdefaultencoding()
        return self._file_encoding

    @file_encoding.setter
    def file_encoding(self, value):
        valid_user_set_encodings = [
            "ascii", "utf-8", "utf-16", "utf-32"
        ]
        if not value or value.lower() not in valid_user_set_encodings:
            raise ValueError(
                "File Encoding must be one of the following: {}".
                    format(valid_user_set_encodings)
             )
        self._file_encoding = value

    @staticmethod
    def _check_and_return_options(options):
        if not options:
            options = dict()
        elif not isinstance(options, dict):
            raise ValueError("Options must be a dictionary.")
        return options

    def _load_data(self, data=None):
        raise NotImplementedError()

    def get_batch_generator(self, batch_size):
        data_length = len(self.data)
        indices = np.random.permutation(data_length)
        for i in range(0, data_length, batch_size):
            if isinstance(self.data, pd.DataFrame):
                yield self.data.iloc[indices[i:i + batch_size], :]
            else:
                yield list(self.data[k] for k in indices[i:i + batch_size])

    @classmethod
    def is_match(cls, input_file_path, options):
        raise NotImplementedError()

    def reload(self, input_file_path, data, options):
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
        if input_file_path and not self.is_match(input_file_path, options):
            raise ValueError(
                "Reloaded dataset does not match the specified data_type"
            )
        self._data = None
        self.input_file_path = None
        self._tmp_file_name = None
        self.options = None
        self._batch_info = dict(perm=list(), iter=0)

    def __len__(self):
        """
        Returns the length of the dataset which is loaded.

        :return: length of the dataset
        """
        return len(self.data)

    @property
    def length(self):
        """
        Returns the length of the dataset which is loaded.

        :return: length of the dataset
        """
        return len(self)
