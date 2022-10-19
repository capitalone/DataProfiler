"""Contains abstract class for data loading and saving."""
import locale
import sys
from collections import OrderedDict
from io import StringIO
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd

from .. import dp_logging
from . import data_utils

logger = dp_logging.get_child_logger(__name__)


class BaseData(object):
    """Abstract class for data loading and saving."""

    data_type: str
    info: Optional[str] = None

    def __init__(
        self, input_file_path: Optional[str], data: Any, options: Dict
    ) -> None:
        """
        Initialize Base class for loading a dataset.

        Options can be specified and maybe
        more specific to the subclasses.

        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        if options is None:
            options = {}

        # Public properties
        self.input_file_path = input_file_path
        self.options: Optional[Dict] = options

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
        #  _file_encoding: contains the suggested file encoding for reading data
        self._data_formats: Dict[str, Any] = OrderedDict()
        self._selected_data_format: Optional[str] = None
        self._data: Optional[Any] = data
        self._batch_info: Dict = dict(perm=list(), iter=0)
        self._tmp_file_name: Optional[str] = None
        self._file_encoding: Optional[str] = options.get("encoding", None)

    @property
    def data(self):
        """Return data."""
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
    def data_format(self) -> Optional[str]:
        """Return data format."""
        return self._selected_data_format

    @data_format.setter
    def data_format(self, value: str):
        allowed_data_formats = list(self._data_formats.keys())
        if value.lower() not in allowed_data_formats:
            raise ValueError(
                "The data format must be one of the following: {}".format(
                    str(allowed_data_formats)
                )
            )
        self._selected_data_format = value.lower()

    @property
    def is_structured(self) -> bool:
        """Determine compatibility with StructuredProfiler."""
        raise NotImplementedError

    @property
    def file_encoding(self) -> Optional[str]:
        """Return file encoding."""
        if not self._file_encoding:
            # get system default, but if set to ascii, just update to utf-8
            file_encoding = "utf-8"
            try:
                file_encoding = locale.getpreferredencoding(False)
            except Exception:
                file_encoding = sys.getfilesystemencoding()
            finally:
                if file_encoding.lower() in ["ansi_x3.4-1968", "ascii"]:
                    file_encoding = "utf-8"
            self._file_encoding = file_encoding

            # set to default, detect if not StringIO
            if self.input_file_path and not isinstance(self.input_file_path, StringIO):
                self._file_encoding = data_utils.detect_file_encoding(
                    self.input_file_path
                )
        return self._file_encoding

    @file_encoding.setter
    def file_encoding(self, value: str) -> None:
        """Set file encoding."""
        valid_user_set_encodings = ["ascii", "utf-8", "utf-16", "utf-32"]
        if not value or value.lower() not in valid_user_set_encodings:
            raise ValueError(
                "File Encoding must be one of the following: {}".format(
                    valid_user_set_encodings
                )
            )
        self._file_encoding = value

    @staticmethod
    def _check_and_return_options(options: Optional[Dict]) -> Dict:
        """Return options or raise error."""
        if not options:
            options = dict()
        elif not isinstance(options, dict):
            raise ValueError("Options must be a dictionary.")
        return options

    def _load_data(self, data: Optional[Any] = None) -> None:
        """Load data."""
        raise NotImplementedError()

    def get_batch_generator(
        self, batch_size: int
    ) -> Generator[Union[pd.DataFrame, List], None, None]:
        """Get batch generator."""
        data_length = len(self.data)
        indices = np.random.permutation(data_length)
        for i in range(0, data_length, batch_size):
            if isinstance(self.data, pd.DataFrame):
                yield self.data.iloc[indices[i : i + batch_size], :]
            else:
                yield list(self.data[k] for k in indices[i : i + batch_size])

    @classmethod
    def is_match(cls, input_file_path: str, options: Optional[Dict]) -> bool:
        """Return true if match, false otherwise."""
        raise NotImplementedError()

    def reload(
        self, input_file_path: Optional[str], data: Any, options: Optional[Dict]
    ) -> None:
        """
        Reload the data class with a new dataset.

        This erases all existing
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
            raise ValueError("Reloaded dataset does not match the specified data_type")
        elif input_file_path:
            self.input_file_path = input_file_path
        self._data = None
        self._tmp_file_name = None
        self.options = None
        self._batch_info = dict(perm=list(), iter=0)

    def __len__(self) -> int:
        """
        Return the length of the dataset which is loaded.

        :return: length of the dataset
        """
        return len(self.data)

    @property
    def length(self) -> int:
        """
        Return the length of the dataset which is loaded.

        :return: length of the dataset
        """
        return len(self)

    def __getattribute__(self, name: Any) -> Any:
        """
        Override getattr for the class.

        Do this such that functions can be applied
        directly to the data class if the function is not part of the data
        class.
        e.g. if data is BaseData where self.data = [1, 2, 3, 1]
        ```
        data.count(1)  # returns 2, bc data.data has the function 'count'
        ```
        """
        try:
            returned = object.__getattribute__(self, name)
        except AttributeError as attr_error:
            class_name = self.__class__.__name__
            data_class_name = self.data.__class__.__name__
            if not f"'{class_name}' object has no attribute '{name}'" == str(
                attr_error
            ):
                raise
            try:
                returned = object.__getattribute__(self.data, name)
            except AttributeError as attr_error:
                if not f"'{data_class_name}' object has no attribute '{name}'" == str(
                    attr_error
                ):
                    raise
                raise AttributeError(
                    f"Neither '{class_name}' nor "
                    f"'{data_class_name}' objects have "
                    f"attribute '{name}'"
                )
        return returned
