"""Contains class for saving and loading text files."""

from io import StringIO
from typing import Dict, List, Optional, Union, cast

from . import data_utils
from .base_data import BaseData


class TextData(BaseData):
    """TextData class to save and load text files."""

    data_type: str = "text"

    def __init__(
        self,
        input_file_path: Optional[str] = None,
        data: Optional[List[str]] = None,
        options: Optional[Dict] = None,
    ) -> None:
        """
        Initialize Data class for loading datasets of type TEXT.

        Can be specified by
        passing in memory data or via a file path. Options pertaining the TEXT
        may also be specified using the options dict parameter.
        Possible Options::

            options = dict(
                data_format= type: str, choices: "text"
                samples_per_line= type: int
            )

        data_format: user selected format in which to return data
        can only be of specified types
        samples_per_line: chunks by which to read in the specified dataset

        :param input_file_path: path to the file being loaded or None
        :type input_file_path: str
        :param data: data being loaded into the class instead of an input file
        :type data: multiple types
        :param options: options pertaining to the data type
        :type options: dict
        :return: None
        """
        if data is not None and not isinstance(data, str):
            raise ValueError("Input data type is not string.")

        options = self._check_and_return_options(options)
        super().__init__(input_file_path, data, options)

        # 'Private' properties
        #  _data_formats: dict containing data_formats (key) and function
        #                 calls (values) which take self._data and convert it
        #                 into the desired data_format for output.
        #  _selected_data_format: user selected format in which to return data
        #                         can only be of types in _data_formats
        #  _delimiter: delimiter used to decipher the csv input file
        #  _selected_columns: columns being selected from the entire dataset
        self._data_formats["text"] = self._get_data_as_text
        self._selected_data_format: str = options.get("data_format", "text")
        self._samples_per_line: int = options.get("samples_per_line", int(5e9))

        if data is not None:
            self._load_data(data)

    @property
    def samples_per_line(self) -> int:
        """Return samples per line."""
        return self._samples_per_line

    @property
    def is_structured(self) -> bool:
        """Determine compatibility with StructuredProfiler."""
        return False

    def _load_data(self, data: Optional[List[str]] = None) -> None:
        """Load data."""
        if data is not None:
            self._data = data
        else:
            self._data = data_utils.read_text_as_list_of_strs(
                cast(str, self.input_file_path), self.file_encoding
            )

    def _get_data_as_text(self, data: Union[str, List[str]]) -> List[str]:
        """Return data as text."""
        if isinstance(data, list) and len(data) and isinstance(data[0], str):
            data = "".join(data)
        elif not isinstance(data, str) and data:
            raise ValueError(
                "Data is not in a str or list of str format and cannot be " "converted."
            )

        data = cast(str, data)
        samples_per_line = min(max(len(data), 1), self.samples_per_line)
        data = [
            data[i * samples_per_line : (i + 1) * samples_per_line]
            for i in range((len(data) + samples_per_line - 1) // samples_per_line)
        ]
        return data

    def tokenize(self) -> None:
        """Tokenize data."""
        raise NotImplementedError("Tokenizing does not currently exist for text data.")

    @classmethod
    def is_match(cls, file_path: str, options: Optional[Dict] = None) -> bool:
        """
        Return True if all are text files.

        :param file_path: path to the file to be examined
        :type file_path: str
        :param options: text file read options
        :type options: dict
        :return: is file a text file or not
        :rtype: bool
        """
        if options is None:
            options = {}

        # if user passes options, this will update them for encodings
        if "encoding" not in options and not isinstance(file_path, StringIO):
            options = {"encoding": data_utils.detect_file_encoding(file_path)}
        return True

    def reload(
        self,
        input_file_path: Optional[str] = None,
        data: Optional[List[str]] = None,
        options: Optional[Dict] = None,
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
        super().reload(input_file_path, data, options)
        TextData.__init__(self, self.input_file_path, data, options)
