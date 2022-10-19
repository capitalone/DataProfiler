"""Contains factory class reading various kinds of data."""
from __future__ import absolute_import, division

from io import BytesIO
from typing import Any, Dict, List, Optional, Union, cast

from .. import dp_logging
from .avro_data import AVROData
from .csv_data import CSVData
from .data_utils import is_valid_url, url_to_bytes
from .graph_data import GraphData
from .json_data import JSONData
from .parquet_data import ParquetData
from .text_data import TextData

logger = dp_logging.get_child_logger(__name__)


class Data(object):
    """Factory class for reading various kinds of data."""

    data_classes: List[Dict] = [
        dict(data_class=JSONData, kwargs=dict()),
        dict(data_class=GraphData, kwargs=dict()),
        dict(data_class=CSVData, kwargs=dict()),
        dict(data_class=ParquetData, kwargs=dict()),
        dict(data_class=AVROData, kwargs=dict()),
        dict(data_class=TextData, kwargs=dict()),
    ]

    def __new__(
        cls,
        input_file_path: Optional[Union[str, BytesIO]] = None,
        data: Optional[Any] = None,
        data_type: Optional[str] = None,
        options: Optional[Dict] = None,
    ):
        """
        Create Factory Data object.

        Auto-detection of data type if not specified for input files.
        Returns the proper data class or specified data class for
        the given data or input file.

        :param input_file_path:
        :param data:
        :param data_type:
        :param options:
        :return:
        """
        if not input_file_path and data is None:
            raise ValueError("At least need to pass data or point to a data file.")

        if input_file_path and data is not None:
            raise ValueError(
                "Either initialize from data or point to a data file. "
                "Cannot do both at the same time."
            )

        if data is not None and not data_type:
            raise ValueError(
                "In memory data must be specified as a specific data type."
            )

        if not options:
            options = cast(Dict, dict())

        if is_valid_url(input_file_path):
            input_file_path = url_to_bytes(input_file_path, options)

        for data_class_info in cls.data_classes:
            data_class = data_class_info["data_class"]
            kwargs = data_class_info["kwargs"]
            options_copy = options.copy()
            options_copy.update(kwargs)
            if (not data_type or data_type == data_class.data_type) and (
                data is not None or data_class.is_match(input_file_path, options_copy)
            ):
                return data_class(input_file_path, data, options_copy)

        raise ValueError("No data class types matched the input.")
