from __future__ import absolute_import
from __future__ import division
import logging

from .csv_data import CSVData
from .json_data import JSONData
from .text_data import TextData
from .parquet_data import ParquetData
from .avro_data import AVROData

logger = logging.getLogger('DataProfiler.data')


class Data(object):

    data_classes = [
        dict(data_class=JSONData, kwargs=dict()),
        dict(data_class=CSVData, kwargs=dict()),
        dict(data_class=ParquetData, kwargs=dict()),
        dict(data_class=AVROData, kwargs=dict()),
        dict(data_class=TextData, kwargs=dict()),
    ]

    def __new__(cls, input_file_path=None, data=None, data_type=None, options=None):
        """
        Factory Data class. Auto-detection of data type if not specified for
        input files. Returns the proper data class or specified data class for
        the given data or input file.
        
        :param input_file_path:
        :param data:
        :param data_type:
        :param options:
        :return:
        """

        if not input_file_path and data is None:
            raise ValueError(
                'At least need to pass data or point to a data file.'
            )

        if input_file_path and data is not None:
            raise ValueError(
                'Either initialize from data or point to a data file. '
                'Cannot do both at the same time.'
            )

        if data is not None and not data_type:
            raise ValueError(
                'In memory data must be specified as a specific data type.'
            )

        if not options:
            options = dict()

        for data_class_info in cls.data_classes:
            data_class = data_class_info["data_class"]
            kwargs = data_class_info["kwargs"]
            options_copy = options.copy()
            options_copy.update(kwargs)
            if (not data_type or data_type == data_class.data_type) and \
                   (data is not None or data_class.is_match(input_file_path, options_copy)):
                return data_class(input_file_path, data, options_copy)

        raise ValueError('No data class types matched the input.')
