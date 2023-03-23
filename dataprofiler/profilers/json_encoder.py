"""Contains ProfilerEncoder class."""

import json

import numpy as np
import pandas as pd

from . import base_column_profilers


class ProfileEncoder(json.JSONEncoder):
    """JSONify profiler objects and it subclasses and contents."""

    def default(self, to_serialize):
        """
        Specify how an object should be serialized.

        :param to_serialize: an object to be serialized
        :type to_serialize: a BaseColumnProfile object
        :return: a datatype serializble by json.JSONEncoder
        """
        if isinstance(to_serialize, base_column_profilers.BaseColumnProfiler):
            return to_serialize.__dict__
        elif isinstance(to_serialize, np.integer):
            return int(to_serialize)
        elif isinstance(to_serialize, pd.Timestamp):
            return to_serialize.isoformat()

        return json.JSONEncoder.default(self, to_serialize)
