"""Contains ProfilerEncoder class."""
import json

import numpy as np
import pandas as pd

from ..labelers.base_data_labeler import BaseDataLabeler
from . import base_column_profilers, column_profile_compilers, numerical_column_stats


class ProfileEncoder(json.JSONEncoder):
    """JSONify profiler objects and it subclasses and contents."""

    def default(self, to_serialize):
        """
        Specify how an object should be serialized.

        :param to_serialize: an object to be serialized
        :type to_serialize: a BaseColumnProfile object
        :return: a datatype serializble by json.JSONEncoder
        """
        if isinstance(
            to_serialize,
            (
                base_column_profilers.BaseColumnProfiler,
                numerical_column_stats.NumericStatsMixin,
                column_profile_compilers.BaseCompiler,
            ),
        ):
            return {"class": type(to_serialize).__name__, "data": to_serialize.__dict__}
        elif isinstance(to_serialize, np.integer):
            return int(to_serialize)
        elif isinstance(to_serialize, np.ndarray):
            return to_serialize.tolist()
        elif isinstance(to_serialize, pd.Timestamp):
            return to_serialize.isoformat()
        elif isinstance(to_serialize, BaseDataLabeler):
            return to_serialize._default_model_loc
        elif callable(to_serialize):
            return to_serialize.__name__
        return json.JSONEncoder.default(self, to_serialize)
