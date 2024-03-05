"""Contains ProfilerEncoder class."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..labelers.base_data_labeler import BaseDataLabeler
from . import (
    base_column_profilers,
    column_profile_compilers,
    numerical_column_stats,
    profile_builder,
    profiler_options,
)


class ProfileEncoder(json.JSONEncoder):
    """JSONify profiler objects and it subclasses and contents."""

    def default(self, to_serialize):
        """
        Specify how an object should be serialized.

        :param to_serialize: an object to be serialized
        :type to_serialize: a BaseColumnProfile object

        :raises: NotImplementedError

        :return: a datatype serializble by json.JSONEncoder
        """
        if isinstance(to_serialize, profile_builder.UnstructuredProfiler):
            raise NotImplementedError(
                "UnstructuredProfiler serialization not supported."
            )

        if isinstance(
            to_serialize,
            (
                base_column_profilers.BaseColumnProfiler,
                numerical_column_stats.NumericStatsMixin,
                column_profile_compilers.BaseCompiler,
                profiler_options.BaseOption,
                profile_builder.BaseProfiler,
                profile_builder.StructuredColProfiler,
            ),
        ):
            return {"class": type(to_serialize).__name__, "data": to_serialize.__dict__}
        elif isinstance(to_serialize, set):
            return list(to_serialize)
        elif isinstance(to_serialize, np.integer):
            return int(to_serialize)
        elif isinstance(to_serialize, np.ndarray):
            return to_serialize.tolist()
        elif isinstance(to_serialize, (pd.Timestamp, datetime)):
            return to_serialize.isoformat()
        elif isinstance(to_serialize, BaseDataLabeler):
            # TODO: This does not allow the user to serialize a model if it is loaded
            # "from_disk". Changes to BaseDataLabeler are needed for this feature
            if to_serialize._default_model_loc is None:
                raise ValueError(
                    "Serialization cannot be done on labelers with "
                    "_default_model_loc not set"
                )

            return {"from_library": to_serialize._default_model_loc}

        elif callable(to_serialize):
            return to_serialize.__name__

        return json.JSONEncoder.default(self, to_serialize)
