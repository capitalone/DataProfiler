"""Contains ProfilerEncoder class."""

import json

from . import base_column_profilers


class ProfileEncoder(json.JSONEncoder):
    """JSONify profiler objects and it subclasses and contents."""

    def default(self, o):
        """
        Specify how an object should be serialized.

        :param o: an object to be serialized
        :type o: a BaseColumnProfile object
        :return: a datatype serializble by json.JSONEncoder
        """
        if isinstance(o, base_column_profilers.BaseColumnProfiler):
            return o.__dict__

        return json.JSONEncoder.default(self, o)
