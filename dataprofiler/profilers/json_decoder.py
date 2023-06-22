"""Contains methods to decode components of a Profiler."""

import json

from .base_column_profilers import BaseColumnProfiler
from .categorical_column_profile import CategoricalColumn


def get_column_profiler_class(class_name: str) -> BaseColumnProfiler:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseColumnProfiler.

    :param class_name: name of BaseColumnProfiler subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of BaseColumnProfiler object
    """
    profiles = {
        CategoricalColumn.__name__: CategoricalColumn,
    }

    profile_class = profiles.get(class_name)
    if profile_class is None:
        raise ValueError(f"Invalid profiler class {class_name} " f"failed to load.")
    profiler: BaseColumnProfiler = profile_class(None)
    return profiler


def load_column_profile(serialized_json: dict) -> BaseColumnProfiler:
    """
    Construct subclass of BaseColumnProfiler given a serialized JSON.

    Expected format of serialized_json (see json_encoder):
        {
            "class": <str name of class that was serialized>
            "data": {
                <attr1>: <value1>
                <attr2>: <value2>
                ...
            }
        }

    :param serialized_json: JSON representation of column profiler that was
        serialized using the custom encoder in profilers.json_encoder
    :type serialized_json: a dict that was created by calling json.loads on
        a JSON representation using the custom encoder
    :return: subclass of BaseColumnProfiler that has been deserialized from
        JSON
    """
    column_profiler = get_column_profiler_class(serialized_json["class"])
    for attr, value in serialized_json["data"].items():
        column_profiler.__setattr__(attr, value)

    return column_profiler


def decode_column_profiler(serialized: str) -> BaseColumnProfiler:
    """
    Construct subclass of BaseColumnProfiler given a serialized JSON.

    :param serialized: JSON representation of column profiler that was
        serialized using the custom encoder in profilers.json_encoder
    :type serialized: a JSON str serialized using the custom decoder
    :return: subclass of BaseColumnProfiler that has been deserialized from
        JSON
    """
    return load_column_profile(json.loads(serialized))
