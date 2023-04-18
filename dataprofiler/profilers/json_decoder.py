"""Contains methods to decode components of a Profiler."""

from .base_column_profilers import BaseColumnProfiler
from .categorical_column_profile import CategoricalColumn


def get_column_profiler_class(class_name: str) -> BaseColumnProfiler:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseColumnProfiler.

    :param class_name: name of BaseColumnProfiler subclass retreived by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of BaseColumnProfiler object
    """
    if class_name == CategoricalColumn.__name__:
        return CategoricalColumn(None)
    else:
        raise ValueError(f"Invalid profiler class {class_name} " f"failed to load.")


def decode_categorical_column(to_decode: dict):
    """
    Specify how CategoricalColumn should be deserialized.

    :param to_decode: an object to be deserialized
    :type to_serialize: a dictionary resullting from json.loads()
    :return: CategoricalColumn object
    """
    decoded = CategoricalColumn(to_decode["name"])
    for attr, value in to_decode.items():
        decoded.__setattr__(attr, value)
