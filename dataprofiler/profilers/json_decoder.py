"""Contains methods to decode components of a Profiler."""
from typing import Dict, Optional, Type

from .base_column_profilers import BaseColumnProfiler
from .categorical_column_profile import CategoricalColumn
from .float_column_profile import FloatColumn
from .int_column_profile import IntColumn


def get_column_profiler_class(class_name: str) -> Type[BaseColumnProfiler]:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseColumnProfiler.

    :param class_name: name of BaseColumnProfiler subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of BaseColumnProfiler object
    """
    profiles: Dict[str, Type[BaseColumnProfiler]] = {
        CategoricalColumn.__name__: CategoricalColumn,
        FloatColumn.__name__: FloatColumn,
        IntColumn.__name__: IntColumn,
    }

    profile_class: Optional[Type[BaseColumnProfiler]] = profiles.get(class_name)
    if profile_class is None:
        raise ValueError(f"Invalid profiler class {class_name} " f"failed to load.")
    return profile_class


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
    column_profiler_cls: Type[BaseColumnProfiler] = get_column_profiler_class(
        serialized_json["class"]
    )
    return column_profiler_cls.load_from_dict(serialized_json["data"])
