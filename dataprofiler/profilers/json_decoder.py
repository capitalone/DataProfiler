"""Contains methods to decode components of a Profiler."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import column_profile_compilers as col_pro_compiler

    from .base_column_profilers import BaseColumnProfiler
    from .profile_builder import BaseProfiler, StructuredColProfiler
    from .profiler_options import BaseOption


# default, but set in the local __init__ to avoid circular imports
_profiles: dict[str, type[BaseColumnProfiler]] = {}
_profilers: dict[str, type[BaseProfiler]] = {}
_compilers: dict[str, type[col_pro_compiler.BaseCompiler]] = {}
_options: dict[str, type[BaseOption]] = {}
_structured_col_profiler: dict[str, type[StructuredColProfiler]] = {}


def get_column_profiler_class(class_name: str) -> type[BaseColumnProfiler]:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseColumnProfiler.

    :param class_name: name of BaseColumnProfiler subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of BaseColumnProfiler object
    """
    profile_class: type[BaseColumnProfiler] | None = _profiles.get(class_name)
    if profile_class is None:
        raise ValueError(f"Invalid profiler class {class_name} " f"failed to load.")
    return profile_class


def get_compiler_class(class_name: str) -> type[col_pro_compiler.BaseCompiler]:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseCompiler.

    :param class_name: name of BaseCompiler subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of BaseCompiler object
    """
    compiler_class: type[col_pro_compiler.BaseCompiler] | None = _compilers.get(
        class_name
    )
    if compiler_class is None:
        raise ValueError(f"Invalid compiler class {class_name} " f"failed to load.")
    return compiler_class


def get_option_class(class_name: str) -> type[BaseOption]:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseOptions.

    :param class_name: name of BaseOptions subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of BaseOptions object
    """
    options_class: type[BaseOption] | None = _options.get(class_name)
    if options_class is None:
        raise ValueError(f"Invalid option class {class_name} " f"failed to load.")

    if class_name == "HistogramOption":
        warnings.warn(
            f"{class_name} will be deprecated in the future. During the JSON encode"
            " process, HistogramOption is mapped to HistogramAndQuantilesOption. "
            "Please begin utilizing the new HistogramAndQuantilesOption class.",
            DeprecationWarning,
        )
    return options_class


def get_profiler_class(class_name: str) -> type[BaseProfiler]:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        BaseProfiler.

    :param class_name: name of BaseProfiler subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class

    :raises: ValueError if the profiler class does not exist

    :return: subclass of BaseProfiler object
    """
    profiler_class: type[BaseProfiler] | None = _profilers.get(class_name)
    if profiler_class is None:
        raise ValueError(f"Invalid profiler class {class_name} " f"failed to load.")
    return profiler_class


def get_structured_col_profiler_class(class_name: str) -> type[StructuredColProfiler]:
    """
    Use name of class to return default-constructed version of that class.

    Raises ValueError if class_name is not name of a subclass of
        StructuredColProfiler.
    :param class_name: name of StructuredColProfiler subclass retrieved by
        calling type(instance).__name__
    :type class_name: str representing name of class
    :return: subclass of StructuredColProfiler object
    """
    struct_col_profiler_class: None | (
        type[StructuredColProfiler]
    ) = _structured_col_profiler.get(class_name)
    if struct_col_profiler_class is None:
        raise ValueError(
            f"Invalid structured col profiler class {class_name} " f"failed to load."
        )
    return struct_col_profiler_class


def load_column_profile(
    serialized_json: dict, config: dict | None = None
) -> BaseColumnProfiler:
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
    :param config: config for overriding data params when loading from dict
    :type config: Dict | None

    :return: subclass of BaseColumnProfiler that has been deserialized from
        JSON

    """
    column_profiler_cls: type[
        BaseColumnProfiler[BaseColumnProfiler]
    ] = get_column_profiler_class(serialized_json["class"])
    return column_profiler_cls.load_from_dict(serialized_json["data"], config)


def load_compiler(
    serialized_json: dict, config: dict | None = None
) -> col_pro_compiler.BaseCompiler:
    """
    Construct subclass of BaseCompiler given a serialized JSON.

    Expected format of serialized_json (see json_encoder):
        {
            "class": <str name of class that was serialized>
            "data": {
                <attr1>: <value1>
                <attr2>: <value2>
                ...
            }
        }

    :param serialized_json: JSON representation of profile compiler that was
        serialized using the custom encoder in profilers.json_encoder
    :type serialized_json: a dict that was created by calling json.loads on
        a JSON representation using the custom encoder
    :param config: config for overriding data params when loading from dict
    :type config: Dict | None
    :return: subclass of BaseCompiler that has been deserialized from
        JSON

    """
    column_profiler_cls: type[col_pro_compiler.BaseCompiler] = get_compiler_class(
        serialized_json["class"]
    )
    return column_profiler_cls.load_from_dict(serialized_json["data"], config)


def load_option(serialized_json: dict, config: dict | None = None) -> BaseOption:
    """
    Construct subclass of BaseOption given a serialized JSON.

    Expected format of serialized_json (see json_encoder):
        {
            "class": <str name of class that was serialized>
            "data": {
                <attr1>: <value1>
                <attr2>: <value2>
                ...
            }
        }

    :param serialized_json: JSON representation of option that was
        serialized using the custom encoder in profilers.json_encoder
    :type serialized_json: a dict that was created by calling json.loads on
        a JSON representation using the custom encoder
    :param config: config for overriding data params when loading from dict
    :type config: Dict | None
    :return: subclass of BaseOption that has been deserialized from
        JSON

    """
    option_cls: type[BaseOption] = get_option_class(serialized_json["class"])
    return option_cls.load_from_dict(serialized_json["data"], config)


def load_profiler(serialized_json: dict, config=None) -> BaseProfiler:
    """
    Construct subclass of BaseProfiler given a serialized JSON.

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
    :param config: config for overriding data params when loading from dict
    :type config: Dict | None
    :return: subclass of BaseProfiler that has been deserialized from
        JSON
    """
    profiler_cls: type[BaseProfiler] = get_profiler_class(serialized_json["class"])
    return profiler_cls.load_from_dict(serialized_json["data"], config)


def load_structured_col_profiler(
    serialized_json: dict, config: dict | None = None
) -> StructuredColProfiler:
    """
    Construct subclass of BaseProfiler given a serialized JSON.

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
    :param config: config for overriding data params when loading from dict
    :type config: Dict | None
    :return: subclass of BaseCompiler that has been deserialized from
        JSON
    """
    profiler_cls: type[StructuredColProfiler] = get_structured_col_profiler_class(
        serialized_json["class"]
    )
    return profiler_cls.load_from_dict(serialized_json["data"], config)
