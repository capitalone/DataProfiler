"""Contains function for generating plugins data."""
from collections import defaultdict
from typing import Any, DefaultDict, Dict

plugins_dict: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)


def plugin_decorator(typ, name):
    """
    Populate plugins_dict with decorated plugin functions.

    :param typ: Broader classification/type of a plugin
    :param name: Specific name of a plugin
    :return: function
    """

    def __inner_factory_function(fn):
        """
        Actual population of plugin_dict.

        :param fn: Plugin function
        :return: function
        """
        global plugins_dict
        plugins_dict[typ][name] = fn
        return fn

    return __inner_factory_function
