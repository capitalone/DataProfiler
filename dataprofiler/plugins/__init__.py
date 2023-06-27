import importlib
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict

plugins_dict: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)


def plugin_decorator(typ, name):
    """
    Populates plugins_dict with decorated plugin functions

    :param typ: Broader classification/type of a plugin
    :param name: Specific name of a plugin
    :return: function
    """

    def __inner_factory_function(fn):
        """
        :param fn: Plugin function
        :return: function
        """
        global plugins_dict
        plugins_dict[typ][name] = fn
        return fn

    return __inner_factory_function


def loadPlugins():
    """
    Digs through plugins folder for possible plugins to be imported
    and consequently added to the plugins_dict if properly decorated

    :return: None
    """
    plugin_path = os.path.dirname(os.path.abspath(__file__))
    for folder in os.listdir(plugin_path):
        option_path = os.path.join(plugin_path, folder)
        if os.path.isdir(option_path):
            for filename in os.listdir(option_path):
                spec = importlib.util.spec_from_file_location(
                    filename, os.path.join(option_path, filename)
                )
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)


def getPlugins(typ):
    """
    Fetches a dictionary of plugins of a certain type

    :param typ: Broader classification/type of a plugin
    :return: dict
    """
    from . import plugins_dict

    return plugins_dict.get(typ)
