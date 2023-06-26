import importlib
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict

plugins_dict: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)


def plugin_decorator(typ, name):
    def __inner_factory_function(fn):
        global plugins_dict
        plugins_dict[typ][name] = fn
        return fn

    return __inner_factory_function


def loadPlugins():
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
    from . import plugins_dict

    return plugins_dict.get(typ)
