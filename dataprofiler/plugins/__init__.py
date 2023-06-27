import importlib
import os

from dataprofiler.plugins.decorators import plugin_decorator, plugins_dict


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
    # from . import plugins_dict

    return plugins_dict.get(typ)
