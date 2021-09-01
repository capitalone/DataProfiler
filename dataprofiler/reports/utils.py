import sys
import warnings


def warn_missing_module(graph_func, module_name):
    """
    Returns a warning if a given graph module doesn't exist

    :param graph_func: Name of the graphing function
    :type graph_func: str
    :param module_name: module name that was missing
    :type module_name: str
    """
    warning_msg = "\n\n!!! WARNING Graphing Failure !!!\n\n"
    warning_msg += "Graph Function: {}".format(graph_func)
    warning_msg += "\nMissing Module: {}".format(module_name)
    warning_msg += "\n\nFor report errors, try installing "
    warning_msg += "the extra reports requirements via:\n\n"
    warning_msg += "$ pip install dataprofiler[reports] --user\n\n"
    warnings.warn(warning_msg, RuntimeWarning, stacklevel=3)


def require_module(names):
    """
    Decorator to check if a set of modules exist in sys.modules prior to running
    the function. If they do not, give a user a warning and do not run the
    function.

    :param names: list of module names to check for in sys.modules
    :type names: list[str]
    """
    def check_module(f):
        def new_f(*args, **kwds):
            for module_name in names:
                if module_name not in sys.modules.keys():
                    # attempt to reload if missing
                    import importlib
                    importlib.reload(sys.modules[f.__module__])
                    if module_name not in sys.modules.keys():
                        warn_missing_module(f.__name__, module_name)
                        return
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f
    return check_module
