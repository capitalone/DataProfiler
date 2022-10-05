"""Contains functions for checking for installations/dependencies."""
import sys
import warnings
from typing import Any, Callable, List


def warn_missing_module(labeler_function: str, module_name: str) -> None:
    """
    Return a warning if a given graph module doesn't exist.

    :param labeler_function: Name of the graphing function
    :type labeler_function: str
    :param module_name: module name that was missing
    :type module_name: str
    """
    warning_msg = "\n\n!!! WARNING Labeler Failure !!!\n\n"
    warning_msg += "Labeler Function: {}".format(labeler_function)
    warning_msg += "\nMissing Module: {}".format(module_name)
    warning_msg += "\n\nFor labeler errors, try installing "
    warning_msg += "the extra labeler requirements via:\n\n"
    warning_msg += "$ pip install -r requirements-ml.txt\n\n"
    warnings.warn(warning_msg, RuntimeWarning, stacklevel=3)


def require_module(names: List[str]) -> Callable:
    """
    Check if a set of modules exists in sys.modules prior to running function.

    If they do not, give a user a warning and do not run the
    function.

    :param names: list of module names to check for in sys.modules
    :type names: list[str]
    """

    def check_module(f: Callable) -> Callable:
        def new_f(*args: Any, **kwds: Any) -> Any:
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
