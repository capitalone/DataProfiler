import logging
import threading
import sys

_logger = None
_logger_lock = threading.Lock()


def get_logger():
    """Access DataProfiler-specific logger"""
    global _logger

    # Return _logger if initialized
    if _logger:
        return _logger

    # Lock to prevent streaming from multiple threads simultaneously
    _logger_lock.acquire()

    try:
        # Initialize specifically to DP logging
        logger = logging.getLogger('DataProfiler')
        logger.setLevel(logging.INFO)

        # Set formatting of logs
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        logger.addHandler(stream_handler)

        _logger = logger
        return _logger
    finally:
        # Unlock before returning
        _logger_lock.release()


def set_verbosity(level):
    """
    Set verbosity level for DataProfiler logger

    :param level: Verbosity level (from logging module)
    :type level: int
    """
    get_logger().setLevel(level)


def get_child_logger(name):
    """
    Returns logger for given filepath

    :param name: name of file in need of accessing child logger
    :type name: str
    :return: Logger instance for given file
    """
    return get_logger().getChild(name.replace('dataprofiler.', ''))
