import logging
import threading
import sys

_dp_logger = None
_dp_logger_lock = threading.Lock()


def get_logger():
    """Access DataProfiler-specific logger"""
    global _dp_logger

    # Return _logger if initialized
    if _dp_logger:
        return _dp_logger

    # Lock to prevent streaming from multiple threads simultaneously
    _dp_logger_lock.acquire()

    try:
        # Initialize specifically to DP logging
        logger = logging.getLogger('DataProfiler')
        logger.setLevel(logging.INFO)

        # Set formatting of logs
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        logger.addHandler(stream_handler)

        _dp_logger = logger
        return _dp_logger
    finally:
        # Unlock before returning
        _dp_logger_lock.release()


def set_verbosity(level):
    """
    Set verbosity level for DataProfiler logger. Must set it to one of
    [logging.NOTSET, logging.DEBUG, logging.INFO,
     logging.WARNING, logging.ERROR, logging.CRITICAL]

    :param level: Verbosity level from logging module
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
