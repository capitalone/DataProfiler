import logging
import threading

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

        # Set formatting of logs
        stream_handle = logging.StreamHandler()
        stream_handle.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        logger.addHandler(stream_handle)

        _logger = logger
        return _logger
    finally:
        # Unlock before returning
        _logger_lock.release()


def set_verbosity(v):
    """Set verbosity level for DataProfiler logger"""
    get_logger().setLevel(v)
