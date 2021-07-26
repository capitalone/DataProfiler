import logging
import threading

_logger = None
_logger_lock = threading.Lock()


def get_logger():
    global _logger

    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        logger = logging.getLogger('DataProfiler')

        stream_handle = logging.StreamHandler()
        stream_handle.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        logger.addHandler(stream_handle)

        _logger = logger
        return _logger
    finally:
        _logger_lock.release()


def set_verbosity(v):
    get_logger().setLevel(v)
