"""File contains the version number for the package."""

MAJOR = 0
MINOR = 8
MICRO = 4
POST = None  # otherwise None

VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

_post_str = ""
if POST:
    _post_str = f".post{POST}"
__version__ = VERSION + _post_str
