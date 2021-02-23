import sys
import warnings


def patched_assert_warns(self):
    # This is taken from: https://github.com/rlworkgroup/dowel/pull/36/files
    # Which took it from:
    #   cpython PR#4800. Fixes assertWarns in the presence of modules
    #   that import in response to getattr calls.
    #   The __warningregistry__'s need to be in a pristine state for tests
    #   to work properly.
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}
    self.warnings_manager = warnings.catch_warnings(record=True)
    self.warnings = self.warnings_manager.__enter__()
    warnings.simplefilter('always', self.expected)
    return self
