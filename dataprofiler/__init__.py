import logging

from .data_readers.data import Data
from .profilers.profile_builder import StructuredProfiler, \
                                       UnstructuredProfiler, Profiler
from .profilers.profiler_options import ProfilerOptions
from .labelers.data_labelers import train_structured_labeler, DataLabeler, \
                                    StructuredDataLabeler, \
                                    UnstructuredDataLabeler
from .validators.base_validators import Validator
from .version import __version__
from . import settings


try:
    import snappy
except ImportError:
    import warnings
    warnings.warn(
        'Snappy must be installed to use parquet/avro datasets.'
        '\n\n'
        'For macOS use Homebrew:\n'
        '\t`brew install snappy`'
        '\n\n'
        'For linux use apt-get:\n`'
        '\tsudo apt-get -y install libsnappy-dev`\n',
        ImportWarning
    )


def set_seed(seed=None):
    # also check it's an integer
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("Seed should be a non-negative integer.")
    settings._seed = seed


def set_verbosity(verbose=None, **kwargs):
    # User can specify kwargs meant to be passed to logging.basicConfig
    if verbose is None:
        # If kwargs given, just pass straight through to basicConfig
        if kwargs:
            logging.basicConfig(**kwargs)
        else:
            raise ValueError("Cannot set verbosity without either verbose "
                             "kwarg or kwargs for logging.basicConfig")
    elif not verbose:
        # Only print warnings and errors
        logging.basicConfig(level=logging.WARNING, **kwargs)
    else:
        # Also print info messages
        logging.basicConfig(level=logging.INFO, **kwargs)
