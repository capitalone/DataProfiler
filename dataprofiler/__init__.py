"""Package for dataprofiler."""
from . import settings
from .data_readers.data import Data
from .dp_logging import get_logger, set_verbosity
from .labelers.data_labelers import (
    DataLabeler,
    StructuredDataLabeler,
    UnstructuredDataLabeler,
    train_structured_labeler,
)
from .plugins import load_plugins
from .profilers.graph_profiler import GraphProfiler
from .profilers.profile_builder import (
    Profiler,
    StructuredProfiler,
    UnstructuredProfiler,
)
from .profilers.profiler_options import ProfilerOptions
from .reports import graphs
from .validators.base_validators import Validator
from .version import __version__


def set_seed(seed=None):
    # also check it's an integer
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("Seed should be a non-negative integer.")
    settings._seed = seed


load_plugins()
