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
    if seed is None or seed >= 0:
        settings._seed = seed
    else:
        warnings.warn("Seed should be a positive integer", RuntimeWarning)