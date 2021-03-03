from .data_readers.data import Data
from .profilers.profile_builder import Profiler
from .profilers.profiler_options import ProfilerOptions
from .labelers.data_labelers import train_structured_labeler, DataLabeler, \
                                    StructuredDataLabeler, \
                                    UnstructuredDataLabeler
from .validators.base_validators import Validator
from .version import __version__

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
