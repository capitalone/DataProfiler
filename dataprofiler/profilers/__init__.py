"""Package for providing statistics and predictions for a given dataset."""
from . import json_decoder
from .base_column_profilers import BaseColumnProfiler
from .categorical_column_profile import CategoricalColumn
from .column_profile_compilers import (
    BaseCompiler,
    ColumnDataLabelerCompiler,
    ColumnPrimitiveTypeProfileCompiler,
    ColumnStatsProfileCompiler,
)
from .data_labeler_column_profile import DataLabelerColumn
from .datetime_column_profile import DateTimeColumn
from .float_column_profile import FloatColumn
from .int_column_profile import IntColumn
from .numerical_column_stats import NumericStatsMixin
from .order_column_profile import OrderColumn
from .profile_builder import Profiler, StructuredProfiler, UnstructuredProfiler
from .text_column_profile import TextColumn
from .unstructured_labeler_profile import UnstructuredLabelerProfile

# set here to avoid circular imports
json_decoder._profiles = {
    CategoricalColumn.__name__: CategoricalColumn,
    FloatColumn.__name__: FloatColumn,
    IntColumn.__name__: IntColumn,
    DateTimeColumn.__name__: DateTimeColumn,
    OrderColumn.__name__: OrderColumn,
    DataLabelerColumn.__name__: DataLabelerColumn,
    TextColumn.__name__: TextColumn,
}


json_decoder._compilers = {
    ColumnPrimitiveTypeProfileCompiler.__name__: ColumnPrimitiveTypeProfileCompiler,
    ColumnStatsProfileCompiler.__name__: ColumnStatsProfileCompiler,
    ColumnDataLabelerCompiler.__name__: ColumnDataLabelerCompiler,
}
