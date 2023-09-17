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
from .profile_builder import (
    Profiler,
    StructuredColProfiler,
    StructuredProfiler,
    UnstructuredProfiler,
)
from .profiler_options import (
    BaseInspectorOptions,
    BooleanOption,
    CategoricalOptions,
    CorrelationOptions,
    DataLabelerOptions,
    DateTimeOptions,
    FloatOptions,
    HistogramAndQuantilesOption,
    HyperLogLogOptions,
    IntOptions,
    ModeOption,
    NumericalOptions,
    OrderOptions,
    PrecisionOptions,
    ProfilerOptions,
    RowStatisticsOptions,
    StructuredOptions,
    TextOptions,
    TextProfilerOptions,
    UniqueCountOptions,
    UnstructuredOptions,
)
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
    ColumnDataLabelerCompiler.__name__: ColumnDataLabelerCompiler,
    ColumnPrimitiveTypeProfileCompiler.__name__: ColumnPrimitiveTypeProfileCompiler,
    ColumnStatsProfileCompiler.__name__: ColumnStatsProfileCompiler,
}

json_decoder._options = {
    BooleanOption.__name__: BooleanOption,
    "HistogramOption": HistogramAndQuantilesOption,
    HistogramAndQuantilesOption.__name__: HistogramAndQuantilesOption,
    ModeOption.__name__: ModeOption,
    BaseInspectorOptions.__name__: BaseInspectorOptions,
    NumericalOptions.__name__: NumericalOptions,
    IntOptions.__name__: IntOptions,
    PrecisionOptions.__name__: PrecisionOptions,
    FloatOptions.__name__: FloatOptions,
    TextOptions.__name__: TextOptions,
    DateTimeOptions.__name__: DateTimeOptions,
    OrderOptions.__name__: OrderOptions,
    CategoricalOptions.__name__: CategoricalOptions,
    CorrelationOptions.__name__: CorrelationOptions,
    UniqueCountOptions.__name__: UniqueCountOptions,
    HyperLogLogOptions.__name__: HyperLogLogOptions,
    RowStatisticsOptions.__name__: RowStatisticsOptions,
    DataLabelerOptions.__name__: DataLabelerOptions,
    TextProfilerOptions.__name__: TextProfilerOptions,
    StructuredOptions.__name__: StructuredOptions,
    UnstructuredOptions.__name__: UnstructuredOptions,
    ProfilerOptions.__name__: ProfilerOptions,
}


json_decoder._profilers = {
    StructuredProfiler.__name__: StructuredProfiler,
}

json_decoder._structured_col_profiler = {
    StructuredColProfiler.__name__: StructuredColProfiler,
}
