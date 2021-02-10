from future.utils import with_metaclass
import abc
from collections import OrderedDict

import pandas as pd

from . import DateTimeColumn, IntColumn, FloatColumn, TextColumn
from . import OrderColumn, CategoricalColumn
from . import DataLabelerColumn
from .profiler_options import StructuredOptions


class BaseColumnProfileCompiler(with_metaclass(abc.ABCMeta, object)):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = list()

    def __init__(self, df_series, options=None):
        if not self._profilers:
            raise NotImplementedError("Must add profilers.")

        self.name = df_series.name
        self._profiles = OrderedDict()
        self._create_profile(df_series, options)

    @property
    @abc.abstractmethod
    def profile(self):
        raise NotImplementedError()

    def _create_profile(self, df_series, options=None):
        """
        Initializes and evaluates all profilers for the given dataframe.
        
        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :param options: Options for the structured profiler
        :type options: StructuredOptions
        :return: None
        :rtype: None
        """

        # convert all the values to string
        df_series = df_series.apply(str)
        
        selected_columns = None
        if options and isinstance(options, StructuredOptions):
            selected_columns = options.enabled_columns

        for column_type in self._profilers:
            # Create profile if options allow for it or if there are no options
            if selected_columns is None or \
                    column_type.col_type in selected_columns:
                column_options = None
                if options and options.properties[column_type.col_type]:
                    column_options = options.properties[column_type.col_type]
                self._profiles[column_type.col_type] = \
                    column_type(df_series.name, options=column_options)
                self._profiles[column_type.col_type].update(df_series)

    def __add__(self, other):
        """
        Merges two profile compilers together overriding the `+` operator.

        :param other: profile compiler being add to this one.
        :type other: BaseColumnProfileCompiler
        :return: merger of the two column profilers
        """
        if type(other) is not type(self):
            raise TypeError('`{}` and `{}` are not of the same profile compiler'
                            ' type.'.format(type(self).__name__,
                                            type(other).__name__))
        elif self.name != other.name:
            raise ValueError('Column profile names are unmatched: {} != {}'
                             .format(self.name, other.name))
        elif set(self._profiles) != set(other._profiles):  # options check
            raise ValueError('Column profilers were not setup with the same '
                             'options, hence they do not calculate the same '
                             'profiles and cannot be added together.')
        merged_profile_compiler = self.__class__(pd.Series([]))
        merged_profile_compiler.name = self.name
        for profile_name in self._profiles:
            merged_profile_compiler._profiles[profile_name] = (
                self._profiles[profile_name] + other._profiles[profile_name]
            )
        return merged_profile_compiler

    def update_profile(self, df_series):
        """
        Initializes the profiles the column dataframe.
        
        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :return: None
        :rtype: None
        """
        df_series = df_series.apply(str)
        for column_profile in self._profiles:
            self._profiles[column_profile].update(df_series)


class ColumnPrimitiveTypeProfileCompiler(BaseColumnProfileCompiler):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        DateTimeColumn,
        IntColumn,
        FloatColumn,
        TextColumn,
    ]

    @property
    def profile(self):
        profile = {
            "data_type_representation": dict(),
            "data_type": None,
            "statistics": dict()
        }
        has_found_match = False
        for _, profiler in self._profiles.items():
            if not has_found_match and profiler.data_type_ratio == 1.0:
                profile.update(
                    {
                        "data_type": profiler.col_type,
                        "statistics": profiler.profile,
                    }
                )
                has_found_match = True
            profile["data_type_representation"].update(
                dict([(profiler.col_type, profiler.data_type_ratio)])
            )
        return profile


class ColumnStatsProfileCompiler(BaseColumnProfileCompiler):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        OrderColumn,
        CategoricalColumn,
    ]

    @property
    def profile(self):
        profile = dict()
        for _, profiler in self._profiles.items():
            profile.update(profiler.profile)
        return profile


class ColumnDataLabelerCompiler(BaseColumnProfileCompiler):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        DataLabelerColumn
    ]

    @property
    def profile(self):
        profile = {
            "data_label": None,
            "statistics": dict()
        }
        # TODO: Only works for last profiler. Abstracted for now.
        for _, profiler in self._profiles.items():
            profile["data_label"] = profiler.data_label
            profile["statistics"].update(profiler.profile)
        return profile
