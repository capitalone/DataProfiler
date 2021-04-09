from future.utils import with_metaclass
import abc
from collections import OrderedDict

import pandas as pd

from . import utils
from . import DateTimeColumn, IntColumn, FloatColumn, TextColumn
from . import OrderColumn, CategoricalColumn
from . import DataLabelerColumn
from .profiler_options import StructuredOptions


class BaseColumnProfileCompiler(with_metaclass(abc.ABCMeta, object)):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = list()

    def __repr__(self):
        return self.__class__.__name__

    def __init__(self, df_series=None, options=None, pool=None):
        if not self._profilers:
            raise NotImplementedError("Must add profilers.")

        self._profiles = OrderedDict()
        if df_series is not None:
            self.name = df_series.name
            self._create_profile(df_series, options, pool)

        
    @property
    @abc.abstractmethod
    def profile(self):
        raise NotImplementedError()
    
    def _create_profile(self, df_series, options=None, pool=None):
        """
        Initializes and evaluates all profilers for the given dataframe.
        
        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :param options: Options for the structured profiler
        :type options: StructuredOptions
        :return: None
        :rtype: None
        """

        if len(self._profilers) == 0:
            return

        selected_col_profiles = None
        if options and isinstance(options, StructuredOptions):
            selected_col_profiles = options.enabled_columns

        # Create profiles
        for col_profile_type in self._profilers:
            
            # Create profile if options allow for it or if there are no options
            if selected_col_profiles is None or \
               col_profile_type.col_type in selected_col_profiles:

                col_options = None
                if options and options.properties[col_profile_type.col_type]:
                    col_options = options.properties[col_profile_type.col_type]
                    
                try:
                    self._profiles[col_profile_type.col_type] = \
                        col_profile_type(df_series.name, options=col_options)
                except Exception as e:
                    utils.warn_on_profile(col_profile_type.col_type, e)

        # Update profile after creation
        self.update_profile(df_series, pool)

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
        merged_profile_compiler = self.__class__()
        merged_profile_compiler.name = self.name
        for profile_name in self._profiles:
            merged_profile_compiler._profiles[profile_name] = (
                self._profiles[profile_name] + other._profiles[profile_name]
            )
        return merged_profile_compiler

    def update_profile(self, df_series, pool=None):
        """
        Updates the profiles from the data frames
        
        :param df_series: a given column, assume df_series in str
        :type df_series: pandas.core.series.Series
        :param pool: pool to utilized for multiprocessing
        :type pool: multiprocessing.Pool
        :return: Self
        :rtype: BaseColumnProfileCompiler
        """
        
        if len(self._profilers) == 0:
            return 
        
        # If single process, loop and return
        if pool is None:
            for col_profile in self._profiles:
                self._profiles[col_profile].update(df_series)
            return self
        
        # If multiprocess, setup pool, etc
        single_process_list = []
        multi_process_dict = {}
        
        # Spin off seperate processes, where possible
        for col_profile in self._profiles:
            
            if self._profiles[col_profile].thread_safe:
                
                try: # Add update function to be applied on the pool
                    multi_process_dict[col_profile] = pool.apply_async(
                        self._profiles[col_profile].update, (df_series,))
                except Exception as e: # Attempt again as a single process
                    self._profiles[col_profile].thread_safe = False
                
            if not self._profiles[col_profile].thread_safe:                
                single_process_list.append(col_profile)

        # Single process thread to loop through any known unsafe
        for col_profile in single_process_list:
            self._profiles[col_profile].update(df_series)
                
        # Loop through remaining multiprocesses and close them out
        single_process_list = []
        for col_profile in multi_process_dict.keys():
            try:
                returned_profile = multi_process_dict[col_profile].get()
                if returned_profile is not None:
                    self._profiles[col_profile] = returned_profile
            except Exception as e: # Attempt again as a single process
                self._profiles[col_profile].thread_safe = False            
                single_process_list.append(col_profile)                
        
        # Single process thread to loop through
        for col_profile in single_process_list:
            self._profiles[col_profile].update(df_series)
        return self


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
                profile.update({
                    "data_type": profiler.col_type,
                    "statistics": profiler.profile,
                })
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
