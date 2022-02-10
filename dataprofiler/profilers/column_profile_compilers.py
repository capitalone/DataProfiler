from future.utils import with_metaclass
import abc
from collections import OrderedDict

from . import utils
from . import DateTimeColumn, IntColumn, FloatColumn, TextColumn
from . import OrderColumn, CategoricalColumn
from . import DataLabelerColumn, UnstructuredLabelerProfile
from .unstructured_text_profile import TextProfiler
from .profiler_options import UnstructuredOptions, StructuredOptions


class BaseCompiler(with_metaclass(abc.ABCMeta, object)):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = list()

    _option_class = None

    def __repr__(self):
        return self.__class__.__name__

    def __init__(self, df_series=None, options=None, pool=None):
        if not self._profilers:
            raise NotImplementedError("Must add profilers.")

        if self._option_class is None:
            raise NotImplementedError("Must set the expected OptionClass.")

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

        if not self._profilers:
            return

        enabled_profiles = None
        if options and isinstance(options, self._option_class):
            enabled_profiles = options.enabled_profiles

        # Create profiles
        for profiler in self._profilers:

            # Create profile if options allow for it or if there are no options
            if enabled_profiles is None or profiler.type in enabled_profiles:

                profiler_options = None
                if options and options.properties[profiler.type]:
                    profiler_options = options.properties[profiler.type]

                try:
                    self._profiles[profiler.type] = profiler(
                        df_series.name, options=profiler_options)
                except Exception as e:
                    utils.warn_on_profile(profiler.type, e)

        # Update profile after creation
        self.update_profile(df_series, pool)

    def __add__(self, other):
        """
        Merges two profile compilers together overriding the `+` operator.

        :param other: profile compiler being add to this one.
        :type other: BaseCompiler
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

    def diff(self, other, options=None):
        """
        Finds the difference between 2 compilers and returns the report

        :param other: profile compiler finding the difference with this one.
        :type other: BaseCompiler
        :return: difference of the profiles
        :rtype: dict
        """
        if type(other) is not type(self):
            raise TypeError('`{}` and `{}` are not of the same profile compiler'
                            ' type.'.format(type(self).__name__,
                                            type(other).__name__))
        return {}

    def update_profile(self, df_series, pool=None):
        """
        Updates the profiles from the data frames

        :param df_series: a given column, assume df_series in str
        :type df_series: pandas.core.series.Series
        :param pool: pool to utilized for multiprocessing
        :type pool: multiprocessing.Pool
        :return: Self
        :rtype: BaseCompiler
        """

        if not self._profilers:
            return

        # If single process, loop and return
        if pool is None:
            for profile_type in self._profiles:
                self._profiles[profile_type].update(df_series)
            return self

        # If multiprocess, setup pool, etc
        single_process_list = []
        multi_process_dict = {}

        # Spin off separate processes, where possible
        for profile_type in self._profiles:

            if self._profiles[profile_type].thread_safe:

                try:  # Add update function to be applied on the pool
                    multi_process_dict[profile_type] = pool.apply_async(
                        self._profiles[profile_type].update, (df_series,))
                except Exception as e:  # Attempt again as a single process
                    self._profiles[profile_type].thread_safe = False

            if not self._profiles[profile_type].thread_safe:
                single_process_list.append(profile_type)

        # Single process thread to loop through any known unsafe
        for profile_type in single_process_list:
            self._profiles[profile_type].update(df_series)

        # Loop through remaining multi-processes and close them out
        single_process_list = []
        for profile_type in multi_process_dict.keys():
            try:
                returned_profile = multi_process_dict[profile_type].get()
                if returned_profile is not None:
                    self._profiles[profile_type] = returned_profile
            except Exception as e:  # Attempt again as a single process
                self._profiles[profile_type].thread_safe = False
                single_process_list.append(profile_type)

                # Single process thread to loop through
        for profile_type in single_process_list:
            self._profiles[profile_type].update(df_series)
        return self


class ColumnPrimitiveTypeProfileCompiler(BaseCompiler):
    
    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        DateTimeColumn,
        IntColumn,
        FloatColumn,
        TextColumn,
    ]
    _option_class = StructuredOptions

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
                    "data_type": profiler.type,
                    "statistics": profiler.profile,
                })
                has_found_match = True
            profile["data_type_representation"].update(
                dict([(profiler.type, profiler.data_type_ratio)])
            )
        return profile

    @property
    def selected_data_type(self):
        """
        Finds the selected data_type in a primitive compiler
        
        :return: name of the selected data type
        :rtype: str
        """
        matched_profile = None
        if self._profiles:
            for key, profiler in self._profiles.items():
                if matched_profile is None and profiler.data_type_ratio == 1.0:
                    matched_profile = key
                    return matched_profile
        return matched_profile

    def diff(self, other, options=None):
        """
        Finds the difference between 2 compilers and returns the report
        
        :param other: profile compiler finding the difference with this one.
        :type other: ColumnPrimitiveTypeProfileCompiler
        :return: difference of the profiles
        :rtype: dict
        """
        # Call super for compiler instance check
        diff_profile = super().diff(other, options)

        # Initialize profile diff dict with data type representation
        diff_profile["data_type_representation"] = dict()

        # Iterate through profiles and find the data_type_ratio diffs
        all_profiles = set(self._profiles.keys()) | set(other._profiles.keys())
        if all_profiles:
            for key in all_profiles:
                if key in self._profiles and key in other._profiles:
                    diff = utils.find_diff_of_numbers(self._profiles[key].data_type_ratio,
                                                      other._profiles[key].data_type_ratio)
                elif key in self._profiles:
                    diff = [self._profiles[key].data_type_ratio, None]
                else:
                    diff = [None, other._profiles[key].data_type_ratio]
                diff_profile["data_type_representation"].update({key: diff})


        # Find data_type diff
        data_type1 = self.selected_data_type
        data_type2 = other.selected_data_type
        if data_type1 is not None or data_type2 is not None:
            diff_profile["data_type"] = utils.find_diff_of_strings_and_bools(
                data_type1, data_type2)
            # Find diff of matching profile statistics
            if diff_profile["data_type"] == "unchanged":
                diff_profile["statistics"] = self._profiles[data_type1]\
                    .diff(other._profiles[data_type2], options)

        # If there is no data, pop the data
        if not diff_profile["data_type_representation"]:
            diff_profile.pop("data_type_representation")

        return diff_profile


class ColumnStatsProfileCompiler(BaseCompiler):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        OrderColumn,
        CategoricalColumn,
    ]
    _option_class = StructuredOptions

    @property
    def profile(self):
        profile = dict()
        for _, profiler in self._profiles.items():
            profile.update(profiler.profile)
        return profile

    def diff(self, other, options=None):
        """
        Finds the difference between 2 compilers and returns the report

        :param other: profile compiler finding the difference with this one.
        :type other: ColumnStatsProfileCompiler
        :return: difference of the profiles
        :rtype: dict
        """
        # Call super for compiler instance check
        diff_profile = super().diff(other, options)

        # Iterate through profiles
        all_profiles = set(self._profiles.keys()) | set(other._profiles.keys())
        for key in all_profiles:
            if key in self._profiles and key in other._profiles:
                diff = self._profiles[key].diff(other._profiles[key], 
                                                options)
                diff_profile.update(diff)

        return diff_profile


class ColumnDataLabelerCompiler(BaseCompiler):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        DataLabelerColumn
    ]
    _option_class = StructuredOptions

    @property
    def profile(self):
        profile = {
            "data_label": None,
            "statistics": dict()
        }
        # TODO: Only works for last profiler. Abstracted for now.
        for _, profiler in self._profiles.items():
            col_profile = profiler.profile
            profile["data_label"] = col_profile.pop("data_label")
            profile["statistics"].update(col_profile)
        return profile
    
    def diff(self, other, options=None):
        """
        Finds the difference between 2 compilers and returns the report

        :param other: profile compiler finding the difference with this one.
        :type other: ColumnDataLabelerCompiler
        :param options: options to change results of the difference
        :type options: dict
        :return: difference of the profiles
        :rtype: dict
        """
        # Call super for compiler instance check
        diff_profile = super().diff(other, options)
        diff_profile["statistics"] = dict()

        # Iterate through profile(s)
        all_profiles = set(self._profiles.keys()) & set(other._profiles.keys())
        for key in all_profiles:
            diff = self._profiles[key].diff(other._profiles[key], options)
            diff_profile["data_label"] = diff.pop("data_label")
            diff_profile["statistics"].update(diff)

        if not diff_profile["statistics"]:
            diff_profile.pop("statistics")
        
        return diff_profile


class UnstructuredCompiler(BaseCompiler):

    # NOTE: these profilers are ordered. Test functionality if changed.
    _profilers = [
        TextProfiler,
        UnstructuredLabelerProfile,
    ]

    _option_class = UnstructuredOptions

    @property
    def profile(self):
        profile = {
            "data_label": dict(),
            "statistics": dict()
        }
        if UnstructuredLabelerProfile.type in self._profiles:
            profile["data_label"] = \
                self._profiles[UnstructuredLabelerProfile.type].profile
        if TextProfiler.type in self._profiles:
            profile["statistics"] = \
                self._profiles[TextProfiler.type].profile
        return profile

    def diff(self, other, options=None):
        """
        Finds the difference between 2 compilers and returns the report

        :param other: profile compiler finding the difference with this one.
        :type other: UnstructuredCompiler
        :param options: options to impact the results of the diff
        :type options: dict
        :return: difference of the profiles
        :rtype: dict
        """
        # Call super for compiler instance check
        diff_profile = super().diff(other, options)

        if "data_labeler" in self._profiles and "data_labeler" in other._profiles:
            diff_profile["data_label"] = self._profiles["data_labeler"].\
                diff(other._profiles["data_labeler"], options)

        if "text" in self._profiles and "text" in other._profiles:
            diff_profile["statistics"] = self._profiles["text"].\
                diff(other._profiles["text"], options)

        return diff_profile
