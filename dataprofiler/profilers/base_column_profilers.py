#!/usr/bin/env python
"""Contains parent column profiler class."""

from __future__ import annotations

import abc
import warnings
from collections import defaultdict
from typing import Any, Callable, Generic, TypeVar

import numpy as np
import pandas as pd

from . import profiler_utils
from .profiler_options import BaseInspectorOptions, BaseOption

BaseColumnProfilerT = TypeVar("BaseColumnProfilerT", bound="BaseColumnProfiler")


class BaseColumnProfiler(Generic[BaseColumnProfilerT], metaclass=abc.ABCMeta):
    """Abstract class for profiling a column of data."""

    col_type = None

    # This specifies the minimum percent of elements in a column to meet the
    # matching condition so that the column is classified as that type.
    _COLUMN_MATCH_THRESHOLD = 0.9

    _SAMPLING_RATIO = 0.20
    _MIN_SAMPLING_COUNT = 500

    def __init__(self, name: str | None, options: BaseOption | None = None):
        """
        Initialize base class properties for the subclass.

        :param name: Name of the dataset
        :type name: String
        """
        self.name: str | None = name
        self.col_index = np.nan
        self.sample_size: int = 0
        self.metadata: dict = dict()
        self.times: dict = defaultdict(float)
        self.thread_safe: bool = True

    # TODO: Not needed for data labeling
    @staticmethod
    def _combine_unique_sets(a: list, b: list) -> list:
        """
        Unify two lists.

        :type a: list
        :type b: list
        :rtype: list
        """
        combined_list: set = set()
        if not a and not b:
            combined_list = set()
        elif not a:
            combined_list = set(b)
        elif not b:
            combined_list = set(a)
        else:
            combined_list = set().union(a, b)
        return list(combined_list)

    @staticmethod
    def _timeit(method: Callable = None, name: str = None) -> Callable:
        """
        Measure execution time of provided method.

        Records time into times dictionary.

        :param method: method to time
        :type method: Callable
        :param name: key argument for the times dictionary
        :type name: str
        """
        return profiler_utils.method_timeit(method, name)

    @staticmethod
    def _filter_properties_w_options(
        calculations: dict, options: BaseInspectorOptions | None
    ) -> None:
        """
        Cycle through the calculations and turns off the ones that are disabled.

        :param calculations: Contains all the column calculations.
        :type calculations: Dict
        :param options: Contains all the options.
        :type options: BaseInspectorOptions
        """
        for prop in list(calculations):
            if options and not options.is_prop_enabled(prop):
                del calculations[prop]

    def _perform_property_calcs(
        self,
        calculations: dict,
        df_series: pd.DataFrame,
        prev_dependent_properties: dict,
        subset_properties: dict,
    ) -> None:
        """
        Cycle through the properties of the columns and calculate them.

        :param calculations: Contains all the column calculations.
        :type calculations: dict
        :param df_series: Data to be profiled
        :type df_series: pandas.Dataframe
        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        for prop in calculations:
            calculations[prop](
                self, df_series, prev_dependent_properties, subset_properties
            )

    @staticmethod
    def _merge_calculations(
        merged_profile_calcs: dict, profile1_calcs: dict, profile2_calcs: dict
    ) -> None:
        """
        Merge the calculations of two profiles to the lowest common denominator.

        :param merged_profile_calcs: default calculations of the merged profile
        :type merged_profile_calcs: dict
        :param profile1_calcs: calculations of profile1
        :type profile1_calcs: dict
        :param profile2_calcs: calculations of profile2
        :type profile2_calcs: dict
        :return: None
        """
        calcs = list(merged_profile_calcs.keys())
        for calc in calcs:
            if calc not in profile1_calcs or calc not in profile2_calcs:
                del merged_profile_calcs[calc]
                if calc in profile1_calcs or calc in profile2_calcs:
                    warnings.warn(
                        "{} is disabled because it is not enabled in "
                        "both profiles.".format(calc),
                        RuntimeWarning,
                    )

    def _add_helper(
        self, other1: BaseColumnProfilerT, other2: BaseColumnProfilerT
    ) -> None:
        """
        Merge the properties of two BaseColumnProfile objects.

        :param other1: first BaseColumn profile
        :param other2: second BaseColumn profile
        :type other1: BaseColumnProfiler
        :type other2: BaseColumnProfiler
        """
        if np.isnan(other1.col_index) and np.isnan(other2.col_index):
            pass
        elif other1.col_index == other2.col_index:
            self.col_index = other1.col_index
        else:
            raise ValueError(
                "Column indexes unmatched: {} != {}".format(
                    other1.col_index, other2.col_index
                )
            )
        if other1.name == other2.name:
            self.name = other1.name
        else:
            raise ValueError(f"Column names unmatched: {other1.name} != {other2.name}")

        self.times = profiler_utils.add_nested_dictionaries(other1.times, other2.times)

        self.sample_size = other1.sample_size + other2.sample_size

    def diff(self, other_profile: BaseColumnProfilerT, options: dict = None) -> dict:
        """
        Find the differences for columns.

        :param other_profile: profile to find the difference with
        :type other_profile: BaseColumnProfiler
        :return: the stat differences
        :rtype: dict
        """
        cls = self.__class__
        if not isinstance(other_profile, cls):
            raise TypeError(
                "Unsupported operand type(s) for diff: '{}' "
                "and '{}'".format(cls.__name__, other_profile.__class__.__name__)
            )
        return {}

    def _update_column_base_properties(self, profile: dict) -> None:
        """
        Update the base properties with the base schema.

        :param profile: profile dictionary of data type
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def __getitem__(self, item: str) -> Any:
        """
        Override for the [] operator to allow access to class properties.

        NOTE: Will be removed when switched over, only used as a method to
        integrate with current setup.
        """
        if not hasattr(self, item):
            raise ValueError(f"The property '{item} does not exist.")
        return getattr(self, item)

    @abc.abstractmethod
    def _update_helper(self, df_series_clean: pd.DataFrame, profile: dict) -> None:
        """Help update the profile."""
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, df_series: pd.DataFrame) -> BaseColumnProfiler:
        """
        Update the profile.

        :param df_series: Data to profile.
        :type df_series: Pandas Dataframe
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def profile(self) -> dict:
        """Return the profile of the column."""
        raise NotImplementedError()

    @abc.abstractmethod
    def report(self, remove_disabled_flag: bool = False) -> dict:
        """
        Return report.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        raise NotImplementedError()

    @classmethod
    def load_from_dict(
        cls: type[BaseColumnProfilerT],
        data: dict[str, Any],
        config: dict | None = None,
    ) -> BaseColumnProfilerT:
        """
        Parse attribute from json dictionary into self.

        :param data: dictionary with attributes and values.
        :type data: dict[string, Any]
        :param config: config for loading column profiler params from dictionary
        :type config: Dict | None

        :return: Profiler with attributes populated.
        :rtype: BaseColumnProfiler
        """
        if config is None:
            config = {}

        class_options = config.get(cls.__name__)
        profile: BaseColumnProfilerT = cls(data["name"], class_options)

        time_vals = data.pop("times")
        setattr(profile, "times", defaultdict(float, time_vals))

        for attr, value in data.items():
            if "__calculations" in attr:
                for metric, function in value.items():
                    if not hasattr(profile, function):
                        raise AttributeError(
                            f"Object {type(profile)} has no attribute {function}."
                        )
                    value[metric] = getattr(profile, function).__func__
            setattr(profile, attr, value)

        return profile


BaseColumnPrimitiveTypeProfilerT = TypeVar(
    "BaseColumnPrimitiveTypeProfilerT", bound="BaseColumnPrimitiveTypeProfiler"
)


class BaseColumnPrimitiveTypeProfiler(
    BaseColumnProfiler[BaseColumnPrimitiveTypeProfilerT],
    metaclass=abc.ABCMeta,
):
    """Abstract class for profiling primative data type for col of data."""

    def __init__(self, name: str | None) -> None:
        """
        Initialize base class properties for the subclass.

        :param name: Name of the data
        :type name: String
        """
        BaseColumnProfiler.__init__(self, name)
        # Number of values that match the column type. eg. how many floats match
        # in the float column
        self.match_count: int = 0
        self.sample_size: int  # inherited from BaseColumnProfiler

    def _update_column_base_properties(self, profile: dict) -> None:
        """
        Update the base properties with the base schema.

        :param profile: profile containg base properties
        :type profile: base data profile dict
        :return: None
        """
        self.match_count += int(profile.pop("match_count"))
        BaseColumnProfiler._update_column_base_properties(self, profile)

    def _add_helper(
        self,
        other1: BaseColumnPrimitiveTypeProfilerT,
        other2: BaseColumnPrimitiveTypeProfilerT,
    ) -> None:
        """
        Merge the properties of two objects inputted.

        :param other1: first profile
        :param other2: second profile
        :type other1: BaseColumnPrimitiveTypeProfiler
        :type other2: BaseColumnPrimitiveTypeProfiler
        """
        super()._add_helper(other1, other2)
        self.match_count = other1.match_count + other2.match_count
