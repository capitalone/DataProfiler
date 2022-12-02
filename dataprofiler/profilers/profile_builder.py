#!/usr/bin/env python
"""Build model for dataset by identifying col type along with its respective params."""
from __future__ import annotations, division, print_function

import copy
import logging
import pickle
import random
import re
import warnings
from collections import OrderedDict, defaultdict
from datetime import datetime
from multiprocessing.pool import Pool
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Type, Union, cast

import networkx as nx
import numpy as np
import pandas as pd

from .. import data_readers, dp_logging
from ..data_readers.data import Data
from ..labelers.base_data_labeler import BaseDataLabeler
from ..labelers.data_labelers import DataLabeler
from . import utils
from .column_profile_compilers import (
    BaseCompiler,
    ColumnDataLabelerCompiler,
    ColumnPrimitiveTypeProfileCompiler,
    ColumnStatsProfileCompiler,
    UnstructuredCompiler,
)
from .graph_profiler import GraphProfiler
from .helpers.report_helpers import _prepare_report, calculate_quantiles
from .profiler_options import (
    BaseOption,
    ProfilerOptions,
    StructuredOptions,
    UnstructuredOptions,
)

logger = dp_logging.get_child_logger(__name__)


class StructuredColProfiler(object):
    """For profiling structured data columns."""

    def __init__(
        self,
        df_series: pd.Series = None,
        sample_size: int = None,
        min_sample_size: int = 5000,
        sampling_ratio: float = 0.2,
        min_true_samples: int = 0,
        sample_ids: np.ndarray = None,
        pool: Pool = None,
        column_index: int = None,
        options: StructuredOptions = None,
    ) -> None:
        """
        Instantiate the StructuredColProfiler class for a given column.

        :param df_series: Data to be profiled
        :type df_series: pandas.core.series.Series
        :param sample_size: Number of samples to use in generating profile
        :type sample_size: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param sample_ids: Randomized list of sample indices
        :type sample_ids: list(list)
        :param pool: pool utilized for multiprocessing
        :type pool: multiprocessing.Pool
        :param column_index: index of the given column
        :type column_index: int
        :param options: Options for the structured profiler.
        :type options: StructuredOptions Object
        """
        self.name: Optional[Union[int, str]] = None
        self.options = options
        self._min_sample_size: int = min_sample_size
        self._sampling_ratio: float = sampling_ratio
        self._min_true_samples: int = min_true_samples
        self.sample_size: int = 0
        self.sample: List[str] = list()
        self.null_count: int = 0
        self.null_types: List[str] = list()
        self.null_types_index: Dict = {}
        self._min_id: Optional[int] = None
        self._max_id: Optional[int] = None
        self._index_shift: Optional[int] = None
        self._last_batch_size: Optional[int] = None
        self.profiles: Dict[str, BaseCompiler] = {}

        NO_FLAG = 0
        self._null_values: Dict[str, Union[re.RegexFlag, int]] = {
            "": NO_FLAG,
            "nan": re.IGNORECASE,
            "none": re.IGNORECASE,
            "null": re.IGNORECASE,
            "  *": NO_FLAG,
            "--*": NO_FLAG,
            "__*": NO_FLAG,
        }
        if options:
            if options.null_values is not None:
                self._null_values = options.null_values.copy()
            if column_index is not None and options.column_null_values is not None:
                self._null_values.update(
                    options.column_null_values.get(column_index, {})
                )

        if df_series is not None and len(df_series) > 0:
            if not sample_size:
                sample_size = self._get_sample_size(df_series)
            if sample_size < len(df_series):
                warnings.warn(
                    "The data will be profiled with a sample size of {}. "
                    "All statistics will be based on this subsample and "
                    "not the whole dataset.".format(sample_size)
                )

            clean_sampled_df, base_stats = self.clean_data_and_get_base_stats(
                df_series=df_series,
                sample_size=sample_size,
                null_values=self._null_values,
                min_true_samples=self._min_true_samples,
                sample_ids=sample_ids,
            )
            self.update_column_profilers(clean_sampled_df, pool)
            self._update_base_stats(base_stats)

    def update_column_profilers(
        self, clean_sampled_df: pd.Series, pool: Pool = None
    ) -> None:
        """
        Calculate type statistics and label dataset.

        :param clean_sampled_df: sampled series with none types dropped
        :type clean_sampled_df: Pandas.Series
        :param pool: pool utilized for multiprocessing
        :type pool: multiprocessing.pool
        """
        if self.name is None:
            self.name = clean_sampled_df.name
        elif self.name != clean_sampled_df.name:
            raise ValueError(
                "Column names have changed, col {} does not match prior name {}",
                clean_sampled_df.name,
                self.name,
            )

        # First run, create the compilers
        if self.profiles is None or len(self.profiles) == 0:
            self.profiles = {
                "data_type_profile": ColumnPrimitiveTypeProfileCompiler(
                    clean_sampled_df, self.options, pool
                ),
                "data_stats_profile": ColumnStatsProfileCompiler(
                    clean_sampled_df, self.options, pool
                ),
            }

            use_data_labeler = True
            if self.options and isinstance(self.options, StructuredOptions):
                use_data_labeler = self.options.data_labeler.is_enabled

            if use_data_labeler:
                self.profiles.update(
                    {
                        "data_label_profile": ColumnDataLabelerCompiler(
                            clean_sampled_df, self.options, pool
                        )
                    }
                )
        else:

            # Profile compilers being updated
            for profile in self.profiles.values():
                profile.update_profile(clean_sampled_df, pool)

    def __add__(self, other: StructuredColProfiler) -> StructuredColProfiler:
        """
        Merge two Structured profiles together overriding the `+` operator.

        :param other: structured profile being add to this one.
        :type other: StructuredColProfiler
        :return: merger of the two structured profiles
        """
        if type(other) is not type(self):
            raise TypeError(
                "`{}` and `{}` are not of the same profiler type.".format(
                    type(self).__name__, type(other).__name__
                )
            )
        elif self.name != other.name:
            raise ValueError(
                "Structured profile names are unmatched: {} != {}".format(
                    self.name, other.name
                )
            )
        elif set(self.profiles) != set(other.profiles):  # options check
            raise ValueError(
                "Structured profilers were not setup with the same"
                " options, hence they do not calculate the same "
                "profiles and cannot be added together."
            )
        merged_profile = StructuredColProfiler(
            df_series=pd.Series([]),
            min_sample_size=max(self._min_sample_size, other._min_sample_size),
            sampling_ratio=max(self._sampling_ratio, other._sampling_ratio),
            min_true_samples=max(self._min_true_samples, other._min_true_samples),
            options=self.options,
        )

        merged_profile.name = self.name
        merged_profile._update_base_stats(
            {
                "sample": self.sample,
                "sample_size": self.sample_size,
                "null_count": self.null_count,
                "null_types": copy.deepcopy(self.null_types_index),
                "min_id": self._min_id,
                "max_id": self._max_id,
            }
        )
        merged_profile._update_base_stats(
            {
                "sample": other.sample,
                "sample_size": other.sample_size,
                "null_count": other.null_count,
                "null_types": copy.deepcopy(other.null_types_index),
                "min_id": other._min_id,
                "max_id": other._max_id,
            }
        )
        samples = list(dict.fromkeys(self.sample + other.sample))
        merged_profile.sample = random.sample(samples, min(len(samples), 5))
        for profile_name in self.profiles:
            merged_profile.profiles[profile_name] = (
                self.profiles[profile_name] + other.profiles[profile_name]
            )
        return merged_profile

    def diff(self, other_profile: StructuredColProfiler, options: Dict = None) -> Dict:
        """
        Find the difference between 2 StructuredCols and return the report.

        :param other_profile: Structured col finding the difference with this
            one.
        :type other_profile: StructuredColProfiler
        :param options: options to change results of the difference
        :type options: dict
        :return: difference of the structured column
        :rtype: dict
        """
        unordered_profile: Dict = dict()
        for key in self.profiles:
            if key in other_profile.profiles:
                comp_diff = self.profiles[key].diff(
                    other_profile.profiles[key], options=options
                )
                unordered_profile = utils.recursive_dict_update(
                    unordered_profile, comp_diff
                )

        name = self.name
        if isinstance(self.name, np.integer):
            name = int(name)  # type: ignore

        unordered_profile.update(
            {
                "column_name": name,
            }
        )

        unordered_profile["statistics"].update(
            {
                "sample_size": utils.find_diff_of_numbers(
                    self.sample_size, other_profile.sample_size
                ),
                "null_count": utils.find_diff_of_numbers(
                    self.null_count, other_profile.null_count
                ),
                "null_types": utils.find_diff_of_lists_and_sets(
                    self.null_types, other_profile.null_types
                ),
                "null_types_index": utils.find_diff_of_dicts_with_diff_keys(
                    self.null_types_index, other_profile.null_types_index
                ),
            }
        )

        if unordered_profile.get("data_type", None) is not None:
            unordered_profile["statistics"].update(
                {
                    "data_type_representation": unordered_profile[
                        "data_type_representation"
                    ]
                }
            )

        dict_order = [
            "column_name",
            "data_type",
            "data_label",
            "categorical",
            "order",
            "statistics",
        ]
        profile = OrderedDict()
        if (
            "data_label_profile" not in self.profiles
            or "data_label_profile" not in other_profile.profiles
        ):
            dict_order.remove("data_label")
        for key in dict_order:
            try:
                profile[key] = unordered_profile[key]
            except KeyError:
                profile[key] = None

        return profile

    def report(self, remove_disabled_flag: bool = False) -> OrderedDict:
        """Return profile."""
        unordered_profile: Dict = dict()
        for profile in self.profiles.values():
            unordered_profile = utils.recursive_dict_update(
                unordered_profile, profile.report(remove_disabled_flag)
            )

        name = self.name
        if isinstance(self.name, np.integer):
            name = int(name)  # type: ignore

        unordered_profile.update(
            {
                "column_name": name,
                "samples": self.sample,
            }
        )

        unordered_profile["statistics"].update(
            {
                "sample_size": self.sample_size,
                "null_count": self.null_count,
                "null_types": self.null_types,
                "null_types_index": self.null_types_index,
            }
        )

        if unordered_profile.get("data_type", None) is not None:
            unordered_profile["statistics"].update(
                {
                    "data_type_representation": unordered_profile[
                        "data_type_representation"
                    ]
                }
            )

        dict_order = [
            "column_name",
            "data_type",
            "data_label",
            "categorical",
            "order",
            "samples",
            "statistics",
        ]
        profile: Dict = OrderedDict()  # type: ignore[no-redef]
        if "data_label_profile" not in self.profiles:
            dict_order.remove("data_label")
        for key in dict_order:
            try:
                profile[key] = unordered_profile[key]
            except KeyError:
                profile[key] = None

        return profile

    @property
    def profile(self) -> Dict:
        """Return a report."""
        return self.report(remove_disabled_flag=False)

    def _update_base_stats(self, base_stats: Dict) -> None:
        self.sample_size += base_stats["sample_size"]
        self._last_batch_size = base_stats["sample_size"]
        self.sample = base_stats["sample"]
        self.null_count += base_stats["null_count"]
        self.null_types = utils._combine_unique_sets(
            self.null_types, list(base_stats["null_types"].keys())
        )

        base_min: int = base_stats["min_id"]
        base_max: int = base_stats["max_id"]
        base_nti = base_stats["null_types"]

        # Check if indices overlap, if they do, adjust attributes accordingly
        if utils.overlap(self._min_id, self._max_id, base_min, base_max):
            warnings.warn(
                f"Overlapping indices detected. To resolve, indices "
                f"where null data present will be shifted forward "
                f"when stored in profile: {self.name}"
            )

            # Shift indices (min, max, and all indices in null types index
            self._index_shift = cast(int, self._max_id) + 1
            base_min = base_min + self._index_shift
            base_max = base_max + self._index_shift

            base_nti = {
                k: {x + self._index_shift for x in v}
                for k, v in base_stats["null_types"].items()
            }

        # Store/compare min/max id with current
        if self._min_id is None:
            self._min_id = base_min
        elif base_min is not None:
            self._min_id = min(self._min_id, base_min)
        if self._max_id is None:
            self._max_id = base_max
        elif base_max is not None:
            self._max_id = max(self._max_id, base_max)

        # Update null row indices
        for null_type, null_rows in base_nti.items():
            if type(null_rows) is list:
                null_rows.sort()
            self.null_types_index.setdefault(null_type, set()).update(null_rows)

    def update_profile(
        self,
        df_series: pd.Series,
        sample_size: int = None,
        min_true_samples: int = None,
        sample_ids: np.ndarray = None,
        pool: Pool = None,
    ) -> None:
        """
        Update the column profiler.

        :param df_series: Data to be profiled
        :type df_series: pandas.core.series.Series
        :param sample_size: Number of samples to use in generating profile
        :type sample_size: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param sample_ids: Randomized list of sample indices
        :type sample_ids: list(list)
        :param pool: pool utilized for multiprocessing
        :type pool: multiprocessing.Pool
        """
        if not sample_size:
            sample_size = len(df_series)
        if not sample_size:
            sample_size = self._get_sample_size(df_series)
        if not min_true_samples:
            min_true_samples = self._min_true_samples

        clean_sampled_df, base_stats = self.clean_data_and_get_base_stats(
            df_series=df_series,
            sample_size=sample_size,
            null_values=self._null_values,
            min_true_samples=min_true_samples,
            sample_ids=sample_ids,
        )

        self._update_base_stats(base_stats)
        self.update_column_profilers(clean_sampled_df, pool)

    def _get_sample_size(self, df_series: pd.Series) -> int:
        """
        Determine the minimum sampling size for detecting column type.

        :param df_series: a column of data
        :type df_series: pandas.core.series.Series
        :return: integer sampling size
        :rtype: int
        """
        len_df = len(df_series)
        if len_df <= self._min_sample_size:
            return int(len_df)
        return max(int(self._sampling_ratio * len_df), self._min_sample_size)

    # TODO: flag column name with null values and potentially return row
    #  index number in the error as well
    @staticmethod
    def clean_data_and_get_base_stats(
        df_series: pd.Series,
        sample_size: int,
        null_values: Dict[str, Union[re.RegexFlag, int]] = None,
        min_true_samples: int = None,
        sample_ids: np.ndarray = None,
    ) -> Tuple[pd.Series, Dict]:
        """
        Identify null characters and return them in a dictionary.

        Remove any nulls in column.

        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :param sample_size: Number of samples to use in generating the profile
        :type sample_size: int
        :param null_values: Dictionary mapping null values to regex flag where
            the key represents the null value to remove from the data and the
            flag represents the regex flag to apply
        :type null_values: Dict[str, Union[re.RegexFlag, int]]
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param sample_ids: Randomized list of sample indices
        :type sample_ids: list(list)
        :return: updated column with null removed and dictionary of null
            parameters
        :rtype: pd.Series, dict
        """
        if min_true_samples is None:
            min_true_samples = 0

        if null_values is None:
            null_values = dict()

        len_df = len(df_series)
        if not len_df:
            return (
                df_series,
                {
                    "sample_size": 0,
                    "null_count": 0,
                    "null_types": dict(),
                    "sample": [],
                    "min_id": None,
                    "max_id": None,
                },
            )

        # Pandas reads empty values in the csv files as nan
        df_series = df_series.apply(str)

        # Record min and max index values if index is int
        is_index_all_ints = True
        try:
            min_id = min(df_series.index)
            max_id = max(df_series.index)
            if not (isinstance(min_id, int) and isinstance(max_id, int)):
                is_index_all_ints = False
        except TypeError:
            is_index_all_ints = False

        if not is_index_all_ints:
            min_id = max_id = None
            warnings.warn(
                "Unable to detect minimum and maximum index values "
                "for overlap detection. Updating/merging profiles "
                "may result in inaccurate null row index reporting "
                "due to unhandled overlapping indices."
            )

        # Select generator depending if sample_ids availability
        if sample_ids is None:
            sample_ind_generator = utils.shuffle_in_chunks(
                len_df, chunk_size=sample_size
            )
        else:
            sample_ind_generator = utils.partition(
                sample_ids[0], chunk_size=sample_size
            )

        na_columns: Dict = dict()
        true_sample_list = set()
        total_sample_size = 0
        query = "|".join(null_values.keys())
        regex = f"^(?:{(query)})$"
        for chunked_sample_ids in sample_ind_generator:
            total_sample_size += len(chunked_sample_ids)

            # Find subset of series based on randomly selected ids
            df_subset = df_series.iloc[chunked_sample_ids]

            # Query should search entire cell for all elements at once
            matches = df_subset.str.match(regex, flags=re.IGNORECASE)

            # Split series into None samples and true samples
            true_sample_list.update(df_subset[~matches].index)

            # Iterate over all the Nones
            for index, cell in df_subset[matches].items():
                na_columns.setdefault(cell, list()).append(index)

            # Ensure minimum number of true samples met
            # and if total_sample_size >= sample size, exit
            if (
                len(true_sample_list) >= min_true_samples
                and total_sample_size >= sample_size
            ):
                break

        # close the generator in case it is not exhausted.
        if sample_ids is None:
            sample_ind_generator.close()

        # If min_true_samples exists, sort
        if min_true_samples > 0 or sample_ids is None:
            true_sample_list = sorted(true_sample_list)  # type: ignore[assignment]

        # Split out true values for later utilization
        df_series = df_series.loc[true_sample_list]
        total_na = total_sample_size - len(true_sample_list)

        base_stats = {
            "sample_size": total_sample_size,
            "null_count": total_na,
            "null_types": na_columns,
            "sample": random.sample(list(df_series.values), min(len(df_series), 5)),
            "min_id": min_id,
            "max_id": max_id,
        }

        return df_series, base_stats


class BaseProfiler(object):
    """Abstract class for profiling data."""

    _default_labeler_type: Optional[str] = None
    _option_class: Optional[Type[BaseOption]] = None
    _allowed_external_data_types: Union[Tuple, type] = ()

    def __init__(
        self,
        data: Optional[Data],
        samples_per_update: int = None,
        min_true_samples: int = 0,
        options: BaseOption = None,
    ) -> None:
        """
        Instantiate the BaseProfiler class.

        :param data: Data to be profiled
        :type data: Data class object
        :param samples_per_update: Number of samples to use in generating
            profile
        :type samples_per_update: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param options: Options for the profiler.
        :type options: ProfilerOptions Object
        :return: Profiler
        """
        if min_true_samples is not None and not isinstance(min_true_samples, int):
            raise ValueError("`min_true_samples` must be an integer or `None`.")

        if self._default_labeler_type is None:
            raise ValueError(
                "`_default_labeler_type` must be set when " "overriding `BaseProfiler`."
            )
        elif self._option_class is None:
            raise ValueError(
                "`_option_class` must be set when overriding " "`BaseProfiler`."
            )
        elif self._allowed_external_data_types == ():
            raise ValueError(
                "`_allowed_external_data_types` must be set when "
                "overriding `BaseProfiler`."
            )

        cast(BaseOption, options).validate()

        self._profile: BaseCompiler = None  # type: ignore
        self.options: Any = options
        self.encoding: Optional[str] = None
        self.file_type: Optional[str] = None
        self._samples_per_update: Optional[int] = samples_per_update
        self._min_true_samples: int = min_true_samples
        self.total_samples: int = 0
        self.times: Dict[str, float] = defaultdict(float)

        # TODO: allow set via options
        self._sampling_ratio: float = 0.2
        self._min_sample_size: int = 5000

        # assign data labeler
        data_labeler_options = self.options.data_labeler
        if (
            data_labeler_options.is_enabled
            and data_labeler_options.data_labeler_object is None
        ):

            try:

                data_labeler = DataLabeler(
                    labeler_type=self._default_labeler_type,
                    dirpath=data_labeler_options.data_labeler_dirpath,
                    load_options=None,
                )
                self.options.set({"data_labeler.data_labeler_object": data_labeler})

            except Exception as e:
                utils.warn_on_profile("data_labeler", e)
                self.options.set({"data_labeler.is_enabled": False})

    def _add_error_checks(self, other: BaseProfiler) -> None:
        """
        Run checks to ensure two profiles can be combined.

        Checks are specific to Profiler type.
        """
        raise NotImplementedError()

    def __add__(self, other: BaseProfiler) -> BaseProfiler:
        """
        Merge two profiles together overriding the `+` operator.

        :param other: profile being added to this one.
        :type other: BaseProfiler
        :return: merger of the two profiles
        :rtype: BaseProfiler
        """
        if type(other) is not type(self):
            raise TypeError(
                "`{}` and `{}` are not of the same profiler type.".format(
                    type(self).__name__, type(other).__name__
                )
            )

        # error checks specific to its profiler
        self._add_error_checks(other)

        merged_profile = self.__class__(
            data=None,
            samples_per_update=self._samples_per_update,
            min_true_samples=self._min_true_samples,
            options=self.options,
        )
        merged_profile.encoding = self.encoding
        if self.encoding != other.encoding:
            merged_profile.encoding = "multiple files"

        merged_profile.file_type = self.file_type
        if self.file_type != other.file_type:
            merged_profile.file_type = "multiple files"

        merged_profile.total_samples = self.total_samples + other.total_samples

        merged_profile.times = utils.add_nested_dictionaries(self.times, other.times)

        return merged_profile

    def diff(self, other_profile: BaseProfiler, options: Dict = None) -> Dict:
        """
        Find the difference of two profiles.

        :param other_profile: profile being added to this one.
        :type other_profile: BaseProfiler
        :return: diff of the two profiles
        :rtype: dict
        """
        if type(other_profile) is not type(self):
            raise TypeError(
                "`{}` and `{}` are not of the same profiler type.".format(
                    type(self).__name__, type(other_profile).__name__
                )
            )

        diff_profile = OrderedDict(
            [
                (
                    "global_stats",
                    {
                        "file_type": utils.find_diff_of_strings_and_bools(
                            self.file_type, other_profile.file_type
                        ),
                        "encoding": utils.find_diff_of_strings_and_bools(
                            self.encoding, other_profile.encoding
                        ),
                    },
                )
            ]
        )

        return diff_profile

    def _get_sample_size(self, data: Union[pd.Series, pd.DataFrame, List]) -> int:
        """
        Determine the minimum sampling size for profiling the dataset.

        :param data: a dataset
        :type data: Union[pd.Series, pd.DataFrame, list]
        :return: integer sampling size
        :rtype: int
        """
        if self._samples_per_update:
            return self._samples_per_update

        len_data = len(data)
        if len_data <= self._min_sample_size:
            return int(len_data)
        return max(int(self._sampling_ratio * len_data), self._min_sample_size)

    @property
    def profile(self) -> BaseCompiler:
        """
        Return the stored profiles for the given profiler.

        :return: BaseCompiler
        """
        return self._profile

    def report(self, report_options: Dict = None) -> Dict:
        """
        Return profile report based on all profiled data fed into the profiler.

        User can specify the output_formats: (pretty, compact, serializable, flat).
            Pretty: floats are rounded to four decimal places, and lists are
                shortened.
            Compact: Similar to pretty, but removes detailed statistics such as
                runtimes, label probabilities, index locations of null types,
                etc.
            Serializable: Output is json serializable and not prettified
            Flat: Nested output is returned as a flattened dictionary

        :var report_options: optional format changes to the report
            `dict(output_format=<FORMAT>)`
        :type report_options: dict
        :return: dictionary report
        :rtype: dict
        """
        raise NotImplementedError()

    def _update_profile_from_chunk(
        self,
        data: Union[pd.Series, pd.DataFrame, List],
        sample_size: int,
        min_true_samples: int = None,
    ) -> None:
        """
        Iterate over the dataset and identify its parameters via profiles.

        :param data: dataset to be profiled
        :type data: Union[pd.Series, pd.DataFrame, list]
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :return: None
        """
        raise NotImplementedError()

    def update_profile(
        self,
        data: Union[data_readers.base_data.BaseData, pd.DataFrame, pd.Series],
        sample_size: int = None,
        min_true_samples: int = None,
    ) -> None:
        """
        Update the profile for data provided.

        User can specify the sample size to profile the data with.
        Additionally, the user can specify the
        minimum number of non-null samples to profile.

        :param data: data to be profiled
        :type data: Union[data_readers.base_data.BaseData, pandas.DataFrame,
            pandas.Series]
        :param sample_size: number of samples to profile from the data
        :type sample_size: int
        :param min_true_samples: minimum number of non-null samples to profile
        :type min_true_samples: int
        :return: None
        """
        encoding = None
        file_type = None

        if min_true_samples is not None and not isinstance(min_true_samples, int):
            raise ValueError("`min_true_samples` must be an integer or `None`.")

        if isinstance(data, data_readers.base_data.BaseData):
            encoding = data.file_encoding
            file_type = data.data_type
            data = data.data
        elif isinstance(data, self._allowed_external_data_types):
            file_type = str(data.__class__)
        else:
            raise TypeError(
                f"Data must either be imported using the data_readers or using "
                f"one of the following: {self._allowed_external_data_types}"
            )

        if not len(data):
            warnings.warn(
                "The passed dataset was empty, hence no data was " "profiled."
            )
            return

        # set sampling properties
        if not min_true_samples:
            min_true_samples = self._min_true_samples
        if not sample_size:
            sample_size = self._get_sample_size(data)

        self._update_profile_from_chunk(data, sample_size, min_true_samples)

        # set file properties since data will be processed
        if encoding is not None:
            self.encoding = encoding
        if file_type is not None:
            self.file_type = file_type

    def _remove_data_labelers(
        self, replacement_type: BaseDataLabeler = BaseDataLabeler()
    ) -> Optional[BaseDataLabeler]:
        """
        Help remove all data labelers before saving to disk.

        :return: data_labeler used for unstructured labelling
        :rtype: BaseDataLabeler
        """
        data_labeler = None
        data_labeler_options = None

        # determine if the data labeler is enabled
        use_data_labeler = True
        if self.options and isinstance(
            self.options, (StructuredOptions, UnstructuredOptions)
        ):
            data_labeler_options = self.options.data_labeler
            use_data_labeler = data_labeler_options.is_enabled

        # remove the data labeler from options
        if (
            use_data_labeler
            and data_labeler_options is not None
            and data_labeler_options.data_labeler_object is not None
        ):
            data_labeler = data_labeler_options.data_labeler_object
            data_labeler_options.data_labeler_object = replacement_type

        # get all profiles, unstructured is a single profile and hence needs to
        # be in a list, whereas structured is already a list
        profilers = [self._profile]
        if isinstance(self, StructuredProfiler):
            profilers = self._profile

        # Remove data labelers for all columns
        for profiler in profilers:

            # profiles stored differently in Struct/Unstruct, this unifies
            # labeler extraction
            # unstructured: _profile is a compiler
            # structured: StructuredColProfiler.profiles['data_label_profile']
            if isinstance(self, StructuredProfiler):
                profiler = profiler.profiles.get("data_label_profile", None)

            if profiler and use_data_labeler and data_labeler is None:
                data_labeler = profiler._profiles["data_labeler"].data_labeler

            if profiler and "data_labeler" in profiler._profiles:
                profiler._profiles["data_labeler"].data_labeler = replacement_type

        return data_labeler

    def _restore_data_labelers(self, data_labeler: BaseDataLabeler = None) -> None:
        """
        Help restore all data labelers after saving to or loading from disk.

        :param data_labeler: unstructured data_labeler
        :type data_labeler: BaseDataLabeler
        """
        # Restore data labeler for options
        use_data_labeler = True
        data_labeler_dirpath = None
        if self.options and isinstance(
            self.options, (StructuredOptions, UnstructuredOptions)
        ):
            data_labeler_options = self.options.data_labeler
            use_data_labeler = data_labeler_options.is_enabled
            data_labeler_dirpath = data_labeler_options.data_labeler_dirpath

        if use_data_labeler:
            try:
                if data_labeler is None:
                    data_labeler = DataLabeler(  # type: ignore[assignment]
                        labeler_type=cast(str, self._default_labeler_type),
                        dirpath=data_labeler_dirpath,
                        load_options=None,
                    )
                self.options.set({"data_labeler.data_labeler_object": data_labeler})

            except Exception as e:
                utils.warn_on_profile("data_labeler", e)
                self.options.set({"data_labeler.is_enabled": False})
                self.options.set({"data_labeler.data_labeler_object": data_labeler})

        # get all profiles, unstructured is a single profile and hence needs to
        # be in a list, whereas structured is already a list
        profilers = [self._profile]
        if isinstance(self, StructuredProfiler):
            profilers = self._profile

        # Restore data labelers for all columns
        for profiler in profilers:

            if use_data_labeler:

                # profiles stored differently in Struct/Unstruct, this unifies
                # label replacement
                # unstructured: _profile is a compiler
                # structured: StructuredColProfiler.profiles['data_label_profile']
                if isinstance(self, StructuredProfiler):
                    profiler = profiler.profiles["data_label_profile"]

                data_labeler_profile = profiler._profiles["data_labeler"]
                data_labeler_profile.data_labeler = data_labeler

    def _save_helper(self, filepath: Optional[str], data_dict: Dict) -> None:
        """
        Save profiler to disk.

        :param filepath: Path of file to save to
        :type filepath: String
        :param data_dict: profile data to be saved
        :type data_dict: dict
        :return: None
        """
        # Set Default filepath
        if filepath is None:
            filepath = "profile-{}.pkl".format(
                datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
            )

        # Remove data labelers as they can't be pickled
        data_labelers = self._remove_data_labelers()

        # add profiler class to data_dict
        data_dict["profiler_class"] = self.__class__.__name__

        # Pickle and save profile to disk
        with open(filepath, "wb") as outfile:
            pickle.dump(data_dict, outfile)

        # Restore all data labelers
        self._restore_data_labelers(data_labelers)

    def save(self, filepath: str = None) -> None:
        """
        Save profiler to disk.

        :param filepath: Path of file to save to
        :type filepath: String
        :return: None
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, filepath: str) -> BaseProfiler:
        """
        Load profiler from disk.

        :param filepath: Path of file to load from
        :type filepath: String
        :return: Profiler being loaded, StructuredProfiler or
            UnstructuredProfiler
        :rtype: BaseProfiler
        """
        # Load profile from disk
        with open(filepath, "rb") as infile:
            data: Dict = pickle.load(infile)

        # remove profiler class if it exists
        profiler_class: Optional[str] = data.pop("profiler_class", None)

        # if the user didn't load from the a given profiler class, we need
        # to determine which profiler is being loaded.
        profiler_cls: Any = cls
        if cls is BaseProfiler:
            if profiler_class == "StructuredProfiler":
                profiler_cls = StructuredProfiler
            elif profiler_class == "UnstructuredProfiler":
                profiler_cls = UnstructuredProfiler
            elif profiler_class is None:  # deprecated case
                profiler_cls = StructuredProfiler
                if "_empty_line_count" in data:
                    profiler_cls = UnstructuredProfiler
            else:
                raise ValueError(
                    f"Invalid profiler class {profiler_class} " f"failed to load."
                )

        profile_options = profiler_cls._option_class()
        profile_options.data_labeler.is_enabled = False
        profiler: BaseProfiler = profiler_cls(None, options=profile_options)

        for key in data:
            setattr(profiler, key, data[key])

        # Restore all data labelers
        profiler._restore_data_labelers()
        return profiler


class UnstructuredProfiler(BaseProfiler):
    """For profiling unstructured data."""

    _default_labeler_type = "unstructured"
    _option_class = UnstructuredOptions
    _allowed_external_data_types = (str, list, pd.Series, pd.DataFrame)

    def __init__(
        self,
        data: Data,
        samples_per_update: int = None,
        min_true_samples: int = 0,
        options: BaseOption = None,
    ) -> None:
        """
        Instantiate the UnstructuredProfiler class.

        :param data: Data to be profiled
        :type data: Data class object
        :param samples_per_update: Number of samples to use in generating
            profile
        :type samples_per_update: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param options: Options for the profiler.
        :type options: ProfilerOptions Object
        :return: UnstructuredProfiler
        """
        if not options:
            options = UnstructuredOptions()
        elif isinstance(options, ProfilerOptions):
            options = options.unstructured_options
        elif not isinstance(options, UnstructuredOptions):
            raise ValueError(
                "The profile options must be passed as a " "ProfileOptions object."
            )

        super().__init__(data, samples_per_update, min_true_samples, options)

        # Unstructured specific properties
        self._empty_line_count: int = 0
        self.memory_size: float = 0
        self.sample: List[str] = []

        if data is not None:
            self.update_profile(data)

    def _add_error_checks(  # type: ignore[override]
        self, other: UnstructuredProfiler
    ) -> None:
        """
        Run checks to ensure two profiles can be combined.

        Checks are specific to UnstructuredProfiler.
        """
        pass

    def __add__(  # type: ignore[override]
        self, other: UnstructuredProfiler
    ) -> UnstructuredProfiler:
        """
        Merge two Unstructured profiles together overriding the `+` operator.

        :param other: unstructured profile being added to this one.
        :type other: UnstructuredProfiler
        :return: merger of the two profiles
        :rtype: UnstructuredProfiler
        """
        merged_profile = cast(UnstructuredProfiler, super().__add__(other))

        # unstruct specific property merging
        merged_profile._empty_line_count = (
            self._empty_line_count + other._empty_line_count
        )
        merged_profile.memory_size = self.memory_size + other.memory_size
        samples = list(dict.fromkeys(self.sample + other.sample))
        merged_profile.sample = random.sample(list(samples), min(len(samples), 5))

        # merge profiles
        merged_profile._profile = self._profile + other._profile

        return merged_profile

    def diff(  # type: ignore[override]
        self, other_profile: UnstructuredProfiler, options: Optional[Dict] = None
    ) -> Dict:
        """
        Find difference between 2 unstuctured profiles and return the report.

        :param other_profile: profile finding the difference with this one.
        :type other_profile: UnstructuredProfiler
        :param options: options to impact the results of the diff
        :type options: dict
        :return: difference of the profiles
        :rtype: dict
        """
        if options is None:
            options = {}

        report = super().diff(other_profile, options)

        report["global_stats"].update(
            {
                "samples_used": utils.find_diff_of_numbers(
                    self.total_samples, other_profile.total_samples
                ),
                "empty_line_count": utils.find_diff_of_numbers(
                    self._empty_line_count, other_profile._empty_line_count
                ),
                "memory_size": utils.find_diff_of_numbers(
                    self.memory_size, other_profile.memory_size
                ),
            }
        )

        report["data_stats"] = self._profile.diff(
            other_profile._profile, options=options
        )
        return _prepare_report(
            report,
            output_format=options.get("output_format", None),
            omit_keys=options.get("omit_keys", None),
        )

    def _update_base_stats(self, base_stats: Dict) -> None:
        """
        Update samples and line count of the class for the given dataset batch.

        :param base_stats: dictionary of basic sampling / data stats
        :type base_stats: dict
        :return: None
        """
        self.total_samples += base_stats["sample_size"]
        self.sample = base_stats["sample"]
        self._empty_line_count += base_stats["empty_line_count"]
        self.memory_size += base_stats["memory_size"]

    def report(self, report_options: Dict = None) -> Dict:
        """
        Return unstructured report based on all profiled data fed into profiler.

        User can specify the output_formats: (pretty, compact, serializable, flat).
            Pretty: floats are rounded to four decimal places, and lists are
                shortened.
            Compact: Similar to pretty, but removes detailed statistics such as
                runtimes, label probabilities, index locations of null types,
                etc.
            Serializable: Output is json serializable and not prettified
            Flat: Nested output is returned as a flattened dictionary

        :var report_options: optional format changes to the report
            `dict(output_format=<FORMAT>)`
        :type report_options: dict
        :return: dictionary report
        :rtype: dict
        """
        if not report_options:
            report_options = {
                "output_format": None,
                "omit_keys": None,
                "remove_disabled_flag": False,
            }

        output_format = report_options.get("output_format", None)
        omit_keys = report_options.get("omit_keys", None)
        remove_disabled_flag = report_options.get("remove_disabled_flag", False)

        report = OrderedDict(
            [
                (
                    "global_stats",
                    {
                        "samples_used": self.total_samples,
                        "empty_line_count": self._empty_line_count,
                        "file_type": self.file_type,
                        "encoding": self.encoding,
                        "memory_size": self.memory_size,
                        "times": self.times,
                    },
                ),
                ("data_stats", OrderedDict()),
            ]
        )
        report["data_stats"] = self._profile.report(remove_disabled_flag)
        return _prepare_report(report, output_format, omit_keys)

    @utils.method_timeit(name="clean_and_base_stats")
    def _clean_data_and_get_base_stats(
        self, data: pd.Series, sample_size: int, min_true_samples: int = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Identify empty rows and return clean version of text data without empty rows.

        :param data: a series of text data
        :type data: pandas.core.series.Series
        :param sample_size: Number of samples to use in generating the profile
        :type sample_size: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :return: updated column with null removed and dictionary of null
            parameters
        :rtype: pd.Series, dict
        """
        if min_true_samples is None:
            min_true_samples = 0

        len_data = len(data)
        if not len_data:
            return (
                data,
                {
                    "sample_size": 0,
                    "empty_line_count": dict(),
                    "sample": [],
                    "memory_size": 0,
                },
            )

        # ensure all data are of type str
        data = data.apply(str)

        # get memory size
        base_stats: Dict = {"memory_size": utils.get_memory_size(data, unit="M")}

        # Setup sample generator
        sample_ind_generator = utils.shuffle_in_chunks(len_data, chunk_size=sample_size)

        true_sample_list = set()
        total_sample_size = 0

        regex = r"^\s*$"
        for chunked_sample_ids in sample_ind_generator:
            total_sample_size += len(chunked_sample_ids)

            # Find subset of series based on randomly selected ids
            data_subset = data.iloc[chunked_sample_ids]

            # Query should search entire cell for all elements at once
            matches = data_subset.str.match(regex, flags=re.IGNORECASE)

            # Split series into None samples and true samples
            true_sample_list.update(data_subset[~matches].index)

            # Ensure minimum number of true samples met
            # and if total_sample_size >= sample size, exit
            if (
                len(true_sample_list) >= min_true_samples
                and total_sample_size >= sample_size
            ):
                break

        # close the generator in case it is not exhausted.
        sample_ind_generator.close()

        true_sample_list = sorted(true_sample_list)  # type: ignore[assignment]

        # Split out true values for later utilization
        data = data.loc[true_sample_list]
        total_empty = total_sample_size - len(true_sample_list)

        base_stats.update(
            {
                "sample_size": total_sample_size,
                "empty_line_count": total_empty,
                "sample": random.sample(list(data.values), min(len(data), 5)),
            }
        )

        return data, base_stats

    def _update_profile_from_chunk(
        self,
        data: Union[pd.Series, pd.DataFrame, List],
        sample_size: int,
        min_true_samples: int = None,
    ) -> None:
        """
        Iterate over the dataset and identify its parameters via profiles.

        :param data: a text dataset
        :type data: Union[pd.Series, pd.DataFrame, list]
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :return: None
        """
        if isinstance(data, pd.DataFrame):
            if len(data.columns) > 1:
                raise ValueError(
                    "The unstructured cannot handle a dataset "
                    "with more than 1 column. Please make sure "
                    "the data format of the dataset is "
                    "appropriate."
                )
            data = data[data.columns[0]]
        elif isinstance(data, (str, list)):
            # we know that if it comes in as a list, it is a 1-d list based
            # bc of our data readers
            # for strings, we just need to put it inside a series for compute.
            data = pd.Series(data)

        # Format the data
        notification_str = "Finding the empty lines in the data..."
        logger.info(notification_str)
        data, base_stats = self._clean_data_and_get_base_stats(
            data, sample_size, min_true_samples
        )
        self._update_base_stats(base_stats)

        if sample_size < len(data):
            warnings.warn(
                "The data will be profiled with a sample size of {}. "
                "All statistics will be based on this subsample and "
                "not the whole dataset.".format(sample_size)
            )

        # process the text data
        notification_str = "Calculating the statistics... "
        logger.info(notification_str)
        pool = None
        if self._profile is None:
            self._profile = UnstructuredCompiler(data, options=self.options, pool=pool)
        else:
            self._profile.update_profile(data, pool=pool)

    def save(self, filepath: str = None) -> None:
        """
        Save profiler to disk.

        :param filepath: Path of file to save to
        :type filepath: String
        :return: None
        """
        # Create dictionary for all metadata, options, and profile
        data_dict = {
            "total_samples": self.total_samples,
            "sample": self.sample,
            "encoding": self.encoding,
            "file_type": self.file_type,
            "_samples_per_update": self._samples_per_update,
            "_min_true_samples": self._min_true_samples,
            "_empty_line_count": self._empty_line_count,
            "memory_size": self.memory_size,
            "options": self.options,
            "_profile": self.profile,
            "times": self.times,
        }
        self._save_helper(filepath, data_dict)


class StructuredProfiler(BaseProfiler):
    """For profiling structured data."""

    _default_labeler_type = "structured"
    _option_class = StructuredOptions
    _allowed_external_data_types = (list, pd.Series, pd.DataFrame)

    def __init__(
        self,
        data: Data,
        samples_per_update: int = None,
        min_true_samples: int = 0,
        options: BaseOption = None,
    ) -> None:
        """
        Instantiate the StructuredProfiler class.

        :param data: Data to be profiled
        :type data: Data class object
        :param samples_per_update: Number of samples to use in generating
            profile
        :type samples_per_update: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param options: Options for the profiler.
        :type options: ProfilerOptions Object
        :return: StructuredProfiler
        """
        if not options:
            options = StructuredOptions()
        elif isinstance(options, ProfilerOptions):
            options = options.structured_options
        elif not isinstance(options, StructuredOptions):
            raise ValueError(
                "The profile options must be passed as a " "ProfileOptions object."
            )

        if isinstance(data, data_readers.text_data.TextData):
            raise TypeError("Cannot provide TextData object to " "StructuredProfiler")

        super().__init__(data, samples_per_update, min_true_samples, options)

        # Structured specific properties
        self.row_has_null_count = 0
        self.row_is_null_count = 0
        self.hashed_row_dict: Dict = dict()
        self._profile: List[BaseCompiler] = []  # type: ignore[assignment]
        self._col_name_to_idx: Dict[Union[str, int], List[int]] = defaultdict(list)
        self.correlation_matrix: np.ndarray = None  # type: ignore[assignment]
        self.chi2_matrix: np.ndarray = None  # type: ignore[assignment]

        # capitalone/synthetic-data specific metrics
        self._null_replication_metrics: Dict = None  # type: ignore[assignment]

        if data is not None:
            self.update_profile(data)

    def _add_error_checks(  # type: ignore[override]
        self, other: StructuredProfiler
    ) -> None:
        """
        Run checks to ensure two profiles can be combined.

        Said checks are specific to StructuredProfiler.
        """
        # Pass with strict = True to enforce both needing to be non-empty
        self_to_other_idx = self._get_and_validate_schema_mapping(
            self._col_name_to_idx, other._col_name_to_idx, True
        )
        if not all(
            [
                isinstance(
                    other._profile[self_to_other_idx[idx]], type(self._profile[idx])
                )
                for idx in range(len(self._profile))
            ]
        ):  # options check
            raise ValueError(
                "The two profilers were not setup with the same "
                "options, hence they do not calculate the same "
                "profiles and cannot be added together."
            )

    def __add__(  # type: ignore[override]
        self, other: StructuredProfiler
    ) -> StructuredProfiler:
        """
        Merge two Structured profiles together overriding the `+` operator.

        :param other: profile being added to this one.
        :type other: StructuredProfiler
        :return: merger of the two profiles
        :rtype: StructuredProfiler
        """
        merged_profile = cast(StructuredProfiler, super().__add__(other))

        # struct specific property merging
        merged_profile.row_has_null_count = (
            self.row_has_null_count + other.row_has_null_count
        )
        merged_profile.row_is_null_count = (
            self.row_is_null_count + other.row_is_null_count
        )
        merged_profile.hashed_row_dict.update(self.hashed_row_dict)
        merged_profile.hashed_row_dict.update(other.hashed_row_dict)

        self_to_other_idx = self._get_and_validate_schema_mapping(
            self._col_name_to_idx, other._col_name_to_idx
        )

        # merge profiles
        for idx in range(len(self._profile)):
            other_idx = self_to_other_idx[idx]
            merged_profile._profile.append(
                self._profile[idx] + other._profile[other_idx]
            )

        # schemas are asserted to be identical
        merged_profile._col_name_to_idx = copy.deepcopy(self._col_name_to_idx)

        # merge correlation
        if self.options.correlation.is_enabled and other.options.correlation.is_enabled:
            merged_profile.correlation_matrix = self._merge_correlation(other)

        # recompute chi2 if needed
        if (
            self.options.chi2_homogeneity.is_enabled
            and other.options.chi2_homogeneity.is_enabled
        ):

            chi2_mat1 = self.chi2_matrix
            chi2_mat2 = other.chi2_matrix
            n1 = self.total_samples - self.row_is_null_count
            n2 = other.total_samples - other.row_is_null_count
            if n1 == 0:
                merged_profile.chi2_matrix = chi2_mat2
            elif n2 == 0:
                merged_profile.chi2_matrix = chi2_mat1
            elif chi2_mat1 is None or chi2_mat2 is None:
                merged_profile.chi2_matrix = None
            else:
                merged_profile.chi2_matrix = merged_profile._update_chi2()

        if (
            self.options.null_replication_metrics.is_enabled
            and other.options.null_replication_metrics.is_enabled
        ):
            merged_profile._null_replication_metrics = (
                self._merge_null_replication_metrics(other)
            )

        return merged_profile

    def diff(  # type: ignore[override]
        self, other_profile: StructuredProfiler, options: Optional[Dict] = None
    ) -> Dict:
        """
        Find the difference between 2 Profiles and return the report.

        :param other_profile: profile finding the difference with this one
        :type other_profile: StructuredProfiler
        :param options: options to change results of the difference
        :type options: dict
        :return: difference of the profiles
        :rtype: dict
        """
        if options is None:
            options = {}

        report = super().diff(other_profile, options)
        report["global_stats"].update(
            {
                "samples_used": utils.find_diff_of_numbers(
                    self._max_col_samples_used, other_profile._max_col_samples_used
                ),
                "column_count": utils.find_diff_of_numbers(
                    len(self._profile), len(other_profile._profile)
                ),
                "row_count": utils.find_diff_of_numbers(
                    self.total_samples, other_profile.total_samples
                ),
                "row_has_null_ratio": utils.find_diff_of_numbers(
                    self._get_row_has_null_ratio(),
                    other_profile._get_row_has_null_ratio(),
                ),
                "row_is_null_ratio": utils.find_diff_of_numbers(
                    self._get_row_is_null_ratio(),
                    other_profile._get_row_is_null_ratio(),
                ),
                "unique_row_ratio": utils.find_diff_of_numbers(
                    self._get_unique_row_ratio(), other_profile._get_unique_row_ratio()
                ),
                "duplicate_row_count": utils.find_diff_of_numbers(
                    self._get_duplicate_row_count(),
                    other_profile._get_row_is_null_ratio(),
                ),
                "correlation_matrix": utils.find_diff_of_matrices(
                    self.correlation_matrix, other_profile.correlation_matrix
                ),
                "chi2_matrix": utils.find_diff_of_matrices(
                    self.chi2_matrix, other_profile.chi2_matrix
                ),
                "profile_schema": defaultdict(list),
            }
        )
        report.update({"data_stats": []})

        # Extract the schema of each profile
        self_profile_schema = defaultdict(list)
        other_profile_schema = defaultdict(list)
        for i in range(len(self._profile)):
            col_name = self._profile[i].name
            self_profile_schema[col_name].append(i)
        for i in range(len(other_profile._profile)):
            col_name = other_profile._profile[i].name
            other_profile_schema[col_name].append(i)

        report["global_stats"][
            "profile_schema"
        ] = utils.find_diff_of_dicts_with_diff_keys(
            self_profile_schema, other_profile_schema
        )

        # Only find the diff of columns if the schemas are exactly the same
        if self_profile_schema == other_profile_schema:
            for i in range(len(self._profile)):
                report["data_stats"].append(
                    self._profile[i].diff(other_profile._profile[i], options=options)
                )

        return _prepare_report(
            report,
            output_format=options.get("output_format", None),
            omit_keys=options.get("omit_keys", None),
        )

    @property
    def _max_col_samples_used(self) -> int:
        """Calculate and return the maximum samples used in cols."""
        samples_used = 0
        for col in self._profile:
            samples_used = max(samples_used, col.sample_size)
        return samples_used

    @property
    def _min_col_samples_used(self) -> int:
        """
        Calculate and return the number of rows that were completely sampled.

        i.e. every column in the Profile was read up to this row (possibly
        further in some cols)
        """
        return min([col.sample_size for col in self._profile], default=0)

    @property
    def _min_sampled_from_batch(self) -> int:
        """Return number of rows that were completely sampled in most recent batch."""
        return min([col._last_batch_size for col in self._profile], default=0)

    @staticmethod
    def _get_and_validate_schema_mapping(
        schema1: Dict[Union[str, int], List[int]],
        schema2: Dict[Union[str, int], List[int]],
        strict: bool = False,
    ) -> Dict[int, int]:
        """
        Validate compatibility between schema1 and schema2.

        Return a dict mapping indices in schema1 to their corresponding indices
        in schema2.
        In __add__: want to map self _profile idx -> other _profile idx
        In _update_profile_from_chunk: want to map data idx -> _profile idx

        :param schema1: a column name to index mapping
        :type schema1: Dict[str, list[int]]
        :param schema2: a column name to index mapping
        :type schema2: Dict[str, list[int]]
        :param strict: whether or not to strictly match (__add__ case)
        :type strict: bool
        :return: a mapping of indices in schema1 to indices in schema2
        :rtype: Dict[int, int]
        """
        len_schema1 = len(schema1)
        len_schema2 = len(schema2)

        # If both non-empty, must be same length
        if 0 < len_schema1 != len_schema2 > 0:
            raise ValueError(
                "Attempted to merge profiles with different " "numbers of columns"
            )

        # In the case of __add__ with one of the schemas not initialized
        if strict and (len_schema1 == 0 or len_schema2 == 0):
            raise ValueError("Cannot merge empty profiles.")

        # In the case of _update_from_chunk with uninitialized schema
        if not strict and len_schema2 == 0:
            return {
                col_ind: col_ind
                for col_ind_list in schema1.values()
                for col_ind in col_ind_list
            }

        # Map indices in schema1 to indices in schema2
        schema_mapping = dict()

        for key in schema1:
            # Pandas columns are int by default, but need to fuzzy match strs
            if isinstance(key, str):
                key = key.lower()
            if key not in schema2:
                raise ValueError(
                    "Columns do not match, cannot update " "or merge profiles."
                )

            elif len(schema1[key]) != len(schema2[key]):
                raise ValueError(
                    f"Different number of columns detected for "
                    f"'{key}', cannot update or merge profiles."
                )

            is_duplicate_col = len(schema1[key]) > 1
            for schema1_col_ind, schema2_col_ind in zip(schema1[key], schema2[key]):
                if is_duplicate_col and (schema1_col_ind != schema2_col_ind):
                    raise ValueError(
                        f"Different column indices under "
                        f"duplicate name '{key}', cannot update "
                        f"or merge unless schema is identical."
                    )
                schema_mapping[schema1_col_ind] = schema2_col_ind

        return schema_mapping

    def report(self, report_options: Dict = None) -> Dict:
        """Return a report."""
        if not report_options:
            report_options = {
                "output_format": None,
                "num_quantile_groups": 4,
                "remove_disabled_flag": False,
            }

        output_format = report_options.get("output_format", None)
        omit_keys = report_options.get("omit_keys", [])
        num_quantile_groups = report_options.get("num_quantile_groups", 4)
        remove_disabled_flag = report_options.get("remove_disabled_flag", False)

        report: Dict = OrderedDict(
            [
                (
                    "global_stats",
                    {
                        "samples_used": self._max_col_samples_used,
                        "column_count": len(self._profile),
                        "row_count": self.total_samples,
                        "row_has_null_ratio": self._get_row_has_null_ratio(),
                        "row_is_null_ratio": self._get_row_is_null_ratio(),
                        "unique_row_ratio": self._get_unique_row_ratio(),
                        "duplicate_row_count": self._get_duplicate_row_count(),
                        "file_type": self.file_type,
                        "encoding": self.encoding,
                        "correlation_matrix": self.correlation_matrix,
                        "chi2_matrix": self.chi2_matrix,
                        "profile_schema": defaultdict(list),  # type: ignore
                        "times": self.times,
                    },
                ),
                ("data_stats", []),
            ]
        )

        for i in range(len(self._profile)):
            col_name = self._profile[i].name
            report["global_stats"]["profile_schema"][col_name].append(i)
            report["data_stats"].append(self._profile[i].report(remove_disabled_flag))
            quantiles = report["data_stats"][i]["statistics"].get("quantiles")
            if quantiles:
                quantiles = calculate_quantiles(num_quantile_groups, quantiles)
                report["data_stats"][i]["statistics"]["quantiles"] = quantiles
            if (
                self.options.null_replication_metrics.is_enabled
                and i in self._null_replication_metrics
            ):
                report["data_stats"][i][
                    "null_replication_metrics"
                ] = self._null_replication_metrics[i]

        return _prepare_report(report, output_format, omit_keys)

    def _get_unique_row_ratio(self) -> float:
        """Return unique row ratio."""
        if self.total_samples:
            return len(self.hashed_row_dict) / self.total_samples
        return 0

    def _get_row_is_null_ratio(self) -> float:
        """Return whether row is null ratio."""
        if self._min_col_samples_used:
            return self.row_is_null_count / self._min_col_samples_used
        return 0

    def _get_row_has_null_ratio(self) -> float:
        """Return whether row has null ratio."""
        if self._min_col_samples_used:
            return self.row_has_null_count / self._min_col_samples_used
        return 0

    def _get_duplicate_row_count(self) -> int:
        """Retun dup row count."""
        return self.total_samples - len(self.hashed_row_dict)

    @utils.method_timeit(name="row_stats")
    def _update_row_statistics(
        self, data: pd.DataFrame, sample_ids: List[int] = None
    ) -> None:
        """
        Iterate over the provided dataset row by row and calculate row stats.

        Specifically, number of unique rows,
        rows containing null values, and total rows reviewed. This
        function is safe to use in batches.

        :param data: a dataset
        :type data: pandas.DataFrame
        :param sample_ids: list of indices in order they were sampled in data
        :type sample_ids: list(int)
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Cannot calculate row statistics on data that is" "not a DataFrame"
            )

        self.total_samples += len(data)
        try:
            self.hashed_row_dict.update(
                dict.fromkeys(pd.util.hash_pandas_object(data, index=False), True)
            )
        except TypeError:
            self.hashed_row_dict.update(
                dict.fromkeys(
                    pd.util.hash_pandas_object(data.astype(str), index=False), True
                )
            )

        # Calculate Null Column Count
        null_rows = set()
        null_in_row_count = set()
        first_col_flag = True
        for column in self._profile:
            null_type_dict = column.null_types_index
            null_row_indices = set()
            if null_type_dict:
                null_row_indices = set.union(*null_type_dict.values())

            # If sample ids provided, only consider nulls in rows that
            # were fully sampled
            if sample_ids is not None:
                # This is the amount (integer) indices were shifted by in the
                # event of overlap
                shift = column._index_shift
                if shift is None:
                    # Shift is None if index is str or if no overlap detected
                    null_row_indices = null_row_indices.intersection(
                        data.index[sample_ids[: self._min_sampled_from_batch]]
                    )
                else:
                    # Only shift if index shift detected (must be ints)
                    null_row_indices = null_row_indices.intersection(
                        data.index[sample_ids[: self._min_sampled_from_batch]] + shift
                    )

            # Find the common null indices between the columns
            if first_col_flag:
                null_rows = null_row_indices
                null_in_row_count = null_row_indices
                first_col_flag = False
            else:
                null_rows = null_rows.intersection(null_row_indices)
                null_in_row_count = null_in_row_count.union(null_row_indices)

        # If sample_ids provided, increment since that means only new data read
        if sample_ids is not None:
            self.row_has_null_count += len(null_in_row_count)
            self.row_is_null_count += len(null_rows)
        else:
            self.row_has_null_count = len(null_in_row_count)
            self.row_is_null_count = len(null_rows)

    def _get_correlation(
        self, clean_samples: Dict, batch_properties: Dict
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix on the cleaned data.

        :param clean_samples: the input cleaned dataset
        :type clean_samples: dict()
        :param batch_properties: mean/std/counts of each batch column necessary
        for correlation computation
        :type batch_properties: dict()
        :return: correlation matrix
        :rtype: pd.DataFrame
        """
        columns = self.options.correlation.columns
        column_ids = list(range(len(self._profile)))
        if columns is not None:
            column_ids = [
                idx for col_name in columns for idx in self._col_name_to_idx[col_name]
            ]
        clean_column_ids = []
        for idx in column_ids:
            data_type = (
                self._profile[idx].profiles["data_type_profile"].selected_data_type
            )
            if data_type not in ["int", "float"]:
                clean_samples.pop(idx)
            else:
                clean_column_ids.append(idx)
        data = pd.DataFrame(clean_samples).apply(pd.to_numeric, errors="coerce")
        means = {index: mean for index, mean in enumerate(batch_properties["mean"])}
        data = data.fillna(value=means)
        data = data[clean_column_ids]

        # Update the counts/std if needed (i.e. if null rows or exist)
        if (len(data) != batch_properties["count"]).any():
            adjusted_stds = np.sqrt(
                batch_properties["std"] ** 2
                * (batch_properties["count"] - 1)
                / (len(data) - 1)
            )
            batch_properties["std"] = adjusted_stds
        # Set count key to a single number now that everything's been adjusted
        batch_properties["count"] = len(data)

        # fill correlation matrix with nan initially
        n_cols = len(self._profile)
        corr_mat = np.full((n_cols, n_cols), np.nan)

        # then, fill in the correlations for valid columns
        rows = [[id] for id in clean_column_ids]
        corr_mat[rows, clean_column_ids] = np.corrcoef(data, rowvar=False)

        return corr_mat

    @utils.method_timeit(name="correlation")
    def _update_correlation(
        self, clean_samples: Dict, prev_dependent_properties: Dict
    ) -> None:
        """
        Update correlation matrix for cleaned data.

        :param clean_samples: the input cleaned dataset
        :type clean_samples: dict()
        """
        batch_properties = self._get_correlation_dependent_properties(clean_samples)
        batch_corr = self._get_correlation(clean_samples, batch_properties)

        self.correlation_matrix = self._merge_correlation_helper(
            self.correlation_matrix,
            prev_dependent_properties["mean"],
            prev_dependent_properties["std"],
            self.total_samples - self.row_is_null_count,
            batch_corr,
            batch_properties["mean"],
            batch_properties["std"],
            batch_properties["count"],
        )

    @utils.method_timeit(name="correlation")
    def _merge_correlation(self, other: StructuredProfiler) -> pd.DataFrame:
        """
        Merge correlation matrix from two profiles.

        :param other: the other profile that needs to be merged
        :return:
        """
        corr_mat1 = self.correlation_matrix
        corr_mat2 = other.correlation_matrix
        n1 = self.total_samples - self.row_is_null_count
        n2 = other.total_samples - other.row_is_null_count
        if n1 == 0:
            return corr_mat2
        if n2 == 0:
            return corr_mat1

        if corr_mat1 is None or corr_mat2 is None:
            return None

        # get column indices without nan
        col_ids1 = np.where(~np.isnan(corr_mat1).all(axis=0))[0]
        col_ids2 = np.where(~np.isnan(corr_mat2).all(axis=0))[0]

        if len(col_ids1) != len(col_ids2) or len(col_ids1) <= 1:
            return None
        if (col_ids1 != col_ids2).any():
            return None

        mean1 = np.array(
            [
                self._profile[idx].profile["statistics"]["mean"]
                for idx in range(len(self._profile))
                if idx in col_ids1
            ]
        )
        std1 = np.array(
            [
                self._profile[idx].profile["statistics"]["stddev"]
                for idx in range(len(self._profile))
                if idx in col_ids1
            ]
        )

        mean2 = np.array(
            [
                other._profile[idx].profile["statistics"]["mean"]
                for idx in range(len(self._profile))
                if idx in col_ids2
            ]
        )
        std2 = np.array(
            [
                other._profile[idx].profile["statistics"]["stddev"]
                for idx in range(len(self._profile))
                if idx in col_ids2
            ]
        )
        return self._merge_correlation_helper(
            corr_mat1, mean1, std1, n1, corr_mat2, mean2, std2, n2
        )

    def _get_correlation_dependent_properties(self, batch: Dict = None) -> Dict:
        """
        Obtain mean/stddev for calculating correlation.

        By default, it will compute it on all columns in the profiler,
        but if a batch is given, it will compute it only for the columns
        in the batch.

        :param batch: Batch data
        :type batch: dict
        :return: dependent properties
        :rtype: dict
        """
        dependent_properties = {
            "mean": np.full(len(self._profile), np.nan),
            "std": np.full(len(self._profile), np.nan),
            "count": np.full(len(self._profile), np.nan),
        }
        for id in range(len(self._profile)):
            compiler = self._profile[id]
            data_type_compiler = compiler.profiles["data_type_profile"]
            data_type = data_type_compiler.selected_data_type
            if data_type in ["int", "float"]:
                data_type_profiler = data_type_compiler._profiles[data_type]
                # Finding dependent values of previous, existing data
                if batch is None:
                    n = data_type_profiler.match_count
                    dependent_properties["mean"][id] = data_type_profiler.mean
                    # Subtract null row count as those aren't included in corr. calc
                    dependent_properties["std"][id] = np.sqrt(
                        data_type_profiler._biased_variance
                        * n
                        / (self.total_samples - self.row_is_null_count - 1)
                    )
                    dependent_properties["count"][id] = n
                # Finding the properties of the batch data if given
                elif id in batch.keys():
                    history = data_type_profiler._batch_history[-1]
                    n = history["match_count"]
                    # Since we impute values, we want the total rows (including nulls)
                    dependent_properties["mean"][id] = history["mean"]
                    dependent_properties["std"][id] = np.sqrt(
                        history["biased_variance"] * n / (n - 1)
                    )
                    dependent_properties["count"][id] = n

        return dependent_properties

    @staticmethod
    def _merge_correlation_helper(
        corr_mat1: pd.DataFrame,
        mean1: np.ndarray,
        std1: np.ndarray,
        n1: int,
        corr_mat2: pd.DataFrame,
        mean2: np.ndarray,
        std2: np.ndarray,
        n2: int,
    ) -> pd.DataFrame:
        """
        Help merge correlation matrix from two profiles.

        :param corr_mat1: correlation matrix of profile1
        :type corr_mat1: pd.DataFrame
        :param mean1: mean of columns of profile1
        :type mean1: np.array
        :param std1: standard deviation of columns of profile1
        :type std1: np.array
        :param corr_mat2: correlation matrix of profile2
        :type corr_mat2: pd.DataFrame
        :param mean2: mean of columns of profile2
        :type mean2: np.array
        :param std2: standard deviation of columns of profile2
        :type std2: np.array
        :return: merged correlation matrix
        """
        if corr_mat1 is None:
            return corr_mat2
        elif corr_mat2 is None:
            return corr_mat1
        elif len(mean1) == 0:
            return corr_mat2
        elif len(mean2) == 0:
            return corr_mat1

        std_mat1 = np.outer(std1, std1)
        std_mat2 = np.outer(std2, std2)
        mean_diff_vector = mean1 - mean2
        mean_diff_mat = np.outer(mean_diff_vector, mean_diff_vector)

        cov1 = corr_mat1 * std_mat1
        cov2 = corr_mat2 * std_mat2

        n = n1 + n2

        cov = cov1 * (n1 - 1) + cov2 * (n2 - 1) + mean_diff_mat * (n1 * n2) / n
        cov = cov / (n - 1)

        delta = mean2 - mean1
        M2_1 = (n1 - 1) * (std1**2)
        M2_2 = (n2 - 1) * (std2**2)
        M2 = M2_1 + M2_2 + delta**2 * n1 * n2 / n
        std = np.sqrt(M2 / (n - 1))

        std_mat = np.outer(std, std)
        corr_mat = cov / std_mat

        return corr_mat

    def _update_chi2(self) -> np.ndarray:
        """
        Calculate p-val from chi-squared test for homogeneity between categorical cols.

        :return: A matrix of p-values corresponding to the results
        of the chi2 test between the columns
        :rtype: np.array(np.array(float))
        """
        n_cols = len(self._profile)
        # Fill matrix with nan initially
        chi2_mat = np.full((n_cols, n_cols), np.nan)
        # Compute chi_sq for each
        for i in range(n_cols):
            data_stats_compiler1 = self._profile[i].profiles["data_stats_profile"]
            profiler1 = data_stats_compiler1._profiles["category"]
            if not profiler1.is_match:
                continue
            for j in range(i, n_cols):
                if i == j:
                    chi2_mat[i][j] = 1
                    continue
                data_stats_compiler2 = self._profile[j].profiles["data_stats_profile"]
                profiler2 = data_stats_compiler2._profiles["category"]
                if not profiler2.is_match:
                    continue

                results = utils.perform_chi_squared_test_for_homogeneity(
                    profiler1.categorical_counts,
                    profiler1.sample_size,
                    profiler2.categorical_counts,
                    profiler2.sample_size,
                )
                chi2_mat[i][j] = results["p-value"]
                chi2_mat[j][i] = results["p-value"]

        return chi2_mat

    def _update_null_replication_metrics(self, clean_samples: Dict) -> None:
        """
        Calculate metrics needed for replicating null values.

        The replication of null values occurs in capitalone/synthetic-data.

        Required for running LDA based binary classifier
        where predicted class label indicates whether
        a value of a column should be NaN (1) or not (0).

        :param clean_samples: input cleaned dataset
        :type clean_samples: dict
        """
        data = pd.DataFrame(clean_samples).apply(pd.to_numeric, errors="coerce")

        get_data_type = lambda profile: profile.profiles[  # NOQA: E731
            "data_type_profile"
        ].selected_data_type
        get_data_type_profiler = lambda profile: profile.profiles[  # NOQA: E731
            "data_type_profile"
        ]._profiles[get_data_type(profile)]

        total_row_sum = np.asarray(
            [
                get_data_type_profiler(profile).sum
                if get_data_type(profile)
                else np.nan
                for profile in self._profile
            ]
        )

        if not isinstance(self._null_replication_metrics, dict):
            self._null_replication_metrics = dict()

        for col_id, profile in enumerate(self._profile):
            null_count = getattr(profile, "null_count")
            if null_count == 0:
                # No missing values to replicate
                continue

            sample_size = getattr(profile, "sample_size")
            true_count = sample_size - null_count

            # null_count and sample_size get updated on profile_update
            # Therefore the priors are calculated from the updated values
            prior_null = null_count / sample_size
            prior_not_null = true_count / sample_size

            # Gets list of null indices of the entire dataset
            null_type_dict = getattr(profile, "null_types_index")
            null_indices = list(set.union(*null_type_dict.values()))

            # Keep only the null indices inside the chunk (reverse index shift)
            if profile._index_shift is not None:
                null_indices = [index - profile._index_shift for index in null_indices]
                null_indices = [index for index in null_indices if index >= 0]

            # Partition data based on whether target column value is null or not
            # Calculate sum, mean of each partition without including current column
            # in calculation
            sum_null = (
                data.loc[data.index.intersection(null_indices), data.columns != col_id]
                .sum()
                .to_numpy()
            )

            # Add old sum_null if exists
            if col_id in self._null_replication_metrics:
                old_class_sum = self._null_replication_metrics[col_id]["class_sum"]
                old_sum_null = old_class_sum[1]

                # np.array for element wise addition
                old_sum_null = np.asarray(old_sum_null)
                sum_null += old_sum_null

            sum_not_null = np.delete(total_row_sum, col_id) - sum_null

            mean_null = sum_null / null_count

            mean_not_null = np.full(len(self._profile) - 1, np.nan)
            if not true_count:
                sum_not_null = np.full(len(self._profile) - 1, np.nan)
            else:
                mean_not_null = sum_not_null / true_count

            # Convert numpy arrays to lists (serializable)
            sum_null = sum_null.tolist()
            sum_not_null = sum_not_null.tolist()

            mean_null = mean_null.tolist()
            mean_not_null = mean_not_null.tolist()

            # Array index serves as class label
            # 0 indicates not null, 1 indicates null
            self._null_replication_metrics[col_id] = {
                "class_prior": [prior_not_null, prior_null],
                "class_sum": [sum_not_null, sum_null],
                "class_mean": [mean_not_null, mean_null],
            }

    def _merge_null_replication_metrics(self, other: StructuredProfiler) -> Dict:
        """
        Merge null replication metrics between two data profiles.

        :param other: profile being added to this one.
        :type other: StructuredProfiler
        :return: merged null replication metrics
        :rtype: dict
        """
        get_data_type = lambda profile: profile.profiles[  # NOQA: E731
            "data_type_profile"
        ].selected_data_type
        get_data_type_profiler = lambda profile: profile.profiles[  # NOQA: E731
            "data_type_profile"
        ]._profiles[get_data_type(profile)]

        self_row_sum = np.asarray(
            [
                get_data_type_profiler(profile).sum
                if get_data_type(profile)
                else np.nan
                for profile in self._profile
            ]
        )
        other_row_sum = np.asarray(
            [
                get_data_type_profiler(profile).sum
                if get_data_type(profile)
                else np.nan
                for profile in other._profile
            ]
        )
        total_row_sum: np.ndarray = self_row_sum + other_row_sum
        merged_properties: Dict = defaultdict(dict)
        for col_id in range(len(self._profile)):
            self_profile = self._profile[col_id]
            other_profile = other._profile[col_id]

            self_null_count = getattr(self_profile, "null_count")
            other_null_count = getattr(other_profile, "null_count")
            null_count = self_null_count + other_null_count
            if null_count == 0:
                continue

            self_sample_size = getattr(self_profile, "sample_size")
            other_sample_size = getattr(other_profile, "sample_size")
            sample_size = self_sample_size + other_sample_size
            true_count = sample_size - null_count

            prior_null = null_count / sample_size
            prior_not_null = true_count / sample_size

            self_sum_null = (
                self._null_replication_metrics[col_id]["class_sum"][1]
                if col_id in self._null_replication_metrics
                else None
            )
            other_sum_null = (
                other._null_replication_metrics[col_id]["class_sum"][1]
                if col_id in other._null_replication_metrics
                else None
            )
            # Initialize zeros array of size (number of columns - 1)
            sum_null = np.zeros(len(self._profile) - 1)

            # Add sum_nulls if they exist
            # Guarantees that at least one of self_sum_null, other_sum_null != None
            if self_sum_null is not None:
                sum_null += np.asarray(self_sum_null)

            if other_sum_null is not None:
                sum_null += np.asarray(other_sum_null)

            sum_not_null = np.delete(total_row_sum, col_id) - sum_null

            mean_null = sum_null / null_count

            mean_not_null = np.full(len(self._profile) - 1, np.nan)
            if not true_count:
                sum_not_null = np.full(len(self._profile) - 1, np.nan)
            else:
                mean_not_null = sum_not_null / true_count

            # Convert numpy arrays to lists (serializable)
            sum_null = sum_null.tolist()
            sum_not_null = sum_not_null.tolist()

            mean_null = mean_null.tolist()
            mean_not_null = mean_not_null.tolist()

            merged_properties[col_id] = {
                # Array index serves as class label
                # 0 indicates not null, 1 indicates null
                "class_prior": [prior_not_null, prior_null],
                "class_sum": [sum_not_null, sum_null],
                "class_mean": [mean_not_null, mean_null],
            }

        return merged_properties

    def _update_profile_from_chunk(
        self,
        data: Union[List, pd.Series, pd.DataFrame],
        sample_size: int,
        min_true_samples: int = None,
    ) -> None:
        """
        Iterate over the columns of a dataset and identify its parameters.

        :param data: a dataset
        :type data: pandas.DataFrame
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :return: None
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, list):
            data = pd.DataFrame(data, dtype=object)

        # Calculate schema of incoming data
        mapping_given = defaultdict(list)
        for col_idx in range(len(data.columns)):
            col = data.columns[col_idx]
            # Pandas columns are int by default, but need to fuzzy match strs
            if isinstance(col, str):
                col = col.lower()
            mapping_given[col].append(col_idx)

        # Validate schema compatibility and index mapping from data to _profile
        col_idx_to_prof_idx = self._get_and_validate_schema_mapping(
            mapping_given, self._col_name_to_idx
        )

        try:
            from tqdm import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False
        finally:
            if not has_tqdm or logger.getEffectiveLevel() > logging.INFO:

                def tqdm(level: Set[int]) -> Generator[int, None, None]:
                    for i, e in enumerate(level):
                        # These will automatically be ignored if user sets
                        # logger level as higher than INFO
                        logger.info(f"Processing Column {i + 1}/{len(level)}")
                        yield e

        # Shuffle indices once and share with columns
        sample_ids = [*utils.shuffle_in_chunks(len(data), len(data))]

        # If there are no minimum true samples, you can sort to save time
        if min_true_samples in [None, 0]:
            sample_ids[0] = sample_ids[0][:sample_size]
            # Sort the sample_ids and replace prior
            sample_ids[0] = sorted(sample_ids[0])

        # Numpy arrays allocate to heap and can be shared between processes
        # Non-locking multiprocessing fails on machines without POSIX (windows)
        # The function handles that situation, but will be single process
        # Newly introduced features (python3.8) improves the situation
        sample_ids: np.ndarray = np.array(sample_ids)  # type: ignore

        # Record the previous mean/std values of columns that need to
        # have correlation updated.
        corr_prev_dependent_properties = self._get_correlation_dependent_properties()

        # Create StructuredColProfilers upon initialization
        # Record correlation between columns in data and index in _profile
        if len(self._profile) == 0:
            # Already calculated incoming schema for validation
            self._col_name_to_idx = mapping_given
            for col_idx in range(data.shape[1]):
                # Add blank StructuredColProfiler to _profile
                self._profile.append(
                    StructuredColProfiler(  # type: ignore
                        sample_size=sample_size,
                        min_true_samples=min_true_samples,  # type: ignore
                        sample_ids=sample_ids,  # type: ignore
                        column_index=col_idx,
                        options=self.options,
                    )
                )

        # Generate pool and estimate datasize
        pool = None
        if self.options.multiprocess.is_enabled:
            est_data_size = data[:50000].memory_usage(index=False, deep=True).sum()
            est_data_size = (est_data_size / min(50000, len(data))) * len(data)
            pool, pool_size = utils.generate_pool(
                max_pool_size=None, data_size=est_data_size, cols=len(data.columns)
            )

        # Format the data
        notification_str = "Finding the Null values in the columns... "
        if pool:
            notification_str += " (with " + str(pool_size) + " processes)"

        # Keys are _profile indices
        clean_sampled_dict = {}
        # Keys are column indices in data
        multi_process_dict = {}
        single_process_list = set()

        if sample_size < len(data):
            warnings.warn(
                "The data will be profiled with a sample size of {}. "
                "All statistics will be based on this subsample and "
                "not the whole dataset.".format(sample_size)
            )

        if pool is not None:
            # Create a bunch of simultaneous column conversions
            for col_idx in range(data.shape[1]):
                col_ser = data.iloc[:, col_idx]
                prof_idx = col_idx_to_prof_idx[col_idx]
                if min_true_samples is None:
                    min_true_samples = self._profile[prof_idx]._min_true_samples
                try:
                    null_values: Dict = self._profile[prof_idx]._null_values.copy()
                    if self.options.column_null_values:
                        null_values.update(
                            self.options.column_null_values.get(col_idx, {})
                        )

                    multi_process_dict[col_idx] = pool.apply_async(
                        self._profile[prof_idx].clean_data_and_get_base_stats,
                        (
                            col_ser,
                            sample_size,
                            null_values,
                            min_true_samples,
                            sample_ids,
                        ),
                    )
                except Exception as e:
                    logger.info(e)
                    single_process_list.add(col_idx)

            # Iterate through multiprocessed columns collecting results
            logger.info(notification_str)
            for col_idx in tqdm(multi_process_dict.keys()):
                try:
                    prof_idx = col_idx_to_prof_idx[col_idx]
                    clean_sampled_dict[prof_idx], base_stats = multi_process_dict[
                        col_idx
                    ].get()
                    self._profile[prof_idx]._update_base_stats(base_stats)
                except Exception as e:
                    logger.info(e)
                    single_process_list.add(col_idx)

            # Clean up any columns which errored
            if len(single_process_list) > 0:
                logger.info(
                    "Errors in multiprocessing occured:",
                    len(single_process_list),
                    "errors, reprocessing...",
                )
                for col_idx in tqdm(single_process_list):
                    col_ser = data.iloc[:, col_idx]
                    prof_idx = col_idx_to_prof_idx[col_idx]
                    if min_true_samples is None:
                        min_true_samples = self._profile[prof_idx]._min_true_samples

                    null_values = self._profile[prof_idx]._null_values.copy()
                    if self.options.column_null_values:
                        null_values.update(
                            self.options.column_null_values.get(col_idx, {})
                        )

                    clean_sampled_dict[prof_idx], base_stats = self._profile[
                        prof_idx
                    ].clean_data_and_get_base_stats(
                        col_ser, sample_size, null_values, min_true_samples, sample_ids
                    )
                    self._profile[prof_idx]._update_base_stats(base_stats)

            pool.close()  # Close pool for new tasks
            pool.join()  # Wait for all workers to complete

        else:  # No pool
            logger.info(notification_str)
            for col_idx in tqdm(range(data.shape[1])):
                col_ser = data.iloc[:, col_idx]
                prof_idx = col_idx_to_prof_idx[col_idx]
                if min_true_samples is None:
                    min_true_samples = self._profile[prof_idx]._min_true_samples

                null_values = self._profile[prof_idx]._null_values.copy()
                if self.options.column_null_values:
                    null_values.update(self.options.column_null_values.get(col_idx, {}))

                clean_sampled_dict[prof_idx], base_stats = self._profile[
                    prof_idx
                ].clean_data_and_get_base_stats(
                    df_series=col_ser,
                    sample_size=sample_size,
                    null_values=null_values,
                    min_true_samples=min_true_samples,
                    sample_ids=sample_ids,
                )
                self._profile[prof_idx]._update_base_stats(base_stats)

        # Process and label the data
        notification_str = "Calculating the statistics... "
        pool = None
        if self.options.multiprocess.is_enabled:
            pool, pool_size = utils.generate_pool(4, est_data_size)
            if pool:
                notification_str += " (with " + str(pool_size) + " processes)"

        logger.info(notification_str)

        for prof_idx in tqdm(clean_sampled_dict.keys()):
            self._profile[prof_idx].update_column_profilers(
                clean_sampled_dict[prof_idx], pool
            )

        if pool is not None:
            pool.close()  # Close pool for new tasks
            pool.join()  # Wait for all workers to complete

        # Only pass along sample ids if necessary
        samples_for_row_stats = None
        if min_true_samples not in [None, 0]:
            samples_for_row_stats = np.concatenate(sample_ids)

        if self.options.correlation.is_enabled:
            self._update_correlation(clean_sampled_dict, corr_prev_dependent_properties)

        if self.options.chi2_homogeneity.is_enabled:
            self.chi2_matrix = self._update_chi2()
        self._update_row_statistics(data, samples_for_row_stats)

        # Calculate metrics specific to capitalone/synthetic-data
        if self.options.null_replication_metrics.is_enabled:
            self._update_null_replication_metrics(clean_sampled_dict)

    def save(self, filepath: str = None) -> None:
        """
        Save profiler to disk.

        :param filepath: Path of file to save to
        :type filepath: String
        :return: None
        """
        # Create dictionary for all metadata, options, and profile
        data_dict = {
            "total_samples": self.total_samples,
            "encoding": self.encoding,
            "file_type": self.file_type,
            "row_has_null_count": self.row_has_null_count,
            "row_is_null_count": self.row_is_null_count,
            "hashed_row_dict": self.hashed_row_dict,
            "_samples_per_update": self._samples_per_update,
            "_min_true_samples": self._min_true_samples,
            "options": self.options,
            "chi2_matrix": self.chi2_matrix,
            "_profile": self.profile,
            "_col_name_to_idx": self._col_name_to_idx,
            "times": self.times,
        }

        self._save_helper(filepath, data_dict)


class Profiler(object):
    """For profiling data."""

    def __new__(  # type: ignore
        cls,
        data: Data,
        samples_per_update: int = None,
        min_true_samples: int = 0,
        options: ProfilerOptions = None,
        profiler_type: str = None,
    ) -> Union[GraphProfiler, StructuredProfiler, UnstructuredProfiler]:
        """
        Instantiate Structured and Unstructured Profilers.

        This is a factory class.

        :param data: Data to be profiled, type allowed depends on the
            profiler_type
        :type data: Data class object
        :param samples_per_update: Number of samples to use to generate profile
        :type samples_per_update: int
        :param min_true_samples: Min number of samples required for the profiler
        :type min_true_samples: int
        :param options: Options for the profiler.
        :type options: ProfilerOptions Object
        :param profiler_type: Type of Profiler ("graph"/"structured"/"unstructured")
        :type profiler_type: str
        :return: Union[GraphProfiler, StructuredProfiler, UnstructuredProfiler]
        """
        if profiler_type is None:
            # defaults as structured
            profiler_type = "structured"
            """
            Graph if instance of GraphData
            Unstructured if data is Data object and is_structured is False
            """
            if isinstance(data, data_readers.graph_data.GraphData) or isinstance(
                data, nx.Graph
            ):
                profiler_type = "graph"
            elif isinstance(data, data_readers.base_data.BaseData):
                if not data.is_structured:
                    profiler_type = "unstructured"
            elif isinstance(data, str):
                profiler_type = "unstructured"
            # the below checks the viable structured formats, on failure raises
            elif not isinstance(data, (list, nx.Graph, pd.DataFrame, pd.Series)):
                raise ValueError(
                    "Data must either be imported using the "
                    "data_readers, nx.Graph, pd.Series, or pd.DataFrame."
                )

        # Construct based off of initial kwarg input or inference
        if profiler_type == "graph":
            profile = GraphProfiler(data, options=options)
            profile.update(data)
            return profile
        elif profiler_type == "structured":
            return StructuredProfiler(
                data, samples_per_update, min_true_samples, options
            )
        elif profiler_type == "unstructured":
            return UnstructuredProfiler(
                data, samples_per_update, min_true_samples, options
            )
        else:
            raise ValueError(
                "Must specify 'profiler_type' to be 'graph', 'structured' or "
                + "'unstructured'."
            )

    @classmethod
    def load(cls, filepath: str) -> BaseProfiler:
        """
        Load profiler from disk.

        :param filepath: Path of file to load from
        :type filepath: String
        :return: Profiler being loaded, StructuredProfiler or
            UnstructuredProfiler
        :rtype: BaseProfiler
        """
        return BaseProfiler.load(filepath)
