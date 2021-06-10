#!/usr/bin/env python
"""
coding=utf-8

Build model for a dataset by identifying type of column along with its
respective parameters.
"""
from __future__ import print_function
from __future__ import division

import copy
import random
import re
from collections import OrderedDict
import warnings
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from . import utils
from .. import data_readers
from .column_profile_compilers import ColumnPrimitiveTypeProfileCompiler, \
    ColumnStatsProfileCompiler, ColumnDataLabelerCompiler, UnstructuredCompiler
from ..labelers.data_labelers import DataLabeler
from .helpers.report_helpers import calculate_quantiles, _prepare_report
from .profiler_options import ProfilerOptions, StructuredOptions, \
    UnstructuredOptions


class StructuredColProfiler(object):

    def __init__(self, df_series=None, sample_size=None, min_sample_size=5000,
                 sampling_ratio=0.2, min_true_samples=None,
                 sample_ids=None, pool=None, options=None):
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
        :param options: Options for the structured profiler.
        :type options: StructuredOptions Object
        """
        self.name = None
        self.options = options
        self._min_sample_size = min_sample_size
        self._sampling_ratio = sampling_ratio
        self._min_true_samples = min_true_samples
        if self._min_true_samples is None:
            self._min_true_samples = 0

        self.sample_size = 0
        self.sample = list()
        self.null_count = 0
        self.null_types = list()
        self.null_types_index = {}
        self._min_id = None
        self._max_id = None
        self._index_shift = None
        self._last_batch_size = None
        self.profiles = {}
                         
        if df_series is not None and len(df_series) > 0:
            
            if not sample_size:
                sample_size = self._get_sample_size(df_series)
            if sample_size < len(df_series):
                warnings.warn("The data will be profiled with a sample size of {}. "
                              "All statistics will be based on this subsample and "
                              "not the whole dataset.".format(sample_size))
                
            clean_sampled_df, base_stats = \
                self.clean_data_and_get_base_stats(
                    df_series=df_series, sample_size=sample_size,
                    min_true_samples=self._min_true_samples, sample_ids=sample_ids)
            self.update_column_profilers(clean_sampled_df, pool)
            self._update_base_stats(base_stats)

    def update_column_profilers(self, clean_sampled_df, pool):
        """
        Calculates type statistics and labels dataset
        
        :param clean_sampled_df: sampled series with none types dropped
        :type clean_sampled_df: Pandas.Series
        :param pool: pool utilized for multiprocessing
        :type pool: multiprocessing.pool
        """

        if self.name is None:
            self.name = clean_sampled_df.name
        if self.name != clean_sampled_df.name:
            raise ValueError(
                'Column names have changed, col {} does not match prior name {}',
                clean_sampled_df.name, self.name
            )
        
        # First run, create the compilers
        if self.profiles is None or len(self.profiles) == 0:
            self.profiles = {
                'data_type_profile':
                ColumnPrimitiveTypeProfileCompiler(
                    clean_sampled_df, self.options, pool),
                'data_stats_profile':
                ColumnStatsProfileCompiler(
                    clean_sampled_df, self.options, pool)
            }
        
            use_data_labeler = True
            if self.options and isinstance(self.options, StructuredOptions):
                use_data_labeler = self.options.data_labeler.is_enabled

            if use_data_labeler:
                self.profiles.update({
                    'data_label_profile':
                    ColumnDataLabelerCompiler(
                        clean_sampled_df, self.options, pool)
                })
        else:

            # Profile compilers being updated
            for profile in self.profiles.values():
                profile.update_profile(clean_sampled_df, pool)

    def __add__(self, other):
        """
        Merges two Structured profiles together overriding the `+` operator.

        :param other: structured profile being add to this one.
        :type other: StructuredColProfiler
        :return: merger of the two structured profiles
        """
        if type(other) is not type(self):
            raise TypeError('`{}` and `{}` are not of the same profiler type.'.
                            format(type(self).__name__, type(other).__name__))
        elif self.name != other.name:
            raise ValueError('Structured profile names are unmatched: {} != {}'
                             .format(self.name, other.name))
        elif set(self.profiles) != set(other.profiles):  # options check
            raise ValueError('Structured profilers were not setup with the same'
                             ' options, hence they do not calculate the same '
                             'profiles and cannot be added together.')
        merged_profile = StructuredColProfiler(
            df_series=pd.Series([]),
            min_sample_size=max(self._min_sample_size, other._min_sample_size),
            sampling_ratio=max(self._sampling_ratio, other._sampling_ratio),
            min_true_samples=max(self._min_true_samples,
                                 other._min_true_samples),
            options=self.options,
        )

        merged_profile.name = self.name
        merged_profile._update_base_stats(
            {"sample": self.sample,
             "sample_size": self.sample_size,
             "null_count": self.null_count,
             "null_types": copy.deepcopy(self.null_types_index),
             "min_id": self._min_id,
             "max_id": self._max_id}
        )
        merged_profile._update_base_stats(
            {"sample": other.sample,
             "sample_size": other.sample_size,
             "null_count": other.null_count,
             "null_types": copy.deepcopy(other.null_types_index),
             "min_id": other._min_id,
             "max_id": other._max_id}
        )
        samples = list(dict.fromkeys(self.sample + other.sample))
        merged_profile.sample = random.sample(samples, min(len(samples), 5))
        for profile_name in self.profiles:
            merged_profile.profiles[profile_name] = (
                self.profiles[profile_name] + other.profiles[profile_name]
            )
        return merged_profile

    @property
    def profile(self):
        unordered_profile = dict()
        for profile in self.profiles.values():
            utils.dict_merge(unordered_profile, profile.profile)

        name = self.name
        if isinstance(self.name, np.integer):
            name = int(name)
            
        unordered_profile.update({
            "column_name": name,
            "samples": self.sample,
        })

        unordered_profile["statistics"].update({
            "sample_size": self.sample_size,
            "null_count": self.null_count,
            "null_types": self.null_types,
            "null_types_index": self.null_types_index
        })
        
        if unordered_profile.get("data_type", None) is not None:
            unordered_profile["statistics"].update({
                "data_type_representation":
                unordered_profile["data_type_representation"]
            })

        dict_order = [
            "column_name",
            "data_type",
            "data_label",
            "categorical",
            "order",
            "samples",
            "statistics",
        ]
        profile = OrderedDict()
        if 'data_label_profile' not in self.profiles:
            dict_order.remove("data_label")
        for key in dict_order:
            try:
                profile[key] = unordered_profile[key]
            except KeyError as e:
                profile[key] = None

        return profile
    
    def _update_base_stats(self, base_stats):
        self.sample_size += base_stats["sample_size"]
        self._last_batch_size = base_stats["sample_size"]
        self.sample = base_stats["sample"]
        self.null_count += base_stats["null_count"]
        self.null_types = utils._combine_unique_sets(
            self.null_types, list(base_stats["null_types"].keys())
        )

        base_min = base_stats["min_id"]
        base_max = base_stats["max_id"]
        base_nti = base_stats["null_types"]

        # Check if indices overlap, if they do, adjust attributes accordingly
        if utils.overlap(self._min_id, self._max_id, base_min, base_max):
            warnings.warn(f"Overlapping indices detected. To resolve, indices "
                          f"where null data present will be shifted forward "
                          f"when stored in profile: {self.name}")

            # Shift indices (min, max, and all indices in null types index
            self._index_shift = self._max_id + 1
            base_min = base_min + self._index_shift
            base_max = base_max + self._index_shift

            base_nti = {k: {x + self._index_shift for x in v} for k, v in
                        base_stats["null_types"].items()}

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

    def update_profile(self, df_series, sample_size=None,
                       min_true_samples=None, sample_ids=None,
                       pool=None):
        """
        Update the column profiler
        
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
            df_series=df_series, sample_size=sample_size,
            min_true_samples=min_true_samples, sample_ids=sample_ids)

        self._update_base_stats(base_stats)
        self.update_column_profilers(clean_sampled_df, pool)

    def _get_sample_size(self, df_series):
        """
        Determines the minimum sampling size for detecting column type.

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
    def clean_data_and_get_base_stats(df_series, sample_size,
                                      min_true_samples=None,
                                      sample_ids=None):
        """
        Identify null characters and return them in a dictionary as well as
        remove any nulls in column.
        
        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :param sample_size: Number of samples to use in generating the profile
        :type sample_size: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param sample_ids: Randomized list of sample indices
        :type sample_ids: list(list)
        :return: updated column with null removed and dictionary of null
            parameters
        :rtype: pd.Series, dict
        """
        NO_FLAG = 0
        null_values_and_flags = {
            "": NO_FLAG,
            "nan": re.IGNORECASE,
            "none": re.IGNORECASE,
            "null": re.IGNORECASE,
            "  *": NO_FLAG,
            "--*": NO_FLAG,
            "__*": NO_FLAG,
        }
        
        if min_true_samples is None:
            min_true_samples = 0
        
        len_df = len(df_series)
        if not len_df:
            return df_series, {
                "sample_size": 0, "null_count": 0,
                "null_types": dict(), "sample": [],
                "min_id": None, "max_id": None
            }

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
            warnings.warn("Unable to detect minimum and maximum index values "
                          "for overlap detection. Updating/merging profiles "
                          "may result in inaccurate null row index reporting "
                          "due to unhandled overlapping indices.")

        # Select generator depending if sample_ids availability
        if sample_ids is None:
            sample_ind_generator = utils.shuffle_in_chunks(
                len_df, chunk_size=sample_size)
        else:
            sample_ind_generator = utils.partition(
                sample_ids[0], chunk_size=sample_size)
            
        na_columns = dict()
        true_sample_list = set()
        total_sample_size = 0
        query = '|'.join(null_values_and_flags.keys())
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
            if len(true_sample_list) >= min_true_samples \
                    and total_sample_size >= sample_size:
                break
            
        # close the generator in case it is not exhausted.
        if sample_ids is None:
            sample_ind_generator.close()

        # If min_true_samples exists, sort
        if min_true_samples > 0 or sample_ids is None:
            true_sample_list = sorted(true_sample_list)

        # Split out true values for later utilization
        df_series = df_series.loc[true_sample_list]
        total_na = total_sample_size - len(true_sample_list)

        base_stats = {
            "sample_size": total_sample_size,
            "null_count": total_na,
            "null_types": na_columns,
            "sample": random.sample(list(df_series.values),
                                    min(len(df_series), 5)),
            "min_id": min_id,
            "max_id": max_id
        }

        return df_series, base_stats


class BaseProfiler(object):

    _default_labeler_type = None
    _option_class = None
    _allowed_external_data_types = None

    def __init__(self, data, samples_per_update=None, min_true_samples=0,
                 options=None):
        """
        Instantiate the BaseProfiler class

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
        if self._default_labeler_type is None:
            raise ValueError('`_default_labeler_type` must be set when '
                             'overriding `BaseProfiler`.')
        elif self._option_class is None:
            raise ValueError('`_option_class` must be set when overriding '
                             '`BaseProfiler`.')
        elif self._allowed_external_data_types is None:
            raise ValueError('`_allowed_external_data_types` must be set when '
                             'overriding `BaseProfiler`.')

        options.validate()

        self._profile = None
        self.options = options
        self.encoding = None
        self.file_type = None
        self._samples_per_update = samples_per_update
        self._min_true_samples = min_true_samples
        self.total_samples = 0

        # TODO: allow set via options
        self._sampling_ratio = 0.2
        self._min_sample_size = 5000

        # assign data labeler
        data_labeler_options = self.options.data_labeler
        if data_labeler_options.is_enabled \
                and data_labeler_options.data_labeler_object is None:

            try:

                data_labeler = DataLabeler(
                    labeler_type=self._default_labeler_type,
                    dirpath=data_labeler_options.data_labeler_dirpath,
                    load_options=None)
                self.options.set(
                    {'data_labeler.data_labeler_object': data_labeler})

            except Exception as e:
                utils.warn_on_profile('data_labeler', e)
                self.options.set({'data_labeler.is_enabled': False})

    def _add_error_checks(self, other):
        """
        Profiler type specific checks to ensure two profiles can be added
        together.
        """
        raise NotImplementedError()

    def __add__(self, other):
        """
        Merges two profiles together overriding the `+` operator.

        :param other: profile being added to this one.
        :type other: BaseProfiler
        :return: merger of the two profiles
        :rtype: BaseProfiler
        """
        if type(other) is not type(self):
            raise TypeError('`{}` and `{}` are not of the same profiler type.'.
                            format(type(self).__name__, type(other).__name__))

        # error checks specific to its profiler
        self._add_error_checks(other)

        merged_profile = self.__class__(
            data=None, samples_per_update=self._samples_per_update,
            min_true_samples=self._min_true_samples, options=self.options
        )
        merged_profile.encoding = self.encoding
        if self.encoding != other.encoding:
            merged_profile.encoding = 'multiple files'

        merged_profile.file_type = self.file_type
        if self.file_type != other.file_type:
            merged_profile.file_type = 'multiple files'

        merged_profile.total_samples = self.total_samples + other.total_samples

        return merged_profile

    def _get_sample_size(self, data):
        """
        Determines the minimum sampling size for profiling the dataset.

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
    def profile(self):
        """
        Returns the stored profiles for the given profiler.

        :return: None
        """
        return self._profile

    def report(self, report_options=None):
        """
        Returns the profile report based on all profiled data fed into the
        profiler. User can specify the output_formats: (pretty, compact,
        serializable, flat).
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

    def _update_profile_from_chunk(self, data, sample_size,
                                   min_true_samples=None):
        """
        Iterate over the dataset and identify its parameters via profiles.

        :param data: dataset to be profiled
        :type data: Union[pd.Series, pd.DataFrame, list]
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :return: list of column profile base subclasses
        :rtype: list(BaseColumnProfiler)
        """
        raise NotImplementedError()

    def update_profile(self, data, sample_size=None, min_true_samples=None):
        """
        Update the profile for data provided. User can specify the sample
        size to profile the data with. Additionally, the user can specify the
        minimum number of non-null samples to profile.

        :param data: data to be profiled
        :type data: Union[data_readers.base_data.BaseData, pandas.DataFrame,
            pandas.Series]
        :param sample_size: number of samples to profile from the data
        :type sample_size: int
        :param min_true_samples: minimum number of non-null samples to profile
        :type min_true_samples
        :return: None
        """
        encoding = None
        file_type = None

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
            warnings.warn("The passed dataset was empty, hence no data was "
                          "profiled.")
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

    def _remove_data_labelers(self):
        """
        Helper method for removing all data labelers before saving to disk.

        :return: data_labeler used for unstructured labelling
        :rtype: DataLabeler
        """
        data_labeler = None
        data_labeler_options = None

        # determine if the data labeler is enabled
        use_data_labeler = True
        if self.options and isinstance(self.options, (StructuredOptions,
                                                      UnstructuredOptions)):
            data_labeler_options = self.options.data_labeler
            use_data_labeler = data_labeler_options.is_enabled

        # remove the data labeler from options
        if use_data_labeler and data_labeler_options is not None \
                and data_labeler_options.data_labeler_object is not None:
            data_labeler = data_labeler_options.data_labeler_object
            data_labeler_options.data_labeler_object = None

        # get all profiles, unstructured is a single profile and hence needs to
        # be in a list, whereas structured is a dict and needs to be a list
        profilers = [self._profile]
        if isinstance(self, StructuredProfiler):
            profilers = self._profile.values()

        # Remove data labelers for all columns
        for profiler in profilers:

            # profiles stored differently in Struct/Unstruct, this unifies
            # labeler extraction
            # unstructured: _profile is a compiler
            # structured: StructuredColProfiler.profiles['data_label_profile']
            if isinstance(self, StructuredProfiler):
                profiler = profiler.profiles.get('data_label_profile', None)

            if profiler and use_data_labeler and data_labeler is None:
                data_labeler = profiler._profiles['data_labeler'].data_labeler

            if profiler and 'data_labeler' in profiler._profiles:
                profiler._profiles['data_labeler'].data_labeler = None

        return data_labeler

    def _restore_data_labelers(self, data_labeler=None):
        """
        Helper method for restoring all data labelers after saving to or
        loading from disk.

        :param data_labeler: unstructured data_labeler
        :type data_labeler: DataLabeler
        """
        # Restore data labeler for options
        use_data_labeler = True
        data_labeler_dirpath = None
        if self.options and isinstance(self.options, (StructuredOptions,
                                                      UnstructuredOptions)):
            data_labeler_options = self.options.data_labeler
            use_data_labeler = data_labeler_options.is_enabled
            data_labeler_dirpath = data_labeler_options.data_labeler_dirpath

        if use_data_labeler:
            try:
                if data_labeler is None:
                    data_labeler = DataLabeler(
                        labeler_type=self._default_labeler_type,
                        dirpath=data_labeler_dirpath,
                        load_options=None)
                self.options.set(
                    {'data_labeler.data_labeler_object': data_labeler})

            except Exception as e:
                utils.warn_on_profile('data_labeler', e)
                self.options.set({'data_labeler.is_enabled': False})
                self.options.set(
                    {'data_labeler.data_labeler_object': data_labeler})

            except Exception as e:
                utils.warn_on_profile('data_labeler', e)
                self.options.set({'data_labeler.is_enabled': False})

        # get all profiles, unstructured is a single profile and hence needs to
        # be in a list, whereas structured is a dict and needs to be a list
        profilers = [self._profile]
        if isinstance(self, StructuredProfiler):
            profilers = self._profile.values()

        # Restore data labelers for all columns
        for profiler in profilers:

            if use_data_labeler:

                # profiles stored differently in Struct/Unstruct, this unifies
                # label replacement
                # unstructured: _profile is a compiler
                # structured: StructuredColProfiler.profiles['data_label_profile']
                if isinstance(self, StructuredProfiler):
                    profiler = profiler.profiles['data_label_profile']

                data_labeler_profile = profiler._profiles['data_labeler']
                data_labeler_profile.data_labeler = data_labeler

    def _save_helper(self, filepath, data_dict):
        """
        Save profiler to disk

        :param filepath: Path of file to save to
        :type filepath: String
        :param data_dict: profile data to be saved
        :type data_dict: dict
        :return: None
        """
        # Set Default filepath
        if filepath is None:
            filepath = "profile-{}.pkl".format(
                datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f"))

        # Remove data labelers as they can't be pickled
        data_labelers = self._remove_data_labelers()

        # add profiler class to data_dict
        data_dict['profiler_class'] = self.__class__.__name__

        # Pickle and save profile to disk
        with open(filepath, "wb") as outfile:
            pickle.dump(data_dict, outfile)

        # Restore all data labelers
        self._restore_data_labelers(data_labelers)

    def save(self, filepath=None):
        """
        Save profiler to disk

        :param filepath: Path of file to save to
        :type filepath: String
        :return: None
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, filepath):
        """
        Load profiler from disk

        :param filepath: Path of file to load from
        :type filepath: String
        :return: Profiler being loaded, StructuredProfiler or
            UnstructuredProfiler
        :rtype: BaseProfiler
        """
        # Load profile from disk
        with open(filepath, "rb") as infile:
            data = pickle.load(infile)

        # remove profiler class if it exists
        profiler_class = data.pop('profiler_class', None)

        # if the user didn't load from the a given profiler class, we need
        # to determine which profiler is being loaded.
        profiler_cls = cls
        if cls is BaseProfiler:
            if profiler_class == 'StructuredProfiler':
                profiler_cls = StructuredProfiler
            elif profiler_class == 'UnstructuredProfiler':
                profiler_cls = UnstructuredProfiler
            elif profiler_class is None:  # deprecated case
                profiler_cls = StructuredProfiler
                if '_empty_line_count' in data:
                    profiler_cls = UnstructuredProfiler
            else:
                raise ValueError(f'Invalid profiler class {profiler_class} '
                                 f'failed to load.')

        profile_options = profiler_cls._option_class()
        profile_options.data_labeler.is_enabled = False
        profiler = profiler_cls(None, options=profile_options)

        for key in data:
            setattr(profiler, key, data[key])

        # Restore all data labelers
        profiler._restore_data_labelers()
        return profiler


class UnstructuredProfiler(BaseProfiler):

    _default_labeler_type = 'unstructured'
    _option_class = UnstructuredOptions
    _allowed_external_data_types = (str, list, pd.Series, pd.DataFrame)

    def __init__(self, data, samples_per_update=None, min_true_samples=0,
                 options=None):
        """
        Instantiate the UnstructuredProfiler class

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
            raise ValueError("The profile options must be passed as a "
                             "ProfileOptions object.")

        super().__init__(data, samples_per_update, min_true_samples, options)

        # Unstructured specific properties
        self._empty_line_count = 0

        if data is not None:
            self.update_profile(data)

    def _add_error_checks(self, other):
        """
        UnstructuredProfiler specific checks to ensure two profiles can be added
        together.
        """
        pass

    def __add__(self, other):
        """
        Merges two Unstructured profiles together overriding the `+` operator.

        :param other: unstructured profile being added to this one.
        :type other: UnstructuredProfiler
        :return: merger of the two profiles
        :rtype: UnstructuredProfiler
        """
        merged_profile = super().__add__(other)

        # unstruct specific property merging
        merged_profile._empty_line_count = (
                self._empty_line_count + other._empty_line_count)
        samples = list(dict.fromkeys(self.sample + other.sample))
        merged_profile.sample = random.sample(list(samples),
                                              min(len(samples), 5))

        # merge profiles
        merged_profile._profile = self._profile + other._profile

        return merged_profile

    def _update_base_stats(self, base_stats):
        """
        Updates the samples and line count of the class for the given dataset
        batch.

        :param base_stats: dictionary of basic sampling / data stats
        :type base_stats: dict
        :return: None
        """
        self.total_samples += base_stats["sample_size"]
        self.sample = base_stats["sample"]
        self._empty_line_count += base_stats["empty_line_count"]

    def report(self, report_options=None):
        """
        Returns the unstructured report based on all profiled data fed into the
        profiler. User can specify the output_formats: (pretty, compact,
        serializable, flat).
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
            }

        output_format = report_options.get("output_format", None)
        omit_keys = report_options.get("omit_keys", None)

        report = OrderedDict([
            ("global_stats", {
                "samples_used": self.total_samples,
                "empty_line_count": self._empty_line_count,
                "file_type": self.file_type,
                "encoding": self.encoding
            }),
            ("data_stats", OrderedDict()),
        ])

        report["data_stats"] = self._profile.profile
        return _prepare_report(report, output_format, omit_keys)

    @staticmethod
    def _clean_data_and_get_base_stats(data, sample_size,
                                       min_true_samples=None):
        """
        Identify empty rows and return a cleaned version of text data without
        empty rows.

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
            return data, {
                "sample_size": 0, "empty_line_count": dict(), "sample": [],
            }

        # ensure all data are of type str
        data = data.apply(str)

        # Setup sample generator
        sample_ind_generator = utils.shuffle_in_chunks(
            len_data, chunk_size=sample_size)

        true_sample_list = set()
        total_sample_size = 0

        regex = f"^\s*$"
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
            if len(true_sample_list) >= min_true_samples \
                    and total_sample_size >= sample_size:
                break

        # close the generator in case it is not exhausted.
        sample_ind_generator.close()

        true_sample_list = sorted(true_sample_list)

        # Split out true values for later utilization
        data = data.loc[true_sample_list]
        total_empty = total_sample_size - len(true_sample_list)

        base_stats = {
            "sample_size": total_sample_size,
            "empty_line_count": total_empty,
            "sample": random.sample(list(data.values),
                                    min(len(data), 5)),
        }

        return data, base_stats

    def _update_profile_from_chunk(self, data, sample_size,
                                   min_true_samples=None):
        """
        Iterate over the dataset and identify its parameters via profiles.

        :param data: a text dataset
        :type data: Union[pd.Series, pd.DataFrame, list]
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :return: list of column profile base subclasses
        :rtype: list(BaseColumnProfiler)
        """

        if isinstance(data, pd.DataFrame):
            if len(data.columns) > 1:
                raise ValueError("The unstructured cannot handle a dataset "
                                 "with more than 1 column. Please make sure "
                                 "the data format of the dataset is "
                                 "appropriate.")
            data = data[data.columns[0]]
        elif isinstance(data, (str, list)):
            # we know that if it comes in as a list, it is a 1-d list based
            # bc of our data readers
            # for strings, we just need to put it inside a series for compute.
            data = pd.Series(data)

        # Format the data
        notification_str = "Finding the empty lines in the data..."
        print(notification_str)
        data, base_stats = self._clean_data_and_get_base_stats(
            data, sample_size, min_true_samples)
        self._update_base_stats(base_stats)

        if sample_size < len(data):
            warnings.warn("The data will be profiled with a sample size of {}. "
                          "All statistics will be based on this subsample and "
                          "not the whole dataset.".format(sample_size))

        # process the text data
        notification_str = "Calculating the statistics... "
        print(notification_str)
        pool = None
        if self._profile is None:
            self._profile = UnstructuredCompiler(data, options=self.options,
                                                 pool=pool)
        else:
            self._profile.update_profile(data, pool=pool)

    def save(self, filepath=None):
        """
        Save profiler to disk

        :param filepath: Path of file to save to
        :type filepath: String
        :return: None
        """
        # Create dictionary for all metadata, options, and profile
        data_dict = {
            "total_samples": self.total_samples,
            "encoding": self.encoding,
            "file_type": self.file_type,
            "_samples_per_update": self._samples_per_update,
            "_min_true_samples": self._min_true_samples,
            "_empty_line_count": self._empty_line_count,
            "options": self.options,
            "_profile": self.profile
        }
        self._save_helper(filepath, data_dict)


class StructuredProfiler(BaseProfiler):

    _default_labeler_type = 'structured'
    _option_class = StructuredOptions
    _allowed_external_data_types = (list, pd.Series, pd.DataFrame)

    def __init__(self, data, samples_per_update=None, min_true_samples=0, 
                 options=None):
        """
        Instantiate the StructuredProfiler class
        
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
            raise ValueError("The profile options must be passed as a "
                             "ProfileOptions object.")

        if isinstance(data, data_readers.text_data.TextData):
            raise TypeError("Cannot provide TextData object to "
                            "StructuredProfiler")

        super().__init__(data, samples_per_update, min_true_samples, options)

        # Structured specific properties
        self.row_has_null_count = 0
        self.row_is_null_count = 0
        self.hashed_row_dict = dict()
        self._profile = dict()

        if data is not None:
            self.update_profile(data)

    def _add_error_checks(self, other):
        """
        StructuredProfiler specific checks to ensure two profiles can be added
        together.
        """
        if set(self._profile) != set(other._profile):
            raise ValueError('Profiles do not have the same schema.')
        elif not all([isinstance(other._profile[p_name],
                                 type(self._profile[p_name]))
                      for p_name in self._profile]):  # options check
            raise ValueError('The two profilers were not setup with the same '
                             'options, hence they do not calculate the same '
                             'profiles and cannot be added together.')

    def __add__(self, other):
        """
        Merges two Structured profiles together overriding the `+` operator.

        :param other: profile being added to this one.
        :type other: StructuredProfiler
        :return: merger of the two profiles
        :rtype: StructuredProfiler
        """
        merged_profile = super().__add__(other)

        # struct specific property merging
        merged_profile.row_has_null_count = \
            self.row_has_null_count + other.row_has_null_count
        merged_profile.row_is_null_count = \
            self.row_is_null_count + other.row_is_null_count
        merged_profile.hashed_row_dict.update(self.hashed_row_dict)
        merged_profile.hashed_row_dict.update(other.hashed_row_dict)

        # merge profiles
        for profile_name in self._profile:
            merged_profile._profile[profile_name] = (
                self._profile[profile_name] + other._profile[profile_name]
            )
        return merged_profile

    @property
    def _max_col_samples_used(self):
        """
        Calculates and returns the maximum samples used in any of the columns.
        """
        samples_used = 0
        columns = list(self._profile.values())
        for col in columns:
            samples_used = max(samples_used, col.sample_size)
        return samples_used

    @property
    def _min_col_samples_used(self):
        """
        Calculates and returns the number of rows that were completely sampled
        i.e. every column in the Profile was read up to this row (possibly
        further in some cols)
        """
        return min([self._profile[col].sample_size
                    for col in self._profile], default=0)

    @property
    def _min_sampled_from_batch(self):
        """
        Calculates and returns the number of rows that were completely sampled
        in the most previous batch
        """
        return min([self._profile[col]._last_batch_size
                    for col in self._profile], default=0)

    def report(self, report_options=None):
        if not report_options:
            report_options = {
                "output_format": None,
                "num_quantile_groups": 4,
            }
            
        output_format = report_options.get("output_format", None)        
        omit_keys = report_options.get("omit_keys", [])
        num_quantile_groups = report_options.get("num_quantile_groups", 4)

        columns = list(self._profile.values())
        report = OrderedDict([
            ("global_stats", {
                "samples_used": self._max_col_samples_used,
                "column_count": len(columns),
                "row_count": self.total_samples,
                "row_has_null_ratio": self._get_row_has_null_ratio(),
                "row_is_null_ratio": self._get_row_is_null_ratio(),
                "unique_row_ratio": self._get_unique_row_ratio(),
                "duplicate_row_count": self._get_duplicate_row_count(),
                "file_type": self.file_type,
                "encoding": self.encoding
            }),
            ("data_stats", OrderedDict()),
        ])
        for key in self._profile.keys():
            report["data_stats"][key] = self._profile[key].profile
            quantiles = report["data_stats"][key]["statistics"].get(
                'quantiles')
            if quantiles:
                quantiles = calculate_quantiles(num_quantile_groups, quantiles)
                report["data_stats"][key]["statistics"]["quantiles"] = quantiles

        return _prepare_report(report, output_format, omit_keys)

    def _get_unique_row_ratio(self):
        return len(self.hashed_row_dict) / self.total_samples

    def _get_row_is_null_ratio(self):
        return 0 if self._min_col_samples_used in {0, None} \
            else self.row_is_null_count / self._min_col_samples_used

    def _get_row_has_null_ratio(self):
        return 0 if self._min_col_samples_used in {0, None} \
            else self.row_has_null_count / self._min_col_samples_used

    def _get_duplicate_row_count(self):
        return self.total_samples - len(self.hashed_row_dict)

    def _update_row_statistics(self, data, sample_ids=None):
        """
        Iterate over the provided dataset row by row and calculate
        the row statistics. Specifically, number of unique rows,
        rows containing null values, and total rows reviewed. This
        function is safe to use in batches.

        :param data: a dataset
        :type data: pandas.DataFrame
        :param sample_ids: list of indices in order they were sampled in data
        :type sample_ids: list(int)
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Cannot calculate row statistics on data that is"
                             "not a DataFrame")
        
        self.total_samples += len(data)
        try:
            self.hashed_row_dict.update(dict.fromkeys(
                pd.util.hash_pandas_object(data, index=False), True
            ))
        except TypeError:
            self.hashed_row_dict.update(dict.fromkeys(
                pd.util.hash_pandas_object(data.astype(str), index=False), True
            ))

        # Calculate Null Column Count
        null_rows = set()
        null_in_row_count = set()
        first_col_flag = True
        for column in self._profile:
            null_type_dict = self._profile[column].null_types_index
            null_row_indices = set()
            if null_type_dict:
                null_row_indices = set.union(*null_type_dict.values())

            # If sample ids provided, only consider nulls in rows that
            # were fully sampled
            if sample_ids is not None:
                # This is the amount (integer) indices were shifted by in the
                # event of overlap
                shift = self._profile[column]._index_shift
                if shift is None:
                    # Shift is None if index is str or if no overlap detected
                    null_row_indices = null_row_indices.intersection(
                        data.index[sample_ids[:self._min_sampled_from_batch]])
                else:
                    # Only shift if index shift detected (must be ints)
                    null_row_indices = null_row_indices.intersection(
                        data.index[sample_ids[:self._min_sampled_from_batch]] +
                        shift)

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

    def _update_profile_from_chunk(self, data, sample_size,
                                   min_true_samples=None):
        """
        Iterate over the columns of a dataset and identify its parameters.
        
        :param data: a dataset
        :type data: pandas.DataFrame
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :return: list of column profile base subclasses
        :rtype: list(BaseColumnProfiler)
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        if len(data.columns) != len(data.columns.unique()):
            raise ValueError('`StructuredProfiler` does not currently support '
                             'data which contains columns with duplicate '
                             'names.')

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(l):
                for i, e in enumerate(l):
                    print("Processing Column {}/{}".format(i+1, len(l)))
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
        sample_ids = np.array(sample_ids)

        # Create structured profile objects
        new_cols = set()
        for col in data.columns:
            if col not in self._profile:
                self._profile[col] = StructuredColProfiler(
                    sample_size=sample_size,
                    min_true_samples=min_true_samples,
                    sample_ids=sample_ids,
                    options=self.options
                )
                new_cols.add(col)
                
        # Generate pool and estimate datasize
        pool = None
        if self.options.multiprocess.is_enabled:
            est_data_size = data[:50000].memory_usage(index=False, deep=True).sum()
            est_data_size = (est_data_size / min(50000, len(data))) * len(data)
            pool, pool_size = utils.generate_pool(
                max_pool_size=None, data_size=est_data_size,
                cols=len(data.columns))

        # Format the data
        notification_str = "Finding the Null values in the columns..."        
        if pool and len(new_cols) > 0:
            notification_str += " (with " + str(pool_size) + " processes)"
        
        clean_sampled_dict = {}
        multi_process_dict = {}
        single_process_list = set()
        
        if sample_size < len(data):
            warnings.warn("The data will be profiled with a sample size of {}. "
                          "All statistics will be based on this subsample and "
                          "not the whole dataset.".format(sample_size))

        if pool is not None:
            # Create a bunch of simultaneous column conversions
            for col in data.columns:
                if min_true_samples is None:
                    min_true_samples = self._profile[col]._min_true_samples
                try:
                    multi_process_dict[col] = pool.apply_async(
                        self._profile[col].clean_data_and_get_base_stats,
                        (data[col], sample_size, min_true_samples, sample_ids))
                except Exception as e:
                    print(e)
                    single_process_list.add(col)
                
            # Iterate through multiprocessed columns collecting results
            print(notification_str)
            for col in tqdm(multi_process_dict.keys()):
                try:
                    clean_sampled_dict[col], base_stats = \
                        multi_process_dict[col].get()
                    self._profile[col]._update_base_stats(base_stats)
                except Exception as e:
                    print(e)
                    single_process_list.add(col)

            # Clean up any columns which errored
            if len(single_process_list) > 0:
                print("Errors in multiprocessing occured:",
                      len(single_process_list), "errors, reprocessing...")
                for col in tqdm(single_process_list):
                    if min_true_samples is None:
                        min_true_samples = self._profile[col]._min_true_samples
                    clean_sampled_dict[col], base_stats = \
                        self._profile[col].clean_data_and_get_base_stats(
                            data[col], sample_size, min_true_samples, sample_ids)
                    self._profile[col]._update_base_stats(base_stats)
            
            pool.close()  # Close pool for new tasks
            pool.join()  # Wait for all workers to complete

        else:  # No pool
            print(notification_str)
            for col in tqdm(data.columns):
                if min_true_samples is None:
                    min_true_samples = self._profile[col]._min_true_samples
                clean_sampled_dict[col], base_stats = \
                    self._profile[col].clean_data_and_get_base_stats(
                        df_series=data[col], sample_size=sample_size,
                        min_true_samples=min_true_samples, sample_ids=sample_ids
                    )
                self._profile[col]._update_base_stats(base_stats)
            
        # Process and label the data
        notification_str = "Calculating the statistics... "
        pool = None
        if self.options.multiprocess.is_enabled:
            pool, pool_size = utils.generate_pool(4, est_data_size)
            if pool:
                notification_str += " (with " + str(pool_size) + " processes)"
        print(notification_str)
        
        for col in tqdm(data.columns):
            self._profile[col].update_column_profilers(
                clean_sampled_dict[col], pool)

        if pool is not None:
            pool.close()  # Close pool for new tasks
            pool.join()  # Wait for all workers to complete

        # Only pass along sample ids if necessary
        samples_for_row_stats = None
        if min_true_samples not in [None, 0]:
            samples_for_row_stats = np.concatenate(sample_ids)

        self._update_row_statistics(data, samples_for_row_stats)

    def save(self, filepath=None):
        """
        Save profiler to disk
        
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
                "_profile": self.profile
               } 

        self._save_helper(filepath, data_dict)


class Profiler(object):

    def __new__(cls, data, samples_per_update=None, min_true_samples=0,
                options=None, profiler_type=None):
        """
        Factory class for instantiating Structured and Unstructured Profilers

        :param data: Data to be profiled, type allowed depends on the
            profiler_type
        :type data: Data class object
        :param samples_per_update: Number of samples to use to generate profile
        :type samples_per_update: int
        :param min_true_samples: Min number of samples required for the profiler
        :type min_true_samples: int
        :param options: Options for the profiler.
        :type options: ProfilerOptions Object
        :param profiler_type: Type of Profiler ("structured"/"unstructured")
        :type profiler_type: str
        :return: BaseProfiler
        """
        if profiler_type is None:
            # defaults as structured
            profiler_type = "structured"
            # Unstructured if data is Data object and is_structured is False
            if isinstance(data, data_readers.base_data.BaseData):
                if not data.is_structured:
                    profiler_type = "unstructured"
            elif isinstance(data, str):
                profiler_type = "unstructured"
            # the below checks the viable structured formats, on failure raises
            elif not isinstance(data, (list, pd.DataFrame, pd.Series)):
                raise ValueError("Data must either be imported using the "
                                 "data_readers, pd.Series, or pd.DataFrame.")

        # Construct based off of initial kwarg input or inference
        if profiler_type == "structured":
            return StructuredProfiler(data, samples_per_update,
                                      min_true_samples, options)
        elif profiler_type == "unstructured":
            return UnstructuredProfiler(data, samples_per_update,
                                        min_true_samples, options)
        else:
            raise ValueError("Must specify 'profiler_type' to be 'structured' "
                             "or 'unstructured'.")

    @classmethod
    def load(cls, filepath):
        """
        Load profiler from disk

        :param filepath: Path of file to load from
        :type filepath: String
        :return: Profiler being loaded, StructuredProfiler or
            UnstructuredProfiler
        :rtype: BaseProfiler
        """
        return BaseProfiler.load(filepath)
