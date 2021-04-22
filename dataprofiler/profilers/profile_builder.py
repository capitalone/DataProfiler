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

import pandas as pd
import numpy as np

from . import utils
from .. import data_readers
from .column_profile_compilers import ColumnPrimitiveTypeProfileCompiler, \
    ColumnStatsProfileCompiler, ColumnDataLabelerCompiler
from ..labelers.data_labelers import DataLabeler
from .helpers.report_helpers import calculate_quantiles, _prepare_report
from .profiler_options import ProfilerOptions, StructuredOptions


class StructuredDataProfile(object):

    def __init__(self, df_series=None, sample_size=None, min_sample_size=5000,
                 sampling_ratio=0.2, min_true_samples=None,
                 sample_ids=None, pool=None, options=None):
        """
        Instantiate the Structured Profiler class for a given column.
        
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
        :type other: StructuredDataProfile
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
        merged_profile = StructuredDataProfile(
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
             "null_types": copy.deepcopy(self.null_types_index)}
        )
        merged_profile._update_base_stats(
            {"sample": other.sample,
             "sample_size": other.sample_size,
             "null_count": other.null_count,
             "null_types": copy.deepcopy(other.null_types_index)}
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
                
        if unordered_profile.get("data_type", None) is not None:
            unordered_profile["statistics"].update({
                "sample_size": self.sample_size,
                "null_count": self.null_count,
                "null_types": self.null_types,
                "null_types_index": self.null_types_index,
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
        self.sample = base_stats["sample"]
        self.null_count += base_stats["null_count"]
        self.null_types = utils._combine_unique_sets(
            self.null_types, list(base_stats["null_types"].keys())
        )

        for null_type, null_rows in base_stats["null_types"].items():
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
                "null_types": dict(), "sample": []
            }

        # Pandas reads empty values in the csv files as nan
        df_series = df_series.apply(str)

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
        if min_true_samples > 0:
            true_sample_list = sorted(true_sample_list)

        # Split out true values for later utilization
        df_series = df_series.loc[true_sample_list]
        total_na = total_sample_size - len(true_sample_list)

        base_stats = {
            "sample_size": total_sample_size,
            "null_count": total_na,
            "null_types": na_columns,
            "sample": random.sample(list(df_series.values),
                                    min(len(df_series), 5))
        }

        return df_series, base_stats


class Profiler(object):

    def __init__(self, data, samples_per_update=None, min_true_samples=0, 
                 profiler_options=None):
        """
        Instantiate the Profiler class
        
        :param data: Data to be profiled
        :type data: Data class object
        :param samples_per_update: Number of samples to use in generating
            profile
        :type samples_per_update: int
        :param min_true_samples: Minimum number of samples required for the
            profiler
        :type min_true_samples: int
        :param profiler_options: Options for the profiler.
        :type profiler_options: ProfilerOptions Object
        :return: Profiler
        """

        if not profiler_options:
            profiler_options = ProfilerOptions()
        elif not isinstance(profiler_options, ProfilerOptions):
            raise ValueError("The profile options must be passed as a "
                             "ProfileOptions object.")
        
        profiler_options.validate()
        self.options = profiler_options
        self.total_samples = 0
        self.encoding = None
        self.file_type = None
        self.row_has_null_count = 0
        self.row_is_null_count = 0
        self.hashed_row_dict = dict()
        self._samples_per_update = samples_per_update
        self._min_true_samples = min_true_samples
        self._profile = dict()

        # matches structured data profile
        # TODO: allow set via options
        self._sampling_ratio = 0.2
        self._min_sample_size = 5000

        if isinstance(data, data_readers.text_data.TextData):
            raise TypeError("Cannot provide TextData object to Profiler")

        # assign data labeler
        data_labeler_options = self.options.structured_options.data_labeler
        if data_labeler_options.is_enabled \
                and data_labeler_options.data_labeler_object is None:

            try:

                data_labeler = DataLabeler(
                    labeler_type='structured',
                    dirpath=data_labeler_options.data_labeler_dirpath,
                    load_options=None)
                self.options.set(
                    {'data_labeler.data_labeler_object': data_labeler})

            except Exception as e:
                utils.warn_on_profile('data_labeler', e)
                self.options.set({'data_labeler.is_enabled': False})

        if len(data):
            self.update_profile(data)

    def __add__(self, other):
        """
        Merges two profiles together overriding the `+` operator.

        :param other: profile being add to this one.
        :type other: Profiler
        :return: merger of the two profiles
        """
        if type(other) is not type(self):
            raise TypeError('`{}` and `{}` are not of the same profiler type.'.
                            format(type(self).__name__, type(other).__name__))
        elif set(self._profile) != set(other._profile):
            raise ValueError('Profiles do not have the same schema.')
        elif not all([isinstance(other._profile[p_name],
                                 type(self._profile[p_name]))
                      for p_name in self._profile]):  # options check
            raise ValueError('The two profilers were not setup with the same '
                             'options, hence they do not calculate the same '
                             'profiles and cannot be added together.')
        merged_profile = Profiler(
            data=pd.DataFrame([]), samples_per_update=self._samples_per_update,
            min_true_samples=self._min_true_samples,
            profiler_options=self.options
        )
        merged_profile.encoding = self.encoding \
            if self.encoding == other.encoding else 'multiple files'
        merged_profile.file_type = self.file_type \
            if self.file_type == other.file_type else 'multiple files'
        merged_profile.row_has_null_count = \
            self.row_has_null_count + other.row_has_null_count
        merged_profile.row_is_null_count = \
            self.row_is_null_count + other.row_is_null_count
        merged_profile.total_samples = self.total_samples + other.total_samples
        merged_profile.hashed_row_dict.update(self.hashed_row_dict)
        merged_profile.hashed_row_dict.update(other.hashed_row_dict)

        for profile_name in self._profile:
            merged_profile._profile[profile_name] = (
                self._profile[profile_name] + other._profile[profile_name]
            )
        return merged_profile

    @property
    def profile(self):
        return self._profile

    def _get_sample_size(self, data):
        """
        Determines the minimum sampling size for detecting column type.

        :param data: data to be profiled
        :type data: Union[data_readers.base_data.BaseData, pandas.DataFrame]
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
        self.hashed_row_dict = dict.fromkeys(
            pd.util.hash_pandas_object(data, index=False), True
        )

        # Calculate Null Column Count
        null_rows = set()
        null_in_row_count = set()
        first_col_flag = True
        for column in self._profile:
            null_type_dict = self._profile[column].null_types_index
            null_row_indices = set()
            if null_type_dict:
                null_row_indices = set.union(*null_type_dict.values())

            if sample_ids is not None:
                # If sample ids provided, only consider nulls in rows that
                # were fully sampled
                null_row_indices = null_row_indices.intersection(
                    data.index[sample_ids[:self._min_col_samples_used]])

            # Find the common null indices between the columns
            if first_col_flag:
                null_rows = null_row_indices
                null_in_row_count = null_row_indices
                first_col_flag = False
            else:
                null_rows = null_rows.intersection(null_row_indices)
                null_in_row_count = null_in_row_count.union(null_row_indices)

        self.row_has_null_count += len(null_in_row_count)
        self.row_is_null_count += len(null_rows)

    def update_profile(self, data, sample_size=None, min_true_samples=None):
        """
        Update the profile for data provided. User can specify the sample
        size to profile the data with. Additionally, the user can specify the
        minimum number of non-null samples to profile.

        :param data: data to be profiled
        :type data: Union[data_readers.base_data.BaseData, pandas.DataFrame]
        :param sample_size: number of samples to profile from the data
        :type sample_size: int
        :param min_true_samples: minimum number of non-null samples to profile
        :type min_true_samples
        :return: None
        """
        if isinstance(data, data_readers.base_data.BaseData):
            self.encoding = data.file_encoding
            self.file_type = data.data_type
            data = data.data
        elif isinstance(data, pd.DataFrame):
            self.file_type = str(data.__class__)
        else:
            raise ValueError(
                "Data must either be imported using the data_readers or "
                "pd.DataFrame."
            )

        if not min_true_samples:
            min_true_samples = self._min_true_samples
        if not sample_size:
            sample_size = self._get_sample_size(data)

        self._update_profile_from_chunk(
            data, sample_size, min_true_samples, self.options)

    def _update_profile_from_chunk(self, df, sample_size=None,
                                   min_true_samples=None, options=None):
        """
        Iterate over the columns of a dataset and identify its parameters.
        
        :param df: a dataset
        :type df: pandas.DataFrame
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :param options: Options for the profiler
        :type options: ProfilerOptions
        :return: list of column profile base subclasses
        :rtype: list(BaseColumnProfiler)
        """

        if len(df.columns) != len(df.columns.unique()):
            raise ValueError('`Profiler` does not currently support data which '
                             'contains columns with duplicate names.')

        try:
            from tqdm import tqdm
        except:
            def tqdm(l):
                for i, e in enumerate(l):
                    print("Processing Column {}/{}".format(i+1, len(l)))
                    yield e

        # Shuffle indices once and share with columns
        sample_ids = [*utils.shuffle_in_chunks(len(df), len(df))]
        
        # If there are no minimum true samples, you can sort to save time
        if min_true_samples in [None, 0]:
            # If there's a sample size, truncate
            if sample_size is not None:
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
        for col in df.columns:
            if col not in self._profile:
                structured_options = None
                if options and options.structured_options:
                    structured_options = options.structured_options
                self._profile[col] = StructuredDataProfile(
                    sample_size=sample_size,
                    min_true_samples=min_true_samples,
                    sample_ids=sample_ids,
                    options=structured_options
                )
                new_cols.add(col)
                
        # Generate pool and estimate datasize
        pool = None
        if options.structured_options.multiprocess.is_enabled:
            est_data_size = df[:50000].memory_usage(index=False, deep=True).sum()
            est_data_size = (est_data_size / min(50000, len(df))) * len(df)
            pool, pool_size = utils.generate_pool(
                max_pool_size=None, data_size=est_data_size, cols=len(df.columns))

        # Format the data
        notification_str = "Finding the Null values in the columns..."        
        if pool and len(new_cols) > 0:
            notification_str += " (with " + str(pool_size) + " processes)"
        
        clean_sampled_dict = {}
        multi_process_dict = {}
        single_process_list = set()
        if not sample_size: sample_size = len(df)
        if sample_size < len(df):
            warnings.warn("The data will be profiled with a sample size of {}. "
                          "All statistics will be based on this subsample and "
                          "not the whole dataset.".format(sample_size))

        if pool is not None:
            # Create a bunch of simultaneous column conversions
            for col in df.columns:
                if min_true_samples is None:
                    min_true_samples = self._profile[col]._min_true_samples
                try:
                    multi_process_dict[col] = pool.apply_async(
                        self._profile[col].clean_data_and_get_base_stats,
                        (df[col], sample_size, min_true_samples, sample_ids))
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
                            df[col], sample_size, min_true_samples, sample_ids)
                    self._profile[col]._update_base_stats(base_stats)
            
            pool.close()  # Close pool for new tasks
            pool.join()  # Wait for all workers to complete

        else:  # No pool
            print(notification_str)
            for col in tqdm(df.columns):
                if min_true_samples is None:
                    min_true_samples = self._profile[col]._min_true_samples
                clean_sampled_dict[col], base_stats = \
                    self._profile[col].clean_data_and_get_base_stats(
                        df_series=df[col], sample_size=sample_size,
                        min_true_samples=min_true_samples, sample_ids=sample_ids
                    )
                self._profile[col]._update_base_stats(base_stats)
            
        # Process and label the data
        notification_str = "Calculating the statistics... "
        pool = None
        if options.structured_options.multiprocess.is_enabled:
            pool, pool_size = utils.generate_pool(4, est_data_size)
            if pool:
                notification_str += " (with " + str(pool_size) + " processes)"
        print(notification_str)
        
        for col in tqdm(df.columns):
            self._profile[col].update_column_profilers(
                clean_sampled_dict[col], pool)

        if pool is not None:
            pool.close()  # Close pool for new tasks
            pool.join()  # Wait for all workers to complete

        # Only pass along sample ids if necessary
        samples_for_row_stats = None
        if min_true_samples not in [None, 0]:
            samples_for_row_stats = sample_ids[0]

        self._update_row_statistics(df, samples_for_row_stats)
