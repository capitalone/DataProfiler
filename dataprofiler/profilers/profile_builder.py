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
import hashlib
from collections import OrderedDict
import warnings

import multiprocessing as mp

import pandas as pd

from . import utils
from .. import data_readers
from .column_profile_compilers import ColumnPrimitiveTypeProfileCompiler, \
    ColumnStatsProfileCompiler, ColumnDataLabelerCompiler
from ..labelers.data_labelers import DataLabeler
from .helpers.report_helpers import calculate_quantiles, _prepare_report
from .profiler_options import ProfilerOptions, StructuredOptions, \
    DataLabelerOptions

class StructuredDataProfile(object):

    def __init__(self, df_series, sample_size=None, min_sample_size=5000,
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
        self.options = options
        self._min_sample_size = min_sample_size
        self._sampling_ratio = sampling_ratio
        self._min_true_samples = min_true_samples
        if self._min_true_samples is None:
            self._min_true_samples = 0

        # if you create your own DF without giving the column name,
        # it labels the name as an int64, however, if you try to
        # `json.dump` an int64, it errors.
        if isinstance(df_series.name, str) or df_series.name is None:
            self.name = df_series.name
        else:
            self.name = int(df_series.name)

        self.sample_size = 0
        self.sample = list()
        self.null_count = 0
        self.null_types = list()
        self.null_types_index = {}
                
        if not sample_size:
            sample_size = self._get_sample_size(df_series)
        if sample_size < len(df_series):
            warnings.warn("The data will be profiled with a sample size of {}. "
                          "All statistics will be based on this subsample and "
                          "not the whole dataset.".format(sample_size))
        clean_sampled_df, base_stats = \
            self.get_base_props_and_clean_null_params(
                df_series, sample_size, sample_ids=sample_ids)
        self._update_base_stats(base_stats)

        self.profiles = {
            'data_type_profile':
            ColumnPrimitiveTypeProfileCompiler(
                clean_sampled_df, self.options, pool),
            'data_stats_profile':
            ColumnStatsProfileCompiler(
                clean_sampled_df, self.options, pool)
        }

        use_data_labeler = True
        if options and isinstance(options, StructuredOptions):
            use_data_labeler = options.data_labeler.is_enabled

        if use_data_labeler:
            self.profiles.update({
                'data_label_profile':
                ColumnDataLabelerCompiler(clean_sampled_df, self.options)
            })

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
                                 other._min_true_samples))

        merged_profile.name = self.name
        merged_profile._update_base_stats(
            {"sample": self.sample, 'sample_size': self.sample_size,
             "null_count": self.null_count,
             "null_types": copy.deepcopy(self.null_types_index)}
        )
        merged_profile._update_base_stats(
            {"sample": other.sample, 'sample_size': other.sample_size,
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
        unordered_profile.update({
            "column_name": self.name,
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
        clean_sampled_df, base_stats = \
            self.get_base_props_and_clean_null_params(
                df_series, sample_size,
                min_true_samples=min_true_samples,
                sample_ids=sample_ids
            )
        self._update_base_stats(base_stats)

        # Profile compilers being updated
        for profile in self.profiles.values():
            profile.update_profile(clean_sampled_df, pool)


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
    def get_base_props_and_clean_null_params(self, df_series, sample_size,
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
        
        len_df = len(df_series)
        if not len_df:
            return df_series, {
                "sample_size": 0, "null_count": 0, "null_types": dict(),
                "sample": []}

        if min_true_samples is None:
            min_true_samples = self._min_true_samples

        # Pandas reads empty values in the csv files as nan
        df_series = df_series.apply(str)

        # Select generator depending if sample_ids availablity
        if sample_ids is None:
            sample_ind_generator = utils.shuffle_in_chunks(
                len_df, chunk_size=sample_size)
        else:
            sample_ind_generator = utils.partition(
                sample_ids[0], chunk_size=sample_size)
            
        na_columns = dict()
        true_sample_list = list()
        total_sample_size = 0
        for chunked_sample_ids in sample_ind_generator:
            total_sample_size += len(chunked_sample_ids)

            df_series_subset = df_series.iloc[chunked_sample_ids]

            query = '(' + '|'.join(null_values_and_flags.keys()) + ')'
            reg_ex_na = f"^{(query)}$"
            matching_na_elements = df_series_subset.str.contains(
                reg_ex_na, flags=re.IGNORECASE)

            for row, elem in matching_na_elements.items():
                if elem:
                    # Since df_series_subset[row] is mutable,
                    # need to make new var
                    row_value = str(df_series_subset[row])
                    na_columns.setdefault(row_value, list()).append(row)
                    
            # Drop the values that matched regex_na
            df_series_subset = df_series_subset[~matching_na_elements]
            
            true_sample_list += df_series_subset.index.tolist()

            if len(true_sample_list) >= min_true_samples and total_sample_size:
                break
            
        # close the generator in case it is not exhausted.
        if sample_ids is None:
            sample_ind_generator.close()

        df_series = df_series.loc[sorted(true_sample_list)]
        non_na = len(df_series)
        total_na = total_sample_size - non_na

        base_stats = {
            # TODO: Is this correct? used to be actual sample size, including
            #  NANs, what now?
            "sample_size": total_sample_size,
            "null_count": total_na,
            "null_types": na_columns,
            "sample": random.sample(list(df_series.values),
                                    min(len(df_series), 5))
        }

        return df_series, base_stats


class Profiler(object):

    def __init__(self, data, samples_per_update=None, min_true_samples=None, 
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
                self.options.set({'data_labeler.data_labeler_object': data_labeler})
                
            except Exception as e:
                utils.warn_on_profile('data_labeler', e)
                self.options.set({'data_labeler.is_enabled': False})

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
            min_true_samples=self._min_true_samples, profiler_options=None
        )
        merged_profile.options = self.options
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
        return 0 if self._max_col_samples_used == 0 \
            else self.row_is_null_count / self._max_col_samples_used

    def _get_row_has_null_ratio(self):
        return 0 if self._max_col_samples_used == 0 \
            else self.row_has_null_count / self._max_col_samples_used

    def _get_duplicate_row_count(self):
        return self.total_samples - len(self.hashed_row_dict)

    def _update_row_statistics(self, data):
        """
        Iterate over the provided dataset row by row and calculate
        the row statistics. Specifically, number of unique rows,
        rows containing null values, and total rows reviewed. This
        function is safe to use in batches.

        :param data: a dataset
        :type data: pandas.DataFrame
        """
        
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
        if not sample_size:
            sample_size = self._samples_per_update
        if not min_true_samples:
            min_true_samples = self._min_true_samples
        if isinstance(data, data_readers.base_data.BaseData):
            self._profile = self._update_profile_from_chunk(
                data.data, self._profile, sample_size, min_true_samples, self.options)
            self._update_row_statistics(data.data)
            self.encoding = data.file_encoding
            self.file_type = data.data_type
        elif isinstance(data, pd.DataFrame):
            self._profile = self._update_profile_from_chunk(
                data, self._profile, sample_size, min_true_samples, self.options)
            self._update_row_statistics(data)
            self.file_type = str(data.__class__)
        else:
            raise ValueError(
                "Data must either be imported using the data_readers or "
                "pd.DataFrame."
            )

    @staticmethod
    def _update_profile_from_chunk(df, profile=None, sample_size=None,
                                   min_true_samples=None, options=None):
        """
        Iterate over the columns of a dataset and identify its parameters.
        
        :param df: a dataset
        :type df: pandas.DataFrame
        :param profile: list of profiled columns [BaseColumnProfiler subclasses]
        :type profile: list
        :param sample_size: number of samples for df to use for profiling
        :type sample_size: int
        :param min_true_samples: minimum number of true samples required
        :type min_true_samples: int
        :param options: Options for the profiler
        :type options: ProfilerOptions
        :return: list of column profile base subclasses
        :rtype: list(BaseColumnProfiler)
        """
        if not profile:
            profile = OrderedDict()

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

        # Shuffle indices ones and share with columns
        sample_ids = [*utils.shuffle_in_chunks(len(df), len(df))]

        pool = None
        if options.structured_options.multiprocess.is_enabled:
            cpu_count = 1
            try:
                cpu_count = mp.cpu_count()
            except NotImplementedError as e:
                cpu_count = 1

            # No additional advantage beyond 8 processes
            # Always leave 1 cores free
            if cpu_count > 2:
                cpu_count = min(cpu_count-1, 8)
                pool = mp.Pool(cpu_count)
                print("Utilizing",cpu_count, "processes for profiling")
        
        for col in tqdm(df.columns):
            if col in profile:
                column_profile = profile[col]
                column_profile.update_profile(
                    df[col],
                    sample_size=sample_size,
                    min_true_samples=min_true_samples,
                    sample_ids=sample_ids,
                    pool=pool
                )
            else:
                structured_options = None
                if options and options.structured_options:
                    structured_options = options.structured_options
                profile[col] = StructuredDataProfile(
                    df[col],
                    sample_size=sample_size,
                    min_true_samples=min_true_samples,
                    sample_ids=sample_ids,
                    pool=pool,
                    options=structured_options
                )

        if pool is not None:
            pool.close() # Close pool for new tasks
            pool.join() # Wait for all workers to complete
            
        return profile

