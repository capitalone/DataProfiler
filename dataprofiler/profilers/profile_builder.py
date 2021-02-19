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

import pandas as pd

from . import utils
from .. import data_readers
from .column_profile_compilers import ColumnPrimitiveTypeProfileCompiler, \
    ColumnStatsProfileCompiler, ColumnDataLabelerCompiler
from .helpers.report_helpers import calculate_quantiles, _prepare_report
from .profiler_options import ProfilerOptions, StructuredOptions


class StructuredDataProfile(object):

    def __init__(self, df_series, sample_size=None, min_sample_size=500,
                 sampling_ratio=0.2, min_true_samples=None,
                 options=None):
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
        clean_sampled_df, base_stats = \
            self.get_base_props_and_clean_null_params(df_series, sample_size)
        self._update_base_stats(base_stats)
        self.profiles = {
            'data_type_profile':
                ColumnPrimitiveTypeProfileCompiler(clean_sampled_df,
                                                   self.options),
            'data_stats_profile':
                ColumnStatsProfileCompiler(clean_sampled_df, self.options)}

        # use the data labeler by default
        use_data_labeler = True
        if options and isinstance(options, StructuredOptions):
            use_data_labeler = options.data_labeler.is_enabled

        if use_data_labeler:
            self.profiles.update(
                {'data_label_profile':
                     ColumnDataLabelerCompiler(clean_sampled_df, self.options)})

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
                    unordered_profile["data_type_representation"],
                "data_label_probability": None,
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
                print(f"Error: {key} has no data")

        return profile

    @staticmethod
    def _combine_unique_sets(a, b):
        """
        Method to union two lists.
        
        :type a: list
        :type b: list
        :rtype: list
        """
        if not a and not b:
            return list()
        elif not a:
            return b
        elif not b:
            return a
        return list(OrderedDict.fromkeys(a + b))

    def _update_base_stats(self, base_stats):
        self.sample_size += base_stats["sample_size"]
        self.sample = base_stats["sample"]
        self.null_count += base_stats["null_count"]
        self.null_types = self._combine_unique_sets(
            self.null_types, list(base_stats["null_types"].keys())
        )

        for null_type, null_rows in base_stats["null_types"].items():
            if type(null_rows) is list:
                null_rows.sort()
            self.null_types_index.setdefault(null_type, []).extend(null_rows)

    def update_profile(self, df_series, sample_size=None, min_true_samples=None):
        if not sample_size:
            sample_size = len(df_series)
        if not sample_size:
            sample_size = self._get_sample_size(df_series)
        clean_sampled_df, base_stats = \
            self.get_base_props_and_clean_null_params(
                df_series, sample_size, min_true_samples=min_true_samples)
        self._update_base_stats(base_stats)
        for profile in self.profiles.values():
            profile.update_profile(clean_sampled_df)

    def _get_sample_size(self, df_series):
        """
        Determines the minimum sampling size for detecting column type.
        
        :param df_series: a column of data
        :type df_series: pandas.core.series.Series
        :return: integer sampling size
        :rtype: int
        """
        len_df = len(df_series)
        if len_df < self._min_sample_size:
            return int(len_df)
        return max(int(self._sampling_ratio * len_df), self._min_sample_size)

    # TODO: flag column name with null values and potentially return row
    #  index number in the error as well
    def get_base_props_and_clean_null_params(self, df_series, sample_size,
                                             min_true_samples=None):
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

        sample_ind_generator = utils.shuffle_in_chunks(
            len_df, chunk_size=sample_size)

        na_columns = dict()
        true_sample_list = list()
        total_sample_size = 0
        for sample_inds in sample_ind_generator:
            total_sample_size += len(sample_inds)

            df_series_subset = df_series.iloc[sample_inds]
            # Check if known null types exist in column
            for na, flags in null_values_and_flags.items():
                # Check for the regex of the na in the string.
                reg_ex_na = f"^{na}$"
                matching_na_elements = df_series_subset.str.contains(
                    reg_ex_na, flags=flags)
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
            
        self.encoding = None
        self.file_type = None
        self.null_in_row_count = 0
        self.hashed_row_dict = dict()
        self.rows_ingested = 0
        self._samples_per_update = samples_per_update
        self._min_true_samples = min_true_samples
        self._profile = dict()
            
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
        merged_profile.null_in_row_count = \
            self.null_in_row_count + other.null_in_row_count
        merged_profile.rows_ingested = self.rows_ingested + other.rows_ingested
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

    def report(self, report_options=None):
        if not report_options:
            report_options = {
                "output_format": None,
                "num_quantile_groups": 4,
            }
        output_format = report_options.get("output_format", None)
        num_quantile_groups = report_options.get("num_quantile_groups", 4)

        columns = list(self._profile.values())
        report = OrderedDict([
            ("global_stats", {
                "samples_used": columns[0].sample_size if columns else 0,
                "column_count": len(columns),
                "unique_row_ratio": self._get_unique_row_ratio(),
                "row_has_null_ratio": self._get_null_row_ratio(),
                "duplicate_row_count": self._get_duplicate_row_count(),
                "file_type": self.file_type,
                "encoding": self.encoding,
                "data_classification": None,
                "covariance": None,
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

        if output_format:
            return _prepare_report(report, output_format=output_format)
        return report

    def _get_unique_row_ratio(self):
        return len(self.hashed_row_dict) / self.rows_ingested

    def _get_null_row_ratio(self):
        return self.null_in_row_count / self.rows_ingested

    def _get_duplicate_row_count(self):
        return self.rows_ingested - len(self.hashed_row_dict)

    def _update_row_statistics(self, data):
        """
        Iterate over the provided dataset row by row and calculate
        the row statistics. Specificaly, number of unique rows,
        rows containing null values, and total rows reviewed. This
        function is safe to use in batches.

        :param data: a dataset
        :type data: pandas.DataFrame
        """
        for index, row in data.iterrows():

            # Hash the row and stores it in the dict, count keys for unique rows
            hashed_row = hashlib.sha256(
                row.to_string().strip().encode()).hexdigest()
            self.hashed_row_dict[hashed_row] = True

            # check if null in row, if any add count
            if row.isnull().any():
                self.null_in_row_count += 1

            # Used for ratios, total ingested rows
            self.rows_ingested += 1

        # Determines null count, transposes column major to row major
        # Any major returns true if null and sums total count of trues
        # This is done quickly and with minimal transform(s)
        # self.null_in_row_count = df.isnull().T.any().sum()

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

        for col in df.columns:
            if col in profile:
                column_profile = profile[col]
                column_profile.update_profile(
                    df[col],
                    sample_size=sample_size,
                    min_true_samples=min_true_samples
                )
            else:
                structured_options = None
                if options and options.structured_options:
                    structured_options = options.structured_options
                profile[col] = StructuredDataProfile(
                    df[col],
                    sample_size=sample_size,
                    min_true_samples=min_true_samples,
                    options=structured_options
                )

        return profile
