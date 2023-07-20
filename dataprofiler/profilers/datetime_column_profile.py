"""Contains class for profiling datetime column."""
from __future__ import annotations

import datetime
import re
import warnings

import numpy as np
import pandas as pd

from . import profiler_utils
from .base_column_profilers import BaseColumnPrimitiveTypeProfiler, BaseColumnProfiler
from .profiler_options import DateTimeOptions


class DateTimeColumn(BaseColumnPrimitiveTypeProfiler["DateTimeColumn"]):
    """
    Datetime column profile subclass of BaseColumnProfiler.

    Represents a column int the dataset which is a datetime column.
    """

    type = "datetime"

    _date_formats = [
        "%Y-%m-%d %H:%M:%S",  # 2013-03-5 15:43:30
        "%Y-%m-%dT%H:%M:%S",  # 2013-03-6T15:43:30
        "%Y-%m-%dT%H:%M:%S.%fZ",  # 2013-03-6T15:43:30.123456Z
        "%Y-%m-%dt%H:%M:%S.%fz",  # 2013-03-6t15:43:30.123456z
        "%m/%d/%y %H:%M",  # 03/10/13 15:43
        "%m/%d/%Y %H:%M",  # 3/8/2013 15:43
        "%Y%m%dT%H%M%S",  # 2013036T154330
        "%Y-%m-%d",  # 2013-03-7
        "%m/%d/%Y",  # 3/8/2013
        "%m/%d/%y",  # 03/10/13
        "%b %d, %Y",  # Mar 11, 2013
        "%B %d, %Y",  # March 9, 2013
        "%d%b%y",  # 12Mar13
        "%b-%d-%y",  # Mar-13-13
        "%m%d%Y",  # 03142013
        "%H:%M:%S.%f",  # 05:46:30.258509
    ]

    _day_suffixes = [
        "st",
        "nd",
        "rd",
        "th",
    ]

    _compiled_day_suffix_regex = re.compile(
        r"(\d{1,2})(" + "|".join(_day_suffixes) + ")"
    )

    def __init__(self, name: str | None, options: DateTimeOptions = None) -> None:
        """
        Initialize it and the column base properties.

        :param name: Name of the data
        :type name: String
        :param options: Options for the datetime column
        :type options: DateTimeOptions
        """
        if options and not isinstance(options, DateTimeOptions):
            raise ValueError(
                "DateTimeColumn parameter 'options' must be of " "type DateTimeOptions."
            )
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)
        self.date_formats: list[str] = []
        self.min = None
        self.max = None
        self._dt_obj_min = None  # datetime obj of min
        self._dt_obj_max = None  # datetime obj of max

        self.__calculations: dict = {}
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other: DateTimeColumn) -> DateTimeColumn:
        """
        Merge the properties of two DateTimeColumn profiles.

        :param other: second profile
        :type other: DateTimeColumn
        """
        if not isinstance(other, DateTimeColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'DateTimeColumn' and '{}'".format(other.__class__.__name__)
            )

        merged_profile = DateTimeColumn(name=None)
        BaseColumnPrimitiveTypeProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )

        if self._dt_obj_min is None:
            merged_profile.min = other.min
            merged_profile._dt_obj_min = other._dt_obj_min
        elif other._dt_obj_min is None or self._dt_obj_min < other._dt_obj_min:
            merged_profile.min = self.min
            merged_profile._dt_obj_min = self._dt_obj_min
        else:
            merged_profile.min = other.min
            merged_profile._dt_obj_min = other._dt_obj_min

        if self._dt_obj_max is None:
            merged_profile.max = other.max
            merged_profile._dt_obj_max = other._dt_obj_max
        elif other._dt_obj_max is None or self._dt_obj_max > other._dt_obj_max:
            merged_profile.max = self.max
            merged_profile._dt_obj_max = self._dt_obj_max
        else:
            merged_profile.max = other.max
            merged_profile._dt_obj_max = other._dt_obj_max

        merged_profile.date_formats = profiler_utils._combine_unique_sets(
            self.date_formats, other.date_formats
        )
        return merged_profile

    def report(self, remove_disabled_flag: bool = False) -> dict:
        """
        Return report.

        Private abstract method.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        return self.profile

    @classmethod
    def load_from_dict(cls, data, config: dict | None = None):
        """
        Parse attribute from json dictionary into self.

        :param data: dictionary with attributes and values.
        :type data: dict[string, Any]
        :param config: config for loading column profiler params from dictionary
        :type config: Dict | None

        :return: Profiler with attributes populated.
        :rtype: DateTimeColumn
        """
        # This is an ambiguous call to super classes.
        # If load_from_dict is part of both super classes there may be issues
        profile = super().load_from_dict(data)

        if profile._dt_obj_min is not None:
            profile._dt_obj_min = pd.Timestamp(profile._dt_obj_min)

        if profile._dt_obj_max is not None:
            profile._dt_obj_max = pd.Timestamp(profile._dt_obj_max)

        return profile

    @property
    def profile(self) -> dict:
        """Return the profile of the column."""
        profile = dict(
            min=self.min,
            max=self.max,
            histogram=None,
            format=self.date_formats,
            times=self.times,
        )
        return profile

    @property
    def data_type_ratio(self) -> float | None:
        """
        Calculate the ratio of samples which match this data type.

        :return: ratio of data type
        :rtype: float
        """
        if self.sample_size:
            return float(self.match_count) / self.sample_size
        return None

    def diff(self, other_profile: DateTimeColumn, options: dict = None) -> dict:
        """
        Generate differences between max, min, and formats of two DateTime cols.

        :return: Dict containing the differences between max, min, and format in their
        appropriate output formats
        :rtype: dict
        """
        # Make sure other_profile's type matches this class
        super().diff(other_profile, options)

        differences = {
            "min": profiler_utils.find_diff_of_dates(
                self._dt_obj_min, other_profile._dt_obj_min
            ),
            "max": profiler_utils.find_diff_of_dates(
                self._dt_obj_max, other_profile._dt_obj_max
            ),
            "format": profiler_utils.find_diff_of_lists_and_sets(
                self.date_formats, other_profile.date_formats
            ),
        }
        return differences

    @staticmethod
    def _validate_datetime(date: str, date_format: str) -> datetime.datetime | float:
        """
        Check to see if a string contains a certain date format.

        :param date: a string that is possibly a date
        :type date: str
        :param date_format: a date regex that will be checked against date
        :type date_format: str
        :return: either the str converted into a date format, or Nan
        """
        try:
            converted_date: (datetime.datetime | float) = datetime.datetime.strptime(
                date, date_format
            )
        except (ValueError, TypeError):
            converted_date = np.nan

        return converted_date

    @staticmethod
    def _replace_day_suffix(date: str, pattern: re.Pattern) -> str | float:
        """
        Check the date for a suffix after the day. Remove suffix if present.

        :param date: a string that is possibly a date
        :type date: str
        :param pattern: precompiled regex pattern that is used to check for day suffixes
        :type pattern: Pattern
        :return: either the date string passed in, or Nan
        """
        try:
            new_date: str | float = pattern.sub(r"\1", date)
        except (TypeError):
            new_date = np.nan
        return new_date

    @classmethod
    def _get_datetime_profile(cls, df_series: pd.Series) -> dict:
        """
        Determine for each val in a col its format and if it's a datetime.

        Also collect datetime stats for the column.

        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :return: parameters for datetime columns
        :rtype: dict
        """
        profile: dict = dict()
        activated_date_formats: list = list()
        len_df = len(df_series)

        is_row_datetime = pd.Series(np.full((len(df_series)), False))

        min_value = None
        max_value = None
        min_value_obj = datetime.datetime.max
        max_value_obj = datetime.datetime.min
        for date_format in cls._date_formats:
            if is_row_datetime.all():
                break
            valid_dates = df_series.apply(
                lambda x: cls._validate_datetime(
                    cls._replace_day_suffix(  # type: ignore
                        x, cls._compiled_day_suffix_regex
                    ),
                    date_format,
                )
            )

            df_dates = valid_dates[~valid_dates.isnull()]

            if "%b" in date_format and not df_dates.empty:
                may_month = 5  # May can be %b or %B we want to force, so check
                all_may = df_dates.apply(lambda x: x.month == may_month).all()
                if all_may:
                    valid_dates[:] = np.nan
                    df_dates = pd.Series([], dtype=object)

            # Create mask to avoid null dates
            null_date_mask = valid_dates.isnull()
            np_date_array = df_dates.values

            # check off any values which were found to be datetime
            is_row_datetime[~is_row_datetime] = (~null_date_mask).values

            if len(df_dates) > 0:

                # Converts to numpy prior to finding max index
                min_idx = np.argmin(np_date_array)
                max_idx = np.argmax(np_date_array)

                # Selects the min, ma value objects for comparison
                tmp_min_value_obj = df_dates.iloc[min_idx]
                tmp_max_value_obj = df_dates.iloc[max_idx]

                # If minimum value, keep reference
                if tmp_min_value_obj < min_value_obj:
                    min_value = df_series[~null_date_mask].iloc[min_idx]
                    min_value_obj = tmp_min_value_obj

                # If maximum value, keep reference
                if tmp_max_value_obj > max_value_obj:
                    max_value = df_series[~null_date_mask].iloc[max_idx]
                    max_value_obj = tmp_max_value_obj

            df_series = df_series[null_date_mask]

            # Get a list of all datetime format identified in column
            new_len = len(df_series)
            if new_len < len_df:
                activated_date_formats.append(date_format)
                len_df = new_len
                if "y" in date_format:
                    warnings.warn(
                        "Years provided were in two digit format. As a result, "
                        "datetime assumes dates < 69 are for 2000s and above "
                        "are for the 1990s. "
                        "https://stackoverflow.com/questions/37766353/"
                        "pandas-to-datetime-parsing-wrong-year",
                        RuntimeWarning,
                    )

        profile["date_formats"] = activated_date_formats
        profile["min"] = min_value
        profile["max"] = max_value
        profile["min_obj"] = (
            min_value_obj.to_datetime()  # type: ignore
            if hasattr(min_value_obj, "to_datetime")
            else min_value_obj
        )
        profile["max_obj"] = (
            max_value_obj.to_datetime()  # type: ignore
            if hasattr(max_value_obj, "to_datetime")
            else max_value_obj
        )
        profile["match_count"] = is_row_datetime.sum()
        return profile

    def _is_subset_datetime_column(self, df_series: pd.Series) -> bool:
        """
        Check whether a subset of the data could be considered datetime.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: True or False
        :rtype: bool
        """
        if len(df_series) == 0:
            return False
        num_samples_to_check = 50
        thresh = 0.10
        sample_size = min(num_samples_to_check, len(df_series))
        profile = self._get_datetime_profile(df_series.sample(sample_size))

        if profile["match_count"] / sample_size < thresh:
            return False
        return True

    @BaseColumnProfiler._timeit(name="datetime")
    def _update_datetime(
        self,
        df_series: pd.DataFrame,
        prev_dependent_properties: dict,
        subset_properties: dict,
    ) -> None:
        """
        Calculate the datetime properties for the profile.

        :param df_series: data to check for datetime values and their properties
        :type df_series: pandas.Dataframe
        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return:
        """
        # date_formats
        profile = self._get_datetime_profile(df_series)
        date_formats = profile.pop("date_formats", [])
        if date_formats:
            self.date_formats = self._combine_unique_sets(
                self.date_formats, date_formats
            )

        # min datetime
        min_obj = profile.pop("min_obj")
        min_dt_value = profile.pop("min")
        if not self._dt_obj_min:
            self._dt_obj_min = min_obj
            self.min = min_dt_value
        elif min_obj and min_obj < self._dt_obj_min:
            self._dt_obj_min = min_obj
            self.min = min_dt_value

        max_obj = profile.pop("max_obj")
        max_dt_value = profile.pop("max")
        if not self._dt_obj_max:
            self._dt_obj_max = max_obj
            self.max = max_dt_value
        elif max_obj and max_obj > self._dt_obj_max:
            self._dt_obj_max = max_obj
            self.max = max_dt_value

        subset_properties.update(profile)

    def _update_helper(self, df_series: pd.Series, profile: dict) -> None:
        """
        Update the column profile properties.

        :param df_series: df series with nulls removed
        :type df_series: pandas.core.series.Series
        :param profile: base properties profile
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series: pd.Series) -> DateTimeColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: None
        """
        if len(df_series) == 0:
            return self

        df_series = df_series.reset_index(drop=True)
        profile = {"sample_size": len(df_series), "match_count": 0}
        if self._is_subset_datetime_column(df_series):
            self._update_datetime(df_series, {}, profile)
            super()._perform_property_calcs(
                self.__calculations,
                df_series=df_series,
                prev_dependent_properties={},
                subset_properties=profile,
            )
        self._update_helper(df_series=df_series, profile=profile)

        return self
