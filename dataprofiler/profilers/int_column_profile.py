"""Int profile analysis for individual col within structured profiling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base_column_profilers import BaseColumnPrimitiveTypeProfiler, BaseColumnProfiler
from .numerical_column_stats import NumericStatsMixin
from .profiler_options import IntOptions


class IntColumn(
    NumericStatsMixin["IntColumn"], BaseColumnPrimitiveTypeProfiler["IntColumn"]
):
    """
    Integer column profile mixin with of numerical stats.

    Represents a column in the dataset which is an integer column.
    """

    type = "int"

    def __init__(self, name: str | None, options: IntOptions = None) -> None:
        """
        Initialize column base properties and itself.

        :param name: Name of the data
        :type name: String
        :param options: Options for the integer column
        :type options: IntOptions
        """
        if options and not isinstance(options, IntOptions):
            raise ValueError(
                "IntColumn parameter 'options' must be of type" " IntOptions."
            )
        NumericStatsMixin.__init__(self, options)
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)
        self.__calculations: dict = {}
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other: IntColumn) -> IntColumn:
        """
        Merge the properties of two IntColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: IntColumn
        :type other: IntColumn
        :return: New IntColumn merged profile
        """
        if not isinstance(other, IntColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'IntColumn' and '{}'".format(other.__class__.__name__)
            )

        merged_profile = IntColumn(None)
        BaseColumnPrimitiveTypeProfiler._add_helper(merged_profile, self, other)
        NumericStatsMixin._add_helper(merged_profile, self, other)
        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )
        return merged_profile

    def report(self, remove_disabled_flag: bool = False) -> dict:
        """
        Return the report.

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
        :rtype: IntColumn
        """
        # This is an ambiguous call to super classes.
        # If load_from_dict is part of both super classes there may be issues
        profile = super().load_from_dict(data)
        profile._reformat_numeric_stats_types_on_serialized_profiles()
        return profile

    @property
    def profile(self) -> dict:
        """
        Return the profile of the column.

        :return:
        """
        return NumericStatsMixin.profile(self)

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

    @classmethod
    def _is_each_row_int(cls, df_series: pd.Series) -> list[bool]:
        """
        Return true if given is numerical and int values.

        e.g.
        For column [1 1 1] returns True
        For column [1.0 1.0 1.0] returns True
        For column [1.0 1.0 1.1] returns False
        For column [1.1 1.1 1.1] returns False

        :param df_series: series of values to evaluate
        :type df_series: pandas.core.series.Series
        :return: is_int_col
        :rtype: list
        """
        len_df = len(df_series)
        if len_df == 0:
            return list()

        return [NumericStatsMixin.is_int(x) for x in df_series]

    def _update_helper(self, df_series_clean: pd.Series, profile: dict) -> None:
        """
        Update col profile properties with clean dataset and its known null params.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: int profile dictionary
        :type profile: dict
        :return: None
        """
        if self._NumericStatsMixin__calculations:
            NumericStatsMixin._update_helper(self, df_series_clean, profile)
        self._update_column_base_properties(profile)

    def update(self, df_series: pd.Series) -> IntColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: updated IntColumn
        :rtype: IntColumn
        """
        if len(df_series) == 0:
            return self

        df_series = df_series.reset_index(drop=True)
        is_each_row_int = self._is_each_row_int(df_series)
        sample_size = len(is_each_row_int)
        match_int_count = np.sum(is_each_row_int)
        profile = dict(match_count=match_int_count, sample_size=sample_size)

        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series[is_each_row_int],
            prev_dependent_properties={},
            subset_properties=profile,
        )

        self._update_helper(df_series_clean=df_series[is_each_row_int], profile=profile)

        return self
