"""Float profile analysis for individual col within structured profiling."""
from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import utils
from .base_column_profilers import BaseColumnPrimitiveTypeProfiler, BaseColumnProfiler
from .numerical_column_stats import NumericStatsMixin
from .profiler_options import FloatOptions


class FloatColumn(NumericStatsMixin, BaseColumnPrimitiveTypeProfiler):
    """
    Float column profile mixin with numerical stats.

    Represents a column in the dataset which is a float column.
    """

    type = "float"

    def __init__(self, name: Optional[str], options: FloatOptions = None) -> None:
        """
        Initialize column base properties and itself.

        :param name: Name of the data
        :type name: String
        :param options: Options for the float column
        :type options: FloatOptions
        """
        if options and not isinstance(options, FloatOptions):
            raise ValueError(
                "FloatColumn parameter 'options' must be of type" " FloatOptions."
            )
        NumericStatsMixin.__init__(self, options)
        BaseColumnPrimitiveTypeProfiler.__init__(self, name)

        self._precision: Dict = {
            "min": None,
            "max": None,
            "sum": None,
            "mean": None,
            "biased_var": None,
            "sample_size": None,
            "confidence_level": 0.999,
        }

        # https://www.calculator.net/confidence-interval-calculator.html
        self.__z_value_precision = 3.291

        self.__precision_sample_ratio = None
        if options and options.precision and options.precision.is_enabled:
            self.__precision_sample_ratio = options.precision.sample_ratio

        self.__calculations = {
            "precision": FloatColumn._update_precision,
        }
        self._filter_properties_w_options(self.__calculations, options)

    def __add__(self, other: FloatColumn) -> FloatColumn:
        """
        Merge the properties of two FloatColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: FloatColumn
        :type other: FloatColumn
        :return: New FloatColumn merged profile
        """
        if not isinstance(other, FloatColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'FloatColumn' and '{}'".format(other.__class__.__name__)
            )

        merged_profile = FloatColumn(None)
        BaseColumnPrimitiveTypeProfiler._add_helper(merged_profile, self, other)
        NumericStatsMixin._add_helper(merged_profile, self, other)

        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )

        if "precision" in merged_profile.__calculations:

            if self._precision["min"] is None:
                merged_profile._precision = copy.deepcopy(other._precision)
            elif other.precision["min"] is None:
                merged_profile._precision = copy.deepcopy(self._precision)
            else:
                merged_profile._precision["min"] = min(
                    self._precision["min"], other._precision["min"]
                )
                merged_profile._precision["max"] = max(
                    self._precision["max"], other._precision["max"]
                )
                merged_profile._precision["sum"] = (
                    self._precision["sum"] + other._precision["sum"]
                )
                merged_profile._precision["sample_size"] = (
                    self._precision["sample_size"] + other._precision["sample_size"]
                )

                merged_profile._precision["mean"] = (
                    merged_profile._precision["sum"]
                    / merged_profile._precision["sample_size"]
                )

                merged_profile._precision["biased_var"] = self._merge_biased_variance(
                    self._precision["sample_size"],
                    self._precision["biased_var"],
                    self._precision["mean"],
                    other._precision["sample_size"],
                    other._precision["biased_var"],
                    other._precision["mean"],
                )

        return merged_profile

    def diff(self, other_profile: FloatColumn, options: Dict = None) -> Dict:
        """
        Find the differences for FloatColumns.

        :param other_profile: profile to find the difference with
        :type other_profile: FloatColumn
        :return: the FloatColumn differences
        :rtype: dict
        """
        differences = NumericStatsMixin.diff(self, other_profile, options=None)
        other_precision = other_profile.profile["precision"]
        precision_diff = dict()
        for key in self.profile["precision"].keys():
            precision_diff[key] = utils.find_diff_of_numbers(
                self.profile["precision"][key], other_precision[key]
            )
        precision_diff.pop("confidence_level")
        differences["precision"] = precision_diff
        return differences

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """Report profile attribute of class; potentially pop val from self.profile."""
        calcs_dict_keys = self._FloatColumn__calculations.keys()
        profile = self.profile

        if remove_disabled_flag:
            profile_keys = list(profile.keys())
            for profile_key in profile_keys:
                if profile_key == "precision":
                    if "precision" in calcs_dict_keys:
                        continue
                profile.pop(profile_key)

        return profile

    @property
    def profile(self) -> Dict:
        """
        Return the profile of the column.

        :return:
        """
        profile = NumericStatsMixin.profile(self)
        profile.update(
            dict(
                precision=dict(
                    min=self.np_type_to_type(self.precision["min"]),
                    max=self.np_type_to_type(self.precision["max"]),
                    mean=self.np_type_to_type(self.precision["mean"]),
                    var=self.np_type_to_type(self.precision["var"]),
                    std=self.np_type_to_type(self.precision["std"]),
                    sample_size=self.np_type_to_type(self.precision["sample_size"]),
                    margin_of_error=self.np_type_to_type(
                        self.precision["margin_of_error"]
                    ),
                    confidence_level=self.np_type_to_type(
                        self.precision["confidence_level"]
                    ),
                )
            )
        )
        return profile

    @property
    def precision(self) -> Dict[str, Optional[float]]:
        """
        Report statistics on the significant figures of each element in the data.

        :return: Precision statistics
        :rtype: dict
        """
        # First add the stats that don't need to be re-calculated
        precision = dict(
            min=self._precision["min"],
            max=self._precision["max"],
            mean=self._precision["mean"],
            sum=self._precision["sum"],
            sample_size=self._precision["sample_size"],
            confidence_level=0.999,
        )
        var = self._correct_bias_variance(
            self._precision["sample_size"], self._precision["biased_var"]
        )

        std = np.sqrt(var)
        margin_of_error = (
            None
            if self._precision["sample_size"] is None
            else self.__z_value_precision
            * std
            / np.sqrt(self._precision["sample_size"])
        )
        precision["var"] = var
        precision["std"] = std
        precision["margin_of_error"] = margin_of_error
        # Set the significant figures
        if self._precision["max"] is not None:
            sigfigs = int(self._precision["max"])
            for key in ["mean", "var", "std", "margin_of_error"]:
                precision[key] = float("{:.{p}g}".format(precision[key], p=sigfigs))

        return precision

    @property
    def data_type_ratio(self) -> Optional[float]:
        """
        Calculate the ratio of samples which match this data type.

        :return: ratio of data type
        :rtype: float
        """
        if self.sample_size:
            return float(self.match_count) / self.sample_size
        return None

    @classmethod
    def _get_float_precision(
        cls, df_series_clean: pd.Series, sample_ratio: float = None
    ) -> Optional[Dict]:
        """
        Determine the precision of the numeric value.

        :param df_series_clean: df series with nulls removed, assumes all values
            are floats as well
        :type df_series_clean: pandas.core.series.Series
        :param sample_ratio: Ratio of samples used for float precision
        :type sample_ratio: float (between 0 and 1)
        :return: string representing its precision print format
        :rtype: int
        """
        len_df = len(df_series_clean)
        if not len_df:
            return None

        # Lead zeros: ^[+-.0\s]+ End zeros: \.?0+(\s|$)
        # Scientific Notation: (?<=[e])(.*) Any non-digits: \D
        r = re.compile(r"^[+-.0\s]+|\.?0+(\s|$)|(?<=[e])(.*)|\D")

        # DEFAULT: Sample the dataset. If small use full dataset,
        # OR 20k samples or 5% of the dataset which ever is larger.
        # If user sets sample ratio, utilize their request
        sample_size = min(len_df, max(20000, int(len_df * 0.05)))
        if sample_ratio is not None and sample_ratio > 0:
            sample_size = int(len_df * sample_ratio)

        # length of sampled cells after all punctuation removed
        len_per_float = (
            df_series_clean.sample(sample_size).replace(to_replace=r, value="").map(len)
        )

        # Determine statistics precision
        precision_sum = len_per_float.sum()
        subset_precision = {
            "min": len_per_float.min(),
            "max": len_per_float.max(),
            "biased_var": float(np.var(len_per_float)),
            "sum": precision_sum,
            "mean": precision_sum / sample_size,
            "sample_size": sample_size,
        }

        return subset_precision

    @classmethod
    def _is_each_row_float(
        cls, df_series: pd.Series
    ) -> Union[List[bool], pd.Series[bool]]:
        """
        Determine if each value in a dataframe is a float.

        Integers and NaNs can be considered a float.
        e.g.
        For column [1, 1, 1] returns [True, True, True]
        For column [1.0, np.NaN, 1.0] returns [True, True, True]
        For column [1.0, "a", "b"] returns [True, False, False]
        :param df_series: series of values to evaluate
        :type df_series: pandas.core.series.Series
        :return: is_float_col
        :rtype: Union[List[bool], pandas.Series[bool]]
        """
        if len(df_series) == 0:
            return list()
        return df_series.map(NumericStatsMixin.is_float)

    @BaseColumnProfiler._timeit(name="precision")
    def _update_precision(
        self,
        df_series: pd.DataFrame,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        """
        Update the precision value of the column.

        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :param df_series: Data to be profiled
        :type df_series: pandas.DataFrame
        :return: None
        """
        sample_ratio = None
        if self.__precision_sample_ratio is not None:
            sample_ratio = self.__precision_sample_ratio

        # (min, max, var, sum, sample_size)
        subset_precision = self._get_float_precision(df_series, sample_ratio)
        if subset_precision is None:
            return
        elif self._precision["min"] is None:
            self._precision.update(subset_precision)
        else:
            # Update the calculations as data is valid
            self._precision["min"] = min(
                self._precision["min"], subset_precision["min"]
            )
            self._precision["max"] = max(
                self._precision["max"], subset_precision["max"]
            )
            self._precision["sum"] += subset_precision["sum"]
            self._precision["sample_size"] += subset_precision["sample_size"]

            self._precision["biased_var"] = self._merge_biased_variance(
                self._precision["sample_size"],
                self._precision["biased_var"],
                self._precision["mean"],
                subset_precision["sample_size"],
                subset_precision["biased_var"],
                subset_precision["mean"],
            )

            self._precision["mean"] = (
                self._precision["sum"] / self._precision["sample_size"]
            )

    def _update_helper(self, df_series_clean: pd.Series, profile: Dict) -> None:
        """
        Update column profile properties with cleaned dataset and its known profile.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: float profile dictionary
        :type profile: dict
        :return: None
        """
        if self._NumericStatsMixin__calculations:
            NumericStatsMixin._update_helper(self, df_series_clean, profile)
        self._update_column_base_properties(profile)

    def _update_numeric_stats(
        self,
        df_series: pd.DataFrame,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        """
        Call the numeric stats update function.

        This is a wrapper to allow for modularity.
        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: Dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: Dict
        :param df_series: Data to be profiled
        :type df_series: Pandas Dataframe
        :return: None
        """
        super(FloatColumn, self)._update_helper(df_series, subset_properties)

    def update(self, df_series: pd.Series) -> FloatColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: updated FloatColumn
        :rtype: FloatColumn
        """
        if len(df_series) == 0:
            return self

        is_each_row_float = self._is_each_row_float(df_series)
        sample_size = len(is_each_row_float)
        float_count = np.sum(is_each_row_float)
        profile = dict(match_count=float_count, sample_size=sample_size)

        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series[is_each_row_float],
            prev_dependent_properties={},
            subset_properties=profile,
        )

        self._update_helper(
            df_series_clean=df_series[is_each_row_float], profile=profile
        )

        return self
