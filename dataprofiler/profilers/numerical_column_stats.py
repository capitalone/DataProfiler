#!/usr/bin/env python
"""Build model for dataset by identifying col type along with its respective params."""
from __future__ import annotations, division, print_function

import abc
import copy
import itertools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
from future.utils import with_metaclass

from . import histogram_utils, utils
from .base_column_profilers import BaseColumnProfiler
from .profiler_options import NumericalOptions


class abstractstaticmethod(staticmethod):
    """For making function an abstract method."""

    __slots__ = ()

    def __init__(self, function: Callable) -> None:
        """Initialize abstract static method."""
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True  # type: ignore

    __isabstractmethod__ = True


class NumericStatsMixin(with_metaclass(abc.ABCMeta, object)):  # type: ignore
    """
    Abstract numerical column profile subclass of BaseColumnProfiler.

    Represents column in the dataset which is a text column.
    Has Subclasses itself.
    """

    type: Optional[str] = None

    def __init__(self, options: NumericalOptions = None) -> None:
        """
        Initialize column base properties and itself.

        :param options: Options for the numerical stats.
        :type options: NumericalOptions
        """
        if options and not isinstance(options, NumericalOptions):
            raise ValueError(
                "NumericalStatsMixin parameter 'options' must be "
                "of type NumericalOptions."
            )
        self.min: Union[int, float, None] = None
        self.max: Union[int, float, None] = None
        self._top_k_modes: int = 5  # By default, return at max 5 modes
        self.sum: Union[int, float] = 0
        self._biased_variance: float = np.nan
        self._biased_skewness: float = np.nan
        self._biased_kurtosis: float = np.nan
        self._median_is_enabled: bool = True
        self._median_abs_dev_is_enabled: bool = True
        self.max_histogram_bin: int = 100000
        self.min_histogram_bin: int = 1000
        self.histogram_bin_method_names: List[str] = [
            "auto",
            "fd",
            "doane",
            "scott",
            "rice",
            "sturges",
            "sqrt",
        ]
        self.histogram_selection: Optional[str] = None
        self.user_set_histogram_bin: Optional[int] = None
        self.bias_correction: bool = True  # By default, we correct for bias
        self._mode_is_enabled: bool = True
        self.num_zeros: int = 0
        self.num_negatives: int = 0
        if options:
            self.bias_correction = options.bias_correction.is_enabled
            self._top_k_modes = options.mode.top_k_modes
            self._median_is_enabled = options.median.is_enabled
            self._median_abs_dev_is_enabled = options.median_abs_deviation.is_enabled
            self._mode_is_enabled = options.mode.is_enabled
            bin_count_or_method = options.histogram_and_quantiles.bin_count_or_method
            if isinstance(bin_count_or_method, str):
                self.histogram_bin_method_names = [bin_count_or_method]
            elif isinstance(bin_count_or_method, list):
                self.histogram_bin_method_names = bin_count_or_method
            elif isinstance(bin_count_or_method, int):
                self.user_set_histogram_bin = bin_count_or_method
                self.histogram_bin_method_names = ["custom"]
        self.histogram_methods: Dict = {}
        self._stored_histogram: Dict = {
            "total_loss": 0,
            "current_loss": 0,
            "suggested_bin_count": self.min_histogram_bin,
            "histogram": {"bin_counts": None, "bin_edges": None},
        }
        self._batch_history: List = []
        for method in self.histogram_bin_method_names:
            self.histogram_methods[method] = {
                "total_loss": 0,
                "current_loss": 0,
                "suggested_bin_count": self.min_histogram_bin,
                "histogram": {"bin_counts": None, "bin_edges": None},
            }
        num_quantiles: int = 1000  # TODO: add to options
        self.quantiles: Union[List[float], Dict] = {
            bin_num: None for bin_num in range(num_quantiles - 1)
        }
        self.__calculations = {
            "min": NumericStatsMixin._get_min,
            "max": NumericStatsMixin._get_max,
            "sum": NumericStatsMixin._get_sum,
            "variance": NumericStatsMixin._get_variance,
            "skewness": NumericStatsMixin._get_skewness,
            "kurtosis": NumericStatsMixin._get_kurtosis,
            "histogram_and_quantiles": NumericStatsMixin._get_histogram_and_quantiles,
            "num_zeros": NumericStatsMixin._get_num_zeros,
            "num_negatives": NumericStatsMixin._get_num_negatives,
        }

        self._filter_properties_w_options(self.__calculations, options)

    def __getattribute__(self, name: str) -> Any:
        """Return computed attribute value."""
        return super(NumericStatsMixin, self).__getattribute__(name)

    def __getitem__(self, item: str) -> Any:
        """Return indexed item."""
        return super(NumericStatsMixin, self).__getitem__(item)

    @property
    def _has_histogram(self) -> bool:
        return self._stored_histogram["histogram"]["bin_counts"] is not None

    @BaseColumnProfiler._timeit(name="histogram_and_quantiles")
    def _add_helper_merge_profile_histograms(
        self, other1: BaseColumnProfiler, other2: BaseColumnProfiler
    ) -> None:
        """
        Add histogram of two profiles together.

        :param other1: profile1 being added to self
        :type other1: BaseColumnProfiler
        :param other2: profile2 being added to self
        :type other2: BaseColumnProfiler
        :return: None
        """
        # get available bin methods and set to current
        bin_methods = [
            x
            for x in other1.histogram_bin_method_names
            if x in other2.histogram_bin_method_names
        ]
        if not bin_methods:
            raise ValueError(
                "Profiles have no overlapping bin methods and "
                "therefore cannot be added together."
            )
        elif other1.user_set_histogram_bin and other2.user_set_histogram_bin:
            if other1.user_set_histogram_bin != other2.user_set_histogram_bin:
                warnings.warn(
                    "User set histogram bin counts did not match. "
                    "Choosing the larger bin count."
                )
            self.user_set_histogram_bin = max(
                other1.user_set_histogram_bin, other2.user_set_histogram_bin
            )

        # initial creation of the profiler creates all methods, but
        # only the methods which intersect should exist.
        self.histogram_bin_method_names = bin_methods
        self.histogram_methods = dict()
        for method in self.histogram_bin_method_names:
            self.histogram_methods[method] = {
                "total_loss": 0,
                "current_loss": 0,
                "histogram": {"bin_counts": None, "bin_edges": None},
            }

        combined_values = np.concatenate(
            [other1._histogram_to_array(), other2._histogram_to_array()]
        )
        bin_counts, bin_edges = self._get_histogram(combined_values)
        self._stored_histogram["histogram"]["bin_counts"] = bin_counts
        self._stored_histogram["histogram"]["bin_edges"] = bin_edges

        histogram_loss = self._histogram_bin_error(combined_values)
        self._stored_histogram["histogram"]["current_loss"] = histogram_loss
        self._stored_histogram["histogram"]["total_loss"] = histogram_loss

        self._get_quantiles()

    def _add_helper(self, other1: NumericStatsMixin, other2: NumericStatsMixin) -> None:
        """
        Help merge profiles.

        :param other1: profile1 being added to self
        :param other2: profile2 being added to self
        :return: None
        """
        BaseColumnProfiler._merge_calculations(
            self._NumericStatsMixin__calculations,
            other1._NumericStatsMixin__calculations,
            other2._NumericStatsMixin__calculations,
        )

        # Check and potentially override bias correction computation
        self.bias_correction = True
        if not other1.bias_correction or not other2.bias_correction:
            self.bias_correction = False

        # Merge variance, histogram, min, max, and sum
        if "variance" in self.__calculations.keys():
            self._biased_variance = self._merge_biased_variance(
                other1.match_count,
                other1._biased_variance,
                other1.mean,
                other2.match_count,
                other2._biased_variance,
                other2.mean,
            )
        if "histogram_and_quantiles" in self.__calculations.keys():
            if other1._has_histogram and other2._has_histogram:
                self._add_helper_merge_profile_histograms(other1, other2)
            elif not other2._has_histogram:
                self.histogram_methods = other1.histogram_methods
                self.quantiles = other1.quantiles
            else:
                self.histogram_methods = other2.histogram_methods
                self.quantiles = other2.quantiles
        if "min" in self.__calculations.keys():
            if other1.min is not None and other2.min is not None:
                self.min = min(other1.min, other2.min)
            elif other2.min is None:
                self.min = other1.min
            else:
                self.min = other2.min
        if "max" in self.__calculations.keys():
            if other1.max is not None and other2.max is not None:
                self.max = max(other1.max, other2.max)
            elif other2.max is None:
                self.max = other1.max
            else:
                self.max = other2.max
        if "sum" in self.__calculations.keys():
            self.sum = other1.sum + other2.sum
        if "skewness" in self.__calculations.keys():
            self._biased_skewness = self._merge_biased_skewness(
                other1.match_count,
                other1._biased_skewness,
                other1._biased_variance,
                other1.mean,
                other2.match_count,
                other2._biased_skewness,
                other2._biased_variance,
                other2.mean,
            )
        if "kurtosis" in self.__calculations.keys():
            self._biased_kurtosis = self._merge_biased_kurtosis(
                other1.match_count,
                other1._biased_kurtosis,
                other1._biased_skewness,
                other1._biased_variance,
                other1.mean,
                other2.match_count,
                other2._biased_kurtosis,
                other2._biased_skewness,
                other2._biased_variance,
                other2.mean,
            )
        if "num_zeros" in self.__calculations.keys():
            self.num_zeros = other1.num_zeros + other2.num_zeros

        if "num_negatives" in self.__calculations.keys():
            self.num_negatives = other1.num_negatives + other2.num_negatives

        # Merge max k mode count
        self._top_k_modes = max(other1._top_k_modes, other2._top_k_modes)
        # Merge median enable/disable option
        self._median_is_enabled = (
            other1._median_is_enabled and other2._median_is_enabled
        )
        # Merge mode enable/disable option
        self._mode_is_enabled = other1._mode_is_enabled and other2._mode_is_enabled
        # Merge median absolute deviation enable/disable option
        self._median_abs_dev_is_enabled = (
            other1._median_abs_dev_is_enabled and other2._median_abs_dev_is_enabled
        )

    def profile(self) -> Dict:
        """
        Return profile of the column.

        :return:
        """
        profile = dict(
            min=self.np_type_to_type(self.min),
            max=self.np_type_to_type(self.max),
            mode=self.np_type_to_type(self.mode),
            median=self.np_type_to_type(self.median),
            sum=self.np_type_to_type(self.sum),
            mean=self.np_type_to_type(self.mean),
            variance=self.np_type_to_type(self.variance),
            stddev=self.np_type_to_type(self.stddev),
            skewness=self.np_type_to_type(self.skewness),
            kurtosis=self.np_type_to_type(self.kurtosis),
            histogram=self._get_best_histogram_for_profile(),
            quantiles=self.quantiles,
            median_abs_deviation=self.np_type_to_type(self.median_abs_deviation),
            num_zeros=self.np_type_to_type(self.num_zeros),
            num_negatives=self.np_type_to_type(self.num_negatives),
            times=self.times,
        )

        return profile

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """
        Call the profile and remove the disabled columns from profile's report.

            "Disabled column" is defined as a column
            that is not present in `self.__calculations` but is present
            in the `self.profile`.

        :var remove_disabled_flag: true/false value to tell the code to remove
            values missing in __calculations
        :type remove_disabled_flag: boolean
        :return: Profile object pop'd based on values missing from __calculations
        :rtype: Profile
        """
        calcs_dict_keys = self._NumericStatsMixin__calculations.keys()
        profile = self.profile()

        if remove_disabled_flag:
            profile_keys = list(profile.keys())
            for profile_key in profile_keys:
                if profile_key in ["mode", "quantiles", "histogram"]:
                    if "histogram_and_quantiles" in calcs_dict_keys:
                        continue
                elif profile_key == "stddev" and "variance" in calcs_dict_keys:
                    continue
                elif profile_key in calcs_dict_keys:
                    continue
                elif profile_key == "times":
                    continue
                profile.pop(profile_key)

        return profile

    def diff(self, other_profile: NumericStatsMixin, options: Dict = None) -> Dict:
        """
        Find the differences for several numerical stats.

        :param other_profile: profile to find the difference with
        :type other_profile: NumericStatsMixin Profile
        :return: the numerical stats differences
        :rtype: dict
        """
        cls = self.__class__
        if not isinstance(other_profile, cls):
            raise TypeError(
                "Unsupported operand type(s) for diff: '{}' "
                "and '{}'".format(cls.__name__, other_profile.__class__.__name__)
            )

        differences = {
            "min": utils.find_diff_of_numbers(self.min, other_profile.min),
            "max": utils.find_diff_of_numbers(self.max, other_profile.max),
            "sum": utils.find_diff_of_numbers(self.sum, other_profile.sum),
            "mean": utils.find_diff_of_numbers(self.mean, other_profile.mean),
            "median": utils.find_diff_of_numbers(self.median, other_profile.median),
            "mode": utils.find_diff_of_lists_and_sets(self.mode, other_profile.mode),
            "median_absolute_deviation": utils.find_diff_of_numbers(
                self.median_abs_deviation, other_profile.median_abs_deviation
            ),
            "variance": utils.find_diff_of_numbers(
                self.variance, other_profile.variance
            ),
            "stddev": utils.find_diff_of_numbers(self.stddev, other_profile.stddev),
            "t-test": self._perform_t_test(
                self.mean,
                self.variance,
                self.match_count,
                other_profile.mean,
                other_profile.variance,
                other_profile.match_count,
            ),
        }
        return differences

    @property
    def mean(self) -> float:
        """Return mean value."""
        if self.match_count == 0:
            return 0
        return float(self.sum) / self.match_count

    @property
    def mode(self) -> List[float]:
        """
        Find an estimate for the mode[s] of the data.

        :return: the mode(s) of the data
        :rtype: list(float)
        """
        if not self._has_histogram or not self._mode_is_enabled:
            return [np.nan]
        return self._estimate_mode_from_histogram()

    @property
    def median(self) -> float:
        """
        Estimate the median of the data.

        :return: the median
        :rtype: float
        """
        if not self._has_histogram or not self._median_is_enabled:
            return np.nan
        return self._get_percentile([50])[0]

    @property
    def variance(self) -> float:
        """Return variance."""
        return (
            self._biased_variance
            if not self.bias_correction
            else self._correct_bias_variance(self.match_count, self._biased_variance)
        )

    @property
    def stddev(self) -> float:
        """Return stddev value."""
        if self.match_count == 0:
            return np.nan
        return np.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """Return skewness value."""
        return (
            self._biased_skewness
            if not self.bias_correction
            else self._correct_bias_skewness(self.match_count, self._biased_skewness)
        )

    @property
    def kurtosis(self) -> float:
        """Return kurtosis value."""
        return (
            self._biased_kurtosis
            if not self.bias_correction
            else self._correct_bias_kurtosis(self.match_count, self._biased_kurtosis)
        )

    @staticmethod
    def _perform_t_test(
        mean1: float, var1: float, n1: int, mean2: float, var2: float, n2: int
    ) -> Dict:
        results: Dict = {
            "t-statistic": None,
            "conservative": {"df": None, "p-value": None},
            "welch": {"df": None, "p-value": None},
        }

        invalid_stats = False
        if n1 <= 1 or n2 <= 1:
            warnings.warn(
                "Insufficient sample size. " "T-test cannot be performed.",
                RuntimeWarning,
            )
            invalid_stats = True
        if np.isnan([mean1, mean2, var1, var2]).any() or None in [
            mean1,
            mean2,
            var1,
            var2,
        ]:
            warnings.warn(
                "Null value(s) found in mean and/or variance values. "
                "T-test cannot be performed.",
                RuntimeWarning,
            )
            invalid_stats = True
        if not var1 and not var2:
            warnings.warn(
                "Data were essentially constant. T-test cannot be performed.",
                RuntimeWarning,
            )
            invalid_stats = True
        if invalid_stats:
            return results

        s_delta = var1 / n1 + var2 / n2
        t = (mean1 - mean2) / np.sqrt(s_delta)
        conservative_df = min(n1, n2) - 1
        welch_df = s_delta**2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )
        results["t-statistic"] = t
        results["conservative"]["df"] = float(conservative_df)
        results["welch"]["df"] = float(welch_df)

        conservative_t = scipy.stats.t(conservative_df)
        conservative_p_val = (1 - conservative_t.cdf(abs(t))) * 2
        welch_t = scipy.stats.t(welch_df)
        welch_p_val = (1 - welch_t.cdf(abs(t))) * 2

        results["conservative"]["p-value"] = float(conservative_p_val)
        results["welch"]["p-value"] = float(welch_p_val)
        return results

    def _update_variance(
        self, batch_mean: float, batch_var: float, batch_count: int
    ) -> float:
        """
        Calculate combined biased variance of the current values and new dataset.

        :param batch_mean: mean of new chunk
        :param batch_var: biased variance of new chunk
        :param batch_count: number of samples in new chunk
        :return: combined biased variance
        :rtype: float
        """
        return self._merge_biased_variance(
            self.match_count,
            self._biased_variance,
            self.mean,
            batch_count,
            batch_var,
            batch_mean,
        )

    @staticmethod
    def _merge_biased_variance(
        match_count1: int,
        biased_variance1: float,
        mean1: float,
        match_count2: int,
        biased_variance2: float,
        mean2: float,
    ) -> float:
        """
        Calculate combined biased variance of the current values and new dataset.

        :param match_count1: number of samples in new chunk 1
        :param mean1: mean of chunk 1
        :param biased_variance1: variance of chunk 1 without bias correction
        :param match_count2: number of samples in new chunk 2
        :param mean2: mean of chunk 2
        :param biased_variance2: variance of chunk 2 without bias correction
        :return: combined variance
        :rtype: float
        """
        if match_count1 < 1:
            return biased_variance2
        elif match_count2 < 1:
            return biased_variance1
        elif np.isnan(biased_variance1) or np.isnan(biased_variance2):
            return np.nan

        curr_count = match_count1
        delta = mean2 - mean1
        m_curr = biased_variance1 * curr_count
        m_batch = biased_variance2 * match_count2
        M2 = (
            m_curr
            + m_batch
            + delta**2 * curr_count * match_count2 / (curr_count + match_count2)
        )
        new_variance = M2 / (curr_count + match_count2)
        return new_variance

    @staticmethod
    def _correct_bias_variance(match_count: int, biased_variance: float) -> float:
        if match_count is None or biased_variance is None or match_count < 2:
            warnings.warn(
                "Insufficient match count to correct bias in variance. Bias correction "
                "can be manually disabled by setting bias_correction.is_enabled to "
                "False in ProfilerOptions.",
                RuntimeWarning,
            )
            return np.nan

        variance = match_count / (match_count - 1) * biased_variance
        return variance

    @staticmethod
    def _merge_biased_skewness(
        match_count1: int,
        biased_skewness1: float,
        biased_variance1: float,
        mean1: float,
        match_count2: int,
        biased_skewness2: float,
        biased_variance2: float,
        mean2: float,
    ) -> float:
        """
        Calculate the combined skewness of two data chunks.

        :param match_count1: # of samples in 1st chunk
        :param biased_skewness1: skewness of 1st chunk without bias correction
        :param biased_variance1: variance of 1st chunk without bias correction
        :param mean1: mean of 1st chunk
        :param match_count2: # of samples in 2nd chunk
        :param biased_skewness2: skewness of 2nd chunk without bias correction
        :param biased_variance2: variance of 2nd chunk without bias correction
        :param mean2: mean of 2nd chunk
        :return: combined skewness
        :rtype: float
        """
        if match_count1 < 1:
            return biased_skewness2
        elif match_count2 < 1:
            return biased_skewness1
        elif np.isnan(biased_skewness1) or np.isnan(biased_skewness2):
            return np.nan

        delta = mean2 - mean1
        N = match_count1 + match_count2
        M2_1 = match_count1 * biased_variance1
        M2_2 = match_count2 * biased_variance2
        M2 = M2_1 + M2_2 + delta**2 * match_count1 * match_count2 / N
        if not M2:
            return 0.0

        M3_1 = biased_skewness1 * np.sqrt(M2_1**3) / np.sqrt(match_count1)
        M3_2 = biased_skewness2 * np.sqrt(M2_2**3) / np.sqrt(match_count2)

        first_term = M3_1 + M3_2
        second_term = (
            delta**3
            * match_count1
            * match_count2
            * (match_count1 - match_count2)
            / N**2
        )
        third_term = 3 * delta * (match_count1 * M2_2 - match_count2 * M2_1) / N
        M3 = first_term + second_term + third_term

        biased_skewness = np.sqrt(N) * M3 / np.sqrt(M2**3)
        return biased_skewness

    @staticmethod
    def _correct_bias_skewness(match_count: int, biased_skewness: float) -> float:
        """
        Apply bias correction to skewness.

        :param match_count: number of samples
        :param biased_skewness: skewness without bias correction
        :return: unbiased estimator of skewness
        :rtype: NaN if sample size is too small, float otherwise
        """
        if np.isnan(biased_skewness) or match_count < 3:
            warnings.warn(
                "Insufficient match count to correct bias in skewness. Bias correction"
                "can be manually disabled by setting bias_correction.is_enabled to"
                "False in ProfilerOptions.",
                RuntimeWarning,
            )
            return np.nan

        skewness = (
            np.sqrt(match_count * (match_count - 1))
            * biased_skewness
            / (match_count - 2)
        )
        return skewness

    @staticmethod
    def _merge_biased_kurtosis(
        match_count1: int,
        biased_kurtosis1: float,
        biased_skewness1: float,
        biased_variance1: float,
        mean1: float,
        match_count2: int,
        biased_kurtosis2: float,
        biased_skewness2: float,
        biased_variance2: float,
        mean2: float,
    ) -> float:
        """
        Calculate the combined kurtosis of two sets of data.

        :param match_count1: # of samples in 1st chunk
        :param biased_kurtosis1: kurtosis of 1st chunk without bias correction
        :param biased_skewness1: skewness of 1st chunk without bias correction
        :param biased_variance1: variance of 1st chunk without bias correction
        :param mean1: mean of 1st chunk
        :param match_count2: # of samples in 2nd chunk
        :param biased_kurtosis2: kurtosis of 2nd chunk without bias correction
        :param biased_skewness2: skewness of 2nd chunk without bias correction
        :param biased_variance2: variance of 2nd chunk without bias correction
        :param mean2: mean of 2nd chunk
        :return: combined skewness
        :rtype: float
        """
        if match_count1 < 1:
            return biased_kurtosis2
        elif match_count2 < 1:
            return biased_kurtosis1
        elif np.isnan(biased_kurtosis1) or np.isnan(biased_kurtosis2):
            return np.nan

        delta = mean2 - mean1
        N = match_count1 + match_count2
        M2_1 = match_count1 * biased_variance1
        M2_2 = match_count2 * biased_variance2
        M2 = M2_1 + M2_2 + delta**2 * match_count1 * match_count2 / N
        if not M2:
            return 0

        M3_1 = biased_skewness1 * np.sqrt(M2_1**3) / np.sqrt(match_count1)
        M3_2 = biased_skewness2 * np.sqrt(M2_2**3) / np.sqrt(match_count2)
        M4_1 = (biased_kurtosis1 + 3) * M2_1**2 / match_count1
        M4_2 = (biased_kurtosis2 + 3) * M2_2**2 / match_count2

        first_term = M4_1 + M4_2
        second_term = (
            delta**4
            * (
                match_count1
                * match_count2
                * (match_count1**2 - match_count1 * match_count2 + match_count2**2)
            )
            / N**3
        )
        third_term = (
            6
            * delta**2
            * (match_count1**2 * M2_2 + match_count2**2 * M2_1)
            / N**2
        )
        fourth_term = 4 * delta * (match_count1 * M3_2 - match_count2 * M3_1) / N
        M4 = first_term + second_term + third_term + fourth_term

        biased_kurtosis = N * M4 / M2**2 - 3
        return biased_kurtosis

    @staticmethod
    def _correct_bias_kurtosis(match_count: int, biased_kurtosis: float) -> float:
        """
        Apply bias correction to kurtosis.

        :param match_count: number of samples
        :param biased_kurtosis: skewness without bias correction
        :return: unbiased estimator of kurtosis
        :rtype: NaN if sample size is too small, float otherwise
        """
        if np.isnan(biased_kurtosis) or match_count < 4:
            warnings.warn(
                "Insufficient match count to correct bias in kurtosis. Bias correction"
                "can be manually disabled by setting bias_correction.is_enabled to"
                "False in ProfilerOptions.",
                RuntimeWarning,
            )
            return np.nan

        kurtosis = (
            (match_count - 1)
            / ((match_count - 2) * (match_count - 3))
            * ((match_count + 1) * (biased_kurtosis + 3) - 3 * (match_count - 1))
        )
        return kurtosis

    def _estimate_mode_from_histogram(self) -> List[float]:
        """
        Estimate the mode of the current data using the histogram.

        If there are multiple modes, returns
        K of them (where K is defined in options given, but
        5 by default)

        :return: The estimated mode of the histogram
        :rtype: list(float)
        """
        bin_counts = self._stored_histogram["histogram"]["bin_counts"]
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]

        # Get the K bin(s) with the highest frequency (one-pass):
        cur_max = -1
        highest_idxs = []
        count = 0
        for i in range(0, len(bin_counts)):
            if bin_counts[i] > cur_max:
                # If a new maximum frequency is found, reset the mode counts
                highest_idxs = [i]
                cur_max = bin_counts[i]
                count = 1
            elif bin_counts[i] == cur_max and count < self._top_k_modes:
                highest_idxs.append(i)
                count += 1
        highest_idxs = np.array(highest_idxs)  # type: ignore

        mode = (
            bin_edges[highest_idxs] + bin_edges[highest_idxs + 1]  # type: ignore
        ) / 2
        return mode.tolist()

    def _estimate_stats_from_histogram(self) -> float:
        # test estimated mean and var
        bin_counts = self._stored_histogram["histogram"]["bin_counts"]
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]
        mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mean = np.average(mids, weights=bin_counts)
        var = np.average((mids - mean) ** 2, weights=bin_counts)
        return var

    def _total_histogram_bin_variance(
        self, input_array: Union[np.ndarray, pd.Series]
    ) -> float:
        # calculate total variance over all bins of a histogram
        bin_counts = self._stored_histogram["histogram"]["bin_counts"]
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]

        # account ofr digitize which is exclusive
        bin_edges = bin_edges.copy()
        bin_edges[-1] += 1e-3

        inds = np.digitize(input_array, bin_edges)
        sum_var = 0
        non_zero_bins = np.where(bin_counts)[0] + 1
        for i in non_zero_bins:
            elements_in_bin = input_array[inds == i]
            bin_var = elements_in_bin.var()
            sum_var += bin_var
        return sum_var

    def _histogram_bin_error(self, input_array: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate error of each value from bin of the histogram it falls within.

        :param input_array: input data used to calculate the histogram
        :type input_array: Union[np.array, pd.pd.Series]
        :return: binning error
        :rtype: float
        """
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]

        # account ofr digitize which is exclusive
        bin_edges = bin_edges.copy()

        temp_last_edge = bin_edges[-1]
        bin_edges[-1] = np.inf

        inds = np.digitize(input_array, bin_edges)
        if temp_last_edge == np.inf:
            inds = np.minimum(inds, len(bin_edges) - 1)

        # reset the edge
        bin_edges[-1] = temp_last_edge

        sum_error = sum(
            (input_array - (bin_edges[inds] + bin_edges[inds - 1]) / 2) ** 2
        )

        return sum_error

    @staticmethod
    def _histogram_loss(
        diff_var: float,
        avg_diffvar: float,
        total_var: float,
        avg_totalvar: float,
        run_time: float,
        avg_runtime: float,
    ) -> float:

        norm_diff_var: float = 0
        norm_total_var: float = 0
        norm_runtime: float = 0
        if avg_diffvar > 0:
            norm_diff_var = float(diff_var - avg_diffvar) / avg_diffvar
        if avg_totalvar > 0:
            norm_total_var = float(total_var - avg_totalvar) / avg_totalvar
        penalized_time = 1  # currently set as 1s
        if (run_time - avg_runtime) >= penalized_time:
            norm_runtime = float(run_time - avg_runtime) / avg_runtime
        return norm_diff_var + norm_total_var + norm_runtime

    def _select_method_for_histogram(
        self,
        current_exact_var: float,
        current_est_var: np.ndarray,
        current_total_var: np.ndarray,
        current_run_time: np.ndarray,
    ) -> str:

        current_diff_var = np.abs(current_exact_var - current_est_var)
        current_avg_diff_var = current_diff_var.mean()
        current_avg_total_var = current_total_var.mean()
        current_avg_run_time = current_run_time.mean()
        min_total_loss = np.inf
        selected_method = ""
        selected_suggested_bin_count = 0
        for method_id, method in enumerate(self.histogram_bin_method_names):
            self.histogram_methods[method]["current_loss"] = self._histogram_loss(
                current_diff_var[method_id],
                current_avg_diff_var,
                current_total_var[method_id],
                current_avg_total_var,
                current_run_time[method_id],
                current_avg_run_time,
            )
            self.histogram_methods[method]["total_loss"] += self.histogram_methods[
                method
            ]["current_loss"]

            if min_total_loss >= self.histogram_methods[method]["total_loss"]:
                # if same loss and less bins, don't save bc higher resolution
                if (
                    self.histogram_methods[method]["suggested_bin_count"]
                    <= selected_suggested_bin_count
                    and min_total_loss == self.histogram_methods[method]["total_loss"]
                ):
                    continue
                min_total_loss = self.histogram_methods[method]["total_loss"]
                selected_method = method
                selected_suggested_bin_count = self.histogram_methods[method][
                    "suggested_bin_count"
                ]

        return selected_method

    def _histogram_to_array(self) -> np.ndarray:
        # Extend histogram to array format
        bin_counts = self._stored_histogram["histogram"]["bin_counts"]
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]
        is_bin_non_zero = bin_counts[:-1] > 0
        bin_left_edge = bin_edges[:-2][is_bin_non_zero]
        hist_to_array = [
            [left_edge] * count
            for left_edge, count in zip(bin_left_edge, bin_counts[:-1][is_bin_non_zero])
        ]
        if not hist_to_array:
            hist_to_array = [[]]

        array_flatten = np.concatenate(
            (
                hist_to_array
                + [[bin_edges[-2]] * int(bin_counts[-1] / 2)]
                + [[bin_edges[-1]] * (bin_counts[-1] - int(bin_counts[-1] / 2))]
            )
        )

        # If we know they are integers, we can limit the data to be as such
        # during conversion
        if not self.__class__.__name__ == "FloatColumn":
            array_flatten = np.round(array_flatten)

        return array_flatten

    def _get_histogram(
        self, values: Union[np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate stored histogram the suggested bin counts for each histogram method.

        Uses np.histogram.

        :param values: input data values
        :type values: Union[np.array, pd.Series]
        :return: bin edges and bin counts
        """
        if len(np.unique(values)) == 1:
            bin_counts = np.array([len(values)])
            if isinstance(values, (np.ndarray, list)):
                unique_value = values[0]
            else:
                unique_value = values.iloc[0]
            bin_edges = np.array([unique_value, unique_value])
            for bin_method in self.histogram_bin_method_names:
                self.histogram_methods[bin_method]["histogram"][
                    "bin_counts"
                ] = bin_counts
                self.histogram_methods[bin_method]["histogram"]["bin_edges"] = bin_edges
                self.histogram_methods[bin_method]["suggested_bin_count"] = 1
        else:
            # if user set the bin count, then use the user set count to
            n_equal_bins = suggested_bin_count = self.min_histogram_bin
            if self.user_set_histogram_bin:
                n_equal_bins = suggested_bin_count = self.user_set_histogram_bin

            if not isinstance(values, np.ndarray):
                values = np.array(values)

            # loop through all methods to get their suggested bin count for
            # reporting
            for i, bin_method in enumerate(self.histogram_bin_method_names):
                if self.user_set_histogram_bin is None:
                    _, suggested_bin_count = histogram_utils._get_bin_edges(
                        values, bin_method, None, None
                    )
                    suggested_bin_count = min(
                        suggested_bin_count, self.max_histogram_bin
                    )
                    n_equal_bins = max(n_equal_bins, suggested_bin_count)
                self.histogram_methods[bin_method]["histogram"]["bin_counts"] = None
                self.histogram_methods[bin_method]["histogram"]["bin_edges"] = None
                self.histogram_methods[bin_method][
                    "suggested_bin_count"
                ] = suggested_bin_count

            # calculate the stored histogram bins
            bin_counts, bin_edges = np.histogram(values, bins=n_equal_bins)
        return bin_counts, bin_edges

    def _merge_histogram(self, values: Union[np.ndarray, pd.Series]) -> None:
        # values is the current array of values,
        # that needs to be updated to the accumulated histogram
        combined_values = np.concatenate([values, self._histogram_to_array()])
        bin_counts, bin_edges = self._get_histogram(combined_values)
        self._stored_histogram["histogram"]["bin_counts"] = bin_counts
        self._stored_histogram["histogram"]["bin_edges"] = bin_edges

    def _update_histogram(self, df_series: pd.Series) -> None:
        """
        Update histogram for each method and the combined method.

        The algorithm 'Follow the best expert' is applied to select the combined
        method:

        N. Cesa-Bianchi and G. Lugosi, Prediction, learning, and games.
        Cambridge University Press, 2006.
        R. D. Kleinberg, A. Niculescu-Mizil, and Y. Sharma, "Regret bounds
        for sleeping experts and bandits," in Proceedings of the 21st Annual
        Conference on Learning Theory - COLT 2008, Helsinki, Finland, 2008,
        pp. 425–436.
        The idea is to select the current best method based on accumulated
        losses up to the current time: all methods are compared using the
        accumulated losses, and the best method with minimal loss is picked

        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :return:
        """
        df_series = df_series.replace([np.inf, -np.inf], np.nan).dropna()
        if df_series.empty:
            return

        if self._has_histogram:
            self._merge_histogram(df_series.tolist())
        else:
            bin_counts, bin_edges = self._get_histogram(df_series)
            self._stored_histogram["histogram"]["bin_counts"] = bin_counts
            self._stored_histogram["histogram"]["bin_edges"] = bin_edges

        # update loss for the stored bins
        histogram_loss = self._histogram_bin_error(df_series)

        self._stored_histogram["current_loss"] = histogram_loss
        self._stored_histogram["total_loss"] += histogram_loss

    def _histogram_for_profile(
        self, histogram_method: str
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Convert the stored histogram into the presentable state.

        Based on the suggested histogram bin count from numpy.histograms.
        The bin count used is stored in 'suggested_bin_count' for each method.

        :param histogram_method: method to use for determining the histogram
            profile
        :type histogram_method: str
        :return: histogram bin edges and bin counts
        :rtype: dict
        """
        bin_counts, bin_edges = (
            self._stored_histogram["histogram"]["bin_counts"],
            self._stored_histogram["histogram"]["bin_edges"],
        )

        current_bin_counts, suggested_bin_count = (
            self.histogram_methods[histogram_method]["histogram"]["bin_counts"],
            self.histogram_methods[histogram_method]["suggested_bin_count"],
        )

        # base case, no need to change if it is already correct
        if not self._has_histogram or current_bin_counts is not None:
            return (
                self.histogram_methods[histogram_method]["histogram"],
                self.histogram_methods[histogram_method]["total_loss"],
            )
        elif len(bin_counts) == suggested_bin_count:
            return (
                self._stored_histogram["histogram"],
                self._stored_histogram["total_loss"],
            )

        # create proper binning
        new_bin_counts = np.zeros((suggested_bin_count,))
        new_bin_edges = np.linspace(
            bin_edges[0], bin_edges[-1], suggested_bin_count + 1
        )

        # allocate bin_counts
        new_bin_id = 0
        hist_loss = 0
        for bin_id, bin_count in enumerate(bin_counts):
            if not bin_count:  # if nothing in bin, nothing to add
                continue

            bin_edge = bin_edges[bin_id : bin_id + 3]

            # if we know not float, we can assume values in bins are integers.
            is_float_profile = self.__class__.__name__ == "FloatColumn"
            if not is_float_profile:
                bin_edge = np.round(bin_edge)

            # loop until we have a new bin which contains the current bin.
            while (
                bin_edge[0] >= new_bin_edges[new_bin_id + 1]
                and new_bin_id < suggested_bin_count - 1
            ):
                new_bin_id += 1

            new_bin_edge = new_bin_edges[new_bin_id : new_bin_id + 3]

            # find where the current bin falls within the new bins
            is_last_bin = new_bin_id == suggested_bin_count - 1
            if bin_edge[1] < new_bin_edge[1] or is_last_bin:
                # current bin is within the new bin
                new_bin_counts[new_bin_id] += bin_count
                hist_loss += (
                    ((new_bin_edge[1] + new_bin_edge[0]) - (bin_edge[1] + bin_edge[0]))
                    / 2
                ) ** 2 * bin_count
            elif bin_edge[0] < new_bin_edge[1]:
                # current bin straddles two of the new bins
                # get the percentage of bin that falls to the left
                percentage_in_left_bin = (new_bin_edge[1] - bin_edge[0]) / (
                    bin_edge[1] - bin_edge[0]
                )
                count_in_left_bin = round(bin_count * percentage_in_left_bin)
                new_bin_counts[new_bin_id] += count_in_left_bin
                hist_loss += (
                    ((new_bin_edge[1] + new_bin_edge[0]) - (bin_edge[1] + bin_edge[0]))
                    / 2
                ) ** 2 * count_in_left_bin

                # allocate leftovers to the right bin
                new_bin_counts[new_bin_id + 1] += bin_count - count_in_left_bin
                hist_loss += (
                    ((new_bin_edge[2] - new_bin_edge[1]) - (bin_edge[1] - bin_edge[0]))
                    / 2
                ) ** 2 * (bin_count - count_in_left_bin)

                # increment bin id to the right bin
                new_bin_id += 1

        return ({"bin_edges": new_bin_edges, "bin_counts": new_bin_counts}, hist_loss)

    def _get_best_histogram_for_profile(self) -> Dict:
        """
        Convert the stored histogram into the presentable state.

        Based on the suggested histogram bin count from numpy.histograms.
        The bin count used is stored in 'suggested_bin_count' for each method.

        :return: histogram bin edges and bin counts
        :rtype: dict
        """
        if self.histogram_selection is None:
            best_hist_loss = None
            for method in self.histogram_methods:
                histogram, hist_loss = self._histogram_for_profile(method)
                self.histogram_methods[method]["histogram"] = histogram
                self.histogram_methods[method]["current_loss"] = hist_loss
                self.histogram_methods[method]["total_loss"] += hist_loss
                if not best_hist_loss or hist_loss < best_hist_loss:
                    self.histogram_selection = method
                    best_hist_loss = hist_loss

        return self.histogram_methods[self.histogram_selection]["histogram"]

    def _get_percentile(
        self, percentiles: Union[np.ndarray, List[float]]
    ) -> List[float]:
        """
        Get value for the number where the given percentage of values fall below it.

        :param percentiles: List of percentage of values to fall before the
            value
        :type percentiles: Union[np.ndarray, list[float]]
        :return: List of corresponding values for which the percentage of values
            in the distribution fall before each percentage
        """
        percentiles = np.array(percentiles)
        bin_counts = self._stored_histogram["histogram"]["bin_counts"]
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]

        zero_inds = bin_counts == 0

        bin_counts = bin_counts.astype(float)
        normalized_bin_counts = bin_counts / np.sum(bin_counts)
        cumsum_bin_counts = np.cumsum(normalized_bin_counts)

        median_value = None
        median_bin_inds = np.abs(cumsum_bin_counts - 0.5) < 1e-10
        if np.sum(median_bin_inds) > 1:
            median_value = np.mean(bin_edges[np.append([False], median_bin_inds)])

        # use the floor by slightly increasing cases where no bin exist.
        cumsum_bin_counts[zero_inds] += 1e-15

        # add initial zero bin
        cumsum_bin_counts = np.append([0], cumsum_bin_counts)

        quantiles: np.ndarray = np.interp(
            percentiles / 100, cumsum_bin_counts, bin_edges
        )
        if median_value:
            quantiles[percentiles == 50] = median_value
        return quantiles.tolist()

    @staticmethod
    def _fold_histogram(
        bin_counts: np.ndarray, bin_edges: np.ndarray, value: float
    ) -> Tuple[List[Union[np.ndarray, List]], List[Union[np.ndarray, List]]]:
        """
        Offset histogram by given value, then fold the histogram at break point.

        :param bin_counts: bin counts of the histogram
        :type bin_counts: np.array
        :param bin_edges: bin edges of the histogram
        :type bin_edges: np.array
        :param value: offset value
        :type value: float
        :return: two histograms represented by bin counts and bin edges
        """
        # normalize bin counts
        bin_counts = bin_counts.astype(float)
        normalized_bin_counts: np.ndarray = bin_counts / np.sum(bin_counts)
        bin_edges = bin_edges - value

        # find the break point to fold the deviation list
        id_zero = None
        for i in range(len(bin_edges)):
            if bin_edges[i] > 0:
                id_zero = i
                break

        # if all bin edges are positive or negative (no break point)
        if id_zero is None:
            return [[], []], [normalized_bin_counts, bin_edges]
        if id_zero == 1:
            return [normalized_bin_counts, bin_edges], [[], []]

        # otherwise, generate two folds of deviation
        bin_counts_pos: np.ndarray = np.append(
            [
                normalized_bin_counts[id_zero - 1]
                * (bin_edges[id_zero] / (bin_edges[id_zero] - bin_edges[id_zero - 1]))
            ],
            normalized_bin_counts[id_zero:],
        )
        bin_edges_pos: np.ndarray = np.append([0], bin_edges[id_zero:])

        bin_counts_neg: np.ndarray = np.append(
            [
                normalized_bin_counts[id_zero - 1]
                - normalized_bin_counts[id_zero - 1]
                * (bin_edges[id_zero] / (bin_edges[id_zero] - bin_edges[id_zero - 1]))
            ],
            normalized_bin_counts[: id_zero - 1][::-1],
        )
        bin_edges_neg: np.ndarray = np.append([0], -bin_edges[:id_zero][::-1])

        if len(bin_edges_neg) > 1 and bin_edges_neg[1] == 0:
            bin_edges_neg = bin_edges_neg[1:]
            bin_counts_neg = bin_counts_neg[1:]
        return [bin_counts_pos, bin_edges_pos], [bin_counts_neg, bin_edges_neg]

    @property
    def median_abs_deviation(self) -> float:
        """
        Get median absolute deviation estimated from the histogram of the data.

            Subtract bin edges from the median value
            Fold the histogram to positive and negative parts around zero
            Impose the two bin edges from the two histogram
            Calculate the counts for the two histograms with the imposed bin edges
            Superimpose the counts from the two histograms
            Interpolate the median absolute deviation from the superimposed counts

        :return: median absolute deviation
        """
        if not self._has_histogram or not self._median_abs_dev_is_enabled:
            return np.nan

        bin_counts = self._stored_histogram["histogram"]["bin_counts"]
        bin_edges = self._stored_histogram["histogram"]["bin_edges"]

        if self._median_is_enabled:
            median = self.median
        else:
            median = self._get_percentile([50])[0]

        # generate two folds of deviation
        histogram_pos, histogram_neg = self._fold_histogram(
            bin_counts, bin_edges, median
        )
        bin_counts_pos, bin_edges_pos = histogram_pos[0], histogram_pos[1]
        bin_counts_neg, bin_edges_neg = histogram_neg[0], histogram_neg[1]

        # if all bin edges are positive or negative (no break point),
        # the median value is actually 0
        if len(bin_counts_pos) == 0 or len(bin_counts_neg) == 0:
            return 0

        # otherwise, superimpose the two histogram and interpolate
        # the median at cumsum count 0.5
        bin_edges_impose: Any = (bin_edges_pos, bin_edges_neg)
        if bin_edges_pos[1] > bin_edges_neg[1]:
            bin_edges_impose = (bin_edges_neg, bin_edges_pos)
        bin_edges_impose = np.array(
            [
                x
                for x in itertools.chain(*itertools.zip_longest(*bin_edges_impose))
                if x is not None
            ][1:]
        )

        bin_edges_impose = bin_edges_impose[
            np.append([True], np.diff(bin_edges_impose) > 1e-14)
        ]

        bin_counts_impose_pos: float = np.interp(
            bin_edges_impose, bin_edges_pos, np.cumsum(np.append([0], bin_counts_pos))
        )
        bin_counts_impose_neg: float = np.interp(
            bin_edges_impose, bin_edges_neg, np.cumsum(np.append([0], bin_counts_neg))
        )
        bin_counts_impose = bin_counts_impose_pos + bin_counts_impose_neg

        median_inds = np.abs(bin_counts_impose - 0.5) < 1e-10
        if np.sum(median_inds) > 1:
            return np.mean(bin_edges_impose[median_inds])

        return np.interp(0.5, bin_counts_impose, bin_edges_impose)

    def _get_quantiles(self) -> None:
        """
        Retrieve quantile set based on specified number of quantiles in self.quantiles.

        :return: list of quantiles
        """
        percentiles: np.ndarray = np.linspace(0, 100, len(self.quantiles) + 2)[1:-1]
        self.quantiles = self._get_percentile(percentiles=percentiles)

    def _update_helper(self, df_series_clean: pd.Series, profile: Dict) -> None:
        """
        Update base numerical profile properties w/ clean dataset and known null params.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: numerical profile dictionary
        :type profile: dict
        :return: None
        """
        if df_series_clean.empty:
            return

        prev_dependent_properties = {
            "mean": self.mean,
            "biased_variance": self._biased_variance,
            "biased_skewness": self._biased_skewness,
            "biased_kurtosis": self._biased_kurtosis,
        }
        subset_properties = copy.deepcopy(profile)
        df_series_clean = df_series_clean.astype(float)
        super(NumericStatsMixin, self)._perform_property_calcs(
            self.__calculations,
            df_series=df_series_clean,
            prev_dependent_properties=prev_dependent_properties,
            subset_properties=subset_properties,
        )
        if len(self._batch_history) == 5:
            self._batch_history.pop(0)
        self._batch_history.append(subset_properties)

    @BaseColumnProfiler._timeit(name="min")
    def _get_min(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        min_value = df_series.min()
        self.min = min_value if not self.min else min(self.min, min_value)
        subset_properties["min"] = min_value

    @BaseColumnProfiler._timeit(name="max")
    def _get_max(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        max_value = df_series.max()
        self.max = max_value if not self.max else max(self.max, max_value)
        subset_properties["max"] = max_value

    @BaseColumnProfiler._timeit(name="sum")
    def _get_sum(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        if np.isinf(self.sum) or (np.isnan(self.sum) and self.match_count > 0):
            return

        sum_value = df_series.sum()
        if np.isinf(sum_value) or (len(df_series) > 0 and np.isnan(sum_value)):
            warnings.warn(
                "Infinite or invalid values found in data. "
                "Future statistics (mean, variance, skewness, kurtosis) "
                "will not be computed.",
                RuntimeWarning,
            )

        subset_properties["sum"] = sum_value
        self.sum = self.sum + sum_value

    @BaseColumnProfiler._timeit(name="variance")
    def _get_variance(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        if np.isinf(self._biased_variance) or (
            np.isnan(self._biased_variance) and self.match_count > 0
        ):
            return

        # Suppress any numpy warnings as we have a custom warning for invalid
        # or infinite data already
        with np.errstate(all="ignore"):
            batch_biased_variance = np.var(df_series)  # Obtains biased variance
        subset_properties["biased_variance"] = batch_biased_variance
        sum_value = subset_properties["sum"]
        batch_count = subset_properties["match_count"]
        batch_mean = 0.0 if not batch_count else float(sum_value) / batch_count
        subset_properties["mean"] = batch_mean
        self._biased_variance = self._merge_biased_variance(
            self.match_count,
            self._biased_variance,
            prev_dependent_properties["mean"],
            batch_count,
            batch_biased_variance,
            batch_mean,
        )

    @BaseColumnProfiler._timeit(name="skewness")
    def _get_skewness(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        """
        Compute and update skewness of current dataset given new chunk.

        :param df_series: incoming data
        :type df_series: pandas series
        :param prev_dependent_properties: pre-update values needed
            for computation
        :type prev_dependent_properties: dict
        :param subset_properties: incoming data statistics
        :type subset_properties: dict
        :return None
        """
        # If skewness is still NaN but has a valid match count, this
        # must mean that there were previous invalid values in
        # the dataset.
        if np.isinf(self._biased_skewness) or (
            np.isnan(self._biased_skewness) and self.match_count > 0
        ):
            return

        batch_biased_skewness = utils.biased_skew(df_series)
        subset_properties["biased_skewness"] = batch_biased_skewness
        batch_count = subset_properties["match_count"]
        batch_biased_var = subset_properties["biased_variance"]
        batch_mean = subset_properties["mean"]

        self._biased_skewness = self._merge_biased_skewness(
            self.match_count,
            self._biased_skewness,
            prev_dependent_properties["biased_variance"],
            prev_dependent_properties["mean"],
            batch_count,
            batch_biased_skewness,
            batch_biased_var,
            batch_mean,
        )

    @BaseColumnProfiler._timeit(name="kurtosis")
    def _get_kurtosis(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        """
        Compute and update kurtosis of current dataset given new chunk.

        :param df_series: incoming data
        :type df_series: pandas series
        :param prev_dependent_properties: pre-update values needed
            for computation
        :type prev_dependent_properties: dict
        :param subset_properties: incoming data statistics
        :type subset_properties: dict
        :return None
        """
        # If kurtosis is still NaN but has a valid match count, this
        # must mean that there were previous invalid values in
        # the dataset.
        if np.isinf(self._biased_kurtosis) or (
            np.isnan(self._biased_kurtosis) and self.match_count > 0
        ):
            return

        batch_biased_kurtosis = utils.biased_kurt(df_series)
        subset_properties["biased_kurtosis"] = batch_biased_kurtosis
        batch_count = subset_properties["match_count"]
        batch_biased_var = subset_properties["biased_variance"]
        batch_biased_skewness = subset_properties["biased_skewness"]
        batch_mean = subset_properties["mean"]

        self._biased_kurtosis = self._merge_biased_kurtosis(
            self.match_count,
            self._biased_kurtosis,
            prev_dependent_properties["biased_skewness"],
            prev_dependent_properties["biased_variance"],
            prev_dependent_properties["mean"],
            batch_count,
            batch_biased_kurtosis,
            batch_biased_skewness,
            batch_biased_var,
            batch_mean,
        )

    @BaseColumnProfiler._timeit(name="histogram_and_quantiles")
    def _get_histogram_and_quantiles(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        try:
            self._update_histogram(df_series)
            self.histogram_selection = None
            if self._has_histogram:
                self._get_quantiles()
        except BaseException:
            warnings.warn(
                "Histogram error. Histogram and quantile results will not be "
                "available"
            )

    @BaseColumnProfiler._timeit(name="num_zeros")
    def _get_num_zeros(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        """
        Get the count of zeros in the numerical column.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :param prev_dependent_properties: previous dependent properties
        :type prev_dependent_properties: dict
        :param subset_properties: subset of properties
        :type subset_properties: dict
        :return: None
        """
        num_zeros_value = (df_series == 0).sum()
        subset_properties["num_zeros"] = num_zeros_value
        self.num_zeros = self.num_zeros + num_zeros_value

    @BaseColumnProfiler._timeit(name="num_negatives")
    def _get_num_negatives(
        self,
        df_series: pd.Series,
        prev_dependent_properties: Dict,
        subset_properties: Dict,
    ) -> None:
        """
        Get the count of negative numbers in the numerical column.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :param prev_dependent_properties: previous dependent properties
        :type prev_dependent_properties: dict
        :param subset_properties: subset of properties
        :type subset_properties: dict
        :return: None
        """
        num_negatives_value = (df_series < 0).sum()
        subset_properties["num_negatives"] = num_negatives_value
        self.num_negatives = self.num_negatives + num_negatives_value

    @abc.abstractmethod
    def update(self, df_series: pd.Series) -> NumericStatsMixin:
        """
        Update the numerical profile properties with an uncleaned dataset.

        :param df_series: df series with nulls removed
        :type df_series: pandas.core.series.Series
        :return: None
        """
        raise NotImplementedError()

    @staticmethod
    def is_float(x: str) -> bool:
        """
        Return True if x is float.

        For "0.80" this function returns True
        For "1.00" this function returns True
        For "1" this function returns True

        :param x: string to test
        :type x: str
        :return: if is float or not
        :rtype: bool
        """
        try:
            float(x)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def is_int(x: str) -> bool:
        """
        Return True if x is integer.

        For "0.80" This function returns False
        For "1.00" This function returns True
        For "1" this function returns True

        :param x: string to test
        :type x: str
        :return: if is integer or not
        :rtype: bool
        """
        try:
            a = float(x)
            b = int(a)
        except (ValueError, OverflowError, TypeError):
            return False
        else:
            return a == b

    @staticmethod
    def np_type_to_type(val: Any) -> Union[int, float]:
        """
        Convert numpy variables to base python type variables.

        :param val: value to check & change
        :type val: numpy type or base type
        :return val: base python type
        :rtype val: int or float
        """
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        return val
