#!/usr/bin/env python
"""
coding=utf-8
Build model for a dataset by identifying type of column along with its
respective parameters.
"""
from __future__ import print_function
from __future__ import division

from future.utils import with_metaclass
import copy
import time
import abc
import warnings

import numpy as np

from . import histogram_utils
from .base_column_profilers import BaseColumnProfiler
from .profiler_options import NumericalOptions


class abstractstaticmethod(staticmethod):

    __slots__ = ()

    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class NumericStatsMixin(with_metaclass(abc.ABCMeta, object)):
    """
    Abstract numerical column profile subclass of BaseColumnProfiler. Represents
    a column in the dataset which is a text column. Has Subclasses itself.
    """
    col_type = None

    def __init__(self, options=None):
        """
        Initialization of column base properties and itself.

        :param options: Options for the numerical stats.
        :type options: NumericalOptions
        """
        self.options = None
        if options and isinstance(options, NumericalOptions):
            self.options = options
        self.min = None
        self.max = None
        self.sum = 0
        self.variance = 0
        self.max_histogram_bin = 10000
        self.histogram_bin_method_names = ['auto', 'fd', 'doane', 'scott',
                                           'rice', 'sturges', 'sqrt']
        self.histogram_methods = {}
        for method in self.histogram_bin_method_names:
            self.histogram_methods[method] = {
                'total_loss': 0,
                'current_loss': 0,
                'histogram': {
                    'bin_counts': None,
                    'bin_edges': None
                }
            }
        self.histogram_selection = None

        self.quantiles = {
            bin_num: None for bin_num in range(1000)
        }

        self.__calculations = {
            "min": NumericStatsMixin._get_min,
            "max": NumericStatsMixin._get_max,
            "sum": NumericStatsMixin._get_sum,
            "variance": NumericStatsMixin._get_variance,
            "histogram_and_quantiles":
                NumericStatsMixin._get_histogram_and_quantiles
        }

        self._filter_properties_w_options(self.__calculations, options)

    def __getattribute__(self, name):
        return super(NumericStatsMixin, self).__getattribute__(name)

    def __getitem__(self, item):
        return super(NumericStatsMixin, self).__getitem__(item)

    @BaseColumnProfiler._timeit(name="histogram_and_quantiles")
    def _add_helper_merge_profile_histograms(self, other1, other2):
        """
        Adds histogram of two profiles together

        :param other1: profile1 being added to self
        :type other1: BaseColumnProfiler
        :param other2: profile2 being added to self
        :type other2: BaseColumnProfiler
        :return: None
        """
        # get available bin methods and set to current
        bin_methods = list(set(other1.histogram_bin_method_names) &
                           set(other2.histogram_bin_method_names))
        if not bin_methods:
            raise ValueError('Profiles have no overlapping bin methods and '
                             'therefore cannot be added together.')
        self.histogram_bin_method_names = bin_methods

        for i, method in enumerate(self.histogram_bin_method_names):
            combined_values = other1._histogram_to_array(
                method) + other2._histogram_to_array(method)
            bin_counts, bin_edges = self._get_histogram(
                combined_values, method)
            self.histogram_methods[method]['histogram']['bin_counts'] = \
                bin_counts
            self.histogram_methods[method]['histogram']['bin_edges'] = bin_edges

        # Select histogram: always choose first profile selected method
        # Either both profiles have the same selection or you at least use one
        # of the profiles selected method
        self.histogram_selection = other1.histogram_selection
        self._get_quantiles()

    def _add_helper(self, other1, other2):
        """
        Helper function for merging profiles.

        :param other1: profile1 being added to self
        :param other2: profile2 being added to self
        :return: None
        """
        # Merge Variance
        self.variance = self._merge_variance(
            other1.match_count, other1.variance, other1.mean,
            other2.match_count, other2.variance, other2.mean)

        # Merge min, max, sum and histograms
        if other1.min is not None and other2.min is not None:
            # update histogram
            self._add_helper_merge_profile_histograms(other1, other2)

            # update min, max, sum
            self.min = min(other1.min, other2.min)
            self.max = max(other1.max, other2.max)
            self.sum = other1.sum + other2.sum

        elif other2.min is None:
            # update histogram
            self.histogram_methods = other1.histogram_methods
            self.quantiles = other1.quantiles

            # update min, max, sum
            self.min = other1.min
            self.max = other1.max
            self.sum = other1.sum
        else:
            # update histogram
            self.histogram_methods = other2.histogram_methods
            self.quantiles = other2.quantiles

            # update min, max, sum
            self.min = other2.min
            self.max = other2.max
            self.sum = other2.sum

    @property
    def mean(self):
        if self.match_count == 0:
            return 0
        return float(self.sum) / self.match_count

    @property
    def stddev(self):
        if self.match_count == 0:
            return np.nan
        return np.sqrt(self.variance)

    def _update_variance(self, batch_mean, batch_var, batch_count):
        """
        Calculate the combined variance of the current values and new dataset.

        :param batch_mean: mean of new chunk
        :param batch_var: variance of new chunk
        :param batch_count: number of samples in new chunk
        :return: combined variance
        :rtype: float
        """
        return self._merge_variance(self.match_count, self.variance, self.mean,
                                    batch_count, batch_var, batch_mean)

    @staticmethod
    def _merge_variance(match_count1, variance1, mean1,
                        match_count2, variance2, mean2):
        """
        Calculate the combined variance of the current values and new dataset.

        :param match_count1: number of samples in new chunk 1
        :param mean1: mean of chunk 1
        :param variance1: variance of chunk 1
        :param match_count2: number of samples in new chunk 2
        :param mean2: mean of chunk 2
        :param variance2: variance of chunk 2
        :return: combined variance
        :rtype: float
        """
        if np.isnan(variance1):
            variance1 = 0
        if np.isnan(variance2):
            variance2 = 0
        if match_count1 < 1:
            return variance2
        elif match_count2 < 1:
            return variance1

        curr_count = match_count1
        delta = mean2 - mean1
        m_curr = variance1 * (curr_count - 1)
        m_batch = variance2 * (match_count2 - 1)
        M2 = m_curr + m_batch + delta ** 2 * curr_count * match_count2 / \
            (curr_count + match_count2)
        new_variance = M2 / (curr_count + match_count2 - 1)
        return new_variance

    def _estimate_stats_from_histogram(self, method):
        # test estimated mean and var
        bin_counts = self.histogram_methods[method]['histogram']['bin_counts']
        bin_edges = self.histogram_methods[method]['histogram']['bin_edges']
        mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mean = np.average(mids, weights=bin_counts)
        var = np.average((mids - mean) ** 2, weights=bin_counts)
        std = np.sqrt(var)
        return mean, var, std

    def _total_histogram_bin_variance(self, input_array, method):
        # calculate total variance over all bins of a histogram
        bin_edges = self.histogram_methods[method]['histogram']['bin_edges']
        inds = np.digitize(input_array, bin_edges)
        sum_var = 0
        for i in range(1, len(bin_edges)):
            elements_in_bin = input_array[inds == i]
            bin_var = elements_in_bin.var() if len(elements_in_bin) > 0 else 0
            sum_var += bin_var
        return sum_var

    @staticmethod
    def _histogram_loss(diff_var, avg_diffvar, total_var,
                        avg_totalvar, run_time, avg_runtime):

        norm_diff_var, norm_total_var, norm_runtime = 0, 0, 0
        if avg_diffvar > 0:
            norm_diff_var = float(diff_var - avg_diffvar) / avg_diffvar
        if avg_totalvar > 0:
            norm_total_var = float(total_var - avg_totalvar) / avg_totalvar
        penalized_time = 1  # currently set as 1s
        if (run_time - avg_runtime) >= penalized_time:
            norm_runtime = float(run_time - avg_runtime) / avg_runtime
        return norm_diff_var + norm_total_var + norm_runtime

    def _select_method_for_histogram(self, current_exact_var, current_est_var,
                                     current_total_var, current_run_time):

        current_diff_var = np.abs(current_exact_var - current_est_var)
        current_avg_diff_var = current_diff_var.mean()
        current_avg_total_var = current_total_var.mean()
        current_avg_run_time = current_run_time.mean()
        min_total_loss = np.inf
        selected_method = ''
        for method_id, method in enumerate(self.histogram_bin_method_names):
            self.histogram_methods[method]['current_loss'] = \
                self._histogram_loss(current_diff_var[method_id],
                                     current_avg_diff_var,
                                     current_total_var[method_id],
                                     current_avg_total_var,
                                     current_run_time[method_id],
                                     current_avg_run_time)
            self.histogram_methods[method]['total_loss'] += \
                self.histogram_methods[method]['current_loss']

            if min_total_loss > self.histogram_methods[method]['total_loss']:
                min_total_loss = self.histogram_methods[method]['total_loss']
                selected_method = method

        return selected_method

    def _histogram_to_array(self, bins):
        # Extend histogram to array format
        bin_counts = self.histogram_methods[bins]['histogram']['bin_counts']
        bin_edges = self.histogram_methods[bins]['histogram']['bin_edges']
        hist_to_array = [[bin_edge] * bin_count for bin_count, bin_edge in
                         zip(bin_counts[:-1], bin_edges[:-2])]
        hist_to_array.append([bin_edges[-2]] * int(bin_counts[-1] / 2))
        hist_to_array.append([bin_edges[-1]] *
                             (bin_counts[-1] - int(bin_counts[-1] / 2)))
        array_flatten = [element for sublist in hist_to_array for
                         element in sublist]
        return array_flatten

    def _get_histogram(self, values, bin_method):
        """
        Get histogram from values and bin method, using np.histogram
        :param values: input values
        :type values: np.array or pd.Series
        :param bin_method: bin method, e.g., sqrt, rice, etc
        :type bin_method: str
        :return: bin edges and bin counts
        """
        if len(np.unique(values)) == 1:
            bin_counts = np.array([len(values)])
            if isinstance(values, (np.ndarray, list)):
                unique_value = values[0]
            else:
                unique_value = values.iloc[0]
            bin_edges = np.array([unique_value, unique_value])
        else:
            values, weights = histogram_utils._ravel_and_check_weights(
                values, None)
            _, n_equal_bins = histogram_utils._get_bin_edges(
                values, bin_method, None, None)
            n_equal_bins = min(n_equal_bins, self.max_histogram_bin)
            bin_counts, bin_edges = np.histogram(values, bins=n_equal_bins)
        return bin_counts, bin_edges

    def _merge_histogram(self, values, bins):
        # values is the current array of values,
        # that needs to be updated to the accumulated histogram
        combined_values = values + self._histogram_to_array(bins)
        bin_counts, bin_edges = self._get_histogram(combined_values, bins)
        self.histogram_methods[bins]['histogram']['bin_counts'] = bin_counts
        self.histogram_methods[bins]['histogram']['bin_edges'] = bin_edges

    def _update_histogram(self, df_series):
        """
        Update histogram for each method and the combined method. The algorithm
        'Follow the best expert' is applied to select the combined method:
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

        current_est_var = np.zeros(len(self.histogram_bin_method_names))
        current_exact_var = np.zeros(len(self.histogram_bin_method_names))
        current_total_var = np.zeros(len(self.histogram_bin_method_names))
        current_run_time = np.zeros(len(self.histogram_bin_method_names))
        for i, method in enumerate(self.histogram_bin_method_names):
            # update histogram for the method
            start_time = time.time()
            bin_counts, bin_edges = self._get_histogram(df_series, method)
            if self.histogram_methods[method]['histogram']['bin_counts'] is None:
                self.histogram_methods[method]['histogram']['bin_counts'] = bin_counts
                self.histogram_methods[method]['histogram']['bin_edges'] = bin_edges
            else:
                self._merge_histogram(df_series.tolist(), bins=method)
            run_time = time.time() - start_time
            # update loss for the method
            current_est_var[i] = self._estimate_stats_from_histogram(method)[1]
            current_exact_var = df_series.values.var()
            current_total_var[i] = self._total_histogram_bin_variance(
                df_series.values, method)
            current_run_time[i] = run_time

        # select the best method and update the total loss
        selected_method = self._select_method_for_histogram(
            current_exact_var, current_est_var,
            current_total_var, current_run_time)
        self.histogram_selection = selected_method

    def _get_percentile(self, percentile):
        """
        Get value for the number where the given percentage of values fall below
        it.

        :param percentile: Percentage of values to fall before the value
        :type percentile: float
        :return: Value for which the percentage of values in the distribution
            fall before the percentage
        """
        selected_method = self.histogram_selection
        bin_counts = \
            self.histogram_methods[selected_method]['histogram']['bin_counts']
        bin_edges = \
            self.histogram_methods[selected_method]['histogram']['bin_edges']
        num_edges = len(bin_edges)

        if percentile == 100:
            return bin_edges[-1]
        percentile = float(percentile) / 100

        accumulated_count = 0
        bin_counts = bin_counts.astype(float)
        normalized_bin_counts = bin_counts / np.sum(bin_counts)

        bin_id = -1
        # keep updating the total counts until it is
        # close to the designated percentile
        while accumulated_count < percentile:
            bin_id += 1
            accumulated_count += normalized_bin_counts[bin_id]

        if accumulated_count == percentile:
            if (num_edges % 2) == 0:
                return 0.5 * (bin_edges[bin_id] + bin_edges[bin_id + 1])
            else:
                return bin_edges[bin_id + 1]
        else:
            if bin_id == 0:
                return 0.5 * (bin_edges[0] + bin_edges[1])
            if (num_edges % 2) == 0:
                return 0.5 * (bin_edges[bin_id - 1] + bin_edges[bin_id])
            else:
                return bin_edges[bin_id]

    def _get_quantiles(self):
        """
        Retrieves the quantile set based on the specified number of quantiles
        in self.quantiles.

        :return: list of quantiles
        """
        size_bins = 100 / len(self.quantiles)
        for bin_num in range(len(self.quantiles) - 1):
            self.quantiles[bin_num] = self._get_percentile(
                percentile=((bin_num + 1) * size_bins))

    def _update_helper(self, df_series_clean, profile):
        """
        Method for updating the base numerical profile properties with a cleaned
        dataset and the known null parameters of the dataset.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: numerical profile dictionary
        :type profile: dict
        :return: None
        """
        if df_series_clean.empty:
            return

        prev_dependent_properties = {"mean": self.mean}
        subset_properties = copy.deepcopy(profile)
        df_series_clean = df_series_clean.astype(float)
        super(NumericStatsMixin, self)._perform_property_calcs(self.__calculations,
                                     df_series=df_series_clean,
                                     prev_dependent_properties=prev_dependent_properties,
                                     subset_properties=subset_properties)

    @BaseColumnProfiler._timeit(name="min")
    def _get_min(self, df_series, prev_dependent_properties,
                 subset_properties):
        min_value = df_series.min()
        self.min = min_value if not self.min else min(self.min, min_value)
        subset_properties["min"] = min_value

    @BaseColumnProfiler._timeit(name="max")
    def _get_max(self, df_series, prev_dependent_properties,
                 subset_properties):
        max_value = df_series.max()
        self.max = max_value if not self.max else max(self.max, max_value)
        subset_properties["max"] = max_value

    @BaseColumnProfiler._timeit(name="sum")
    def _get_sum(self, df_series, prev_dependent_properties,
                 subset_properties):
        sum_value = df_series.sum()
        subset_properties["sum"] = sum_value
        self.sum = self.sum + sum_value

    @BaseColumnProfiler._timeit(name="variance")
    def _get_variance(self, df_series, prev_dependent_properties,
                      subset_properties):
        variance = df_series.var()
        subset_properties["variance"] = variance
        sum_value = subset_properties["sum"]
        batch_count = subset_properties["match_count"]
        batch_mean = 0. if not batch_count else \
            float(sum_value) / batch_count
        self.variance = self._merge_variance(self.match_count, self.variance,
                                             prev_dependent_properties["mean"],
                                             batch_count,
                                             variance,
                                             batch_mean)

    @BaseColumnProfiler._timeit(name="histogram_and_quantiles")
    def _get_histogram_and_quantiles(self, df_series,
                                     prev_dependent_properties,
                                     subset_properties):
        try:
            self._update_histogram(df_series)
            if self.histogram_selection is not None:
                self._get_quantiles()
        except BaseException:
            warnings.warn(
                'Histogram error. Histogram and quantile results will not be '
                'available')

    @abc.abstractmethod
    def update(self, df_series):
        """
        Abstract Method for updating the numerical profile properties with an
        uncleaned dataset.

        :param df_series: df series with nulls removed
        :type df_series: pandas.core.series.Series
        :return: None
        """
        raise NotImplementedError()

    @staticmethod
    def is_float(x):
        """
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
    def is_int(x):
        """
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
