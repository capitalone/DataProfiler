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
        if options and not isinstance(options, NumericalOptions):
            raise ValueError("NumericalStatsMixin parameter 'options' must be "
                             "of type NumericalOptions.")
        self.min = None
        self.max = None
        self.sum = 0
        self.variance = 0
        self.max_histogram_bin = 100000
        self.min_histogram_bin = 1000
        self.histogram_bin_method_names = [
            'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'
        ]
        self.histogram_selection = None
        self.user_set_histogram_bin = None
        if options:
            bin_count_or_method = \
                options.histogram_and_quantiles.bin_count_or_method
            if isinstance(bin_count_or_method, str):
                self.histogram_bin_method_names = [bin_count_or_method]
            elif isinstance(bin_count_or_method, list):
                self.histogram_bin_method_names = bin_count_or_method
            elif isinstance(bin_count_or_method, int):
                self.user_set_histogram_bin = bin_count_or_method
                self.histogram_bin_method_names = ['custom']
        self.histogram_methods = {}
        self._stored_histogram = {
                'total_loss': 0,
                'current_loss': 0,
                'suggested_bin_count': self.min_histogram_bin,
                'histogram': {
                    'bin_counts': None,
                    'bin_edges': None
                }
            }
        for method in self.histogram_bin_method_names:
            self.histogram_methods[method] = {
                'total_loss': 0,
                'current_loss': 0,
                'suggested_bin_count': self.min_histogram_bin,
                'histogram': {
                    'bin_counts': None,
                    'bin_edges': None
                }
            }
        num_quantiles = 1000  # TODO: add to options
        self.quantiles = {bin_num: None for bin_num in range(num_quantiles - 1)}
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

    @property
    def _has_histogram(self):
        return self._stored_histogram['histogram']['bin_counts'] is not None

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
        bin_methods = [x for x in other1.histogram_bin_method_names
                       if x in other2.histogram_bin_method_names]
        if not bin_methods:
            raise ValueError('Profiles have no overlapping bin methods and '
                             'therefore cannot be added together.')
        elif other1.user_set_histogram_bin and other2.user_set_histogram_bin:
            if other1.user_set_histogram_bin != other2.user_set_histogram_bin:
                warnings.warn('User set histogram bin counts did not match. '
                              'Choosing the larger bin count.')
            self.user_set_histogram_bin = max(other1.user_set_histogram_bin,
                                              other2.user_set_histogram_bin)

        # initial creation of the profiler creates all methods, but
        # only the methods which intersect should exist.
        self.histogram_bin_method_names = bin_methods
        self.histogram_methods = dict()
        for method in self.histogram_bin_method_names:
            self.histogram_methods[method] = {
                'total_loss': 0,
                'current_loss': 0,
                'histogram': {
                    'bin_counts': None,
                    'bin_edges': None
                }
            }

        combined_values = np.concatenate([other1._histogram_to_array(),
                                          other2._histogram_to_array()])
        bin_counts, bin_edges = self._get_histogram(combined_values)
        self._stored_histogram['histogram']['bin_counts'] = bin_counts
        self._stored_histogram['histogram']['bin_edges'] = bin_edges

        histogram_loss = self._histogram_bin_error(combined_values)
        self._stored_histogram['histogram']['current_loss'] = histogram_loss
        self._stored_histogram['histogram']['total_loss'] = histogram_loss

        self._get_quantiles()

    def _add_helper(self, other1, other2):
        """
        Helper function for merging profiles.

        :param other1: profile1 being added to self
        :param other2: profile2 being added to self
        :return: None
        """

        BaseColumnProfiler._merge_calculations(
            self._NumericStatsMixin__calculations,
            other1._NumericStatsMixin__calculations,
            other2._NumericStatsMixin__calculations)

        # Merge variance, histogram, min, max, and sum
        if "variance" in self.__calculations.keys():
            self.variance = self._merge_variance(
                other1.match_count, other1.variance, other1.mean,
                other2.match_count, other2.variance, other2.mean)
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

    def _estimate_stats_from_histogram(self):
        # test estimated mean and var
        bin_counts = self._stored_histogram['histogram']['bin_counts']
        bin_edges = self._stored_histogram['histogram']['bin_edges']
        mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mean = np.average(mids, weights=bin_counts)
        var = np.average((mids - mean) ** 2, weights=bin_counts)
        return var

    def _total_histogram_bin_variance(self, input_array):
        # calculate total variance over all bins of a histogram
        bin_counts = self._stored_histogram['histogram']['bin_counts']
        bin_edges = self._stored_histogram['histogram']['bin_edges']

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

    def _histogram_bin_error(self, input_array):
        """
        Calculate the error of each value from the bin of the histogram it
        falls within.

        :param input_array: input data used to calculate the histogram
        :type input_array: Union[np.array, pd.Series]
        :return: binning error
        :rtype: float
        """
        bin_counts = self._stored_histogram['histogram']['bin_counts']
        bin_edges = self._stored_histogram['histogram']['bin_edges']

        # account ofr digitize which is exclusive
        bin_edges = bin_edges.copy()
        bin_edges[-1] += 1e-3

        inds = np.digitize(input_array, bin_edges)

        # reset the edge
        bin_edges[-1] -= 1e-3

        sum_error = sum(
            (input_array - (bin_edges[inds] + bin_edges[inds - 1])/2) ** 2
        )
        return sum_error

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
        selected_suggested_bin_count = 0
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

            if min_total_loss >= self.histogram_methods[method]['total_loss']:
                # if same loss and less bins, don't save bc higher resolution
                if (self.histogram_methods[method]['suggested_bin_count']
                        <= selected_suggested_bin_count
                        and min_total_loss ==
                        self.histogram_methods[method]['total_loss']):
                    continue
                min_total_loss = self.histogram_methods[method]['total_loss']
                selected_method = method
                selected_suggested_bin_count = \
                    self.histogram_methods[method]['suggested_bin_count']

        return selected_method

    def _histogram_to_array(self):
        # Extend histogram to array format
        bin_counts = self._stored_histogram['histogram']['bin_counts']
        bin_edges = self._stored_histogram['histogram']['bin_edges']
        is_bin_non_zero = bin_counts[:-1] > 0
        bin_left_edge = bin_edges[:-2][is_bin_non_zero]
        hist_to_array = [
            [left_edge] * count for left_edge, count
            in zip(bin_left_edge, bin_counts[:-1][is_bin_non_zero])
        ]
        if not hist_to_array:
            hist_to_array = [[]]

        array_flatten = np.concatenate(
            (hist_to_array + [[bin_edges[-2]] * int(bin_counts[-1] / 2)] +
            [[bin_edges[-1]] * (bin_counts[-1] - int(bin_counts[-1] / 2))]))

        # If we know they are integers, we can limit the data to be as such
        # during conversion
        if not self.__class__.__name__ == 'FloatColumn':
            array_flatten = np.round(array_flatten)

        return array_flatten

    def _get_histogram(self, values):
        """
        Calculates the stored histogram the suggested bin counts for each
        histogram method, uses np.histogram

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
                self.histogram_methods[bin_method]['histogram'][
                    'bin_counts'] = bin_counts
                self.histogram_methods[bin_method]['histogram'][
                    'bin_edges'] = bin_edges
                self.histogram_methods[bin_method]['suggested_bin_count'] = 1
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
                        values, bin_method, None, None)
                    suggested_bin_count = min(suggested_bin_count,
                                              self.max_histogram_bin)
                    n_equal_bins = max(n_equal_bins, suggested_bin_count)
                self.histogram_methods[bin_method]['histogram'][
                    'bin_counts'] = None
                self.histogram_methods[bin_method]['histogram'][
                    'bin_edges'] = None
                self.histogram_methods[bin_method]['suggested_bin_count'] = \
                    suggested_bin_count

            # calculate the stored histogram bins
            bin_counts, bin_edges = np.histogram(values, bins=n_equal_bins)
        return bin_counts, bin_edges

    def _merge_histogram(self, values):
        # values is the current array of values,
        # that needs to be updated to the accumulated histogram
        combined_values = np.concatenate([values, self._histogram_to_array()])
        bin_counts, bin_edges = self._get_histogram(combined_values)
        self._stored_histogram['histogram']['bin_counts'] = bin_counts
        self._stored_histogram['histogram']['bin_edges'] = bin_edges

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

        if self._has_histogram:
            self._merge_histogram(df_series.tolist())
        else:
            bin_counts, bin_edges = self._get_histogram(df_series)
            self._stored_histogram['histogram']['bin_counts'] = bin_counts
            self._stored_histogram['histogram']['bin_edges'] = bin_edges

        # update loss for the stored bins
        histogram_loss = self._histogram_bin_error(df_series)
        self._stored_histogram['current_loss'] = histogram_loss
        self._stored_histogram['total_loss'] += histogram_loss
        
    def _histogram_for_profile(self, histogram_method):
        """
        Converts the stored histogram into the presentable state based on the
        suggested histogram bin count from numpy.histograms. The bin count used
        is stored in 'suggested_bin_count' for each method.
        
        :param histogram_method: method to use for determining the histogram
            profile
        :type histogram_method: str
        :return: histogram bin edges and bin counts
        :rtype: dict
        """
        bin_counts, bin_edges = (
            self._stored_histogram['histogram']['bin_counts'],
            self._stored_histogram['histogram']['bin_edges'],
        )

        current_bin_counts, suggested_bin_count = (
            self.histogram_methods[histogram_method]['histogram']['bin_counts'],
            self.histogram_methods[histogram_method]['suggested_bin_count'],
        )
        
        # base case, no need to change if it is already correct
        if not self._has_histogram or current_bin_counts is not None:
            return (self.histogram_methods[histogram_method]['histogram'],
                    self.histogram_methods[histogram_method]['total_loss'])
        elif len(bin_counts) == suggested_bin_count:
            return (self._stored_histogram['histogram'],
                    self._stored_histogram['total_loss'])

        # create proper binning
        new_bin_counts = np.zeros((suggested_bin_count,))
        new_bin_edges = np.linspace(
            bin_edges[0], bin_edges[-1], suggested_bin_count + 1)
        
        # allocate bin_counts
        new_bin_id = 0
        hist_loss = 0
        for bin_id, bin_count in enumerate(bin_counts):
            if not bin_count:  # if nothing in bin, nothing to add
                continue

            bin_edge = bin_edges[bin_id: bin_id + 3]

            # if we know not float, we can assume values in bins are integers.
            is_float_profile = self.__class__.__name__ == 'FloatColumn'
            if not is_float_profile:
                bin_edge = np.round(bin_edge)

            # loop until we have a new bin which contains the current bin.
            while (bin_edge[0] >= new_bin_edges[new_bin_id + 1]
                   and new_bin_id < suggested_bin_count - 1):
                new_bin_id += 1

            new_bin_edge = new_bin_edges[new_bin_id: new_bin_id + 3]
            
            # find where the current bin falls within the new bins
            is_last_bin = new_bin_id == suggested_bin_count -1
            if bin_edge[1] < new_bin_edge[1] or is_last_bin:
                # current bin is within the new bin
                new_bin_counts[new_bin_id] += bin_count
                hist_loss += ((
                    (new_bin_edge[1] + new_bin_edge[0])
                    - (bin_edge[1] + bin_edge[0])) / 2) ** 2 * bin_count
            elif bin_edge[0] < new_bin_edge[1]:
                # current bin straddles two of the new bins
                # get the percentage of bin that falls to the left
                percentage_in_left_bin = (
                    (new_bin_edge[1] - bin_edge[0])
                    / (bin_edge[1] - bin_edge[0])
                )
                count_in_left_bin = round(bin_count * percentage_in_left_bin)
                new_bin_counts[new_bin_id] += count_in_left_bin
                hist_loss += ((
                    (new_bin_edge[1] + new_bin_edge[0])
                    - (bin_edge[1] + bin_edge[0])) / 2) ** 2 * count_in_left_bin

                # allocate leftovers to the right bin
                new_bin_counts[new_bin_id + 1] += bin_count - count_in_left_bin
                hist_loss += ((
                    (new_bin_edge[2] - new_bin_edge[1])
                    - (bin_edge[1] - bin_edge[0])
                ) / 2)**2 * (bin_count - count_in_left_bin)

                # increment bin id to the right bin
                new_bin_id += 1

        return ({'bin_edges': new_bin_edges, 'bin_counts': new_bin_counts},
                hist_loss)

    def _get_best_histogram_for_profile(self):
        """
        Converts the stored histogram into the presentable state based on the
        suggested histogram bin count from numpy.histograms. The bin count used
        is stored in 'suggested_bin_count' for each method.

        :return: histogram bin edges and bin counts
        :rtype: dict
        """
        if self.histogram_selection is None:
            best_hist_loss = np.inf
            for method in self.histogram_methods:
                histogram, hist_loss = self._histogram_for_profile(method)
                self.histogram_methods[method]['histogram'] = histogram
                self.histogram_methods[method]['current_loss'] = hist_loss
                self.histogram_methods[method]['total_loss'] += hist_loss
                if hist_loss < best_hist_loss:
                    self.histogram_selection = method
                    best_hist_loss = hist_loss
        return self.histogram_methods[self.histogram_selection]['histogram']

    def _get_percentile(self, percentiles):
        """
        Get value for the number where the given percentage of values fall below
        it.

        :param percentiles: List of percentage of values to fall before the
            value
        :type percentiles: list[float]
        :return: List of corresponding values for which the percentage of values
            in the distribution fall before each percentage
        """
        bin_counts = self._stored_histogram['histogram']['bin_counts']
        bin_edges = self._stored_histogram['histogram']['bin_edges']

        zero_inds = bin_counts == 0

        bin_counts = bin_counts.astype(float)
        normalized_bin_counts = bin_counts / np.sum(bin_counts)
        cumsum_bin_counts = np.cumsum(normalized_bin_counts)

        median_value = None
        median_bin_inds = cumsum_bin_counts == 0.5
        if np.sum(median_bin_inds) > 1:
            median_value = np.mean(bin_edges[np.append([False], median_bin_inds)])

        # use the floor by slightly increasing cases where no bin exist.
        cumsum_bin_counts[zero_inds] += 1e-15

        # add initial zero bin
        cumsum_bin_counts = np.append([0], cumsum_bin_counts)

        quantiles = np.interp(percentiles / 100,
                              cumsum_bin_counts, bin_edges).tolist()
        if median_value:
            quantiles[499] = median_value
        return quantiles

    def _get_quantiles(self):
        """
        Retrieves the quantile set based on the specified number of quantiles
        in self.quantiles.

        :return: list of quantiles
        """
        percentiles = np.linspace(0, 100, len(self.quantiles) + 2)[1:-1]
        self.quantiles = self._get_percentile(
            percentiles=percentiles)

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
            self.histogram_selection = None
            if self._has_histogram:
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

    @staticmethod
    def np_type_to_type(val):
        """
        Converts numpy variables to base python type variables
        
        :param val: value to check & change
        :type val: numpy type or base type
        :return val: base python type
        :rtype val: int or float
        """
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.float):
            return float(val)
        return val
