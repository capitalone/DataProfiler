"""
Histogram-related functions.

As noted below, this file contains modified versions of code from:
https://github.com/numpy/numpy/tree/main

A copy of the license for numpy is available here:
https://github.com/numpy/numpy/blob/main/LICENSE.txt
"""
import operator
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.lib.histograms import (  # type: ignore[attr-defined]
    _get_outer_edges,
    _hist_bin_selectors,
    _unsigned_subtract,
)


def _get_maximum_from_profile(profile):
    """
    Get the maximum of a dataset from a profile.

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: dataset maximum
    """
    return (
        profile.max
        if profile.max is not None
        else profile._stored_histogram["histogram"]["bin_edges"][-1]
    )


def _get_minimum_from_profile(profile):
    """
    Get the minimum of a dataset from a profile.

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: dataset minimum
    """
    return (
        profile.min
        if profile.min is not None
        else profile._stored_histogram["histogram"]["bin_edges"][0]
    )


def _get_dataset_size_from_profile(profile):
    """
    Get the dataset size from a profile.

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: dataset size
    """
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])
    return dataset_size


def _ptp(maximum: float, minimum: float):
    """Peak-to-peak using minimum and maximum value of a dataset.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param maximum: The maximium of a dataset for peak-to-peak analysis
    :type maximum: float
    :param minimum: The minimum of a dataset for peak-to-peak analysis
    :type minimum: float

    :return: the difference between the maximum and minimum
    """
    return np.subtract(maximum, minimum)


def _calc_doane_bin_width_from_profile(profile):
    """
    Doane's histogram bin estimator reworked to use profiles.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    dataset_size = _get_dataset_size_from_profile(profile)
    minimum = _get_minimum_from_profile(profile)
    maximum = _get_maximum_from_profile(profile)

    if dataset_size > 2:
        sg1 = np.sqrt(
            6.0 * (dataset_size - 2) / ((dataset_size + 1.0) * (dataset_size + 3))
        )
        sigma = profile.stddev
        if sigma > 0.0:
            g1 = profile._biased_skewness
            return _ptp(maximum, minimum) / (
                1.0 + np.log2(dataset_size) + np.log2(1.0 + np.absolute(g1) / sg1)
            )
    return 0.0


def _calc_rice_bin_width_from_profile(profile):
    """
    Rice histogram bin estimator reworked to use profiles.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    dataset_size = _get_dataset_size_from_profile(profile)
    minimum = _get_minimum_from_profile(profile)
    maximum = _get_maximum_from_profile(profile)

    return _ptp(maximum, minimum) / (2.0 * dataset_size ** (1.0 / 3))


def _calc_sturges_bin_width_from_profile(profile):
    """
    Sturges histogram bin estimator reworked to use profiles.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    dataset_size = _get_dataset_size_from_profile(profile)
    minimum = _get_minimum_from_profile(profile)
    maximum = _get_maximum_from_profile(profile)

    return _ptp(maximum, minimum) / (np.log2(dataset_size) + 1.0)


def _calc_sqrt_bin_width_from_profile(profile):
    """
    Square root histogram bin estimator reworked to use profiles.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    dataset_size = _get_dataset_size_from_profile(profile)
    minimum = _get_minimum_from_profile(profile)
    maximum = _get_maximum_from_profile(profile)

    return _ptp(maximum, minimum) / np.sqrt(dataset_size)


def _calc_fd_bin_width_from_profile(profile):
    """
    Execute Freedman-Diaconis histogram binning reworked to use profiles.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(profile._get_percentile([75]), profile._get_percentile([25]))
    dataset_size = _get_dataset_size_from_profile(profile)

    return 2.0 * iqr * dataset_size ** (-1.0 / 3.0)


def _calc_auto_bin_width_from_profile(profile):
    """
    Histogram bin estimator that uses Freedman-Diaconis and Sturges estimators.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    fd_bw = _calc_fd_bin_width_from_profile(profile)
    sturges_bw = _calc_sturges_bin_width_from_profile(profile)
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        return sturges_bw


def _calc_scott_bin_width_from_profile(profile):
    """
    Scott histogram bin estimator reworked to use profiles.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin

    :return: An estimate of the optimal bin width for the given data.
    """
    dataset_size = _get_dataset_size_from_profile(profile)
    std = profile.stddev

    return (24.0 * np.pi**0.5 / dataset_size) ** (1.0 / 3.0) * std


_hist_bin_width_selectors_for_profile = {
    "auto": _calc_auto_bin_width_from_profile,
    "doane": _calc_doane_bin_width_from_profile,
    "fd": _calc_fd_bin_width_from_profile,
    "rice": _calc_rice_bin_width_from_profile,
    "scott": _calc_scott_bin_width_from_profile,
    "sqrt": _calc_sqrt_bin_width_from_profile,
    "sturges": _calc_sturges_bin_width_from_profile,
}


def _get_bin_edges(
    a: np.ndarray,
    bins: Union[str, int, List],
    range: Optional[Tuple[int, int]],
    weights: Optional[np.ndarray],
) -> Tuple[None, int]:
    """
    Compute the bins used internally by `histogram`.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param a: Ravelled data array
    :type a: ndarray
    :param bins: Forwarded arguments from `histogram`
    :type bins: str, int, List, or None
    :param range: Forwarded arguments from `histogram`
    :type range: Tuple[int, int] or None
    :param weights: Ravelled weights array
    :type weights: ndarray or None

    :return: The upper bound, lowerbound, and number of bins, used in the optimized
        implementation of `histogram` that works on uniform bins.
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, str):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ValueError(f"{bin_name!r} is not a valid estimator for `bins`")
        if weights is not None:
            raise TypeError(
                "Automated estimation of the number of "
                "bins is not supported for weighted data"
            )

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = a >= first_edge
            keep &= a <= last_edge
            if not np.logical_and.reduce(keep):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                n_equal_bins = int(
                    np.ceil(_unsigned_subtract(last_edge, first_edge) / width)
                )
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)  # type: ignore
        except TypeError as e:
            raise TypeError("`bins` must be an integer, a string, or an array") from e
        if n_equal_bins < 1:
            raise ValueError("`bins` must be positive, when an integer")

    else:
        raise ValueError("`bins` must be 1d, when an array")

    return bin_edges, n_equal_bins


def _calculate_bins_from_profile(profile, bin_method):
    """
    Compute the bins used internally by `histogram`.

    Function follows the numpy implementation within:
    https://github.com/numpy/numpy/blob/main/numpy/lib/histograms.py

    :param profile: Input data's data profile that is to be histogrammed
    :type profile: NumericStatsMixin
    :param bin_method: the method used to calculate bins based on the profile given
    :type bin_method: string

    :return: ideal number of bins for a particular histogram calcuation method
    """
    # if `bins` is a string for an automatic method,
    # this will replace it with the number of bins calculated
    if bin_method not in _hist_bin_width_selectors_for_profile:
        raise ValueError(f"{bin_method!r} is not a valid estimator for `bins`")

    dataset_size = _get_dataset_size_from_profile(profile)
    minimum = _get_minimum_from_profile(profile)
    maximum = _get_maximum_from_profile(profile)

    if dataset_size == 0:
        n_equal_bins = 1
    else:
        # Do not call selectors on empty arrays
        width = _hist_bin_width_selectors_for_profile[bin_method](profile)
        if width and not np.isnan(width):
            n_equal_bins = int(np.ceil(_ptp(maximum, minimum) / width))
        else:
            # Width can be zero for some estimators, e.g. FD when
            # the IQR of the data is zero.
            n_equal_bins = 1
    return n_equal_bins
