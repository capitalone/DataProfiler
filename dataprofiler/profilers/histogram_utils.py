"""
Histogram-related functions.

https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/histograms.py
"""
import operator
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.lib.histograms import (  # type: ignore
    _get_outer_edges,
    _hist_bin_selectors,
    _unsigned_subtract,
)


def rework_ptp(maximum, minimum):
    """Peak-to-peak value of x.

    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    return np.subtract(maximum, minimum)


def rework_hist_bin_doane(profile):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])

    minimum = (
        profile.min
        if profile.min is not None
        else profile._stored_histogram["histogram"]["bin_edges"][0]
    )

    maximum = (
        profile.max
        if profile.max is not None
        else profile._stored_histogram["histogram"]["bin_edges"][-1]
    )
    if dataset_size > 2:
        sg1 = np.sqrt(
            6.0 * (dataset_size - 2) / ((dataset_size + 1.0) * (dataset_size + 3))
        )
        sigma = profile.stddev
        if sigma > 0.0:
            g1 = profile._biased_skewness
            return rework_ptp(maximum, minimum) / (
                1.0 + np.log2(dataset_size) + np.log2(1.0 + np.absolute(g1) / sg1)
            )
    return 0.0


def rework_hist_bin_rice(profile):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])

    minimum = (
        profile.min
        if profile.min is not None
        else profile._stored_histogram["histogram"]["bin_edges"][0]
    )

    maximum = (
        profile.max
        if profile.max is not None
        else profile._stored_histogram["histogram"]["bin_edges"][-1]
    )
    return rework_ptp(maximum, minimum) / (2.0 * dataset_size ** (1.0 / 3))


def rework_hist_bin_sturges(profile):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])

    minimum = (
        profile.min
        if profile.min is not None
        else profile._stored_histogram["histogram"]["bin_edges"][0]
    )

    maximum = (
        profile.max
        if profile.max is not None
        else profile._stored_histogram["histogram"]["bin_edges"][-1]
    )
    return rework_ptp(maximum, minimum) / (np.log2(dataset_size) + 1.0)


def rework_hist_bin_sqrt(profile):
    """
    Square root histogram bin estimator.

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])

    minimum = (
        profile.min
        if profile.min is not None
        else profile._stored_histogram["histogram"]["bin_edges"][0]
    )

    maximum = (
        profile.max
        if profile.max is not None
        else profile._stored_histogram["histogram"]["bin_edges"][-1]
    )
    return rework_ptp(maximum, minimum) / np.sqrt(dataset_size)


def rework_hist_bin_fd(profile):
    """
    Execute Freedman-Diaconis histogram binning.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 0 for the bin width.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(profile._get_percentile([75]), profile._get_percentile([25]))
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])
    return 2.0 * iqr * dataset_size ** (-1.0 / 3.0)


def rework_hist_bin_auto(profile):
    """
    Histogram bin estimator that uses Freedman-Diaconis and Sturges estimators.

    Chooses the minimum width of the estimators if the FD bin width is non-zero.
    If the bin width from the FD estimator is 0, the Sturges estimator is used.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x` and bad for data with limited
    variance. The Sturges estimator is quite good for small (<1000) datasets
    and is the default in the R language. This method gives good off-the-shelf
    behaviour.

    .. versionchanged:: 1.15.0
    If there is limited variance the IQR can be 0, which results in the
    FD bin width being 0 too. This is not a valid bin width, so
    ``np.histogram_bin_edges`` chooses 1 bin instead, which may not be optimal.
    If the IQR is 0, it's unlikely any variance-based estimators will be of
    use, so we revert to the Sturges estimator, which only uses the size of the
    dataset in its calculation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    fd_bw = rework_hist_bin_fd(profile)
    sturges_bw = rework_hist_bin_sturges(profile)
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        return sturges_bw


def rework_hist_bin_scott(profile):
    """
    Scott histogram bin estimator.

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])
    std = profile.stddev
    return (24.0 * np.pi**0.5 / dataset_size) ** (1.0 / 3.0) * std


def _get_bin_edges(
    a: np.ndarray,
    bins: Union[str, int, List],
    range: Optional[Tuple[int, int]],
    weights: Optional[np.ndarray],
) -> Tuple[None, int]:
    """
    Compute the bins used internally by `histogram`.

    Parameters
    ==========
    a : ndarray
        Ravelled data array
    bins, range
        Forwarded arguments from `histogram`.
    weights : ndarray, optional
        Ravelled weights array, or None
    Returns
    =======
    bin_edges : ndarray
        Array of bin edges
    uniform_bins : (Number, Number, int):
        The upper bound, lowerbound, and number of bins, used in the optimized
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


def get_suggested_bins(profile, bin_method):
    """Get ideal binning for the specified calculation method."""
    # parse the overloaded bins argument

    rework_hist_bin_selectors = {
        "auto": rework_hist_bin_auto,
        "doane": rework_hist_bin_doane,
        "fd": rework_hist_bin_fd,
        "rice": rework_hist_bin_rice,
        "scott": rework_hist_bin_scott,
        "sqrt": rework_hist_bin_sqrt,
        "sturges": rework_hist_bin_sturges,
    }

    n_equal_bins = None
    if isinstance(bin_method, str):
        bin_name = bin_method
    # if `bins` is a string for an automatic method,
    # this will replace it with the number of bins calculated
    if bin_name not in _hist_bin_selectors:
        raise ValueError(f"{bin_name!r} is not a valid estimator for `bins`")

    try:
        dataset_size = profile.match_count
    except AttributeError:
        dataset_size = sum(profile._stored_histogram["histogram"]["bin_counts"])

    minimum = (
        profile.min
        if profile.min is not None
        else profile._stored_histogram["histogram"]["bin_edges"][0]
    )
    maximum = (
        profile.max
        if profile.max is not None
        else profile._stored_histogram["histogram"]["bin_edges"][-1]
    )

    if dataset_size == 0:
        n_equal_bins = 1
    else:
        # Do not call selectors on empty arrays
        width = rework_hist_bin_selectors[bin_name](profile)
        if width and not np.isnan(width):
            n_equal_bins = int(np.ceil(_unsigned_subtract(maximum, minimum) / width))
        else:
            # Width can be zero for some estimators, e.g. FD when
            # the IQR of the data is zero.
            n_equal_bins = 1
    return n_equal_bins
