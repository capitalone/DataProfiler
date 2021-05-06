"""
https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/histograms.py
Histogram-related functions
"""
import operator

import numpy as np
from numpy.lib.histograms import _hist_bin_selectors, _get_outer_edges, \
    _unsigned_subtract


def _get_bin_edges(a, bins, range, weights):
    """
    Computes the bins used internally by `histogram`.
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
            raise ValueError(
                "{!r} is not a valid estimator for `bins`".format(bin_name))
        if weights is not None:
            raise TypeError("Automated estimation of the number of "
                            "bins is not supported for weighted data")

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not np.logical_and.reduce(keep):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError(
                '`bins` must be an integer, a string, or an array') from e
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

    else:
        raise ValueError('`bins` must be 1d, when an array')

    return bin_edges, n_equal_bins
