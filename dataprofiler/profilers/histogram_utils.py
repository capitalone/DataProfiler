"""
https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/histograms.py
Histogram-related functions
"""
import functools
import operator
import warnings

import numpy as np
from numpy.core import overrides

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def _ptp(x):
    """Peak-to-peak value of x.
    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    return _unsigned_subtract(x.max(), x.min())


def _hist_bin_sqrt(*input_data):
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
    x = input_data[0]
    return _ptp(x) / np.sqrt(x.size)


def _hist_bin_sturges(*input_data):
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
    x = input_data[0]
    return _ptp(x) / (np.log2(x.size) + 1.0)


def _hist_bin_rice(*input_data):
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
    x = input_data[0]
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(*input_data):
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
    x = input_data[0]
    return (24.0 * np.pi ** 0.5 / x.size) ** (1.0 / 3.0) * np.std(x)


def _hist_bin_stone(*input_data):
    """
    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).
    The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.
    The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.
    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule
    This paper by Stone appears to be the origination of this rule.
    http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf
    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    range : (float, float)
        The lower and upper range of the bins.
    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    x, range = input_data
    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    nbins_upper_bound = max(100, int(np.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn("The number of bins estimated may be suboptimal.",
                      RuntimeWarning, stacklevel=3)
    return ptp_x / nbins


def _hist_bin_doane(*input_data):
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
    x = input_data[0]
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            # These three operations add up to
            # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
            # but use only one temp array instead of three
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return _ptp(x) / (1.0 + np.log2(x.size) +
                              np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0


def _hist_bin_fd(*input_data):
    """
    The Freedman-Diaconis histogram bin estimator.
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
    x = input_data[0]
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(*input_data):
    """
    Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators if the FD bin width is non-zero.
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
    x, range = input_data
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    #del range  # unused
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        return sturges_bw


# Private dict initialized at module load time
_hist_bin_selectors = {'stone': _hist_bin_stone,
                       'auto': _hist_bin_auto,
                       'doane': _hist_bin_doane,
                       'fd': _hist_bin_fd,
                       'rice': _hist_bin_rice,
                       'scott': _hist_bin_scott,
                       'sqrt': _hist_bin_sqrt,
                       'sturges': _hist_bin_sturges}


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """
    a = np.asarray(a)

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == np.bool_:
        warnings.warn("Converting input from {} to {} for compatibility."
                      .format(a.dtype, np.uint8),
                      RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result
    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong
    }
    dt = np.result_type(a, b)
    try:
        dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned
        return np.subtract(a, b, casting='unsafe', dtype=dt)


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


