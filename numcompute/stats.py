import numpy as np


# ----------------------------
# Streaming Statistics
# ----------------------------

class RunningStats:
    """
    Compute running mean and variance using Welford’s algorithm.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        """
        Update statistics with new data.

        Parameters
        ----------
        x : iterable of shape (n,)
        """
        for val in x:
            self.n += 1
            delta = val - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (val - self.mean)

    def variance(self):
        """
        Return sample variance.

        Returns
        -------
        float
        """
        return self.M2 / (self.n - 1 + 1e-8)


# ----------------------------
# Descriptive Statistics
# ----------------------------

def mean(x, axis=None):
    """
    Compute mean (NaN-safe).

    Parameters
    ----------
    x : np.ndarray
    axis : int or None

    Returns
    -------
    np.ndarray or float
    """
    return np.nanmean(x, axis=axis)


def median(x, axis=None):
    """
    Compute median (NaN-safe).
    """
    return np.nanmedian(x, axis=axis)


def std(x, axis=None):
    """
    Compute standard deviation (NaN-safe).
    """
    return np.nanstd(x, axis=axis)


def min_val(x, axis=None):
    """
    Compute minimum (NaN-safe).
    """
    return np.nanmin(x, axis=axis)


def max_val(x, axis=None):
    """
    Compute maximum (NaN-safe).
    """
    return np.nanmax(x, axis=axis)


# ----------------------------
# Distribution Functions
# ----------------------------

def histogram(x, bins=10):
    """
    Compute histogram.

    Parameters
    ----------
    x : np.ndarray
    bins : int

    Returns
    -------
    tuple (counts, bin_edges)
    """
    return np.histogram(x, bins=bins)


def quantiles(x, q):
    """
    Compute quantiles (NaN-safe).

    Parameters
    ----------
    x : np.ndarray
    q : float or array-like

    Returns
    -------
    np.ndarray or float
    """
    return np.nanpercentile(x, q)