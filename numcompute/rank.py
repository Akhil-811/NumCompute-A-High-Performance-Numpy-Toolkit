import numpy as np


# ----------------------------
# Existing Functions (Keep)
# ----------------------------

def rankdata(x):
    temp = x.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(x))
    return ranks


def rank_with_ties(x):
    sorted_idx = np.argsort(x)
    ranks = np.zeros(len(x), dtype=float)
    i = 0

    while i < len(x):
        j = i
        while j + 1 < len(x) and x[sorted_idx[j]] == x[sorted_idx[j+1]]:
            j += 1

        avg_rank = (i + j) / 2
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank

        i = j + 1

    return ranks


def percentile(x):
    return rank_with_ties(x) / (len(x) - 1 + 1e-8)


# ----------------------------
# Spec-Compliant Functions
# ----------------------------

def rank(x, method='average'):
    """
    Rank data with different tie handling methods.

    Parameters
    ----------
    x : np.ndarray
    method : {'average', 'dense', 'ordinal'}

    Returns
    -------
    np.ndarray
    """

    x = np.asarray(x)

    if method == 'ordinal':
        return rankdata(x)

    elif method == 'dense':
        unique = np.unique(x)
        mapping = {v: i for i, v in enumerate(unique)}
        return np.array([mapping[v] for v in x], dtype=float)

    elif method == 'average':
        return rank_with_ties(x)

    else:
        raise ValueError("Invalid method")


def percentile_full(x, q, interpolation='linear'):
    """
    Compute percentile with interpolation options.

    Parameters
    ----------
    x : np.ndarray
    q : float (0-100)
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}

    Returns
    -------
    float
    """
    x = np.sort(x)
    n = len(x)

    pos = (q / 100) * (n - 1)

    if interpolation == 'lower':
        return x[int(np.floor(pos))]

    elif interpolation == 'higher':
        return x[int(np.ceil(pos))]

    elif interpolation == 'midpoint':
        lower = x[int(np.floor(pos))]
        upper = x[int(np.ceil(pos))]
        return (lower + upper) / 2

    elif interpolation == 'linear':
        lower_idx = int(np.floor(pos))
        upper_idx = int(np.ceil(pos))

        if lower_idx == upper_idx:
            return x[lower_idx]

        weight = pos - lower_idx
        return x[lower_idx] * (1 - weight) + x[upper_idx] * weight

    else:
        raise ValueError("Invalid interpolation method")