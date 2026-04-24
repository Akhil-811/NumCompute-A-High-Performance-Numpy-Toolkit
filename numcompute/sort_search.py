import numpy as np


# ----------------------------
# Sorting Utilities
# ----------------------------

def stable_sort(arr):
    """
    Perform stable sort.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)

    Returns
    -------
    np.ndarray of shape (n,)
        Sorted array (stable).
    """
    return np.sort(arr, kind='stable')


def multi_key_sort(arr, keys):
    """
    Sort array by multiple keys.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
    keys : list of np.ndarray (each of shape (n,))

    Returns
    -------
    np.ndarray of shape (n,)
        Array sorted based on multiple keys.
    """
    return arr[np.lexsort(keys[::-1])]


# ----------------------------
# Top-K / Selection
# ----------------------------

def top_k(arr, k):
    """
    Return indices of top-k largest elements.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
    k : int

    Returns
    -------
    np.ndarray of shape (k,)
        Indices of top-k elements.
    """
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]


def quickselect(arr, k):
    """
    Return k-th smallest element.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
    k : int

    Returns
    -------
    float or int
        k-th smallest value.
    """
    return np.partition(arr, k)[k]


# ----------------------------
# Searching
# ----------------------------

def binary_search(arr, target):
    """
    Perform binary search on sorted array.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
    target : scalar

    Returns
    -------
    int
        Index if found, else -1.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_with_insertion(arr, target):
    """
    Binary search with insertion index.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
    target : scalar

    Returns
    -------
    tuple (int, bool)
        insertion index and existence flag.
    """
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    exists = left < len(arr) and arr[left] == target
    return left, exists