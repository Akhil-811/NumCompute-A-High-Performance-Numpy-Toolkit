import numpy as np


# ----------------------------
# Sorting Utilities
# ----------------------------

def stable_sort(arr):
    """
    Perform stable sort.

    Returns
    -------
    np.ndarray
        Sorted array (stable).
    """
    return np.sort(arr, kind='stable')


def multi_key_sort(arr, keys):
    """
    Sort array by multiple keys.

    Parameters
    ----------
    arr : np.ndarray
    keys : list of arrays (same length as arr)

    Returns
    -------
    np.ndarray
        Sorted array based on keys
    """
    return arr[np.lexsort(keys[::-1])]


# ----------------------------
# Top-K / Selection
# ----------------------------

def top_k(arr, k):
    """
    Return indices of top-k largest elements.
    """
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]


def quickselect(arr, k):
    """
    Return k-th smallest element.
    """
    return np.partition(arr, k)[k]


# ----------------------------
# Searching
# ----------------------------

def binary_search(arr, target):
    """
    Classic binary search.

    Returns
    -------
    int
        Index if found, else -1
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
    Spec-compliant binary search.

    Returns
    -------
    tuple (index, exists)
        index : insertion index
        exists : bool
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