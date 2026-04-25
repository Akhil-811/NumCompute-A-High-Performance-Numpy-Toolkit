import numpy as np
from numcompute.sort_search import top_k, quickselect, binary_search

def test_top_k():
    arr = np.array([1,5,3])
    idx = top_k(arr, 2)
    assert len(idx) == 2

def test_quickselect():
    arr = np.array([5,1,3])
    assert quickselect(arr, 1) == 3

def test_binary_search_found():
    arr = np.array([1,2,3,4])
    assert binary_search(arr, 3) != -1

def test_binary_search_not_found():
    arr = np.array([1,2,3])
    assert binary_search(arr, 5) == -1