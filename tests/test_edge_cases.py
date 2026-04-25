def test_empty_array():
    import numpy as np
    x = np.array([])
    assert x.size == 0

def test_all_equal():
    import numpy as np
    x = np.array([5,5,5])
    from numcompute.rank import rank_with_ties
    r = rank_with_ties(x)
    assert len(set(r)) == 1

def test_large_k():
    import numpy as np
    from numcompute.sort_search import top_k
    x = np.array([1,2,3])
    idx = top_k(x, 3)
    assert len(idx) == 3

def test_nan_handling():
    import numpy as np
    from numcompute.preprocessing import Imputer
    X = np.array([[np.nan,1]])
    Xi = Imputer().fit(X).transform(X)
    assert not np.isnan(Xi).any()

def test_non_contiguous():
    import numpy as np
    x = np.arange(10)[::2]
    assert x.flags['C_CONTIGUOUS'] == False