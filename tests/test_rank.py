import numpy as np
from numcompute.rank import rank_with_ties, percentile

def test_rank_ties():
    x = np.array([1,2,2])
    r = rank_with_ties(x)
    assert r[1] == r[2]

def test_percentile():
    x = np.array([10,20,30])
    p = percentile(x)
    assert p.max() <= 1