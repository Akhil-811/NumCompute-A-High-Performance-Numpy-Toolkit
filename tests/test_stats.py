import numpy as np
from numcompute.stats import RunningStats, histogram, quantiles

def test_running_stats():
    rs = RunningStats()
    rs.update([1,2,3])
    assert abs(rs.mean - 2) < 1e-6

def test_histogram():
    x = np.array([1,2,3,4])
    hist = histogram(x, bins=2)
    assert len(hist[0]) == 2

def test_quantiles():
    x = np.array([1,2,3])
    q = quantiles(x, [50])
    assert q[0] == 2