import numpy as np
from numcompute.optim import finite_diff_grad, jacobian

def test_gradient():
    f = lambda x: x[0]**2
    g = finite_diff_grad(f, np.array([2.0]))
    assert abs(g[0] - 4) < 1e-2

def test_jacobian():
    f = lambda x: np.array([x[0] + x[1]])
    J = jacobian(f, np.array([1.0,2.0]))
    assert J.shape == (1,2)