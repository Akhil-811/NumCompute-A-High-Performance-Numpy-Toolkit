import numpy as np


# ----------------------------
# Existing Functions (Keep)
# ----------------------------

def finite_diff_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        grad[i] = (f(x1) - f(x2)) / (2 * eps)

    return grad

#--------------------
# Jacobian Functions
#--------------------


def jacobian(f, x, eps=1e-5):
    fx = f(x)
    J = np.zeros((len(fx), len(x)))

    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += eps
        J[:, i] = (f(x1) - fx) / eps

    return J


# ----------------------------
# Spec-Compliant Functions
# ----------------------------

def grad(f, x, h=1e-5, method='central'):
    """
    Compute gradient using finite differences.

    Parameters
    ----------
    f : function
        Scalar function f(x)

    x : np.ndarray
        Input vector

    h : float
        Step size

    method : {'central', 'forward'}

    Returns
    -------
    np.ndarray
        Gradient vector
    """

    x = np.asarray(x)
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x1 = x.copy()

        if method == 'central':
            x2 = x.copy()
            x1[i] += h
            x2[i] -= h
            grad[i] = (f(x1) - f(x2)) / (2 * h)

        elif method == 'forward':
            x1[i] += h
            grad[i] = (f(x1) - f(x)) / h

        else:
            raise ValueError("method must be 'central' or 'forward'")

    return grad