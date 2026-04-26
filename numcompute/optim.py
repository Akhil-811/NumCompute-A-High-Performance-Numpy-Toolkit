import numpy as np


# ----------------------------
# Existing Functions 
# ----------------------------

def finite_diff_grad(f, x, eps=1e-5):
    """
    Compute gradient using central finite differences.

    Parameters
    ----------
    f : function
    x : np.ndarray of shape (n,)
    eps : float

    Returns
    -------
    np.ndarray of shape (n,)
    """
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        grad[i] = (f(x1) - f(x2)) / (2 * eps)

    return grad


def jacobian(f, x, eps=1e-5):
    """
    Compute Jacobian matrix using forward differences.

    Parameters
    ----------
    f : function returning np.ndarray
    x : np.ndarray of shape (n,)
    eps : float

    Returns
    -------
    np.ndarray of shape (m, n)
    """
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
    x : np.ndarray of shape (n,)
    h : float
    method : {'central', 'forward'}

    Returns
    -------
    np.ndarray of shape (n,)
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