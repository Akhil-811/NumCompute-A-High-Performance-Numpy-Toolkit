import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b), axis=-1)

def logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def batch_iterator(X, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size]

def topk_indices(arr, k):
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]