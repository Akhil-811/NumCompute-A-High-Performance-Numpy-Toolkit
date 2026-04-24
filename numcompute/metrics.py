import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            matrix[i, j] = np.sum((y_true == c1) & (y_pred == c2))
    return matrix


def roc_auc(y_true, y_scores):
    sorted_idx = np.argsort(y_scores)
    y_true = y_true[sorted_idx]
    cum_pos = np.cumsum(y_true)
    cum_neg = np.cumsum(1 - y_true)
    auc = np.sum(cum_pos * (1 - y_true)) / (cum_pos[-1] * cum_neg[-1] + 1e-8)
    return auc