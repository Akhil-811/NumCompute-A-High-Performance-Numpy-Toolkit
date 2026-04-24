import numpy as np
from numcompute.metrics import accuracy, precision, recall, f1_score, mse, confusion_matrix, roc_auc

def test_accuracy():
    assert accuracy(np.array([1,0]), np.array([1,0])) == 1

def test_precision():
    assert precision(np.array([1,0]), np.array([1,1])) <= 1

def test_recall():
    assert recall(np.array([1,1]), np.array([1,0])) <= 1

def test_f1():
    assert f1_score(np.array([1,1]), np.array([1,0])) <= 1

def test_mse():
    assert mse(np.array([1,2]), np.array([1,2])) == 0

def test_confusion_matrix():
    cm = confusion_matrix(np.array([1,0]), np.array([1,0]))
    assert cm.shape == (2,2)

def test_roc_auc():
    auc = roc_auc(np.array([0,1]), np.array([0.1,0.9]))
    assert auc >= 0