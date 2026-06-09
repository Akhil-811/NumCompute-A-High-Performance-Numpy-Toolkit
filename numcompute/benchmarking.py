"""
benchmarking.py

Performance benchmarking utilities for NumCompute.

Provides:

- benchmark()
- compare_models()

Supports benchmarking of:
- Functions
- Decision Trees
- Random Forests
- Streaming learning workflows
"""

import time
import numpy as np


# ============================================================
# Generic Function Benchmark
# ============================================================

def benchmark(func, *args, repeat=5):
    """
    Benchmark a function.

    Parameters
    ----------
    func : callable

    *args :
        Arguments passed to the function.

    repeat : int, default=5
        Number of benchmark runs.

    Returns
    -------
    float
        Fastest execution time (seconds).

    Time Complexity
    ---------------
    O(repeat × function_cost)

    Space Complexity
    ----------------
    O(repeat)
    """

    times = []

    for _ in range(repeat):

        start = time.perf_counter()

        func(*args)

        elapsed = (
            time.perf_counter() - start
        )

        times.append(elapsed)

    return min(times)


# ============================================================
# Model Benchmark
# ============================================================

def benchmark_model(
    model,
    X,
    y
):
    """
    Benchmark model training and prediction.

    Parameters
    ----------
    model : object
        Model implementing:
        fit()
        predict()

    X : np.ndarray of shape
        (n_samples, n_features)

    y : np.ndarray of shape
        (n_samples,)

    Returns
    -------
    dict
        {
            "train_time": float,
            "predict_time": float
        }
    """

    start = time.perf_counter()

    model.fit(X, y)

    train_time = (
        time.perf_counter() - start
    )

    start = time.perf_counter()

    model.predict(X)

    predict_time = (
        time.perf_counter() - start
    )

    return {
        "train_time": train_time,
        "predict_time": predict_time
    }


# ============================================================
# Streaming Benchmark
# ============================================================

def benchmark_stream(
    trainer,
    X,
    y,
    chunk_size
):
    """
    Benchmark streaming training.

    Parameters
    ----------
    trainer : StreamTrainer

    X : np.ndarray

    y : np.ndarray

    chunk_size : int

    Returns
    -------
    float
        Total streaming training time.
    """

    start = time.perf_counter()

    trainer.fit_stream(
        X,
        y,
        chunk_size=chunk_size
    )

    return (
        time.perf_counter() - start
    )


# ============================================================
# Compare Models
# ============================================================

def compare_models(
    models,
    names,
    X,
    y
):
    """
    Benchmark multiple models.

    Parameters
    ----------
    models : list
        Models to compare.

    names : list[str]

    X : np.ndarray

    y : np.ndarray

    Returns
    -------
    list[dict]

    Example
    -------
    >>> compare_models(
    ...     [tree, forest],
    ...     ["Tree", "Forest"],
    ...     X,
    ...     y
    ... )
    """

    if len(models) != len(names):
        raise ValueError(
            "models and names must have same length"
        )

    results = []

    for model, name in zip(
        models,
        names
    ):

        metrics = benchmark_model(
            model,
            X,
            y
        )

        metrics["model"] = name

        results.append(metrics)

    return results