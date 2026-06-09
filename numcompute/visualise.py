"""
visualise.py

Visualization utilities for NumCompute.

Provides lightweight plotting functions for:

- Streaming metrics
- Model comparison
- Predictions vs ground truth

Uses only:
- NumPy
- matplotlib

Assignment 2 compliant.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Metric Over Time
# ============================================================

def plot_metric_over_time(
    values,
    metric_name="Metric",
    title=None,
    figsize=(8, 5)
):
    """
    Plot metric values across streaming chunks.

    Parameters
    ----------
    values : array-like of shape (n_chunks,)
        Metric values recorded over time.

    metric_name : str, default="Metric"
        Label for y-axis.

    title : str or None
        Plot title.

    figsize : tuple, default=(8, 5)

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If values is empty.

    Time Complexity
    ---------------
    O(n)

    Space Complexity
    ----------------
    O(n)
    """

    values = np.asarray(values)

    if len(values) == 0:
        raise ValueError(
            "values must contain at least one element"
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        np.arange(1, len(values) + 1),
        values,
        marker="o"
    )

    ax.set_xlabel("Chunk")
    ax.set_ylabel(metric_name)

    if title is None:
        title = f"{metric_name} Over Time"

    ax.set_title(title)

    ax.grid(True)

    plt.tight_layout()

    return ax


# ============================================================
# Model Comparison
# ============================================================

def compare_models(
    model_names,
    scores,
    metric_name="Score",
    title=None,
    figsize=(8, 5)
):
    """
    Compare model performance.

    Parameters
    ----------
    model_names : list[str]
        Names of models.

    scores : array-like of shape (n_models,)
        Scores for each model.

    metric_name : str
        Metric label.

    title : str or None

    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If lengths differ.

    Time Complexity
    ---------------
    O(n)

    Space Complexity
    ----------------
    O(n)
    """

    scores = np.asarray(scores)

    if len(model_names) != len(scores):
        raise ValueError(
            "model_names and scores must have same length"
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(model_names, scores)

    ax.set_ylabel(metric_name)

    if title is None:
        title = f"Model Comparison ({metric_name})"

    ax.set_title(title)

    plt.tight_layout()

    return ax


# ============================================================
# Predictions vs Ground Truth
# ============================================================

def plot_predictions(
    y_true,
    y_pred,
    title="Predictions vs Ground Truth",
    figsize=(8, 5)
):
    """
    Visualize predictions against targets.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth values.

    y_pred : np.ndarray of shape (n_samples,)
        Predicted values.

    title : str

    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If shapes do not match.

    Time Complexity
    ---------------
    O(n)

    Space Complexity
    ----------------
    O(n)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have same shape"
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        y_true,
        y_pred
    )

    min_val = min(
        np.min(y_true),
        np.min(y_pred)
    )

    max_val = max(
        np.max(y_true),
        np.max(y_pred)
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)

    plt.tight_layout()

    return ax


# ============================================================
# Streaming Model Comparison
# ============================================================

def compare_streaming_models(
    histories,
    labels,
    metric_name="Accuracy",
    figsize=(8, 5)
):
    """
    Compare streaming metric histories.

    Parameters
    ----------
    histories : list of array-like
        Metric histories.

    labels : list[str]
        Model labels.

    metric_name : str

    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If lengths differ.

    Time Complexity
    ---------------
    O(total_points)

    Space Complexity
    ----------------
    O(total_points)
    """

    if len(histories) != len(labels):
        raise ValueError(
            "histories and labels must have same length"
        )

    fig, ax = plt.subplots(figsize=figsize)

    for history, label in zip(
        histories,
        labels
    ):
        history = np.asarray(history)

        ax.plot(
            np.arange(1, len(history) + 1),
            history,
            marker="o",
            label=label
        )

    ax.set_xlabel("Chunk")
    ax.set_ylabel(metric_name)

    ax.set_title(
        f"Streaming {metric_name} Comparison"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return ax
