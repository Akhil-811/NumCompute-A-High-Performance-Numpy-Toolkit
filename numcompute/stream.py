"""
stream.py

Streaming learning utilities for NumCompute.

Provides:

- StreamTrainer

Supports:

- Chunk-wise training
- Incremental learning
- Metric tracking
- Model comparison
- Visualisation integration

Assignment 2 compliant.
"""

import numpy as np

from numcompute.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    mse
)


# ============================================================
# Stream Trainer
# ============================================================

class StreamTrainer:
    """
    Train models incrementally using streamed chunks.

    Parameters
    ----------
    model : object
        Model implementing:

        - fit(X, y)
        - partial_fit(X, y)
        - predict(X)

    task : {"classification", "regression"}
        Learning task type.

    Notes
    -----
    The trainer simulates online learning by feeding
    chunks sequentially to the model.

    Metric history is stored after each chunk.
    """

    def __init__(
        self,
        model,
        task="classification"
    ):

        if task not in (
            "classification",
            "regression"
        ):
            raise ValueError(
                "task must be "
                "'classification' or 'regression'"
            )

        self.model = model
        self.task = task

        self.logs = {
            "chunk": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "mse": []
        }

    # --------------------------------------------------------
    # Chunk Generator
    # --------------------------------------------------------

    @staticmethod
    def split_stream(
        X,
        y,
        chunk_size
    ):
        """
        Split dataset into chunks.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        y : np.ndarray of shape
            (n_samples,)

        chunk_size : int

        Yields
        ------
        tuple
            (X_chunk, y_chunk)

        Time Complexity
        ---------------
        O(n)

        Space Complexity
        ----------------
        O(chunk_size)
        """

        n_samples = len(X)

        for start in range(
            0,
            n_samples,
            chunk_size
        ):

            end = start + chunk_size

            yield (
                X[start:end],
                y[start:end]
            )

    # --------------------------------------------------------
    # Streaming Training
    # --------------------------------------------------------

    def fit_stream(
        self,
        X,
        y,
        chunk_size=100
    ):
        """
        Train model incrementally.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        y : np.ndarray of shape
            (n_samples,)

        chunk_size : int

        Returns
        -------
        StreamTrainer

        Notes
        -----
        Metrics are computed after every chunk.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self.reset_logs()

        for chunk_id, (
            X_chunk,
            y_chunk
        ) in enumerate(
            self.split_stream(
                X,
                y,
                chunk_size
            ),
            start=1
        ):

            if hasattr(
                self.model,
                "partial_fit"
            ):

                self.model.partial_fit(
                    X_chunk,
                    y_chunk
                )

            else:

                self.model.fit(
                    X_chunk,
                    y_chunk
                )

            preds = self.model.predict(
                X_chunk
            )

            self.logs["chunk"].append(
                chunk_id
            )

            if self.task == "classification":

                self.logs[
                    "accuracy"
                ].append(
                    accuracy(
                        y_chunk,
                        preds
                    )
                )

                self.logs[
                    "precision"
                ].append(
                    precision(
                        y_chunk,
                        preds
                    )
                )

                self.logs[
                    "recall"
                ].append(
                    recall(
                        y_chunk,
                        preds
                    )
                )

                self.logs[
                    "f1"
                ].append(
                    f1_score(
                        y_chunk,
                        preds
                    )
                )

            else:

                self.logs[
                    "mse"
                ].append(
                    mse(
                        y_chunk,
                        preds
                    )
                )

        return self

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        X,
        y
    ):
        """
        Evaluate model on a dataset.

        Parameters
        ----------
        X : np.ndarray

        y : np.ndarray

        Returns
        -------
        dict
        """

        preds = self.model.predict(X)

        if self.task == "classification":

            return {
                "accuracy": accuracy(
                    y,
                    preds
                ),
                "precision": precision(
                    y,
                    preds
                ),
                "recall": recall(
                    y,
                    preds
                ),
                "f1": f1_score(
                    y,
                    preds
                )
            }

        return {
            "mse": mse(
                y,
                preds
            )
        }

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    def get_logs(self):
        """
        Return metric history.

        Returns
        -------
        dict
        """
        return self.logs

    def reset_logs(self):
        """
        Reset metric history.
        """

        for key in self.logs:
            self.logs[key] = []

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    def summary(self):
        """
        Return final metrics.

        Returns
        -------
        dict
        """

        if self.task == "classification":

            return {
                "final_accuracy":
                    self.logs["accuracy"][-1]
                    if self.logs["accuracy"]
                    else None,

                "final_precision":
                    self.logs["precision"][-1]
                    if self.logs["precision"]
                    else None,

                "final_recall":
                    self.logs["recall"][-1]
                    if self.logs["recall"]
                    else None,

                "final_f1":
                    self.logs["f1"][-1]
                    if self.logs["f1"]
                    else None
            }

        return {
            "final_mse":
                self.logs["mse"][-1]
                if self.logs["mse"]
                else None
        }