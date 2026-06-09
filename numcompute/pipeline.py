import numpy as np

# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """
    Sequential processing pipeline.

    Applies a series of transformers and/or estimators in order.
    Supports both batch learning (`fit`) and streaming learning
    (`partial_fit`).

    Parameters
    ----------
    steps : list of tuple(str, object)
        List of (name, component) pairs.

        Components may implement:

        - fit(X)
        - transform(X)
        - partial_fit(X)
        - partial_fit(X, y)

    Notes
    -----
    Pipeline behavior:

    During fit():
        fit -> transform -> fit -> transform ...

    During partial_fit():
        partial_fit -> transform -> partial_fit -> transform ...

    The final component may be an estimator that accepts y.

    Time Complexity
    ---------------
    O(sum(component costs))

    Space Complexity
    ----------------
    Depends on individual components.
    """

    def __init__(self, steps):
        self.steps = steps

    # --------------------------------------------------------
    # Batch Learning
    # --------------------------------------------------------

    def fit(self, X, y=None):
        """
        Fit all components.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        y : np.ndarray of shape (n_samples,), optional
            Target values.

        Returns
        -------
        Pipeline
            Fitted pipeline.

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features)"
            )

        for i, (_, step) in enumerate(self.steps):

            is_last = (i == len(self.steps) - 1)

            # final estimator
            if is_last and y is not None:
                if hasattr(step, "fit"):
                    step.fit(X, y)
            else:
                if hasattr(step, "fit"):
                    step.fit(X)

                if hasattr(step, "transform"):
                    X = step.transform(X)

        return self

    # --------------------------------------------------------
    # Streaming Learning
    # --------------------------------------------------------

    def partial_fit(self, X, y=None):
        """
        Incrementally update pipeline using a data chunk.

        Parameters
        ----------
        X : np.ndarray of shape (chunk_size, n_features)

        y : np.ndarray of shape (chunk_size,), optional
            Target values.

        Returns
        -------
        Pipeline
            Updated pipeline.

        Notes
        -----
        This method enables online learning by updating
        each component incrementally.

        Time Complexity
        ---------------
        O(chunk_size × n_features)

        Space Complexity
        ----------------
        O(n_features)
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (chunk_size, n_features)"
            )

        for i, (_, step) in enumerate(self.steps):

            is_last = (i == len(self.steps) - 1)

            if is_last and y is not None:

                if hasattr(step, "partial_fit"):
                    step.partial_fit(X, y)

            else:

                if hasattr(step, "partial_fit"):
                    step.partial_fit(X)

                elif hasattr(step, "fit"):
                    step.fit(X)

                if hasattr(step, "transform"):
                    X = step.transform(X)

        return self

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    def transform(self, X):
        """
        Transform data through all transformer stages.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Transformed output.
        """
        X = np.asarray(X)

        for _, step in self.steps:

            if hasattr(step, "transform"):
                X = step.transform(X)

        return X

    # --------------------------------------------------------
    # Fit + Transform
    # --------------------------------------------------------

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        y : np.ndarray, optional

        Returns
        -------
        np.ndarray
            Transformed output.
        """
        return self.fit(X, y).transform(X)


# ============================================================
# FeatureUnion
# ============================================================

class FeatureUnion:
    """
    Parallel feature transformer.

    Applies multiple transformers independently and
    concatenates their outputs.

    Parameters
    ----------
    transformers : list of tuple(str, object)

    Notes
    -----
    Each transformer must implement:

    - fit(X)
    - transform(X)

    Optional:

    - partial_fit(X)

    Time Complexity
    ---------------
    O(sum(transformer costs))

    Space Complexity
    ----------------
    O(total output features)
    """

    def __init__(self, transformers):
        self.transformers = transformers

    # --------------------------------------------------------
    # Batch Learning
    # --------------------------------------------------------

    def fit(self, X):
        """
        Fit all transformers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        FeatureUnion
        """
        X = np.asarray(X)

        for _, transformer in self.transformers:

            if hasattr(transformer, "fit"):
                transformer.fit(X)

        return self

    # --------------------------------------------------------
    # Streaming Learning
    # --------------------------------------------------------

    def partial_fit(self, X):
        """
        Incrementally update transformers.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        Returns
        -------
        FeatureUnion
        """
        X = np.asarray(X)

        for _, transformer in self.transformers:

            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X)

            elif hasattr(transformer, "fit"):
                transformer.fit(X)

        return self

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    def transform(self, X):
        """
        Transform and concatenate outputs.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Shape:
            (n_samples, total_output_features)
        """
        X = np.asarray(X)

        outputs = []

        for _, transformer in self.transformers:

            if hasattr(transformer, "transform"):
                outputs.append(
                    transformer.transform(X)
                )

        return np.hstack(outputs)

    # --------------------------------------------------------
    # Fit + Transform
    # --------------------------------------------------------

    def fit_transform(self, X):
        """
        Fit and transform.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        return self.fit(X).transform(X)

