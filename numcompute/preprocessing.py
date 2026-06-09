import numpy as np


# ----------------------------
# StandardScaler
# ----------------------------

class StandardScaler:
    """
    Standardize features using z-score normalization.

    The transformation is defined as:

        z = (x - mean) / std

    Supports both batch learning via ``fit()`` and
    streaming learning via ``partial_fit()``.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Running feature means.

    std_ : np.ndarray of shape (n_features,)
        Running feature standard deviations.

    count_ : np.ndarray of shape (n_features,)
        Number of valid observations seen per feature.

    M2_ : np.ndarray of shape (n_features,)
        Running sum of squared deviations used by
        Welford's algorithm.

    Notes
    -----
    NaN values are ignored during fitting.

    Streaming updates use a vectorized form of
    Chan-Welford statistics merging.

    Time Complexity
    ---------------
    fit(X):
        O(n_samples * n_features)

    partial_fit(X):
        O(n_samples * n_features)

    transform(X):
        O(n_samples * n_features)

    Space Complexity
    ----------------
    O(n_features)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.count_ = None
        self.M2_ = None

    def fit(self, X):
        """
        Compute scaling statistics from a full dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        StandardScaler
            Fitted scaler.

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features)"
            )

        self.mean_ = np.nanmean(X, axis=0)

        variance = np.nanvar(X, axis=0)

        self.std_ = np.sqrt(variance)

        self.std_[self.std_ == 0] = 1.0

        self.count_ = np.sum(
            ~np.isnan(X),
            axis=0
        )

        centered = X - self.mean_

        centered[np.isnan(centered)] = 0

        self.M2_ = np.sum(
            centered ** 2,
            axis=0
        )

        return self

    def partial_fit(self, X):
        """
        Incrementally update scaling statistics.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        Returns
        -------
        StandardScaler
            Updated scaler.

        Raises
        ------
        ValueError
            If X is not 2-dimensional.

        Notes
        -----
        Uses vectorized Chan-Welford updates for
        numerical stability.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (chunk_size, n_features)"
            )

        chunk_count = np.sum(
            ~np.isnan(X),
            axis=0
        )

        chunk_mean = np.nanmean(
            X,
            axis=0
        )

        chunk_var = np.nanvar(
            X,
            axis=0
        )

        chunk_M2 = chunk_var * chunk_count

        if self.mean_ is None:

            self.mean_ = chunk_mean

            self.count_ = chunk_count

            self.M2_ = chunk_M2

        else:

            delta = chunk_mean - self.mean_

            total_count = (
                self.count_ + chunk_count
            )

            valid = total_count > 0

            new_mean = self.mean_.copy()

            new_mean[valid] = (
                self.mean_[valid] +
                delta[valid] *
                chunk_count[valid] /
                total_count[valid]
            )

            self.M2_[valid] = (
                self.M2_[valid]
                + chunk_M2[valid]
                + (
                    delta[valid] ** 2
                )
                * self.count_[valid]
                * chunk_count[valid]
                / total_count[valid]
            )

            self.mean_ = new_mean
            self.count_ = total_count

        variance = self.M2_ / np.maximum(
            self.count_,
            1
        )

        self.std_ = np.sqrt(variance)

        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X):
        """
        Standardize data.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape
            (n_samples, n_features)

        Raises
        ------
        ValueError
            If scaler has not been fitted.
        """

        if self.mean_ is None:
            raise ValueError(
                "Scaler has not been fitted."
            )

        X = np.asarray(X, dtype=float)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit and transform data.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape
            (n_samples, n_features)
        """

        return self.fit(X).transform(X)

# ----------------------------
# MinMaxScaler
# ----------------------------

class MinMaxScaler:
    """
    Scale features to the range [0, 1].

    The transformation is defined as:

        x_scaled = (x - min) / (max - min)

    Supports both batch learning via ``fit()`` and
    streaming learning via ``partial_fit()``.

    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Running minimum values.

    max_ : np.ndarray of shape (n_features,)
        Running maximum values.

    range_ : np.ndarray of shape (n_features,)
        Feature ranges.

    Notes
    -----
    NaN values are ignored during fitting.

    Time Complexity
    ---------------
    fit(X):
        O(n_samples * n_features)

    partial_fit(X):
        O(n_samples * n_features)

    transform(X):
        O(n_samples * n_features)

    Space Complexity
    ----------------
    O(n_features)
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        """
        Compute scaling statistics.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        MinMaxScaler

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features)"
            )

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)

        self.range_ = self.max_ - self.min_

        self.range_[self.range_ == 0] = 1.0

        return self

    def partial_fit(self, X):
        """
        Incrementally update min/max values.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        Returns
        -------
        MinMaxScaler

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (chunk_size, n_features)"
            )

        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self.min_ is None:

            self.min_ = chunk_min
            self.max_ = chunk_max

        else:

            self.min_ = np.minimum(
                self.min_,
                chunk_min
            )

            self.max_ = np.maximum(
                self.max_,
                chunk_max
            )

        self.range_ = self.max_ - self.min_

        self.range_[self.range_ == 0] = 1.0

        return self

    def transform(self, X):
        """
        Scale data.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape
            (n_samples, n_features)

        Raises
        ------
        ValueError
            If scaler has not been fitted.
        """

        if self.min_ is None:
            raise ValueError(
                "Scaler has not been fitted."
            )

        X = np.asarray(X, dtype=float)

        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        """
        Fit and transform data.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """

        return self.fit(X).transform(X)

# ----------------------------
# Imputer
# ----------------------------

class Imputer:
    """
    Replace missing values using feature means.

    Supports both batch learning via ``fit()`` and
    streaming learning via ``partial_fit()``.

    Missing values are represented by NaN.

    Attributes
    ----------
    fill_ : np.ndarray of shape (n_features,)
        Current imputation values.

    sum_ : np.ndarray of shape (n_features,)
        Running feature sums.

    count_ : np.ndarray of shape (n_features,)
        Running counts of valid observations.

    Notes
    -----
    If an entire feature contains only NaNs,
    the imputation value defaults to 0.

    Time Complexity
    ---------------
    fit(X):
        O(n_samples * n_features)

    partial_fit(X):
        O(n_samples * n_features)

    transform(X):
        O(n_samples * n_features)

    Space Complexity
    ----------------
    O(n_features)
    """

    def __init__(self):
        self.fill_ = None
        self.sum_ = None
        self.count_ = None

    def fit(self, X):
        """
        Compute imputation values.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        Imputer

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features)"
            )

        with np.errstate(all="ignore"):

            self.sum_ = np.nansum(
                X,
                axis=0
            )

            self.count_ = np.sum(
                ~np.isnan(X),
                axis=0
            )

        self.fill_ = (
            self.sum_
            / np.maximum(
                self.count_,
                1
            )
        )

        self.fill_ = np.where(
            np.isnan(self.fill_),
            0,
            self.fill_
        )

        return self

    def partial_fit(self, X):
        """
        Incrementally update imputation statistics.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        Returns
        -------
        Imputer

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (chunk_size, n_features)"
            )

        chunk_sum = np.nansum(
            X,
            axis=0
        )

        chunk_count = np.sum(
            ~np.isnan(X),
            axis=0
        )

        if self.sum_ is None:

            self.sum_ = chunk_sum
            self.count_ = chunk_count

        else:

            self.sum_ += chunk_sum
            self.count_ += chunk_count

        self.fill_ = (
            self.sum_
            / np.maximum(
                self.count_,
                1
            )
        )

        self.fill_ = np.where(
            np.isnan(self.fill_),
            0,
            self.fill_
        )

        return self

    def transform(self, X):
        """
        Replace missing values.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape
            (n_samples, n_features)

        Raises
        ------
        ValueError
            If imputer has not been fitted.
        """

        if self.fill_ is None:
            raise ValueError(
                "Imputer has not been fitted."
            )

        X = np.asarray(X, dtype=float)

        return np.where(
            np.isnan(X),
            self.fill_,
            X
        )

    def fit_transform(self, X):
        """
        Fit and transform data.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """

        return self.fit(X).transform(X)


# ----------------------------
# OneHotEncoder
# ----------------------------

class OneHotEncoder:
    """
    One-hot encode categorical features.

    Supports both batch learning via ``fit()`` and
    streaming learning via ``partial_fit()``.

    Categories discovered in future chunks are
    automatically added to the encoder.

    Attributes
    ----------
    categories_ : list[np.ndarray]
        Unique categories for each feature.

    Notes
    -----
    Categories are stored in sorted order to ensure
    deterministic encoding.

    Unknown categories encountered after fitting
    are encoded as all zeros unless discovered via
    partial_fit().

    Time Complexity
    ---------------
    fit(X):
        O(n_samples * n_features)

    partial_fit(X):
        O(n_samples * n_features)

    transform(X):
        O(n_samples * total_categories)

    Space Complexity
    ----------------
    O(total_categories)
    """

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """
        Learn categories from a full dataset.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        OneHotEncoder

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

        self.categories_ = [
            np.unique(X[:, i])
            for i in range(X.shape[1])
        ]

        return self

    def partial_fit(self, X):
        """
        Incrementally discover categories.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        Returns
        -------
        OneHotEncoder

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
        """

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (chunk_size, n_features)"
            )

        if self.categories_ is None:

            self.categories_ = [
                np.unique(X[:, i])
                for i in range(X.shape[1])
            ]

            return self

        for i in range(X.shape[1]):

            new_categories = np.unique(
                X[:, i]
            )

            self.categories_[i] = np.unique(
                np.concatenate(
                    [
                        self.categories_[i],
                        new_categories
                    ]
                )
            )

        return self

    def transform(self, X):
        """
        Transform categorical features into
        one-hot encoded representation.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape
            (n_samples, total_categories)

        Raises
        ------
        ValueError
            If encoder has not been fitted.
        """

        if self.categories_ is None:
            raise ValueError(
                "Encoder has not been fitted."
            )

        X = np.asarray(X)

        encoded_features = []

        for feature_idx in range(X.shape[1]):

            categories = self.categories_[
                feature_idx
            ]

            one_hot = (
                X[:, feature_idx][:, None]
                == categories
            ).astype(float)

            encoded_features.append(
                one_hot
            )

        return np.hstack(
            encoded_features
        )

    def fit_transform(self, X):
        """
        Fit and transform data.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """

        return self.fit(X).transform(X)