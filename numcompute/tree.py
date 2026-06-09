"""
tree.py

Decision tree infrastructure for NumCompute.

Provides:

- TreeNode
- Impurity functions
- Split evaluation utilities
- BaseDecisionTree

Used by:
- DecisionTreeClassifier
- DecisionTreeRegressor
- RandomForestClassifier
"""

from dataclasses import dataclass
import numpy as np


# ============================================================
# Tree Node
# ============================================================

@dataclass
class TreeNode:
    """
    Node in a decision tree.

    Parameters
    ----------
    feature_index : int or None
        Feature used for splitting.

    threshold : float or None
        Split threshold.

    left : TreeNode or None
        Left child.

    right : TreeNode or None
        Right child.

    value : object or None
        Leaf prediction.

    Notes
    -----
    Internal tree structure.
    """

    feature_index: int = None
    threshold: float = None

    left: object = None
    right: object = None

    value: object = None

    @property
    def is_leaf(self):
        """
        Returns
        -------
        bool
            True if node is a leaf.
        """
        return self.value is not None


# ============================================================
# Impurity Functions
# ============================================================

def gini(y):
    """
    Compute Gini impurity.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    float

    Time Complexity
    ---------------
    O(n)
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    probs = counts / counts.sum()

    return 1.0 - np.sum(probs ** 2)


def entropy(y):
    """
    Compute entropy.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    float

    Time Complexity
    ---------------
    O(n)
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    probs = counts / counts.sum()

    probs = probs[probs > 0]

    return -np.sum(probs * np.log2(probs))


def variance(y):
    """
    Compute target variance.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    float

    Time Complexity
    ---------------
    O(n)
    """
    if len(y) == 0:
        return 0.0

    return np.var(y)


# ============================================================
# Base Decision Tree
# ============================================================

class BaseDecisionTree:
    """
    Base class for decision tree models.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum tree depth.

    min_samples_split : int, default=2
        Minimum samples required to split.

    random_state : int or None
        Random seed.

    Notes
    -----
    This class should not be used directly.

    Extended by:
    - DecisionTreeClassifier
    - DecisionTreeRegressor
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.root_ = None

        # streaming support
        self._X_seen = None
        self._y_seen = None

    # --------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------

    def _split(self, X_column, threshold):
        """
        Split indices using threshold.

        Parameters
        ----------
        X_column : np.ndarray
            Feature column.

        threshold : float

        Returns
        -------
        tuple
            (left_indices, right_indices)
        """

        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]

        return left_idx, right_idx

    def _majority_class(self, y):
        """
        Deterministic majority class.

        Ties resolved by selecting
        smallest label.

        Parameters
        ----------
        y : np.ndarray

        Returns
        -------
        int
        """
        classes, counts = np.unique(
            y,
            return_counts=True
        )

        max_count = counts.max()

        winners = classes[counts == max_count]

        return np.min(winners)

    def _mean_value(self, y):
        """
        Regression leaf value.

        Parameters
        ----------
        y : np.ndarray

        Returns
        -------
        float
        """
        return float(np.mean(y))

    # --------------------------------------------------------
    # Streaming Support
    # --------------------------------------------------------

    def partial_fit(self, X, y):
        """
        Incrementally update model.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        y : np.ndarray of shape
            (chunk_size,)

        Returns
        -------
        self

        Notes
        -----
        Stores all observed chunks and
        rebuilds tree.

        Assignment-compliant streaming
        strategy.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        if self._X_seen is None:

            self._X_seen = X.copy()
            self._y_seen = y.copy()

        else:

            self._X_seen = np.vstack(
                [self._X_seen, X]
            )

            self._y_seen = np.concatenate(
                [self._y_seen, y]
            )

        self.fit(
            self._X_seen,
            self._y_seen
        )

        return self

    # --------------------------------------------------------
    # Prediction Traversal
    # --------------------------------------------------------

    def _predict_one(self, x, node):
        """
        Traverse tree.

        Parameters
        ----------
        x : np.ndarray

        node : TreeNode

        Returns
        -------
        prediction
        """

        if node.is_leaf:
            return node.value

        feature_value = x[node.feature_index]

        if feature_value <= node.threshold:

            return self._predict_one(
                x,
                node.left
            )

        return self._predict_one(
            x,
            node.right
        )

    def predict(self, X):
        """
        Predict outputs.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """

        X = np.asarray(X)

        return np.array([
            self._predict_one(
                row,
                self.root_
            )
            for row in X
        ])
# ============================================================
# Decision Tree Classifier
# ============================================================

class DecisionTreeClassifier(BaseDecisionTree):
    """
    Decision Tree Classifier.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum tree depth.

    min_samples_split : int, default=2
        Minimum samples required to split.

    criterion : {"gini", "entropy"}, default="gini"
        Split criterion.

    random_state : int or None, default=None

    Notes
    -----
    Supports:
    - fit()
    - partial_fit()
    - predict()
    - predict_proba()

    NaN values are converted using np.nan_to_num
    during split evaluation.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        random_state=None
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        if criterion not in ("gini", "entropy"):
            raise ValueError(
                "criterion must be 'gini' or 'entropy'"
            )

        self.criterion = criterion
        self.classes_ = None

    # --------------------------------------------------------
    # Criterion
    # --------------------------------------------------------

    def _impurity(self, y):
        if self.criterion == "gini":
            return gini(y)

        return entropy(y)

    # --------------------------------------------------------
    # Information Gain
    # --------------------------------------------------------

    def _information_gain(
        self,
        y,
        left_idx,
        right_idx
    ):
        """
        Compute information gain.
        """

        n = len(y)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0.0

        parent = self._impurity(y)

        left_impurity = self._impurity(
            y[left_idx]
        )

        right_impurity = self._impurity(
            y[right_idx]
        )

        child = (
            len(left_idx) / n
        ) * left_impurity + (
            len(right_idx) / n
        ) * right_impurity

        return parent - child

    # --------------------------------------------------------
    # Best Split
    # --------------------------------------------------------

    def _best_split(self, X, y):
        """
        Find best feature and threshold.

        Returns
        -------
        tuple
            (
                best_feature,
                best_threshold,
                best_gain
            )
        """

        n_samples, n_features = X.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        X = np.nan_to_num(X)

        for feature_idx in range(n_features):

            column = X[:, feature_idx]

            thresholds = np.unique(column)

            for threshold in thresholds:

                left_idx, right_idx = self._split(
                    column,
                    threshold
                )

                gain = self._information_gain(
                    y,
                    left_idx,
                    right_idx
                )

                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return (
            best_feature,
            best_threshold,
            best_gain
        )

    # --------------------------------------------------------
    # Tree Construction
    # --------------------------------------------------------

    def _build_tree(
        self,
        X,
        y,
        depth=0
    ):
        """
        Recursively build tree.
        """

        n_samples = len(y)

        unique_classes = np.unique(y)

        # pure node
        if len(unique_classes) == 1:

            return TreeNode(
                value=unique_classes[0]
            )

        # max depth reached
        if depth >= self.max_depth:

            return TreeNode(
                value=self._majority_class(y)
            )

        # insufficient samples
        if n_samples < self.min_samples_split:

            return TreeNode(
                value=self._majority_class(y)
            )

        feature_idx, threshold, gain = (
            self._best_split(X, y)
        )

        # no useful split
        if (
            feature_idx is None
            or gain <= 0
        ):
            return TreeNode(
                value=self._majority_class(y)
            )

        left_idx, right_idx = self._split(
            X[:, feature_idx],
            threshold
        )

        if (
            len(left_idx) == 0
            or len(right_idx) == 0
        ):
            return TreeNode(
                value=self._majority_class(y)
            )

        left_child = self._build_tree(
            X[left_idx],
            y[left_idx],
            depth + 1
        )

        right_child = self._build_tree(
            X[right_idx],
            y[right_idx],
            depth + 1
        )

        return TreeNode(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(self, X, y):
        """
        Train classifier.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        y : np.ndarray of shape
            (n_samples,)

        Returns
        -------
        self
        """

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                "X must be 2-dimensional"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have same length"
            )

        self.classes_ = np.unique(y)

        self.root_ = self._build_tree(
            X,
            y
        )

        return self

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    def predict_proba(self, X):
        """
        Estimate class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Shape:
            (n_samples, n_classes)
        """

        preds = self.predict(X)

        probs = np.zeros(
            (
                len(preds),
                len(self.classes_)
            )
        )

        for i, pred in enumerate(preds):

            idx = np.where(
                self.classes_ == pred
            )[0][0]

            probs[i, idx] = 1.0

        return probs
    
# ============================================================
# Decision Tree Regressor
# ============================================================

class DecisionTreeRegressor(BaseDecisionTree):
    """
    Decision Tree Regressor.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum tree depth.

    min_samples_split : int, default=2
        Minimum samples required to split.

    random_state : int or None, default=None
        Random seed.

    Notes
    -----
    Uses variance reduction as the split criterion.

    Supports:
    - fit()
    - partial_fit()
    - predict()

    Streaming updates are supported through partial_fit().
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        random_state=None
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    # --------------------------------------------------------
    # Variance Reduction
    # --------------------------------------------------------

    def _variance_reduction(
        self,
        y,
        left_idx,
        right_idx
    ):
        """
        Compute variance reduction.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)

        left_idx : np.ndarray
        right_idx : np.ndarray

        Returns
        -------
        float
        """

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0.0

        parent_var = variance(y)

        left_var = variance(y[left_idx])
        right_var = variance(y[right_idx])

        n = len(y)

        weighted_child_var = (
            len(left_idx) / n
        ) * left_var + (
            len(right_idx) / n
        ) * right_var

        return parent_var - weighted_child_var

    # --------------------------------------------------------
    # Best Split
    # --------------------------------------------------------

    def _best_split(self, X, y):
        """
        Find best feature and threshold.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        y : np.ndarray of shape
            (n_samples,)

        Returns
        -------
        tuple
            (
                best_feature,
                best_threshold,
                best_gain
            )
        """

        n_samples, n_features = X.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        X = np.nan_to_num(X)

        for feature_idx in range(n_features):

            column = X[:, feature_idx]

            thresholds = np.unique(column)

            for threshold in thresholds:

                left_idx, right_idx = self._split(
                    column,
                    threshold
                )

                gain = self._variance_reduction(
                    y,
                    left_idx,
                    right_idx
                )

                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return (
            best_feature,
            best_threshold,
            best_gain
        )

    # --------------------------------------------------------
    # Tree Construction
    # --------------------------------------------------------

    def _build_tree(
        self,
        X,
        y,
        depth=0
    ):
        """
        Recursively build regression tree.

        Parameters
        ----------
        X : np.ndarray

        y : np.ndarray

        depth : int

        Returns
        -------
        TreeNode
        """

        n_samples = len(y)

        if depth >= self.max_depth:

            return TreeNode(
                value=self._mean_value(y)
            )

        if n_samples < self.min_samples_split:

            return TreeNode(
                value=self._mean_value(y)
            )

        if np.allclose(y, y[0]):

            return TreeNode(
                value=float(y[0])
            )

        feature_idx, threshold, gain = (
            self._best_split(X, y)
        )

        if (
            feature_idx is None
            or gain <= 0
        ):
            return TreeNode(
                value=self._mean_value(y)
            )

        left_idx, right_idx = self._split(
            X[:, feature_idx],
            threshold
        )

        if (
            len(left_idx) == 0
            or len(right_idx) == 0
        ):
            return TreeNode(
                value=self._mean_value(y)
            )

        left_child = self._build_tree(
            X[left_idx],
            y[left_idx],
            depth + 1
        )

        right_child = self._build_tree(
            X[right_idx],
            y[right_idx],
            depth + 1
        )

        return TreeNode(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(self, X, y):
        """
        Train regression tree.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        y : np.ndarray of shape
            (n_samples,)

        Returns
        -------
        DecisionTreeRegressor

        Raises
        ------
        ValueError
            If input dimensions are invalid.

        Time Complexity
        ---------------
        O(n_features * n_samples^2)
        worst case.

        Space Complexity
        ----------------
        O(n_samples)
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be 2-dimensional"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have same length"
            )

        self.root_ = self._build_tree(
            X,
            y
        )

        return self