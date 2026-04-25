import numpy as np


# ----------------------------
# Pipeline
# ----------------------------

class Pipeline:
    def __init__(self, steps):
        """
        Parameters
        ----------
        steps : list of (name, transformer)
        """
        self.steps = steps

    def fit(self, X):
        X = np.asarray(X)

        for _, step in self.steps:
            if hasattr(step, "fit"):
                step.fit(X)

            if hasattr(step, "transform"):
                X = step.transform(X)

        return self

    def transform(self, X):
        X = np.asarray(X)

        for _, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)

        return X

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)


# ----------------------------
# FeatureUnion
# ----------------------------

class FeatureUnion:
    def __init__(self, transformers):
        """
        Parameters
        ----------
        transformers : list of (name, transformer)
        """
        self.transformers = transformers

    def fit(self, X):
        X = np.asarray(X)

        for _, t in self.transformers:
            if hasattr(t, "fit"):
                t.fit(X)

        return self

    def transform(self, X):
        X = np.asarray(X)

        outputs = []
        for _, t in self.transformers:
            if hasattr(t, "transform"):
                outputs.append(t.transform(X))

        return np.hstack(outputs)

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)