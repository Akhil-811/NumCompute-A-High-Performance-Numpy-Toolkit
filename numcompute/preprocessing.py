import numpy as np


# ----------------------------
# StandardScaler
# ----------------------------

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X)

        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)

        # avoid division by zero
        self.std[self.std == 0] = 1

        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------
# MinMaxScaler
# ----------------------------

class MinMaxScaler:
    def fit(self, X):
        X = np.asarray(X)

        self.min = np.nanmin(X, axis=0)
        self.max = np.nanmax(X, axis=0)
        self.range = self.max - self.min

        # avoid division by zero
        self.range[self.range == 0] = 1

        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.min) / self.range

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------
# Imputer
# ----------------------------

class Imputer:
    def fit(self, X):
        X = np.asarray(X)

        with np.errstate(all='ignore'):
            self.fill = np.nanmean(X, axis=0)

        self.fill = np.where(np.isnan(self.fill), 0, self.fill)

        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.where(np.isnan(X), self.fill, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------
# OneHotEncoder
# ----------------------------

class OneHotEncoder:
    def fit(self, X):
        X = np.asarray(X)
        self.categories = [np.unique(col) for col in X.T]
        return self

    def transform(self, X):
        X = np.asarray(X)

        result = []
        for i, col in enumerate(X.T):
            onehot = (col[:, None] == self.categories[i]).astype(float)
            result.append(onehot)

        return np.hstack(result)

    def fit_transform(self, X):
        return self.fit(X).transform(X)