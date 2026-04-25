import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler

def test_pipeline():
    X = np.array([[1,2],[3,4]])
    pipe = Pipeline([("scale", StandardScaler())])
    X2 = pipe.fit(X).transform(X)
    assert X2.shape == X.shape