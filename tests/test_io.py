import numpy as np
from numcompute.io import read_csv

def test_read_csv_basic(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("1,2\n3,4\n")
    data = next(read_csv(file))
    assert data.shape == (2,2)

def test_read_csv_nan(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("1,\n,3\n")
    data = next(read_csv(file))
    assert np.isnan(data).sum() == 2