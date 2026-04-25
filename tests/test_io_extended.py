import numpy as np
from numcompute.io import read_csv

def test_read_csv_full_generator(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("1,2\n3,4\n")

    gen = read_csv(file)
    data = next(gen)

    assert isinstance(data, np.ndarray)
    assert data.shape == (2,2)


def test_read_csv_chunking(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("1,2\n3,4\n5,6\n")

    chunks = list(read_csv(file, chunk_size=2))

    assert len(chunks) == 2
    assert chunks[0].shape == (2,2)
    assert chunks[1].shape == (1,2)


def test_read_csv_nan_values(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("1,\n,3\n")

    data = next(read_csv(file))
    assert np.isnan(data).sum() == 2