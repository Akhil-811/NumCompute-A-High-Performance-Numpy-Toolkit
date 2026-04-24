import numpy as np


def read_csv(filepath, delimiter=",", chunk_size=None, dtype=float):
    """
    Load a CSV file as NumPy array(s), with optional chunking.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    delimiter : str, optional (default=",")
        Delimiter used in the CSV file.

    chunk_size : int or None, optional
        If None, yields the full array.
        If int, yields chunks of shape (chunk_size, n_features).

    dtype : data-type, optional (default=float)
        Desired data type of output array.

    Yields
    ------
    np.ndarray of shape (n_samples, n_features)
        If chunk_size is None.

    np.ndarray of shape (chunk_size, n_features)
        If chunk_size is provided.

    Notes
    -----
    - Uses np.genfromtxt for robust CSV parsing.
    - Missing values are automatically converted to np.nan.
    - Supports large files via chunked iteration.

    Time Complexity
    ---------------
    O(n) where n = number of elements.

    Space Complexity
    ----------------
    O(n) for full load, O(chunk_size) for chunked mode.
    """

    # Load entire file using NumPy (SPEC REQUIREMENT)
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        dtype=dtype,
        missing_values="",
        filling_values=np.nan
    )

    # Yield data to maintain a consistent iterator-based interface, enabling both full and chunked processing
    if chunk_size is None:
        yield data
    else:
        n = len(data)
        for i in range(0, n, chunk_size):
            yield data[i:i + chunk_size]