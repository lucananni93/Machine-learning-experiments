import numpy as np


def check_array(X, dims=2):
    if X.ndim == dims:
        return X
    elif X.ndim < dims:
        for _ in range(dims - X.ndim):
            X = X[np.newaxis, :]
        return X
    else:
        raise ValueError("Input array has more dimensions than required")
