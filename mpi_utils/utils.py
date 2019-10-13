"""
Utilities for mpi4py.
"""
import numpy as np

from mpi4py import MPI


_np2mpi = {np.dtype(np.float32): MPI.FLOAT,
           np.dtype(np.float64): MPI.DOUBLE,
           np.dtype(np.int): MPI.LONG,
           np.dtype(np.intc): MPI.INT}


def check_valid_ndarray(X):
    """Checks whether X is a ndarray and returns a contiguous version.
    """
    if X is None:
        return X
    if not isinstance(X, np.ndarray):
        raise ValueError('Must be a numpy ndarray.')
    return np.ascontiguousarray(X)
