"""
Helper functions for working with MPI.
"""
import h5py
import numpy as np
from mpi4py import MPI

from .utils import _np2mpi


def load_data_MPI(h5_name, keys, root=0, comm=None):
    """Load data from an h5 file and broadcast it across MPI ranks.

    This is a helper function. It is also possible to load the data
    without this function.

    Parameters
    ----------
    h5_name : str
        Path to h5 file.
    X_key : str
        Key for the features dataset. (default: 'X')
    y_key : str
        Key for the targets dataset. (default: 'y')

    Returns
    -------
    X : ndarray
        Features on all MPI ranks.i
    y : ndarray
        Targets on all MPI ranks.
    """

    if isinstance(keys, str):
        keys = [keys]
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.rank
    shapes = [None] * len(keys)
    dtypes = [None] * len(keys)
    if rank == root:
        with h5py.File(h5_name, 'r') as f:
            arrays = [f[key][:] for key in keys]
            shapes = [f[key].shape for key in keys]
            dtypes = [f[key].dtype for key in keys]

    shapes = [comm.bcast(shape, root=root) for shape in shapes]
    dtypes = [comm.bcast(dtypes, root=root) for dtype in dtypes]
    if rank != root:
        arrays = [np.empty(shape, dtype=dtype)
                  for shape, dtype in zip(shapes, dtypes)]
    [comm.Bcast([arr, _np2mpi[np.dtype(arr.dtype)]], root=root)
     for arr in arrays]
    return tuple(arrays)
