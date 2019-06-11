"""
Helper functions for working with MPI.
"""
import h5py
import numpy as np

from .utils import _np2mpi


def load_data_MPI(h5_name, keys, root=0):
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
        Features on all MPI ranks.
    y : ndarray
        Targets on all MPI ranks.
    """

    comm = MPI.COMM_WORLD
    rank = comm.rank
    Xshape = None
    Xdtype = None
    yshape = None
    ydtype = None
    if rank == root:
        with h5py.File(h5_name, 'r') as f:
            X = f[X_key][()]
            Xshape = X.shape
            Xdtype = X.dtype
            y = f[y_key][()]
            yshape = y.shape
            ydtype = y.dtype
    Xshape = comm.bcast(Xshape, root=root)
    Xdtype = comm.bcast(Xdtype, root=root)
    yshape = comm.bcast(yshape, root=root)
    ydtype = comm.bcast(ydtype, root=root)
    if rank != root:
        X = np.empty(Xshape, dtype=Xdtype)
        y = np.empty(yshape, dtype=ydtype)
    comm.Bcast([X, _np2mpi[np.dtype(X.dtype)]], root=root)
    comm.Bcast([y, _np2mpi[np.dtype(y.dtype)]], root=root)
    return X, y
