import h5py, pytest
import numpy as np

from numpy.testing import assert_array_equal
from mpi4py import MPI

from mpi_utils.disk import load_data_MPI


def test_load_data_MPI(tmpdir):
    """Tests loading data from an HDF5 file into all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    root = 0
    X = np.random.randn(5, 10)
    y = np.random.randint(5, size=5)

    fname = tmpdir.join('temp.h5')
    if rank == root:
        with h5py.File(str(fname), 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('Xp', data=X)
            f.create_dataset('y', data=y)
            f.create_dataset('yp', data=y)

    # Default keys
    X_load, y_load = load_data_MPI(fname)
    if rank == root:
        assert_array_equal(X, X_load)
        assert_array_equal(y, y_load)

    # Set keys
    X_load, y_load = load_data_MPI(fname,
                                   X_key='Xp',
                                   y_key='yp')
    if rank == root:
        assert_array_equal(X, X_load)
        assert_array_equal(y, y_load)
