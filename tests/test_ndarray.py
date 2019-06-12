import pytest
import numpy as np
from numpy.testing import assert_array_equal
from mpi4py import MPI

from mpi_utils.ndarray import (Bcast_from_root, Gatherv_rows,
                               Gather_ndlist)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Bcast_from_root():
    """Test the Bcast_from_root function for broadcasting
    an array from root to all ranks.
    """
    comm = MPI.COMM_WORLD
    root = 0

    dims = [2, 3, 5]

    for dtype in [int, float]:
        for ndim in range(1, 4):
            my_dim = dims[:ndim]
            X = None
            if comm.rank == root:
                X = np.arange(np.prod(my_dim), dtype=dtype)
                X = X.reshape(my_dim)
            X = Bcast_from_root(X, comm, root)
            Xp = np.arange(np.prod(my_dim), dtype=dtype)
            Xp = Xp.reshape(my_dim)
            assert_array_equal(X, Xp)
            assert X.dtype == dtype
            assert X.ndim == len(my_dim)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Gatherv_rows():
    """Test the Gatherv_rows function for Gathering and
    concatenating ndarrys along their first axes to root.
    """
    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank
    size = comm.size

    for dtype in [int, float]:
        # Multiple rows per rank
        X = np.arange(151 * 3, dtype=dtype).reshape(151, 3)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype

        # Fewer rows than ranks
        X = np.arange(2 * 3, dtype=dtype).reshape(2, 3)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype

        # Multiple rows per rank, 3d
        X = np.arange(151 * 2 * 3, dtype=dtype).reshape(151, 2, 3)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype

        # Fewer rows than ranks, 3d
        X = np.arange(2 * 3 * 5, dtype=dtype).reshape(2, 3, 5)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Gatherv_random_rows():
    """Test Gatherv_rows for gathering ndarrays with random
    shapes along their first axis
    """

    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank

    data = np.random.normal(size=(np.random.randint(1, 10), 1000))
    sizes = comm.gather(data.shape[0], root=root)
    data = Gatherv_rows(data, comm, root)

    if rank == root:
        assert(data.shape[0] == np.sum(sizes))


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Gather_ndlist():
    """Test Gather_ndlist for concatenating lists of various lengths,
    with each element being an nd-array of various dimension"""

    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank
    numprocs = comm.size

    # generate list of differently sized np arrays
    size = rank + 1
    data = [np.arange(size**2).reshape(size, size) for i in range(size)]

    data = Gather_ndlist(data, comm, root=root)

    # Check dimensions
    if rank == root:
        # Expected number of arrays and respective dimensions
        narrays = np.sum([(n + 1) for n in range(numprocs)])

        expected_arrays = []
        for n in range(numprocs):
            expected_arrays.extend([np.arange((n + 1)**2).reshape(n + 1, n + 1)
                                    for i in range(n + 1)])
        print(expected_arrays)
        array_shapes = [arr.shape for arr in expected_arrays]

        # Right number of arrays?
        assert(len(data) == narrays)

        # Right dimensions in each array?
        assert(np.all([data[i].shape == array_shapes[i]
                      for i in range(narrays)]))

        # Right values in each array?
        assert(np.all([np.array_equal(data[i], expected_arrays[i])
                       for i in range(narrays)]))
