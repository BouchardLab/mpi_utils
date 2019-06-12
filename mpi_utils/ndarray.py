"""
Functions that help with dealing with multidimensional ndarrays.
"""
import numpy as np

from mpi4py import MPI
from .utils import _np2mpi


def Bcast_from_root(send, comm, root=0):
    """Broadcast an array from root to all MPI ranks.

    Parameters
    ----------
    send : ndarray or None
        Array to send from root to all ranks. send in other ranks
        has no effect.
    comm : MPI.COMM_WORLD
        MPI communicator.
    root : int, default 0
        This rank contains the array to send.

    Returns
    -------
    send : ndarray
        Each rank will have a copy of the array from root.
    """
    rank = comm.rank
    if rank == 0:
        dtype = send.dtype
        shape = send.shape
    else:
        dtype = None
        shape = None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    if rank != 0:
        send = np.empty(shape, dtype=dtype)
    comm.Bcast([send, _np2mpi[np.dtype(dtype)]], root=root)
    return send


def Gatherv_rows(send, comm, root=0):
    """Concatenate arrays along the first axis using Gatherv.

    Parameters
    ----------
    send : ndarray
        The arrays to concatenate. All dimensions must be equal except for the
        first.
    comm : MPI.COMM_WORLD
        MPI communicator.
    root : int, default 0
        This rank will contain the Gatherv'ed array.

    Returns
    -------
    rec : ndarray or None
        Gatherv'ed array on root or None on other ranks.
    """

    rank = comm.rank
    dtype = send.dtype
    shape = send.shape
    tot = np.zeros(1, dtype=int)

    # Gather the sizes of the first dimension on root
    rank_sizes = comm.gather(shape[0], root=root)
    comm.Reduce(np.array(shape[0], dtype=int),
                [tot, _np2mpi[tot.dtype]], op=MPI.SUM, root=root)
    if rank == root:
        rec_shape = (tot[0],) + shape[1:]
        rec = np.empty(rec_shape, dtype=dtype)
        sizes = [size * np.prod(rec_shape[1:]) for size in rank_sizes]
        disps = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    else:
        rec = None
        sizes = None
        disps = None

    comm.Gatherv(send, [rec, sizes, disps, _np2mpi[dtype]], root=root)
    return rec


def Gather_ndlist(send, comm, root=0):
    """Gather lists of arbitrarily shaped ndarrays at each rank into a single
    concatenated list of ndarrays on root. Works by raveling and concatenating
    the arrays at each rank, gathering using Gatherv, and then separating and
    reshaping the arrays on root. All arrays should be the same dtype.

    Parameters
    ----------
    send : list of ndarrays
        The list of ndarrays to concatenate. Arrays may be arbitrarily shaped.
    comm : MPI.COMM_WORLD
        MPI communicator.
    root : int, default 0
        This rank will contain the concatenated list of arrays

    Returns
    -------
    array_list : list of ndarrays or None
        Final concatenated list of arrays on root, or None on other ranks.
    """

    rank = comm.rank
    dtype = send[0].dtype
    shapes = [x.shape for x in send]

    # Gather size tuples - this is a list of lists of tuples
    rank_shapes = comm.gather(shapes, root=root)

    # Ravel list of arrays
    raveled_arrays = np.array([arr.ravel() for arr in send]).ravel()

    if rank == root:
        # Sizes of each of the raveled arrays
        sizes = [np.sum([np.product(tup) for tup in tup_list])
                 for tup_list in rank_shapes]
        total_length = np.sum(sizes)
        rec = np.empty(total_length, dtype=dtype)
        # Location in rec where each incoming object should be placed
        displs = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    else:
        sizes = None
        total_length = None
        rec = None
        displs = None

    # Stack end to end
    comm.Gatherv(raveled_arrays, [rec, sizes, displs, _np2mpi[dtype]],
                 root=root)

    # Separate and re-shape
    if rank == root:

        # Flatten list of lists of shapes into a list of shapes
        rank_shapes = [tup for tup_list in rank_shapes for tup in tup_list]

        # Separate
        rank_sizes = [np.product(tup) for tup in rank_shapes]
        split_locs = np.cumsum(rank_sizes)[:-1]
        array_list = np.array_split(rec, split_locs)

        # Re-shape
        array_list = [np.reshape(array_list[i], rank_shapes[i])
                      for i in range(len(rank_shapes))]
    else:
        array_list = None

    return array_list
