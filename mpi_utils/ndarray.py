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

    comm.Gatherv(send, [rec, sizes, disps, _np2mpi[dtype]], root=0)
    return rec
