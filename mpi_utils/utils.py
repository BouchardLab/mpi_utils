"""
Utilities for mpi4py.
"""
import numpy as np

from mpi4py import MPI


_np2mpi = {np.dtype(np.float32): MPI.FLOAT,
           np.dtype(np.float64): MPI.DOUBLE,
           np.dtype(np.int): MPI.LONG,
           np.dtype(np.intc): MPI.INT}
