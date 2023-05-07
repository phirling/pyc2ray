# Utility functions for density field
import numpy as np

def constant_field(ndens,N):
    """Create array of constant value

    Parameters
    ----------
    ndens : float
        Value of the constant density
    N : int
        Mesh size

    Returns
    -------
    density : 3D-array
        Density array
    """
    return ndens * np.ones((N,N,N),order='F')