# ===================================================================================================
# This module manages the initialization of the OCTA extension library. It ensures that GPU memory
# has been allocated when GPU-accelerated functions are called.
# ===================================================================================================

from .load_extensions import load_octa
libocta = load_octa()

__all__ = ['cuda_is_init','device_init','device_close','photo_table_to_device']

# This flag indicates whether GPU memory has been correctly allocated before calling any methods.
# NOTE: there is no check if the allocated memory has the correct mesh size when calling a function,
# so the user is responsible for that.
cuda_init = False

def cuda_is_init():
    global cuda_init
    return cuda_init

def device_init(N):
    """Initialize GPU and allocate memory for grid data

    Parameters
    ----------
    N : int
        Mesh size in grid coordinates
    """
    global cuda_init
    if libocta is not None:
        libocta.device_init(N)
        cuda_init = True
    else:
        raise RuntimeError("Could not initialize GPU: octa library not loaded")

def device_close():
    """Deallocate GPU memory
    """
    global cuda_init
    if cuda_init:
        libocta.device_close()
        cuda_init = False
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)")
    
def photo_table_to_device(table):
    """Copy radiation table to GPU

    """
    global cuda_init
    NumTau = table.shape[0]
    if cuda_init:
        libocta.photo_table_to_device(table,NumTau)
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)") 