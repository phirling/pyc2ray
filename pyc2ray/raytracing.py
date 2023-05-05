import numpy as np
from . import libc2ray
from .octa_core import gpu, cuda_init
if gpu:
    from octa_core import libocta
from .utils import printlog

# ===================================================================================================
# Raytrace all sources: find column density -> find photoionization rates
# ===================================================================================================

def do_all_sources(srcflux,srcpos,r_RT,subboxsize,sig,dr,ndens,xh_av,loss_fraction=1e-2):
    """Raytrace all sources and compute photoionization rates

    Parameters
    ----------
    srcflux : 1D-array
        Array containing the normalization for the ionizing flux of each source. Shape: (NumSources)
    srcpos : 2D-array
        Array containing the position of each source. Shape: (NumSources,3)
    r_RT : int
        Size of maximum subbox to raytrace. When compiled without subbox, sets the constant subbox size
    subboxsize : int
        Increment for subbox technique
    sig : float
        Constant photoionization cross-section of hydrogen in cm^2.
    dr : float
        Cell dimension in each direction in cm
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    xh_av : 3D-array
        Current time-averaged value of ionization fraction in each cell
    loss_fraction : float (default: 1e-2)
        Fraction of remaining photons below we stop ray-tracing (subbox technique)

    Returns
    -------
    nsubbox : int
        Total number of subbox increments used across all sources
    photonloss : float
        Flux of photons that leaves the subbox used for RT
    """

    m1 = ndens.shape[0]
    coldensh_out = np.zeros((m1,m1,m1),order='F')
    phi_ion = np.zeros((m1,m1,m1),order='F')


    nsubbox, photonloss = c2r.raytracing.do_all_sources(srcflux,srcpos,r_RT,subboxsize,coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction)
    return phi_ion, nsubbox, photonloss

# ===================================================================================================
# Raytrace all sources: OCTA GPU version
# ===================================================================================================

def do_all_sources_octa(srcflux,srcpos,r_RT,sig,dr,ndens,xh_av,phi_ion,N):
    """Do raytracing for a list of sources and collect photoioniztaion rates

    Parameters
    ----------
    srcflux : 1D-array
        Strength of each source in ionizing photons per second
    srcpos : 1D-array
        Flattened array containing the positions of the sources. Use read_sources(...,mode="c2ray_octa")
        to correctly flatten the array
    r_RT : float
        Size of the octahedron to raytrace in grid coordinates (can be float)
    sig : float
        Constant photoionization cross-section of hydrogen in cm^2
    dr : float
        Cell dimension in each direction in cm
    ndens : 1D or 3D-array
        Number density of hydrogen in cm^-3. Will be flattened internally if 3D
    xh_av : 1D or 3D-array
        Current time-averaged ionization fraction. Will be flattened internally if 3D
    phi_ion : 1D or 3D-array
        Photoionization rates in s^-1. Will be flattened internally if 3D
    N : int
        Mesh size
    """
    if cuda_init:
        numsrc = srcflux.shape[0]
        phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        cdh_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        # Create flattened copies of arrays
        xh_av_flat = np.ravel(xh_av).astype('float64',copy=True)
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)

        libocta.do_all_sources(srcpos,srcflux,r_RT,cdh_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,numsrc,N)

        phi_ion = np.reshape(phi_ion_flat, (N,N,N))
        return phi_ion
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")