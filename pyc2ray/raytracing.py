import numpy as np
from .utils import printlog
from .load_extensions import load_c2ray, load_octa
from .octa_core import cuda_is_init

# Load extension modules
libc2ray = load_c2ray()
libocta = load_octa()

__all__ = ['do_all_sources','do_all_sources_octa']

# ===================================================================================================
# Raytrace all sources: find column density -> find photoionization rates
# ===================================================================================================

def do_all_sources(dr,normflux,srcpos,max_subbox,subboxsize,ndens,xh_av,sig,
             photo_thin_table,minlogtau,dlogtau,loss_fraction=1e-2):
    """Raytrace all sources and compute photoionization rates

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

    nsubbox, photonloss = libc2ray.raytracing.do_all_sources(normflux,srcpos,max_subbox,subboxsize,
                                                                 coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction,
                                                                 photo_thin_table,minlogtau,dlogtau)

    return phi_ion, nsubbox, photonloss

# ===================================================================================================
# Raytrace all sources: OCTA GPU version
# ===================================================================================================

def do_all_sources_octa(dr,srcflux,srcpos,r_RT,ndens,xh_av,sig,
                  minlogtau,dlogtau,NumTau):
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
    if cuda_is_init():
        NumSrc = srcflux.shape[0]
        N = ndens.shape[0]           # Mesh size
        phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        cdh_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        # Create flattened copies of arrays
        xh_av_flat = np.ravel(xh_av).astype('float64',copy=True)
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)

        # Copy density field to GPU (!! do_all_sources does not touch the density field !!)
        libocta.density_to_device(ndens_flat,N)

        libocta.do_all_sources(srcpos,srcflux,r_RT,cdh_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,NumSrc,N,minlogtau,dlogtau,NumTau)

        phi_ion = np.reshape(phi_ion_flat, (N,N,N))
        return phi_ion
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")