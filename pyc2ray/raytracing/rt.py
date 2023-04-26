import numpy as np
from .. import c2ray as c2r

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