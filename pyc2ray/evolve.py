from . import libc2ray
from .octa_core import gpu, cuda_init
if gpu:
    from octa_core import libocta
import numpy as np
from .utils import printlog

# ===================================================================================================
# evolve3D routine: iterate between raytracing <-> chemistry to solve for the ionization fraction
# ===================================================================================================

def evolve3D(dt,dr,srcflux,srcpos,max_subbox,subboxsize,temp,ndens,xh,sig,bh00,albpow,colh0,temph0,abu_c,
             loss_fraction=1e-2,logfile="pyC2Ray.log",quiet=False):
    
    """Evolves the ionization fraction over one timestep for the whole grid.

    For a given list of sources and hydrogen number density, computes the evolution of
    the ionization fraction over a timestep due to the radiative transfer from the sources.

    Parameters
    ----------
    dt : float
        Timestep in seconds
    dr : float
        Cell dimension in each direction in cm
    srcflux : 1D-array
        Array containing the normalization for the ionizing flux of each source. Shape: (NumSources)
    srcpos : 2D-array
        Array containing the position of each source. Shape: (NumSources,3)
    max_subbox : int
        Size of maximum subbox to raytrace
    subboxsize : int
        Increment size of subbox (e.g. 10 will raytrace 10^3, 20^3, 30^3,... until photon loss is below loss_fraction)
    temp : 3D-array
        The initial temperature of each cell in K
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    xh : 3D-array
        Initial ionized fraction of each cell
    sig : float
        Constant photoionization cross-section of hydrogen in cm^2. TODO: replace by general (frequency-dependent)
        case.
    bh00 : float
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation
    albpow : float
        Power-law index for the H recombination parameter
    colh0 : float
        Hydrogen collisional ionization parameter
    temph0 : float
        Hydrogen ionization energy expressed in K
    abu_c : float
        Carbon abundance
    loss_fraction : float (default: 1e-2)
        Fraction of remaining photons below we stop ray-tracing (subbox technique)
    logfile : str
        Name of the file to append logs to. Default: pyC2Ray.log
    quiet : bool
        Don't write logs to stdout. Default is false

    Returns
    -------
    xh_intermed : 3D-array
        The updated ionization fraction of each cell at the end of the timestep
    phi_ion : 3D-array
        Photoionization rate of each cell due to all sources
    coldensh_out : 3D-array
        Outgoing column density of each cell due to the last source (for debugging, will be removed later on)
    """

    m1 = ndens.shape[0]         # Mesh size
    NumSrc = srcpos.shape[1]    # Number of sources
    NumCells = m1*m1*m1         # Number of cells/points
    # TODO: In c2ray, evolution around a source happens in subboxes of increasing sizes. For now, here, always do the whole grid.
    # last_l = np.ones(3)         # mesh position of left end point for RT
    # last_r = m1 * np.ones(3)    # mesh position of right end point for RT
    conv_flag = NumCells        # Flag that counts the number of non-converged cells (initialized to non-convergence)

    # Convergence Criteria
    convergence_fraction=1.0e-4
    conv_criterion = min(int(convergence_fraction*NumCells), (NumSrc-1)/3)
    
    # Initialize convergence metrics
    prev_sum_xh1_int = 2*NumCells
    prev_sum_xh0_int = 2*NumCells
    converged = False
    niter = 0

    # initialize average and intermediate results to values at beginning of timestep
    xh_av = np.copy(xh)
    xh_intermed = np.copy(xh)

    #print(f"Convergence Criterion (Number of points): {conv_criterion : n}")
    printlog(f"Convergence Criterion (Number of points): {conv_criterion : n}",logfile,quiet)

    # Iterate until convergence in <x> and <y>
    while not converged:
        niter += 1

        # Set rates to 0
        # phi_ion[:,:,:] = 0.0
        phi_ion = np.zeros((m1,m1,m1),order='F')
        coldensh_out = np.zeros((m1,m1,m1),order='F')

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        nsubbox, photonloss = libc2ray.raytracing.do_all_sources(srcflux,srcpos,max_subbox,subboxsize,coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction)

        #print(f"Average number of subboxes: {nsubbox/NumSrc:n}, Total photon loss: {photonloss:.3e}")
        printlog(f"Average number of subboxes: {nsubbox/NumSrc:n}, Total photon loss: {photonloss:.3e}",logfile,quiet)

        # Apply these rates to compute the updated ionization fraction
        conv_flag = libc2ray.chemistry.global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)
        
        # Test Convergence
        sum_xh1_int = np.sum( xh_intermed )
        sum_xh0_int = np.sum( 1.0 - xh_intermed )

        if sum_xh1_int > 0.0:
            rel_change_xh1 = np.abs( (sum_xh1_int - prev_sum_xh1_int) / sum_xh1_int )
        else:
            rel_change_xh1 = 1.0

        if sum_xh0_int > 0.0:
            rel_change_xh0 = np.abs( (sum_xh0_int - prev_sum_xh0_int) / sum_xh0_int )
        else:
            rel_change_xh0 = 1.0

        # Display convergence
        #print(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}")
        printlog(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}",logfile,quiet)

        converged = (conv_flag < conv_criterion) or ( (rel_change_xh1 < convergence_fraction) and (rel_change_xh0 < convergence_fraction))

        # Set previous metrics to current ones and repeat if not converged
        prev_sum_xh1_int = sum_xh1_int
        prev_sum_xh0_int = sum_xh0_int

    # When converged, return the updated ionization fractions at the end of the timestep
    return xh_intermed, phi_ion, coldensh_out


# ===================================================================================================
# evolve3D routine with OCTA raytracing
# ===================================================================================================

def evolve3D_octa(dt,dr,srcflux,srcpos,r_RT,temp,ndens,xh,sig,bh00,albpow,colh0,temph0,abu_c,N,
                  logfile="pyC2Ray.log",quiet=False):
    """Evolves the ionization fraction over one timestep for the whole grid, using OCTA raytracing

    For a given list of sources and hydrogen number density, computes the evolution of
    the ionization fraction over a timestep due to the radiative transfer from the sources.

    Parameters
    ----------
    dt : float
        Timestep in seconds
    dr : float
        Cell dimension in each direction in cm
    srcflux : 1D-array
        Array containing the normalization for the ionizing flux of each source. Shape: (NumSources)
    srcpos : 1D-array
        Array containing the position of each source. This array needs to be flattened correctly, e.g. using the readsources()
        method and setting the mode="pyc2ray_octa" flag.
    r_RT : int
        Determines the size of the octahedron to raytrace: the condition is that it contains a sphere of radius r_RT. This means that
        the height of the octahedron (from the source position to the top vertex) will be sqrt(3)*r_RT.
    temp : 3D-array
        The initial temperature of each cell in K
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    xh : 3D-array
        Initial ionized fraction of each cell
    sig : float
        Constant photoionization cross-section of hydrogen in cm^2. TODO: replace by general (frequency-dependent)
        case.
    bh00 : float
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation
    albpow : float
        Power-law index for the H recombination parameter
    colh0 : float
        Hydrogen collisional ionization parameter
    temph0 : float
        Hydrogen ionization energy expressed in K
    abu_c : float
        Carbon abundance
    logfile : str
        Name of the file to append logs to. Default: pyC2Ray.log
    quiet : bool
        Don't write logs to stdout. Default is false

    Returns
    -------
    xh_intermed : 3D-array
        The updated ionization fraction of each cell at the end of the timestep
    phi_ion : 3D-array
        Photoionization rate of each cell due to all sources
    coldensh_out : 3D-array
        Outgoing column density of each cell due to the last source (for debugging, will be removed later on)
    """
    # Allow a call only if 1. the octa library is present and 2. the GPU memory has been allocated using device_init()
    if cuda_init:
        NumSrc = srcflux.shape[0]    # Number of sources
        NumCells = N*N*N         # Number of cells/points
        conv_flag = NumCells        # Flag that counts the number of non-converged cells (initialized to non-convergence)

        # Convergence Criteria
        convergence_fraction=1.0e-4
        conv_criterion = min(int(convergence_fraction*NumCells), (NumSrc-1)/3)
        
        # Initialize convergence metrics
        prev_sum_xh1_int = 2*NumCells
        prev_sum_xh0_int = 2*NumCells
        converged = False
        niter = 0

        # initialize average and intermediate results to values at beginning of timestep
        xh_av = np.copy(xh)
        xh_intermed = np.copy(xh)

        # Create flattened copies of arrays for CUDA
        xh_av_flat = np.ravel(xh).astype('float64',copy=True)
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)

        # Initialize Flat Column density & ionization rate arrays. These are used to store the
        # output of OCTA. TODO: python column density array is actually not needed ?
        coldensh_out_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))

        printlog(f"Convergence Criterion (Number of points): {conv_criterion : n}",logfile,quiet)

        # Iterate until convergence in <x> and <y>
        while not converged:
            niter += 1

            # Rates are set to zero on the GPU in the octa code

            # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
            libocta.do_all_sources(srcpos,srcflux,r_RT,coldensh_out_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,NumSrc,N)

            # Reshape for C2Ray Fortran Chemistry
            phi_ion = np.reshape(phi_ion_flat, (N,N,N))
            
            # Apply these rates to compute the updated ionization fraction
            conv_flag = libc2ray.chemistry.global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)
            
            # Test Convergence
            sum_xh1_int = np.sum( xh_intermed )
            sum_xh0_int = np.sum( 1.0 - xh_intermed )

            if sum_xh1_int > 0.0:
                rel_change_xh1 = np.abs( (sum_xh1_int - prev_sum_xh1_int) / sum_xh1_int )
            else:
                rel_change_xh1 = 1.0

            if sum_xh0_int > 0.0:
                rel_change_xh0 = np.abs( (sum_xh0_int - prev_sum_xh0_int) / sum_xh0_int )
            else:
                rel_change_xh0 = 1.0

            # Display convergence
            printlog(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}",logfile,quiet)

            # Set convergence criterion
            converged = (conv_flag < conv_criterion) or ( (rel_change_xh1 < convergence_fraction) and (rel_change_xh0 < convergence_fraction))

            # Set previous metrics to current ones and repeat if not converged
            prev_sum_xh1_int = sum_xh1_int
            prev_sum_xh0_int = sum_xh0_int

            # Flatten the updated time-average fraction for the next OCTA iteration
            xh_av_flat = np.ravel(xh_av)

        # When converged, return the updated ionization fractions at the end of the timestep
        return xh_intermed, phi_ion
    

    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)")