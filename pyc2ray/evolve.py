import numpy as np
from .utils import printlog
from .utils.sourceutils import format_sources
from .load_extensions import load_c2ray, load_asora
from .asora_core import cuda_is_init

# Load extension modules
libc2ray = load_c2ray()
libasora = load_asora()

__all__ = ['evolve3D', 'evolve3D_gpu']

# ===================================================================================================
# evolve3D routine: iterate between raytracing <-> chemistry to solve for the ionization fraction
# ===================================================================================================
def evolve3D(dt,dr,normflux,srcpos,max_subbox,subboxsize,temp,ndens,xh,sig,bh00,albpow,colh0,temph0,abu_c,
             photo_thin_table,minlogtau,dlogtau,loss_fraction=1e-2,logfile="pyC2Ray.log",quiet=False):
    
    """Evolves the ionization fraction over one timestep for the whole grid.

    For a given list of sources and hydrogen number density, computes the evolution of
    the ionization fraction over a timestep due to the radiative transfer from the sources.

    Parameters
    ----------
    dt : float
        Timestep in seconds
    dr : float
        Cell dimension in each direction in cm
    normflux : 1D-array of shape (numsrc)
        Array containing the normalization factor (relative to S_star = 1e48 by default) of the total ionizing flux of each source
    srcpos : 2D-array of shape (3,numsrc)
        Array containing the 3D grid position of each source
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
    photo_thin_table : 1D-array
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    dlogtau : float
        Step size of the logτ-table
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
        nsubbox, photonloss = libc2ray.raytracing.do_all_sources(normflux,srcpos,max_subbox,subboxsize,
                                                                 coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction,
                                                                 photo_thin_table,minlogtau,dlogtau)

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
        printlog(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}",logfile,quiet)

        converged = (conv_flag < conv_criterion) or ( (rel_change_xh1 < convergence_fraction) and (rel_change_xh0 < convergence_fraction))

        # Set previous metrics to current ones and repeat if not converged
        prev_sum_xh1_int = sum_xh1_int
        prev_sum_xh0_int = sum_xh0_int

    # When converged, return the updated ionization fractions at the end of the timestep
    return xh_intermed, phi_ion


# ===================================================================================================
# evolve3D routine with GPU raytracing (ASORA)
# ===================================================================================================
def evolve3D_gpu(dt,dr,normflux,srcpos,r_RT,temp,ndens,xh,sig,bh00,albpow,colh0,temph0,abu_c,
                  minlogtau,dlogtau,NumTau,logfile="pyC2Ray.log",quiet=False):
    """Evolves the ionization fraction over one timestep for the whole grid, using ASORA raytracing

    For a given list of sources and hydrogen number density, computes the evolution of
    the ionization fraction over a timestep due to the radiative transfer from the sources.

    !! Calling this function assumes that the radiation tables have been copied to the device using
    photo_table_to_device()

    Parameters
    ----------
    dt : float
        Timestep in seconds
    dr : float
        Cell dimension in each direction in cm
    normflux : 1D-array of shape (numsrc)
        Array containing the normalization factor (relative to S_star = 1e48 by default) of the total ionizing flux of each source
    srcpos : 2D-array of shape (3,numsrc)
        Array containing the 3D grid position of each source
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
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    dlogtau : float
        Step size of the logτ-table
    NumTau : int
        Number of table points
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
    # Allow a call only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
    if cuda_is_init():
        NumSrc = normflux.shape[0]   # Number of sources
        N = temp.shape[0]           # Mesh size
        NumCells = N*N*N            # Number of cells/points
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

        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
        xh_av_flat = np.ravel(xh).astype('float64',copy=True)
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)
        srcpos_flat, normflux_flat = format_sources(srcpos,normflux)

        # Initialize Flat Column density & ionization rate arrays. These are used to store the
        # output of the raytracing module. TODO: python column density array is actually not needed ?
        coldensh_out_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))

        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources does not touch the density field !!)
        libasora.density_to_device(ndens_flat,N)

        printlog(f"Convergence Criterion (Number of points): {conv_criterion : n}",logfile,quiet)

        # Iterate until convergence in <x> and <y>
        while not converged:
            niter += 1

            # Rates are set to zero on the GPU in the asora code

            # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
            libasora.do_all_sources(srcpos_flat,normflux_flat,r_RT,coldensh_out_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,NumSrc,N,minlogtau,dlogtau,NumTau)

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
            
            # Flatten the updated time-average fraction for the next ASORA call
            xh_av_flat = np.ravel(xh_av)

        # When converged, return the updated ionization fractions at the end of the timestep
        return xh_intermed, phi_ion
    

    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")







# Work in progress

# =========================================================================
# This file contains the main time-evolution subroutine, which updates
# the ionization state of the whole grid over one timestep, using the
# C2Ray method.
#
# The raytracing step can use either the sequential (subbox, cubic)
# technique which runs in Fortran on the CPU or the accelerated technique,
# which runs using the ASORA library on the GPU.
# 
# When using the latter, some notes apply:
# For performance reasons, the program minimizes the frequency at which
# data is moved between the CPU and the GPU (this is a big bottleneck).
# In particular, the radiation tables, which in principle shouldn't change
# over the run of a simulation, need to be copied separately to the GPU
# using the photo_table_to_device() method of the module. This is done
# automatically when using the C2Ray subclasses but must be done manually
# if for some reason you are calling the evolve3D routine directly without
# using the C2Ray subclasses.
# =========================================================================

def evolve3D_unified(dt,dr,
        src_pos,src_flux,
        r_RT,use_gpu,max_subbox,loss_fraction,
        temp,ndens,xh,
        photo_thin_table,minlogtau,dlogtau,
        sig,bh00,albpow,colh0,temph0,abu_c,
        logfile="pyC2Ray.log",quiet=False):

    """Evolves the ionization fraction over one timestep for the whole grid

    Warning: Calling this function with use_gpu = True assumes that the radiation
    tables have previously been copied to the GPU using photo_table_to_device()

    Parameters
    ----------
    dt : float
        Timestep in seconds
    dr : float
        Cell dimension in each direction in cm
    src_pos : 2D-array of shape (3,numsrc)
        Array containing the 3D grid position of each source, in Fortran indexing (from 1)
    src_flux : 1D-array of shape (numsrc)
        Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default)
    r_RT : int
        Parameter which determines the size of the raytracing volume around each source:
        * When using CPU (cubic) RT, this sets the increment of the cubic region (subbox) that will be treated.
        Raytracing stops when either max_subbox is reached or when the photon loss is low enough. For example, if
        r_RT = 5, the size of the cube around the source will grow as 10^3, 20^3, ...
        * When using GPU (octahedral) RT with ASORA, this sets the size of the octahedron such that a sphere of
        radius r_RT fits inside the octahedron.
    use_gpu : bool
        Whether or not to use the GPU-accelerated ASORA library for raytracing.
    max_subbox : int
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true
    loss_fraction : float
        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true
    temp : 3D-array
        The initial temperature of each cell in K
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    xh : 3D-array
        The initial ionized fraction of each cell
    photo_thin_table : 1D-array
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
        in a separate (previous) step, using photo_table_to_device()
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    dlogtau : float
        Step size of the logτ-table  
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
    xh_new : 3D-array
        The updated ionization fraction of each cell at the end of the timestep
    phi_ion : 3D-array
        Photoionization rate of each cell due to all sources
    """

    # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
    if (use_gpu and not cuda_is_init()):
        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")

    # Set some constant sizes
    NumSrc = src_flux.shape[0]   # Number of sources
    N = temp.shape[0]           # Mesh size
    NumCells = N*N*N            # Number of cells/points
    conv_flag = NumCells        # Flag that counts the number of non-converged cells (initialized to non-convergence)
    NumTau = photo_thin_table.shape[0]

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

    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
    if use_gpu:
        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
        xh_av_flat = np.ravel(xh).astype('float64',copy=True)
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)
        srcpos_flat, normflux_flat = format_sources(src_pos,src_flux)

        # Initialize Flat Column density & ionization rate arrays. These are used to store the
        # output of the raytracing module. TODO: python column density array is actually not needed ?
        coldensh_out_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))

        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
        libasora.density_to_device(ndens_flat,N)

    # -----------------------------------------------------------
    # Start Evolve step, Iterate until convergence in <x> and <y>
    # -----------------------------------------------------------
    printlog(f"Convergence Criterion (Number of points): {conv_criterion : n}",logfile,quiet)

    while not converged:
        niter += 1

        # --------------------
        # (1): Raytracing Step
        # --------------------
        # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
        if not use_gpu:
            phi_ion = np.zeros((m1,m1,m1),order='F')
            coldensh_out = np.zeros((m1,m1,m1),order='F')

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        if use_gpu:
            # Use GPU raytracing
            libasora.do_all_sources(srcpos_flat,normflux_flat,r_RT,coldensh_out_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,NumSrc,N,minlogtau,dlogtau,NumTau)
        else:
            # Use CPU raytracing with subbox optimization
            nsubbox, photonloss = libc2ray.raytracing.do_all_sources(src_flux,src_pos,max_subbox,r_RT,coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction,photo_thin_table,minlogtau,dlogtau)
            printlog(f"Average number of subboxes: {nsubbox/NumSrc:n}, Total photon loss: {photonloss:.3e}",logfile,quiet)

        # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
        if use_gpu:
            phi_ion = np.reshape(phi_ion_flat, (N,N,N))
        
        # ---------------------
        # (2): ODE Solving Step
        # ---------------------
        # Apply the global rates to compute the updated ionization fraction
        conv_flag = libc2ray.chemistry.global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)

        # ----------------------------
        # (3): Test Global Convergence
        # ----------------------------
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

        converged = (conv_flag < conv_criterion) or ( (rel_change_xh1 < convergence_fraction) and (rel_change_xh0 < convergence_fraction))

        # Set previous metrics to current ones and repeat if not converged
        prev_sum_xh1_int = sum_xh1_int
        prev_sum_xh0_int = sum_xh0_int

        # Finally, when using GPU, need to reshape x back for the next ASORA call
        if (use_gpu and not converged):
            xh_av_flat = np.ravel(xh_av)

    # When converged, return the updated ionization fractions at the end of the timestep
    xh_new = xh_intermed
    return xh_new, phi_ion
