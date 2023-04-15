import c2ray as c2r
import numpy as np

import matplotlib.pyplot as plt

def evolve3D(dt,dr,srcflux,srcpos,r_RT,temp,ndens,xh,sig,bh00,albpow,colh0,temph0,abu_c):
    """Evolves the ionization fraction over one timestep for the whole grid.

    For a given list of sources and hydrogen number density, computes the evolution of
    the ionization fraction over a timestep due to the radiative transfer from the sources.
    This is the python function that replaces evolve3D in the Fortran version. It calls
    f2py-compiled subroutines that do the computationally expensive part.

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
    r_RT : int
        Size of cube to raytrace around source (half side length of the cube). Set to negative value to
        RT whole box
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

    print(f"Convergence Criterion (Number of points): {conv_criterion : n}")

    # Iterate until convergence in <x> and <y>
    while not converged:
        niter += 1

        # Set rates to 0
        # phi_ion[:,:,:] = 0.0
        phi_ion = np.zeros((m1,m1,m1),order='F')
        coldensh_out = np.zeros((m1,m1,m1),order='F')

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        for ns in range(1,NumSrc+1): # (1-indexation in Fortran)
            c2r.raytracing.do_source(srcflux,srcpos,ns,r_RT,coldensh_out,sig,dr,ndens,xh_av,phi_ion)

        # Apply these rates to compute the updated ionization fraction
        conv_flag = c2r.chemistry.global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)
        
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
        print(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}")

        converged = (conv_flag < conv_criterion) or ( (rel_change_xh1 < convergence_fraction) and (rel_change_xh0 < convergence_fraction))

        # Set previous metrics to current ones and repeat if not converged
        prev_sum_xh1_int = sum_xh1_int
        prev_sum_xh0_int = sum_xh0_int

    # When converged, return the updated ionization fractions at the end of the timestep
    return xh_intermed, phi_ion, coldensh_out