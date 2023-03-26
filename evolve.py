import c2ray_core as c2r
import numpy as np

import matplotlib.pyplot as plt
"""
This is the python function that replaces evolve3D in the Fortran version. It calls
f2py-compiled subroutines that do the computationally expensive part.
"""
def evolve3D(dt,dr,srcflux,srcpos,temp,ndens,coldensh_out,xh,phi_ion,sig,bh00,albpow,colh0,temph0,abu_c):
    

    m1 = ndens.shape[0]         # Mesh size
    NumSrc = srcpos.shape[1]    # Number of sources
    NumCells = m1*m1*m1         # Number of cells/points
    # TODO: In c2ray, evolution around a source happens in subboxes of increasing sizes. For now, here, always do the whole grid.
    last_l = np.ones(3)         # mesh position of left end point for RT
    last_r = m1 * np.ones(3)    # mesh position of right end point for RT
    conv_flag = NumCells        # Flag that counts the number of non-converged cells (initialized to non-convergence)

    # Convergence Criteria
    convergence_fraction=1.0e-4
    conv_criterion = min(int(convergence_fraction*NumCells), (NumSrc-1)/3)
    
    # TODO: add MPI distribution of sources. For now, use "static serial" mode

    # Initialize convergence metrics
    #rel_change_xh = 2*convergence_fraction
    prev_sum_xh1_int = 2*NumCells

    converged = False
    niter = 0

    # initialize average and intermediate results to initial values
    xh_av = np.copy(xh)
    xh_intermed = np.copy(xh)

    print(f"Convergence Criterion (Number of points): {conv_criterion : n}")
    while not converged:
        niter += 1

        sum_xh1_int = np.sum( xh_intermed )

        if sum_xh1_int > 0.0:
            rel_change_xh = np.abs( (sum_xh1_int - prev_sum_xh1_int) / sum_xh1_int )
        else:
            rel_change_xh = 1.0

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        # First, reset rates:
        phi_ion[:,:,:] = 0.0  #= np.zeros((m1,m1,m1),order='F')
        for ns in range(1,NumSrc+1): # (1-indexation in Fortran)
            c2r.raytracing.do_source(srcflux,srcpos,ns,last_l,last_r,coldensh_out,sig,dr,ndens,xh_av,phi_ion)

        # Now, apply these rates to compute the updated ionization fraction
        conv_flag = c2r.chemistry.global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)
        print(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh : .2e}")

        converged = (conv_flag < conv_criterion) or (rel_change_xh < convergence_fraction)

        #plt.imshow(xh_intermed[2,:,:],norm='log',cmap='rainbow')
        #plt.show()
        prev_sum_xh1_int = sum_xh1_int
        #if niter > 4:
        #    break
    return xh_intermed