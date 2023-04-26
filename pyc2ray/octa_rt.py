from . import c2ray as c2r
import numpy as np
from .common import printlog
try:
    from . import octa
    gpu = True
except ImportError:
    gpu = False

cuda_init = False

def device_init(N):
    """Initialize GPU and allocate memory for grid data

    Parameters
    ----------
    N : int
        Mesh size in grid coordinates
    """
    if gpu:
        global cuda_init
        octa.device_init(N)
        cuda_init = True
    else:
        raise RuntimeError("Could not initialize GPU: octa library not found")

def device_close():
    """Deallocate GPU memory
    """
    if cuda_init:
        octa.device_close()
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)")

def do_source_octa(srcflux,srcpos,ns,r_RT,sig,dr,ndens,xh_av,phi_ion,N):
    """Do raytracing for a single source [Deprecated]
    """
    if cuda_init:
        numsrc = srcflux.shape[0]
        cdh = np.ravel(np.zeros((N,N,N),dtype='float64'))
        octa.do_source(srcpos,srcflux,ns,r_RT,cdh,sig,dr,ndens,xh_av,phi_ion,numsrc,N)
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)")

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

        octa.do_all_sources(srcpos,srcflux,r_RT,cdh_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,numsrc,N)

        phi_ion = np.reshape(phi_ion_flat, (N,N,N))
        return phi_ion
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)")


def evolve3D_octa(dt,dr,srcflux,srcpos,r_RT,temp,ndens,xh,sig,bh00,albpow,colh0,temph0,abu_c,N,logfile="pyC2Ray.log",quiet=False):
    if cuda_init:
        NumSrc = srcflux.shape[0]    # Number of sources
        NumCells = N*N*N         # Number of cells/points
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

        # Create flattened copies of arrays
        xh_av_flat = np.ravel(xh).astype('float64',copy=True)
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)

        printlog(f"Convergence Criterion (Number of points): {conv_criterion : n}",logfile,quiet)

        # Iterate until convergence in <x> and <y>
        while not converged:
            niter += 1

            # Set rates to 0
            # phi_ion[:,:,:] = 0.0
            coldensh_out_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
            phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))

            # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
            octa.do_all_sources(srcpos,srcflux,r_RT,coldensh_out_flat,sig,dr,ndens_flat,xh_av_flat,phi_ion_flat,NumSrc,N)

            # Reshape for Fortran Chemistry
            phi_ion = np.reshape(phi_ion_flat, (N,N,N))
            
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
            #print(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}")
            printlog(f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100 : .3f} % ), Relative change in ionfrac: {rel_change_xh1 : .2e}",logfile,quiet)

            converged = (conv_flag < conv_criterion) or ( (rel_change_xh1 < convergence_fraction) and (rel_change_xh0 < convergence_fraction))

            # Set previous metrics to current ones and repeat if not converged
            prev_sum_xh1_int = sum_xh1_int
            prev_sum_xh0_int = sum_xh0_int

            # Reshape back
            xh_av_flat = np.ravel(xh_av)

        # When converged, return the updated ionization fractions at the end of the timestep
        return xh_intermed, phi_ion
    else:
        raise RuntimeError("GPU not initialized. Please initialize it by calling octa.device_init(N)")