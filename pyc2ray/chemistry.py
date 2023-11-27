import numpy as np
from .load_extensions import load_c2ray

libc2ray = load_c2ray()

__all__ = ['global_pass']

# TODO: Add chemistry wrapper function

def global_pass(dt,ndens,temp,xh,phi_ion,bh00,albpow,colh0,temph0,abu_c):
    """ Do chemistry on the whole grid

    Parameters
    ----------
    dt : float
        Timestep in seconds
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    temp : 3D-array
        The initial temperature of each cell in K
    xh : 3D-array
        Initial ionized fraction of each cell
    xh_av : 3D-array
        Current time-averaged ionization fraction over the timestep ()
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
    """
    
    xh_av = np.copy(xh)
    xh_intermed = np.copy(xh)

    temp_av = np.copy(temp)
    temp_intermed = np.copy(temp)

    conv_flag = libc2ray.chemistry.global_pass(dt,ndens,temp,temp_av,temp_intermed,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)
    #print("Number of non-converged cells: ",conv_flag)

    return xh_intermed, temp_intermed