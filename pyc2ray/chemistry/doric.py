from .. import c2ray as c2r

def global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c):
    """ Apply ionization rates to the whole grid

    Solves the chemistry equation for the ionization fraction x on the whole grid using
    the "doric" method.

    Parameters
    ----------
    dt : float
        Timestep in seconds
    ndens : 3D-array
        Total Hydrogen number density of each cell in cm^-3
    temp : 3D-array
        Initial temperature of each cell in Kelvin
    xh : 3D-array
        Value of the ionization fraction of each cell at the beginning of the timestep (initial xHII)
    xh_av : 3D-array
        Current (intermediate) value of the time-averaged ionization fraction of each cell
    xh_intermed : 3D-array
        Current (intermediate) value of the final ionization fraction of each cell
    phi_ion : 3D-array
        Total ionization rate of each cell, in s^-1
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
    conv_flag : int
        Number of non-converged cells
    """

    conv_flag = c2r.chemistry.global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c)
    return conv_flag