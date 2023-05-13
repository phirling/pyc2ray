import numpy as np
from scipy.integrate import quad,quad_vec
import astropy.constants as ac

h_over_k = (ac.h/(ac.k_B)).cgs.value
two_pi_over_c_square = 2*np.pi/ac.c.cgs.value**2

# These parameters will need to be handled differently
ev2fr=0.241838e15
eth0=13.598
ion_freq_HI=ev2fr*eth0
freq0 = ion_freq_HI
pl_index = 2.8

def BB_SED(freq,T,R_star):
    if (freq*h_over_k/T < 700.0):
        sed = 4*np.pi*R_star**2*two_pi_over_c_square*freq**2/(np.exp(freq*h_over_k/T)-1.0)
    else:
        sed = 0.0
    return sed

def integrate_SED(freq_min,freq_max,T,R_star):
    res = quad(BB_SED,freq_min,freq_max,args=(T,R_star,))
    return res[0]

def cross_section_freq_dependence(freq,grey):
    if grey:
        return 1.0
    else:
        return (freq/freq0)**(-pl_index)

def photo_integrand_vec(freq,tau,T,R_star,grey):
    return BB_SED(freq,T,R_star) * np.exp(-tau*cross_section_freq_dependence(freq,grey))

def make_photo_table(tau,freq_min,freq_max,T,R_star,grey):
    integrand_ = lambda f : photo_integrand_vec(f,tau,T,R_star,grey)
    table = quad_vec(integrand_,freq_min,freq_max)
    return table[0]