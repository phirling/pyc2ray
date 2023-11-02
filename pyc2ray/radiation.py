import numpy as np
from scipy.integrate import quad,quad_vec #, romberg <- TODO: add option for Romberg integration
import astropy.constants as ac

# For detailed comparisons with C2Ray, we use the same exact value for the constants
# This can be changed to the astropy values once consistency between the two codes has been established

#h_over_k = (ac.h/(ac.k_B)).cgs.value
h_over_k = 6.6260755e-27 / 1.381e-16

#two_pi_over_c_square = 2*np.pi/ac.c.cgs.value**2
pi =3.141592654
c = 2.997925e+10
two_pi_over_c_square = 2.0*pi/(c*c)

__all__ = ['BlackBodySource','make_tau_table']

class BlackBodySource:
    def __init__(self, temp, grey, freq0, pl_index) -> None:
        self.temp = temp
        self.grey = grey
        self.freq0 = freq0
        self.pl_index = pl_index
        self.R_star = 1.0

    def SED(self,freq):
        if (freq*h_over_k/self.temp < 700.0):
            sed = 4*np.pi*self.R_star**2*two_pi_over_c_square*freq**2/(np.exp(freq*h_over_k/self.temp)-1.0)
        else:
            sed = 0.0
        return sed
    
    def integrate_SED(self,f1,f2):
        res = quad(self.SED,f1,f2)
        return res[0]
    
    def normalize_SED(self,f1,f2,S_star_ref):
        S_unscaled = self.integrate_SED(f1,f2)
        S_scaling = S_star_ref / S_unscaled
        self.R_star = np.sqrt(S_scaling) * self.R_star

    def cross_section_freq_dependence(self,freq):
        if self.grey:
            return 1.0
        else:
            return (freq/self.freq0)**(-self.pl_index)
    
    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently
    # for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self,freq,tau):
        itg = self.SED(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0,itg,0.0)
    
    def _photo_thin_integrand_vec(self,freq,tau):
        itg = self.SED(freq) * self.cross_section_freq_dependence(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0,itg,0.0)
    
    def make_photo_table(self,tau,freq_min,freq_max,S_star_ref):
        self.normalize_SED(freq_min,freq_max,S_star_ref)
        integrand_thin = lambda f : self._photo_thin_integrand_vec(f,tau)
        integrand_thick = lambda f : self._photo_thick_integrand_vec(f,tau)
        table_thin = quad_vec(integrand_thin,freq_min,freq_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,freq_min,freq_max,epsrel=1e-12)[0]
        return table_thin, table_thick
    
    # TODO: add Romberg integrator
    # def make_photo_table(self,tau,freq_min,freq_max,S_star_ref):
    #     self.normalize_SED(freq_min,freq_max,S_star_ref)
    #     table = np.empty(tau.shape)
    #     for i,ttau in enumerate(tau):
    #         integrand_ = lambda f : self._photo_integrand_vec(f,ttau)
    #         tpoint = romberg(integrand_,freq_min,freq_max,divmax=12,show=False)
    #         table[i] = tpoint
    #     return table

def make_tau_table(minlogtau,maxlogtau,NumTau):
    """Utility function to create optical depth array for C2Ray

    Parameters
    ----------
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    minlogtau : float
        Base 10 log of the maximum value of the table in τ
    NumTau : int
        Number of points in the table, excluding τ = 0
    
    Returns
    -------
    tau : 1D-array of shape (NumTau + 1)
        Array of optical depths log-distributed between minlogtau and maxlogtau. The 0-th
        entry is τ = 0 and so the array has shape NumTau+1 (same convention as c2ray)
    dlogtau : float
        Table step size in log10
    """
    dlogtau = (maxlogtau-minlogtau)/NumTau
    tau = np.empty(NumTau+1)
    tau[0] = 0.0
    tau[1:] = 10**(minlogtau + np.arange(NumTau)*dlogtau)
    return tau, dlogtau