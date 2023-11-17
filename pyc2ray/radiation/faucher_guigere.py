import numpy as np
from scipy.integrate import quad,quad_vec
from astropy.constants import h as hplanck_ac
from astropy.constants import Ryd as Ryd_ac
from astropy.constants import c as c_ac
import os

"""
Calibrated UV background spectra from Faucher-Guigère et al 2009
"""
sigma_0 = 6.3e-18
hplanck = hplanck_ac.cgs.value
dir_path = os.path.dirname(os.path.realpath(__file__))
ion_freq_HI = (Ryd_ac * c_ac).cgs.value

def format_raw_spectrum_data(raw_data):
    """
    Raw spectrum data:
    dim0: nu/nu_HI
    dim1: J_nu in units [1e-21 erg s^-1 cm^-2 sr^-1 Hz^-1]

    Formatted spectrum data (for integration):
    dim0: log10(nu/nu_HI)       # We denote x = nu/nu_HI, and y = log10(x)
    dim1: log10(J_nu), with J_nu in units [erg s^-1 cm^-2 sr^-1 Hz^-1]
    """
    dim0 = np.log10(raw_data[0])
    dim1 = np.log10(1e-21 * raw_data[1])
    return np.array([dim0,dim1])

# Read in data from Faucher-Guigère et al. 2009
redshifts = [0,1,2,3]
spectra = {}
for z in redshifts:
    try:
        fn = f"z={z:n}.csv"
        kn = f"z={z:n}"
        fpath = os.path.join(dir_path,'FG_data',fn)
        raw_data = np.loadtxt(fpath,delimiter=',').T
        spectra[kn] = format_raw_spectrum_data(raw_data)
    except:
        raise RuntimeError(f"Error reading data file " + fpath)
    
__all__ = ['UVBSource_FG2009']

class UVBSource_FG2009:
    """A point-source equivalent of the Faucher-Guigère 2009 Ultraviolet Background (UVB)
    """
    def __init__(self,grey, zred) -> None:
        self.grey = grey
        self.zred = zred
        kn = f"z={zred:n}"
        try:
            self.data = spectra[kn]
        except:
            raise ValueError(f"Redshift z = {zred:.3f} not available")

    def logJnu(self,y):
        val = np.interp(y,self.data[0],self.data[1])
        return val
    
    def cross_section_freq_dependence(self,y,Z=1):
        xx = 10**y / Z**2
        e = np.sqrt(xx-1)
        val = 1.0/Z**2 * (1/xx)**4 * np.exp( 4 - (4*np.arctan(e))/e ) / (1-np.exp(-2*np.pi/e))
        return np.where(y <= 0, 0, val)
    
    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently
    # for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self,y,tau):
        itg = 10**self.logJnu(y) * np.log(10) * np.exp(-tau*self.cross_section_freq_dependence(y)) / hplanck
        # To avoid overflow in the exponential, check
        return np.where(tau*self.cross_section_freq_dependence(y) < 700.0,itg,0.0)
    
    def _photo_thin_integrand_vec(self,y,tau):
        itg = 10**self.logJnu(y) * np.log(10) * self.cross_section_freq_dependence(y) * np.exp(-tau*self.cross_section_freq_dependence(y)) / hplanck
        return np.where(tau*self.cross_section_freq_dependence(y) < 700.0,itg,0.0)
    
    # Heating rates WIP
    def _heat_thick_integrand_vec(self,y,tau):
        photo_thick = self._photo_thick_integrand_vec(y,tau)
        return hplanck * ion_freq_HI * (10**y - 1) * photo_thick
    
    def _heat_thin_integrand_vec(self,y,tau):
        photo_thin = self._photo_thin_integrand_vec(y,tau)
        return hplanck * ion_freq_HI * (10**y - 1) * photo_thin
    
    def make_photo_table(self,tau,y_min,y_max):
        # self.normalize_SED(freq_min,freq_max,S_star_ref)
        integrand_thin = lambda y : self._photo_thin_integrand_vec(y,tau)
        integrand_thick = lambda y : self._photo_thick_integrand_vec(y,tau)
        table_thin = quad_vec(integrand_thin,y_min,y_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,y_min,y_max,epsrel=1e-12)[0]
        return table_thin, table_thick
    
    def make_heat_table(self,tau,y_min,y_max):
        integrand_thin = lambda y : self._heat_thin_integrand_vec(y,tau)
        integrand_thick = lambda y : self._heat_thick_integrand_vec(y,tau)
        table_thin = quad_vec(integrand_thin,y_min,y_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,y_min,y_max,epsrel=1e-12)[0]
        return table_thin, table_thick