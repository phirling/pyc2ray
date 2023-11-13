from .c2ray_base import C2Ray, YEAR, Mpc
from .utils.sourceutils import read_test_sources
import numpy as np
from astropy import units as u
from astropy import constants as ac
import pickle as pkl
from .radiation import make_tau_table
from .radiation import BlackBodySource, UVBSource_FG2009
from .asora_core import device_init, device_close, photo_table_to_device

__all__ = ['C2Ray_Minihalo']

ev2fr = 0.241838e15

# ======================================================================
# This file contains the C2Ray_Minihalos subclass of C2Ray, which is a
# version used for the Master project to simulate reionization of
# dark matter minihalos (~10^9 Msun)
# ======================================================================

class C2Ray_Minihalo(C2Ray):
    def __init__(self, paramfile, Nmesh, use_gpu,boxsize):
        """A C2Ray Minihalo simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation
        Nmesh : int
            Mesh size (number of cells in each dimension)
        use_gpu : bool
            Whether to use the GPU-accelerated ASORA library for raytracing
        boxsize : float
            Box size in Mpc (overrides the value from paramfile)
        """
        self.boxsize_c = boxsize * Mpc
        super().__init__(paramfile, Nmesh, use_gpu)
        self.printlog('Running: "C2Ray for Minihalo" \n')

    def read_sources(self,file,numsrc,S_star_ref = 1e48):
        """ Read in a source file formatted for Test-C2Ray

        Read in a file that gives source positions and total ionizing flux
        in s^-1, formatted like for the original C2Ray code, e.g.

        1.0
        40 40 40 1e54 1.0

        for one source at (40,40,40) (Fortran indexing) and of total flux
        1e54 photons/s. The last row is conventional.

        Parameters
        ----------
        file : string
            Name of the file to read
        numsrc : int
            Numer of sources to read from the file
        S_star_ref : float, optional
            Flux of the reference source. Default: 1e48
            There is no real reason to change this, but if it is changed, the value in src/c2ray/photorates.f90
            has to be changed accordingly and the library recompiled.
            
        Returns
        -------
        src_pos : 2D-array of shape (3,numsrc)
            Source positions
        src_flux : 1D-array of shape (numsrc)
            Normalization of the strength of each source, i.e. total ionizing flux / reference flux
        """
        return read_test_sources(file,numsrc,S_star_ref)
        

    def write_output(self,z):
        """Write ionization fraction & ionization rates as pickle files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.pkl"
        with open(self.results_basename + "xfrac" + suffix,"wb") as f:
            pkl.dump(self.xh,f)
        with open(self.results_basename + "IonRates" + suffix,"wb") as f:
            pkl.dump(self.phi_ion,f)

    def write_output_numbered(self,n):
        """Write ionization fraction & ionization rates as pickle files with number rather than redshift

        Parameters
        ----------
        n : int
            Number of the file
        """
        suffix = f"_{n:n}.pkl"
        with open(self.results_basename + "xfrac" + suffix,"wb") as f:
            pkl.dump(self.xh,f)
        with open(self.results_basename + "IonRates" + suffix,"wb") as f:
            pkl.dump(self.phi_ion,f)

    def set_constant_average_density(self,ndens,z):
        """Helper function to set the density grid to a constant value

        Parameters
        ----------
        ndens : float
            Value of the hydrogen density in cm^-3. When in a cosmological
            run, this is the comoving density, or the proper density at
            z = 0.
        z : float
            Redshift to scale the density. When cosmological is false,
            this parameter has no effect and the initial redshift specified
            in the parameter file is used at each call.
        """
        # This is the same as in C2Ray
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0
        self.ndens = ndens * np.ones(self.shape,order='F') * (1+redshift)**3

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the minihalo case
    # =====================================================================================================
    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.dr_c = self.boxsize_c / self.N

        self.printlog(f"Welcome! Mesh size is N = {self.N:n}.")
        self.printlog(f"Simulation Box size (comoving Mpc): {self.boxsize_c/Mpc:.3e}")

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # Set R_max (LLS 3) in cell units
        self.R_max_LLS = self._ld['Photo']['R_max_cMpc'] * Mpc * self.N / (self.boxsize_c)
        self.printlog(f"Maximum comoving distance for photons from source (type 3 LLS): {self._ld['Photo']['R_max_cMpc'] : .3e} comoving Mpc")
        self.printlog(f"This corresponds to                                             {self.R_max_LLS : .3f} grid cells.")

    def _radiation_init(self):
        """Set up radiation tables for ionization/heating rates
        """
        # Create optical depth table (log-spaced)
        self.minlogtau = self._ld['Photo']['minlogtau']
        self.maxlogtau = self._ld['Photo']['maxlogtau']
        self.NumTau = self._ld['Photo']['NumTau']
        self.SourceType = self._ld['Photo']['SourceType']
        self.grey = self._ld['Photo']['grey']

        if self.grey:
            self.printlog(f"Warning: Using grey opacity")
        else:
            self.printlog(f"Using power-law opacity with {self.NumTau:n} table points between tau=10^({self.minlogtau:n}) and tau=10^({self.maxlogtau:n})")

        # The actual table has NumTau + 1 points: the 0-th position is tau=0 and the
        # remaining NumTau points are log-spaced from minlogtau to maxlogtau (same as in C2Ray)
        self.tau, self.dlogtau = make_tau_table(self.minlogtau,self.maxlogtau,self.NumTau)

        ion_freq_HI = ev2fr * self.eth0
        ion_freq_HeII = ev2fr * self.ethe1

        if self.SourceType == 'blackbody':
            freq_min = ion_freq_HI
            freq_max = 10*ion_freq_HeII

            # Initialize Black-Body Source
            self.bb_Teff = self._ld['BlackBodySource']['Teff']
            self.cs_pl_idx_h = self._ld['BlackBodySource']['cross_section_pl_index']
            radsource = BlackBodySource(self.bb_Teff, self.grey, ion_freq_HI, self.cs_pl_idx_h)
            
            # Print info
            self.printlog(f"Using Black-Body sources with effective temperature T = {radsource.temp :.1e} K and Radius {(radsource.R_star/ac.R_sun.to('cm')).value : .3e} rsun")
            self.printlog(f"Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz")
            self.printlog(f"This is Energy:           {freq_min/ev2fr:.3e} to {freq_max/ev2fr:.3e} eV")

            # Integrate table
            self.printlog("Integrating photoionization rates tables...")
            self.photo_thin_table, self.photo_thick_table = radsource.make_photo_table(self.tau,freq_min,freq_max,1e48)

        elif self.SourceType == 'uvb_fg2009':
            self.uvb_zred = self._ld['UVBSource_FG2009']['uvb_zred']
            radsource = UVBSource_FG2009(self.grey,2)

            # Print info
            self.printlog(f"Using z = {self.uvb_zred:.1f} Faucher-Guigère 2009 UV Background sources")

            # For now, we simply integrate until 100 * the HI ionization frequency to get the rates
            self.printlog("Integrating photoionization rates tables...")
            self.photo_thin_table, self.photo_thick_table = radsource.make_photo_table(self.tau,0,2) # nb integration bounds are given in log10(freq/freq_HI)
        else:
            raise NameError("Unknown source type : ",self.SourceType)
        
        # Copy radiation table to GPU
        if self.gpu:
            photo_table_to_device(self.photo_thin_table,self.photo_thick_table)
            self.printlog("Successfully copied radiation tables to GPU memory.")
    
    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.time = self.age_0
        self.zred = self.zred_0

    def _material_init(self):
        """Initialize material properties of the grid
        """
        xh0 = self._ld['Material']['xh0']
        temp0 = self._ld['Material']['temp0']

        self.ndens = np.empty(self.shape,order='F')
        self.xh = xh0 * np.ones(self.shape,order='F')
        self.temp = temp0 * np.ones(self.shape,order='F')
        self.phi_ion = np.zeros(self.shape,order='F')
        self.avg_dens = self._ld['Material']['avg_dens']

    def _output_init(self):
        """ Set up output & log file
        """
        self.results_basename = self._ld['Output']['results_basename']
        self.logfile = self.results_basename + self._ld['Output']['logfile']
        title = '                 _________   ____            \n    ____  __  __/ ____/__ \ / __ \____ ___  __\n   / __ \/ / / / /    __/ // /_/ / __ `/ / / /\n  / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ / \n / .___/\__, /\____//____/_/ |_|\__,_/\__, /  \n/_/    /____/                        /____/   \n'
        with open(self.logfile,"w") as f:
            f.write("\nLog file for pyC2Ray \n\n")
            #f.write(title + "\nLog file for pyC2Ray. \n\n") # Clear file and write header line
        self.printlog(title)