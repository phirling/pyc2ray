import yaml
import atexit
import re
import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader
from .utils.logutils import printlog
from .evolve import evolve3D
from .asora_core import device_init, device_close, photo_table_to_device
from .radiation import BlackBodySource, make_tau_table

from .utils.other_utils import get_redshifts_from_output, find_bins
import tools21cm as t2c
from .utils import get_source_redshifts
import h5py

__all__ = ['C2Ray_244Test']

pc = 3.086e18            # C2Ray value: 3.086e18
YEAR = (1*u.yr).to('s').value           # C2Ray value: 3.15576E+07
ev2fr = 0.241838e15                     # eV to Frequency (Hz)
ev2k = 1.0/8.617e-05                    # eV to Kelvin
kpc = 1e3*pc                            # kiloparsec in cm
Mpc = 1e6*pc                            # megaparsec in cm
msun2g = 1.98892e33       # solar mass to grams
m_p = 1.672661e-24

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================

class C2Ray_244Test():
    def __init__(self,paramfile,Nmesh,use_gpu):
        """Basis class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation
        Nmesh : int
            Mesh size (number of cells in each dimension)
        use_gpu : bool
            Whether to use the GPU-accelerated ASORA library for raytracing

        """
        # Read YAML parameter file and set main properties
        self._read_paramfile(paramfile)
        self.N = Nmesh
        self.shape = (Nmesh,Nmesh,Nmesh)

        # Set Raytracing mode
        if use_gpu:
            self.gpu = True
            # Allocate GPU memory
            device_init(Nmesh)
            # Register deallocation function (automatically calls this on program termination)
            atexit.register(self._gpu_close)
        else:
            self.gpu = False

        # Initialize Simulation
        self._param_init()
        self._output_init()
        self._grid_init()
        self._cosmology_init()
        self._redshift_init()
        self._material_init()
        self._sources_init()
        self._radiation_init()
        if self.gpu: self.printlog("Using ASORA Raytracing")
        else: self.printlog("Using CPU Raytracing")
        self.printlog("Starting simulation... \n\n")
        self.printlog('Running: "C2Ray for 244 Mpc/h test"')

    # =====================================================================================================
    # TIME-EVOLUTION METHODS
    # =====================================================================================================
    def set_timestep(self, z1, z2, num_timesteps):
        """Compute timestep to use between redshift slices

        Parameters
        ----------
        z1 : float
            Initial redshift
        z2 : float
            Next redshift
        num_timesteps : int
            Number of timesteps between the two slices
        
        Returns
        -------
        dt : float
            Timestep to use in seconds
        """
        #t1 = self.cosmology.lookback_time(z1).to('s').value
        #t2 = self.cosmology.lookback_time(z2).to('s').value
        t2 = self.zred2time(z1)
        t1 = self.zred2time(z2)
        # we do t1-t2 since ti are lookback times (not ages)
        dt = (t1-t2)/num_timesteps
        return dt
    
    def evolve3D(self, dt, src_flux, src_pos, r_RT, max_subbox):
        """Evolve the grid over one timestep

        Raytrace all sources, compute cumulative photoionization rate of each cell and
        do chemistry. This is done until convergence in the ionized fraction is reached.

        Parameters
        ----------
        dt : float
            Timestep in seconds (typically generated using set_timestep method)
        src_flux : 1D-array of shape (numsrc)
            Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default)
        src_pos : 2D-array of shape (3,numsrc)
            Array containing the 3D grid position of each source, in Fortran indexing (from 1)
        r_RT : int
            Parameter which determines the size of the raytracing volume around each source:
            * When using CPU (cubic) RT, this sets the increment of the cubic region (subbox) that will be treated.
            Raytracing stops when either max_subbox is reached or when the photon loss is low enough. For example, if
            r_RT = 5, the size of the cube around the source will grow as 10^3, 20^3, ...
            * When using GPU (octahedral) RT with ASORA, this sets the size of the octahedron such that a sphere of
            radius r_RT fits inside the octahedron.
        max_subbox : int
            Maximum size of the subbox when using cubic raytracing. When using ASORA, this
            parameter has no effect.
        """
        self.xh, self.phi_ion = evolve3D(
            dt, self.dr,
            src_flux, src_pos,
            r_RT, self.gpu, max_subbox, self.loss_fraction,
            self.temp, self.ndens, self.xh,
            self.photo_thin_table, self.minlogtau, self.dlogtau,
            self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c,
            self.logfile
            )

    def cosmo_evolve(self, dt):
        """Evolve cosmology over a timestep

        Note that if cosmological is set to false in the parameter file, this
        method does nothing!

        Following the C2Ray convention, we set the redshift according to the
        half point of the timestep.
        """
        # Time step
        t_now = self.time
        t_half = t_now + 0.5*dt
        t_after = t_now + dt
        self.printlog(' This is time : %f\t %f' %((t_now*u.s).to('yr').value, (t_after*u.s).to('yr').value))
        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            zfactor = (1+z_half) / (1+self.zred)
            dilution_factor = zfactor**3
            self.ndens *= dilution_factor

            # Set cell size to current proper size
            #TODO: it should be: self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr = self.dr_c / zfactor #/ (1 + z_half)

        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after

    def printlog(self,s,quiet=False):
        """Print to log file and standard output

        Parameters
        ----------
        s : str
            String to print
        quiet : bool
            Whether to print only to log file or also to standard output (default)
        """
        if self.logfile is None:
            raise RuntimeError("Please set the log file in output_ini")
        else:
            printlog(s,self.logfile,quiet)

    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self,t):
        """Calculate the redshift corresponding to an age t in seconds
        """
        # TODO: it should be then z_at_value(self.cosmology.age, t*u.s).value
        # in C2Ray is defined: time2zred = -1+(1.+zred_t0)*(t0/(t0+time))**(2./3.)        
        return -1+(1.+self.zred_0)*(self.age_0/(self.age_0+t))**(2./3.)

    def zred2time(self, z, unit='s'):
        """Calculate the age corresponding to a redshift z

        Parameters
        ----------
        z : float
            Redshift at which to get age
        unit : str (optional)
            Unit to get age in astropy naming. Default: seconds
        """
        # TODO : it should be then self.cosmology.age(z).to(unit).value
        # In C2Ray is defined: zred2time = t0*( ((1.0+zred_t0)/(1.0+zred1))**1.5 - 1.0 )
        return self.age_0*(((1.0+self.zred_0)/(1.0+z))**1.5 - 1.0)
        

    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

    def _param_init(self):
        """ Set up general constants and parameters

        Computes additional required quantities from the read-in parameters
        and stores them as attributes
        """
        self.eth0 = self._ld['CGS']['eth0']
        self.ethe0 = self._ld['CGS']['ethe0']
        self.ethe1 = self._ld['CGS']['ethe1']
        self.bh00 = self._ld['CGS']['bh00']
        self.fh0 = self._ld['CGS']['fh0']
        self.xih0 = self._ld['CGS']['xih0']
        self.albpow = self._ld['CGS']['albpow']
        self.abu_h = self._ld['Abundances']['abu_h']
        self.abu_he = self._ld['Abundances']['abu_he']
        self.mean_molecular = self.abu_h + 4.0*self.abu_he
        self.abu_c = self._ld['Abundances']['abu_c']
        self.colh0 = self._ld['CGS']['colh0_fact']*self.fh0*self.xih0/self.eth0**2
        self.temph0=self.eth0*ev2k
        self.sig = self._ld['Photo']['sigma_HI_at_ion_freq']
        self.loss_fraction = self._ld['Raytracing']['loss_fraction']

    def _cosmology_init(self):
        """ Set up cosmology from parameters (H0, Omega,..)
        """
        h = self._ld['Cosmology']['h']
        Om0 = self._ld['Cosmology']['Omega0']
        Ob0 = self._ld['Cosmology']['Omega_B']
        Tcmb0 = self._ld['Cosmology']['cmbtemp']
        H0 = 100*h
        self.cosmology = FlatLambdaCDM(H0, Om0, Tcmb0, Ob0=Ob0)

        self.cosmological = self._ld['Cosmology']['cosmological']
        self.zred_0 = self._ld['Cosmology']['zred_0']

        H0 *= 1e5/Mpc

        self.age_0 = 2.*(1.+self.zred_0)**(-1.5)/(3.*H0*np.sqrt(Om0))
        #self.age_0 = self.zred2time(self.zred_0)

        # Scale quantities to the initial redshift
        if self.cosmological:
            self.printlog(f"Cosmology is on, scaling comoving quantities to the initial redshift, which is z0 = {self.zred_0:.3f}...")
            #TODO: it should be: self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
            self.dr = self.dr_c / (1 + self.zred_0)
        else:
            self.printlog("Cosmology is off.")

    def _radiation_init(self):
        """Set up radiation tables for ionization/heating rates
        """
        # Create optical depth table (log-spaced)
        self.minlogtau = self._ld['Photo']['minlogtau']
        self.maxlogtau = self._ld['Photo']['maxlogtau']
        self.NumTau = self._ld['Photo']['NumTau']

        # The actual table has NumTau + 1 points: the 0-th position is tau=0 and the
        # remaining NumTau points are log-spaced from minlogtau to maxlogtau (same as in C2Ray)
        self.tau, self.dlogtau = make_tau_table(self.minlogtau,self.maxlogtau,self.NumTau)

        ion_freq_HI = ev2fr * self.eth0
        ion_freq_HeII = ev2fr * self.ethe1

        freq_min = ion_freq_HI
        freq_max = 10*ion_freq_HeII

        # Initialize Black-Body Source
        self.bb_Teff = self._ld['Photo']['Teff']
        self.grey = self._ld['Photo']['grey']
        self.cs_pl_idx_h = self._ld['Photo']['cross_section_pl_index']
        radsource = BlackBodySource(self.bb_Teff, self.grey, ion_freq_HI, self.cs_pl_idx_h)

        if self.grey:
            self.printlog(f"Warning: Using grey opacity")
        else:
            self.printlog(f"Using power-law opacity with {self.NumTau:n} table points between tau=10^({self.minlogtau:n}) and tau=10^({self.maxlogtau:n})")

        # Integrate table. 
        # TODO: make this more customizeable
        self.photo_thin_table = radsource.make_photo_table(self.tau,freq_min,freq_max,1e48)
        
        self.printlog(f"Using Black-Body sources with effective temperature T = {radsource.temp :.1e} K and Radius {(radsource.R_star/c.R_sun.to('cm')).value : .3e} rsun")
        self.printlog(f"Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz")
        self.printlog(f"This is Energy:           {freq_min/ev2fr:.3e} to {freq_max/ev2fr:.3e} eV")

        # Copy radiation table to GPU
        if self.gpu:
            photo_table_to_device(self.photo_thin_table)
            self.printlog("Successfully copied radiation tables to GPU memory.")

    # =====================================================================================================
    # OTHER PRIVATE METHODS
    # =====================================================================================================

    def _read_paramfile(self,paramfile):
        """ Read in YAML parameter file
        """
        loader = SafeLoader
        # Configure to read scientific notation as floats rather than strings
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.')
        )
        with open(paramfile,'r') as f:
            self._ld = yaml.load(f,loader)

    def _gpu_close(self):
        """ Deallocate GPU memory
        """
        device_close()

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================
      
    def read_sources(self, file, mass, ts): # >:( trgeoip
        """Read sources from a C2Ray-formatted file

        The way sources are dealt with is still open and will change significantly
        in the final version. For now, this method is provided:

        It reads source positions and strengths (total ionizing flux in
        photons/second) from a file that is formatted for the original C2Ray,
        and computes the source strength as normalization factors relative
        to a reference strength (1e48 by default). These normalization factors
        are then used during raytracing to compute the photoionization rate.
        (same procedure as in C2Ray)

        Moreover, the method formats the source positions correctly depending
        on whether OCTA is used or not. This is because, while the default CPU
        raytracing takes a 3D-array of any type as argument, OCTA assumes that the
        source position array is flattened and has a C single int type (int32),
        and that the normalization (strength) array has C double float type (float64).

        Parameters
        ----------
        file : str
            Filename to read
        n : int
            Number of sources to read from the file
        
        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        numsrc : int
            Number of sources read from the file
        """
        S_star_ref = 1e48
        
        # TODO: automatic selection of low mass or high mass. For the moment only high mass
        #mass2phot = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)    
        # TODO: for some reason the difference with the orginal Fortran run is of the molecular weight
        #self.printlog('%f' %self.mean_molecular )
        mass2phot = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (m_p * ts * self.cosmology.Om0)    
        
        if file.endswith('.hdf5'):
            f = h5py.File(file, 'r')
            srcpos = f['sources_positions'][:].T
            assert srcpos.shape[0] == 3
            normflux = f['sources_mass'][:] * mass2phot / S_star_ref
            f.close()
        else:
            # use original C2Ray source file
            src = t2c.SourceFile(filename=file, mass=mass)
            srcpos = src.sources_list[:, :3].T
            normflux = src.sources_list[:, -1] * mass2phot / S_star_ref

        self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' %(normflux.size, file))
        self.printlog(' Total Flux : %e' %np.sum(normflux*S_star_ref))
        self.printlog(' Source lifetime : %f Myr' %((ts*u.s).to('Myr').value))
        self.printlog(' min, max source mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]' %(normflux.min()/mass2phot*S_star_ref, normflux.max()/mass2phot*S_star_ref, normflux.min()*S_star_ref, normflux.mean()*S_star_ref, normflux.max()*S_star_ref))
        return srcpos, normflux
    
    def read_density(self, z):
        """ Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either N-body or hydro-dynamical simulations. The field is then smoothed on a coarse mesh grid.

        Parameters
        ----------
        n : int
            Number of sources to read from the file
        
        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            density mesh-grid in csg units
        """
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0

        # redshift bin for the current redshift based on the density redshift
        #low_z, high_z = find_bins(redshift, self.zred_density)
        high_z = self.zred_density[np.argmin(np.abs(self.zred_density[self.zred_density >= redshift] - redshift))]

        if(high_z != self.prev_zdens):
            file = '%scoarser_densities/%.3fn_all.dat' %(self.inputs_basename, high_z)
            self.printlog(f'\n---- Reading density file:\n '+file)
            self.ndens = t2c.DensityFile(filename=file).cgs_density / (self.mean_molecular * m_p) * (1+redshift)**3
            self.printlog(' min, mean and max density : %.3e  %.3e  %.3e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))
            self.prev_zdens = high_z
        else:
            # no need to re-read the same file again
            # TODO: in the future use this values for a 3D interpolation for the density (can be extended to sources too)
            pass

    def write_output(self,z):
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.dat"
        t2c.save_cbin(filename=self.results_basename + "xfrac" + suffix, data=self.xh, bits=64, order='F')
        t2c.save_cbin(filename=self.results_basename + "IonRates" + suffix, data=self.phi_ion, bits=32, order='F')

        self.printlog('\n--- Reionization History ----')
        self.printlog(' min, mean, max xHII : %.3e  %.3e  %.3e' %(self.xh.min(), self.xh.mean(), self.xh.max()))
        self.printlog(' min, mean, max Irate : %.3e  %.3e  %.3e [1/s]' %(self.phi_ion.min(), self.phi_ion.mean(), self.phi_ion.max()))
        self.printlog(' min, mean, max density : %.3e  %.3e  %.3e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))

    
    # =====================================================================================================
    # Below are the overridden initialization routines specific to the CubeP3M case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.zred_density = t2c.get_dens_redshifts(self.inputs_basename+'coarser_densities/')[::-1]
        #self.zred_sources = get_source_redshifts(self.inputs_basename+'sources/')[::-1]
        # TODO: waiting for next tools21cm release
        self.zred_sources = t2c.get_source_redshifts(self.inputs_basename+'sources/')[::-1]
        if(self.resume):
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename)) 
            #self.age_0 = self.zred2time(self.zred_0)
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1
            self.zred = self.zred_0

        self.time = self.zred2time(self.zred)
        #self.time = self.age_0

    def _material_init(self):
        """Initialize material properties of the grid
        """
        if(self.resume):
            # get fields at the resuming redshift
            self.ndens = t2c.DensityFile(filename='%scoarser_densities/%.3fn_all.dat' %(self.inputs_basename, self.prev_zdens)).cgs_density / (self.mean_molecular * m_p) * (1+self.zred)**3
            #self.ndens = self.read_density(z=self.zred)
            self.xh = t2c.read_cbin(filename='%sxfrac_%.3f.dat' %(self.results_basename, self.zred), bits=64, order='F')
            # TODO: implement heating
            temp0 = self._ld['Material']['temp0']
            self.temp = temp0 * np.ones(self.shape, order='F')
            self.phi_ion = t2c.read_cbin(filename='%sIonRates_%.3f.dat' %(self.results_basename, self.zred), bits=32, order='F')
        else:
            xh0 = self._ld['Material']['xh0']
            temp0 = self._ld['Material']['temp0']
            avg_dens = self._ld['Material']['avg_dens']

            self.ndens = avg_dens * np.empty(self.shape, order='F')
            self.xh = xh0 * np.ones(self.shape, order='F')
            self.temp = temp0 * np.ones(self.shape, order='F')
            self.phi_ion = np.zeros(self.shape, order='F')
    
    def _output_init(self):
        """ Set up output & log file
        """
        self.results_basename = self._ld['Output']['results_basename']
        self.inputs_basename = self._ld['Output']['inputs_basename']

        self.logfile = self.results_basename + self._ld['Output']['logfile']
        title = '                 _________   ____            \n    ____  __  __/ ____/__ \ / __ \____ ___  __\n   / __ \/ / / / /    __/ // /_/ / __ `/ / / /\n  / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ / \n / .___/\__, /\____//____/_/ |_|\__,_/\__, /  \n/_/    /____/                        /____/   \n'
        if(self._ld['Grid']['resume']):
            with open(self.logfile,"r") as f: 
                log = f.readlines()
            with open(self.logfile,"w") as f: 
                log.append("\n\nResuming"+title[8:]+"\n\n")
                f.write(''.join(log))
        else:
            with open(self.logfile,"w") as f: 
                # Clear file and write header line
                f.write(title+"\nLog file for pyC2Ray.\n\n") 

    def _sources_init(self):
        """Initialize settings to read source files
        """
        self.fgamma_hm = self._ld['Sources']['fgamma_hm']
        self.fgamma_lm = self._ld['Sources']['fgamma_lm']
        self.ts = (self._ld['Sources']['ts'] * u.Myr).cgs.value


    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.boxsize_c = self._ld['Grid']['boxsize'] * Mpc / self._ld['Cosmology']['h']
        self.dr_c = self.boxsize_c / self.N

        self.printlog(f"Welcome! Mesh size is N = {self.N:n}.")
        self.printlog(f"Simulation Box size (comoving Mpc): {self.boxsize_c/Mpc:.3e}")

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c
        self.resume = self._ld['Grid']['resume']

