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
from .radiation import BlackBodySource

# ======================================================================
# This file defines the abstract C2Ray object class, which is the basis
# for a c2ray simulation. It deals with parameters, I/O, cosmology,
# and other things such as memory allocation when using the GPU.
# Any concrete simulation uses subclasses of C2Ray, with methods specific
# to certain input files (e.g. CubeP3M)
#
#
# -- Notes on cosmology: --
# * In C2Ray, the scale factor is 1 at z = 0. The box size is given
# in comoving units, i.e. it is the proper size at z = 0. At the
# start (in cosmo_ini), the cell size & volume are scaled down to
# the first redshift slice of the program.
# 
# * There are 2 types of redshift evolution: (1) when the program
# reaches a new "slice" (where a density file would be read etc)
# and (2) at each timestep BETWEEN slices. Basically, at (1), the
# density is set, and during (2), this density is diluted due to
# the expansion.
#
# * During this dilution (at each timestep between slices), C2Ray
# has the convention that the redshift is incremented not by the
# value that corresponds to a full timestep in cosmic time, but by
# HALF a timestep.
#    ||          |           |           |           |               ||
#    ||    z1    |     z2    |     z3    |     z4    |       ...     ||
#    ||          |           |           |           |               ||
#    t0          t1          t2          t3          t4
# 
#   ("||" = slice,    "|" = timestep,   "1,2,3,4,.." indexes the timestep)
# 
# In terms of attributes, C2Ray.time always contains the time at the
# end of the current timestep, while C2Ray.zred contains the redshift
# at half the current timestep. This is relevant to understand the
# cosmo_evolve routine below (which is based on what is done in the
# original C2Ray)
# 
# This induces a potential bug: when a slice is reached and the
# density is set, the density corresponds to zslice while
# C2Ray.zred is at the redshift "half a timestep before".
# The best solution I've found here is to just save the comoving cell
# size dr_c and always set the current cell size to dr = a(z)*dr_c,
# rather than "diluting" dr iteratively like the density.
#
# TODO ideas:
# * Add "default" values for YAML parameter file so that if the user
# omits a value in the file, a default value is used instead rather
# than throwing an error
# ======================================================================

# Conversion Factors.
ev2k = 1.0/8.617e-05         # eV to Kelvin
pc = (1*u.pc).to('cm').value # parsec in cm
kpc = 1e3*pc                 # kiloparsec in cm
Mpc = 1e6*pc                 # megaparsec in cm
YEAR = (1*u.yr).to('s').value
ev2fr = 0.241838e15
msun2g = (1*u.Msun).to('g').value  # solar mass to grams


class C2Ray:
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
        self._grid_init()
        self._cosmology_init()
        self._output_init()
        self._redshift_init()
        self._material_init()
        self._sources_init()
        self._radiation_init()

    # =====================================================================================================
    # TIME-EVOLUTION METHODS
    # =====================================================================================================
    def set_timestep(self,z1,z2,num_timesteps):
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
        t1 = self.cosmology.lookback_time(z1).to('s').value
        t2 = self.cosmology.lookback_time(z2).to('s').value
        # we do t1-t2 since ti are lookback times (not ages)
        dt = (t1-t2)/num_timesteps
        self.printlog(f"dt [years]: {dt/YEAR:.3e}")
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

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor

            # Set cell size to current proper size
            self.dr = self.dr_c * self.cosmology.scale_factor(z_half)

        self.printlog(f"dr [kpc]: {self.dr/kpc:.3e}")

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
        printlog(s,self.logfile,quiet)


    def write_output(self,z):
        pass


    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self,t):
        """Calculate the redshift corresponding to an age t in seconds
        """
        return z_at_value(self.cosmology.age, t*u.s).value

    def zred2time(self,z,unit='s'):
        """Calculate the age corresponding to a redshift z

        Parameters
        ----------
        z : float
            Redshift at which to get age
        unit : str (optional)
            Unit to get age in astropy naming. Default: seconds
        """
        return self.cosmology.age(z).to(unit).value
    

    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

    def _param_init(self):
        """ Set up general constants and parameters

        Computes additional required quantities from the read-in parameters
        and stores them as attributes
        """
        self.eth0 = self._ld['CGS']['eth0']
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
        self.age_0 = self.zred2time(self.zred_0)

        # Scale quantities to the initial redshift
        if self.cosmological:
            self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c

    def _radiation_init(self):
        """Set up radiation tables for ionization/heating rates
        """
        # Create optical depth table (log-spaced)
        self.minlogtau = self._ld['Photo']['minlogtau']
        self.maxlogtau = self._ld['Photo']['maxlogtau']
        self.NumTau = self._ld['Photo']['NumTau']
        self.dlogtau = (self.maxlogtau - self.minlogtau) / (self.NumTau)

        # The table has NumTau + 1 points: the 0-th position is tau=0 and the
        # remaining NumTau points are log-spaced from minlogtau to maxlogtau (same as in C2Ray)
        self.tau = np.empty(self.NumTau + 1)
        self.tau[0] = 0.0
        for i in range(1,self.NumTau+1):
            self.tau[i] = 10.0**(self.minlogtau+self.dlogtau*(i-1))
        
        # Initialize Black-Body Source
        self.bb_Teff = self._ld['Photo']['Teff']
        self.grey = self._ld['Photo']['grey']
        self.cs_pl_idx_h = self._ld['Photo']['cross_section_pl_index']
        self.printlog(f"Using Black-Body sources with effective temperature T = {self.bb_Teff :.1e} K")

        ion_freq_HI = ev2fr * self.eth0
        radsource = BlackBodySource(self.bb_Teff, self.grey, ion_freq_HI, self.cs_pl_idx_h)

        if self.grey:
            self.printlog(f"Using grey opacity")
        else:
            self.printlog(f"Using power-law opacity with {self.NumTau:n} table points between tau=10^({self.minlogtau:n}) and tau=10^({self.maxlogtau:n})")

        # Integrate table. TODO: make this more customizeable
        self.photo_thin_table = radsource.make_photo_table(self.tau,ion_freq_HI,10*ion_freq_HI,1e48)

        # Copy radiation table to GPU
        if self.gpu:
            photo_table_to_device(self.photo_thin_table)
            self.printlog("Successfully copied radiation tables to GPU memory.")

    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.boxsize_c = self._ld['Grid']['boxsize'] * Mpc
        self.dr_c = self.boxsize_c / self.N

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c


    # The following initialization methods are simulation kind-dependent and need to be
    # overridden in the subclasses
    def _output_init(self):
        """ Set up output & log file
        """
        pass

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        pass

    def _material_init(self):
        """Initialize material properties of the grid
        """
        pass

    def _sources_init(self):
        """Initialize settings to read source files
        """
        pass

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