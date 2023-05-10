import yaml
import atexit
import re
import pickle as pkl
import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader
from .utils.logutils import printlog
from .utils.sourceutils import read_sources
from .evolve import evolve3D, evolve3D_octa
from .octa_core import device_init, device_close

# ==================================================================
# This file defines the C2Ray object class, which is the basis
# for a c2ray simulation. It deals with parameters, I/O, cosmology,
# and other things such as memory allocation when using octa.
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
# ==================================================================

# Conversion Factors. These will be replaced by astropy constants later on
ev2k = 1.0/8.617e-05        # eV to Kelvin
pc = 3.086e18               # parsec
kpc = 1e3*pc                # kiloparsec
Mpc = 1e6*pc                # megaparsec
YEAR = 3.15576E+07

class C2Ray:
    def __init__(self,paramfile,Nmesh,use_octa):
        """A C2Ray Simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation
        Nmesh : int
            Mesh size (number of cells in each dimension)
        use_octa : bool
            Whether to use the OCTA library for raytracing

        """
        # Read YAML parameter file and set main properties
        self._read_paramfile(paramfile)
        self.N = Nmesh
        self.shape = (Nmesh,Nmesh,Nmesh)

        # Set Raytracing mode
        if use_octa:
            self.octa = True
            # Allocate GPU memory
            device_init(Nmesh)
            # Register deallocation function (automatically calls this on program termination)
            atexit.register(self._octa_close)
        else:
            self.octa = False

        # Initialize Simulation
        self._param_init()
        self._grid_init()
        self._cosmology_init()
        self._output_init()
        self._material_init()
        self._redshift_init()

    def evolve3D(self,dt,srcflux,srcpos,r_RT,max_subbox):
        """Evolve the grid over one timestep

        Raytrace all sources, compute cumulative photoionization rate of each cell and
        do chemistry. This is done until convergence in the ionized fraction is reached.
        TODO: pass convergence criteria as arguments (they are fixed at compile-time for now)

        Parameters
        ----------
        dt : float
            Timestep in seconds (typically generated using set_timestep method)
        srcflux : 1D-array
            Strength of each source
        srcpos : array
            Positions of the sources. Shape depends on the raytracing algorithm used. Use
            read_sources to automatically format the array in the correct way. TODO: make this automatic
        r_RT : int
            When using C2Ray raytracing: size of the subbox to use. When using OCTA, determines the
            size of the octahedron
        max_subbox : int
            Maximum size of the subbox when using C2Ray raytracing. When using OCTA, this
            parameter has no effect.

        """
        if self.octa:
            self.xh, self.phi_ion = evolve3D_octa(dt, self.dr, srcflux, srcpos, r_RT, self.temp, self.ndens, self.xh, self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c, self.N, self.logfile)
        else:
            self.xh, self.phi_ion = evolve3D(dt, self.dr, srcflux, srcpos, max_subbox,r_RT, self.temp, self.ndens, self.xh, self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c,self.loss_fraction, self.logfile)


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

    def read_sources(self,file,n): # >:( trgeoip
        """Read sources from a C2Ray-formatted file

        This is limited to the test case for now (total ionizing
        flux of the sources is known)

        Parameters
        ----------
        file : str
            Filename to read
        n : int
            Number of sources to read from the file
        
        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for
            the chosen raytracing algorithm
        srcflux : array
            Total flux of ionizing photons of the sources
        numsrc : int
            Number of sources read from the file
        """
        if self.octa: mode = 'pyc2ray_octa'
        else: mode = 'pyc2ray'
        return read_sources(file, n, mode)
    
    def density_init(self,z):
        """Set density at redshift z

        For now, this simply sets the density to a constant value,
        specified in the parameter file, that is scaled to redshift
        if the run is cosmological.

        Parameters
        ----------
        z : float
            Redshift slice
        
        """
        self.set_constant_average_density(self.avg_dens,z)

    def cosmo_evolve(self,dt):
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

    def generate_redshift_array(self,num_zred,delta_t):
        """Helper function to generate a list of equally-time-spaced redshifts

        Generate num_zred redshifts that correspond to cosmic ages
        separated by a constant time interval delta_t. Useful for the
        test case of C2Ray. The initial redshift is set in the parameter file.

        Parameters
        ----------
        num_zred : int
            Number of redshifts to generate
        delta_t : float
            Spacing between redshifts in years

        Returns
        -------
        zred_array : 1D-array
            List of redshifts (including initial one)
        """
        step = delta_t * YEAR
        zred_array = np.empty(num_zred)
        for i in range(num_zred):
            zred_array[i] = self.time2zred(self.age_0 + i*step)
        return zred_array

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
    
    def scale_density(self,z):
        self.ndens /= self.cosmology.scale_factor(z)**3

    # ==================================================================
    # Private methods of class
    # ==================================================================

    def _material_init(self):
        xh0 = self._ld['Material']['xh0']
        temp0 = self._ld['Material']['temp0']

        self.ndens = np.empty(self.shape,order='F')
        self.xh = xh0 * np.ones(self.shape,order='F')
        self.temp = temp0 * np.ones(self.shape,order='F')
        self.phi_ion = np.zeros(self.shape,order='F')

        self.avg_dens = self._ld['Material']['avg_dens']

    def _param_init(self):
        """ Set up constants and parameters

        Computes additional required quantities from the read-in parameters
        and stores them as attributes
        """
        self.eth0 = self._ld['CGS']['eth0']
        self.bh00 = self._ld['CGS']['bh00']
        self.fh0 = self._ld['CGS']['fh0']
        self.xih0 = self._ld['CGS']['xih0']
        self.albpow = self._ld['CGS']['albpow']
        self.abu_c = self._ld['Abundances']['abu_c']
        self.colh0 = self._ld['CGS']['colh0_fact']*self.fh0*self.xih0/self.eth0**2
        self.temph0=self.eth0*ev2k
        self.sig = self._ld['Photo']['sigma_HI_at_ion_freq']
        self.loss_fraction = self._ld['Raytracing']['loss_fraction']
    
    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.boxsize_c = self._ld['Grid']['boxsize'] * Mpc
        self.dr_c = self.boxsize_c / self.N

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

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
        
    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.time = self.age_0
        self.zred = self.zred_0

    def _output_init(self):
        """ Set up output & log file
        """
        self.results_basename = self._ld['Output']['results_basename']
        self.logfile = self.results_basename + self._ld['Output']['logfile']
        with open(self.logfile,"w") as f: f.write("Log file for pyC2Ray. \n\n") # Clear file and write header line

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

    def _octa_close(self):
        """ Deallocate GPU memory
        """
        device_close()