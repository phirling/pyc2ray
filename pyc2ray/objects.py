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
from .utils.sourceutils import read_sources
from .evolve import evolve3D, evolve3D_octa
from .octa_core import device_init, device_close, cuda_is_init

# ==================================================================
# This file defines the C2Ray object class, which is the basis
# for a c2ray simulation. It deals with parameters, I/O, cosmology,
# and other things such as memory allocation when using octa.
#
# I'm not sure what is best in terms of which features are
# managed internally by the class, and which ones should appear
# explicitely in a simulation script. For now, the actual grid
# arrays are kept outside and the class only handles parameters,
# computations, etc.
# ==================================================================

# Conversion Factors. These will be replaced by astropy constants later on
ev2k = 1.0/8.617e-05        # eV to Kelvin
pc = 3.086e18               # parsec
kpc = 1e3*pc                # kiloparsec
Mpc = 1e6*pc                # megaparsec

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
        # Read YAML parameter file
        self._read_paramfile(paramfile)
        self.N = Nmesh

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
        self._cosmology_init()
        self._output_init()
        self._grid_init()

    def evolve3D(self,dt,srcflux,srcpos,r_RT,temp,ndens,xh,max_subbox):
        """Evolve the grid over one timestep

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """
        if self.octa:
            return evolve3D_octa(dt, self.dr, srcflux, srcpos, r_RT, temp, ndens, xh, self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c, self.N, self.logfile)
        else:
            return evolve3D(dt, self.dr, srcflux, srcpos, max_subbox, r_RT, temp, ndens, xh, self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c,self.loss_fraction, self.logfile)

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
        return dt

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
        return self.cosmology.age(z).to(unit)

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
        else: mode = 'c2ray'
        return read_sources(file, n, mode)
    # ==================================================================
    # Private methods of class
    # ==================================================================

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
        self.boxsize = self._ld['Grid']['boxsize'] * Mpc
        self.dr = self.boxsize / self.N

    def _cosmology_init(self):
        """ Set up cosmology from parameters (H0, Omega,..)
        """
        self.cosmological = self._ld['Cosmology']['cosmological']
        h = self._ld['Cosmology']['h']
        Om0 = self._ld['Cosmology']['Omega0']
        Ob0 = self._ld['Cosmology']['Omega_B']
        Tcmb0 = self._ld['Cosmology']['cmbtemp']
        H0 = 100*h
        self.cosmology = FlatLambdaCDM(H0, Om0, Tcmb0, Ob0=Ob0)

    def _output_init(self):
        """ Set up output & log file
        """
        rb = self._ld['Output']['results_basename']
        self.logfile = rb + self._ld['Output']['logfile']
        with open(self.logfile,"w") as f: f.write("Log file for pyC2Ray. \n\n") # Clear file and write header line

    def _read_paramfile(self,paramfile):
        """ Read in YAML parameter file
        """
        loader = SafeLoader
        # Configure to read scientific notation as floats
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