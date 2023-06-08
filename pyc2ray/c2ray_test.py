from .c2ray_base import C2Ray, YEAR, Mpc
from .utils.sourceutils import read_sources
import numpy as np
import pickle as pkl

__all__ = ['C2Ray_test']

class C2Ray_test(C2Ray):
    def __init__(self, paramfile, Nmesh, use_octa):
        super().__init__(paramfile, Nmesh, use_octa)

    def read_sources(self,file,n): # >:( trgeoip
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
            Grid positions of the sources formatted in a suitable way for
            the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
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
    
    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.boxsize_c = self._ld['Grid']['boxsize'] * Mpc
        self.dr_c = self.boxsize_c / self.N

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

    def _output_init(self):
        """ Set up output & log file
        """
        self.results_basename = self._ld['Output']['results_basename']
        self.logfile = self.results_basename + self._ld['Output']['logfile']
        with open(self.logfile,"w") as f: f.write("Log file for pyC2Ray. \n\n") # Clear file and write header line

    def _material_init(self):
        xh0 = self._ld['Material']['xh0']
        temp0 = self._ld['Material']['temp0']

        self.ndens = np.empty(self.shape,order='F')
        self.xh = xh0 * np.ones(self.shape,order='F')
        self.temp = temp0 * np.ones(self.shape,order='F')
        self.phi_ion = np.zeros(self.shape,order='F')
        self.avg_dens = self._ld['Material']['avg_dens']

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.time = self.age_0
        self.zred = self.zred_0