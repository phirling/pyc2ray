import yaml
import numpy as np
import h5py
import os
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

class IOManager:
    """Class that handles all read/write operations of C2Ray
    At construction, the class reads in a YAML parameter file and provides a
    convenient method to access these parameters from outside the class.

    Additionally, it provides methods to read in data (density,radiation tables,etc)
    and to write logs and outputs.
    """
    def __init__(self,paramfile) -> None:
        # Read in parameter file
        self.paramfile = paramfile
        with open(self.paramfile,'r') as f:
            self.ld = yaml.load(f,SafeLoader)
        
        # Log/Debugging files
        self.timefile = self.param('Log','timefile')
        self.logfile = self.param('Log','logfile')
        self.write_log("Log file for C2-Ray run \n",mode='w')

        # Input files
        self.redfile = self.param('Input','redfile')
        self.densityfile_prefix = self.param('Input','density_prefix')

        # Output Files
        self.photonstatsfile1 = self.param('Output','photon_statistics_file_1')
        self.photonstatsfile2 = self.param('Output','photon_statistics_file_2')

    def param(self,group,key):
        p = None
        try:
            p = self.ld[group][key]
        except KeyError:
            print(f"Invalid key: {group}->{key} is not a valid member of {self.paramfile}.")
            return None
        if p is None:
            return ""
        else:
            return p
    
    def write_log(self,s,mode='a'):
        with open(self.logfile,mode) as f:
            f.write(s + "\n")
    
    def setup_output(self):
        # TODO: Here need to setup output streams

        # Setup photon stats files
        if self.param('Output','do_photonstatistics'):
            self.open_photonstatistics_files()
            self.initialize_photonstatistics() # TODO: define this
        
    def read_density(self,z):
        s = self.construct_density_filename(z)
        with h5py.File(s,'r') as f:
            dens = np.array(f['density'])
        return dens
        
    def construct_density_filename(self,z):
        s = f"{self.densityfile_prefix}_{z:n}.hdf5"
        return s
    
    # ============================== Output ================================ #
    def open_photonstatistics_files(self):
        with open(self.photonstatsfile1,'w') as f:
            f.write(
                f"Columns: redshift, "
                "total number of photons used on the grid, "
                "total number of photons produced on the grid, "
                "photon conservation number, " +
                "fraction new ionization, fraction recombinations, "
                "fraction LLS losses (seems to be wrong), "
                "fraction photon losses, fraction collisional ionization, "
                "grand total photon conservation number"
            )
        with open(self.photonstatsfile2,'w') as f:
            f.write(
                f"Columns: redshift, total number of ions, "
                "grand total ionizing photons, mean ionization fraction "
                "(by volume and mass)"
            )
    
    def initialize_photonstatistics(self):
        pass
    
    def write_photonstatistics(self,data):
        pass
    # ====================================================================== #

    # ============================== Redshift & Density ================================ #
    def redshift_ini(self):
        NumZred, zred_array = self.read_redshifts()
        self.write_log(f"Read {NumZred} redshifts from {self.redfile}.")
        self.write_log("Checking density files...")
        # Check files
        for z in zred_array:
            s = self.construct_density_filename(z)
            if not os.path.isfile(s):
                self.write_log(f"The density file {s} doesn't exist")
                raise ValueError(f"The density file {s} doesn't exist")
        self.write_log("All density files checked successfully!")
        return NumZred, zred_array


    def read_redshifts(self):
        with open(self.redfile,'r') as f:
            raw_z = np.loadtxt(f)
            NumZred = raw_z[0]
            zred_array = raw_z[1:]
        if NumZred % 1 != 0:
            raise ValueError("Number of redshifts must be integer")
        else:
            NumZred = int(NumZred)
        return NumZred, zred_array