import numpy as np
import astropy.units as u
import pickle as pkl


"""
ABOUT
Generate random source lists used for benchmarks and tests

"""

# Set random seed for reproducibility
np.random.seed(100)

# Parameters
N       = 300               # Grid Size
numsrc  = 100               # Number of sources
fname = "sourcelist.txt"    # File name
flux = 5.0e55               # Flux of each source. TODO: maybe try with random flux too ?

# Source Setup
srcpos = 1+np.random.randint(0,N,size=3*numsrc)
srcpos = srcpos.reshape((numsrc,3),order='C')
srcflux = flux * np.ones((numsrc,1))
zerocol = np.zeros((numsrc,1)) # By convention for c2ray

output = np.hstack((srcpos,srcflux,zerocol))

with open(fname,'w') as f:
    f.write(f"{numsrc:n}\n")

with open(fname,'a') as f:
    np.savetxt(f,output,("%i %i %i %.0e %.1f"))