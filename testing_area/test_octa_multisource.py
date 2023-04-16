import RTC                              # C++ Module (CUDA, CUDA GPU)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time
import argparse
from tomography import zTomography # Custom module to visualize datacube
import pickle as pkl

np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("-r",type=int,default=50)
parser.add_argument("-srcx",type=int,default=150)
parser.add_argument("-zslice",type=int,default=None)
parser.add_argument("--pickle",action='store_true')
args = parser.parse_args()

"""
ABOUT
Test the c2ray version of the RT

"""

""" /////////////////////////////// Main Setup //////////////////////////////////////// """

# Test Parameters
N       = 300       # Grid Size
srcx    = args.srcx       # Source x-position (x=y=z)
numsrc  = 50         # Number of sources
if args.zslice is None:
    zslice = srcx
else:
    zslice = args.zslice
plot_final    = 0         # Whether or not to plot results
rad = args.r

# Numerical/Physical Setup
dt = 1.0 * u.Myr.to('s')                # Timestep
boxsize = 14 * u.kpc.to('cm')           # Simulation box size
dxbox = boxsize / N                     # Cell Size (1D)
dr = dxbox * np.ones(3)                 # Cell Size (3D)
sig = 6.30e-18

# Initial conditions
avgdens = 1.0e-3                        # Constant Hydrogen number density
xhav = 1.2e-3                           # Initial ionization fraction
temp0 = 1e4

""" ////////////////////////// C++ (OCTA) Version Setup /////////////////////////////// """

# Source Setup
srcpos = np.random.randint(0,N,size=3*numsrc,dtype='int32')
print("Source positions: ")
print(srcpos)
srcflux = 5.0e48 * np.ones(numsrc)

# Initialize Arrays
cdh2 = np.ravel(np.zeros((N,N,N),dtype='float64'))
ndens = 1e-3*np.ravel(np.ones((N,N,N),dtype='float64') )
phi_ion2 = np.ravel(np.zeros((N,N,N),dtype='float64') )
xh_av = 1.2e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )

# Initialize GPU and allocate memory
RTC.device_init(N)

""" ////////////////////////////////// Run Tests ////////////////////////////////////// """

print(f"Doing radius r = {rad:.2f}")
print("Running OCTA GPU...")
t5 = time.time()
RTC.octa_gpu_allsources(srcpos,srcflux,rad,cdh2,sig,dxbox,ndens,xh_av,phi_ion2,numsrc,N)
t6 = time.time()
cdh2 = cdh2.reshape((N,N,N))
phi_ion2 = phi_ion2.reshape((N,N,N))

RTC.device_close() # Deallocate GPU memory
print(f"Done. took {t6-t5:.2f} second(s).")

""" ///////////////////////////////// Visualization /////////////////////////////////// """
loggamma = np.where(phi_ion2 != 0.0,np.log(phi_ion2),np.nan) #= np.log(phi_ion_f)

if args.pickle:
    with open(f"octa_{numsrc:n}_sources_r={rad:n}.pkl","wb") as f:
        pkl.dump(loggamma, f)

tomo = zTomography(loggamma, zslice)
plt.show()