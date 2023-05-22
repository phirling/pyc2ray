import sys
sys.path.append("../")

import pyc2ray as pc2r
from pyc2ray.visualization import zTomography_rates
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time
import argparse
import pickle as pkl

np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("-N",type=int,default=300)
parser.add_argument("-r",type=int,default=50)
parser.add_argument("-numsrc",type=int,default=10)
parser.add_argument("--pickle",action='store_true')
parser.add_argument("--plot",action='store_true')
args = parser.parse_args()

"""
ABOUT
Test the octa version of the RT

"""

""" /////////////////////////////// Main Setup //////////////////////////////////////// """

# Test Parameters
sourcefile = "sourcelist.txt"
N       = args.N       # Grid Size
numsrc  = args.numsrc         # Number of sources
plot_final    = args.plot         # Whether or not to plot results
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

# TODO: replace by new method. doesn't work yet for some reason.
# srcpos, srcflux, numsrc = read_sources(sourcefile,numsrc,"pyc2ray_octa")

# Initialize Arrays
# cdh2 = np.ravel(np.zeros((N,N,N),dtype='float64'))
# ndens = 1e-3*np.ravel(np.ones((N,N,N),dtype='float64') )
# phi_ion2 = np.ravel(np.zeros((N,N,N),dtype='float64') )
# xh_av = 1.2e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )

ndens = 1e-3*np.ones((N,N,N),dtype='float64')
phi_ion2 = np.zeros((N,N,N),dtype='float64')
xh_av = 1.2e-3 * np.ones((N,N,N),dtype='float64')

# Initialize GPU and allocate memory
#octa.device_init(N)
pc2r.device_init(N)
""" ////////////////////////////////// Run Tests ////////////////////////////////////// """

print(f"Doing radius r = {rad:.2f}")
print("Running OCTA GPU...")
t5 = time.time()
#octa.do_all_sources(srcpos,srcflux,rad,cdh2,sig,dxbox,ndens,xh_av,phi_ion2,numsrc,N)
phi_ion_2 = pc2r.do_all_sources_octa(srcflux,srcpos,rad,sig,dxbox,ndens,xh_av,phi_ion2,N)
t6 = time.time()
#cdh2 = cdh2.reshape((N,N,N))
phi_ion2 = phi_ion2.reshape((N,N,N))

octa.device_close() # Deallocate GPU memory
print(f"Done. took {t6-t5:.2f} second(s).")

""" ///////////////////////////////// Visualization /////////////////////////////////// """
print(f"Average ionization rate: {phi_ion2.mean():.5e}")

loggamma = np.where(phi_ion2 != 0.0,np.log(phi_ion2),np.nan) #= np.log(phi_ion_f)

if args.pickle:
    with open(f"octa_{numsrc:n}_sources_r={rad:n}.pkl","wb") as f:
        pkl.dump(loggamma, f)

if plot_final:
    tomo = zTomography_rates(loggamma, N // 2)
    plt.show()