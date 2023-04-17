import c2ray as c2r                     # Fortran Module (c2ray)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time
import argparse
from tomography import zTomography # Custom module to visualize datacube
import pickle as pkl
from readsources import read_sources

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
Test the c2ray version of the RT

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

""" /////////////////////////// C2Ray Version Setup /////////////////////////////////// """

# Source Setup
srcpos_f, srcflux, numsrc = read_sources(sourcefile,numsrc,"pyc2ray")

# Initialize Arrays
ndens_f = avgdens * np.ones((N,N,N),order='F')
xh_f = xhav*np.ones((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')
coldensh_out_f = np.zeros((N,N,N),order='F')

""" ////////////////////////////////// Run Tests ////////////////////////////////////// """

print(f"Doing radius r = {rad:.2f}")
print("Running c2ray...")
phi_ion_f[:,:,:] = 0.0
t1 = time.time()
c2r.raytracing.do_all_sources(srcflux,srcpos_f,rad,coldensh_out_f,sig,dr,ndens_f,xh_f,phi_ion_f)
t2 = time.time()

print(f"Done. took {t2-t1:.2f} second(s).")

""" ///////////////////////////////// Visualization /////////////////////////////////// """

print(f"Average ionization rate: {phi_ion_f.mean():.5e}")

loggamma = np.where(phi_ion_f != 0.0,np.log(phi_ion_f),np.nan)

if args.pickle:
    with open(f"c2ray_{numsrc:n}_sources_r={rad:n}.pkl","wb") as f:
        pkl.dump(loggamma, f)

if plot_final:
    tomo = zTomography(loggamma, N // 2)
    plt.show()