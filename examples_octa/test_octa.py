import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time
import argparse
import sys
sys.path.append("../")
import pyc2ray as pc2r

parser = argparse.ArgumentParser()
parser.add_argument("-r",type=int,default=50)
parser.add_argument("-srcx",type=int,default=150)
parser.add_argument("-zslice",type=int,default=None)
args = parser.parse_args()

"""
ABOUT
Test the c2ray version of the RT

"""

""" /////////////////////////////// Main Setup //////////////////////////////////////// """

# Test Parameters
N       = 300       # Grid Size
srcx    = args.srcx       # Source x-position (x=y=z)
numsrc  = 1         # Number of sources
if args.zslice is None:
    zslice = srcx
else:
    zslice = args.zslice
plot_interm    = 1         # Whether or not to plot results
plot_final    = 1         # Whether or not to plot results
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
srcpos = np.ravel(np.array([[srcx],[srcx],[srcx]],dtype='int32')) # C++ version uses flattened arrays
srcflux = np.empty(numsrc)
srcflux[0] = 5.0e48

# Initialize Arrays
cdh2 = np.ravel(np.zeros((N,N,N),dtype='float64'))
ndens = 1e-3*np.ravel(np.ones((N,N,N),dtype='float64') )
phi_ion2 = np.ravel(np.zeros((N,N,N),dtype='float64') )
xh_av = 1.2e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )

# Initialize GPU and allocate memory
pc2r.device_init(N)

""" ////////////////////////////////// Run Tests ////////////////////////////////////// """

print(f"Doing radius r = {rad:.2f}")
print("Running OCTA GPU...")
t5 = time.time()
# pc2r.do_source(srcpos,srcflux,0,rad,cdh2,sig,dxbox,ndens,xh_av,phi_ion2,numsrc,N)
pc2r.do_source_octa(srcflux,srcpos,0,rad,sig,dxbox,ndens,xh_av,phi_ion2,N)
t6 = time.time()
cdh2 = cdh2.reshape((N,N,N))
phi_ion2 = phi_ion2.reshape((N,N,N))

pc2r.device_close() # Deallocate GPU memory

""" ///////////////////////////////// Visualization /////////////////////////////////// """

print("Making Figure...")
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10.5,3.7))

# Left: column density
ax1.set_title(f"Column Density, N={N}",fontsize=12)
im1 = ax1.imshow(cdh2[:,:,zslice],origin='lower')
c1 = plt.colorbar(im1,ax=ax1)

# Middle: ionization rate
ax2.set_title(f"Ionization Rate",fontsize=12)
# For some reason this gets mapped wrong with log, do manually:
loggamma = np.where(phi_ion2 != 0.0,np.log(phi_ion2),np.nan) #= np.log(phi_ion_f)
im2 = ax2.imshow(loggamma[:,:,zslice],origin='lower',cmap='inferno')
c2 = plt.colorbar(im2,ax=ax2)
c2.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)

fig.tight_layout()

plt.show()