import c2ray as c2r                     # Fortran Module (c2ray)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time
import argparse

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

""" /////////////////////////// C2Ray Version Setup /////////////////////////////////// """

# Source Setup
srcpos_f = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
srcpos_f[:,0] = np.array([srcx+1,srcx+1,srcx+1])
srcflux[0] = 5.0e48 # Strength of source (not actually used here but required argument)

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
c2r.raytracing.do_source(srcflux,srcpos_f,1,rad,coldensh_out_f,sig,dr,ndens_f,xh_f,phi_ion_f)
t2 = time.time()

""" ///////////////////////////////// Visualization /////////////////////////////////// """

print("Making Figure...")
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10.5,3.7))

# Left: column density
ax1.set_title(f"Column Density, N={N}",fontsize=12)
im1 = ax1.imshow(coldensh_out_f[:,:,zslice],origin='lower')
c1 = plt.colorbar(im1,ax=ax1)

# Middle: ionization rate
ax2.set_title(f"Ionization Rate",fontsize=12)
# For some reason this gets mapped wrong with log, do manually:
loggamma = np.where(phi_ion_f != 0.0,np.log(phi_ion_f),np.nan) #= np.log(phi_ion_f)
im2 = ax2.imshow(loggamma[:,:,zslice],origin='lower',cmap='inferno')
c2 = plt.colorbar(im2,ax=ax2)
c2.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)

fig.tight_layout()

plt.show()