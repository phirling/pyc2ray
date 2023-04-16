import c2ray as c2r                     # Fortran Module (c2ray)
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
srcpos_f = 1+np.random.randint(0,N,size=3*numsrc)
print("Source positions: ")
print(srcpos_f)
srcpos_f = srcpos_f.reshape((3,numsrc),order='F')
print("After reshape: ")
print(srcpos_f)
srcflux = 5.0e48 * np.ones(numsrc)
# srcpos_f = np.empty((3,numsrc),dtype='int')
# srcflux = np.empty(numsrc)
# srcpos_f[:,0] = np.array([srcx+1,srcx+1,srcx+1])
# srcflux[0] = 5.0e48 # Strength of source (not actually used here but required argument)


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

loggamma = np.where(phi_ion_f != 0.0,np.log(phi_ion_f),np.nan) #= np.log(phi_ion_f)

if args.pickle:
    with open(f"c2ray_{numsrc:n}_sources_r={rad:n}.pkl","wb") as f:
        pkl.dump(loggamma, f)
"""
print("Making Figure...")
fig, (ax1) = plt.subplots(1, 1,figsize=(6,6))

zz = zslice
# ionization rate
ax1.set_title(f"Ionization Rate",fontsize=12)
# For some reason this gets mapped wrong with log, do manually:
im2 = ax1.imshow(loggamma[:,:,zz],origin='lower',cmap='inferno')
im2.zz = zz

c2 = plt.colorbar(im2,ax=ax1)
c2.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)


def switch(event):
    up = event.key == 'up'
    down = event.key == 'down'
    zz = im2.zz
    if up:
        zz += 10
    elif down:      
        zz -= 10
    if up or down:
        if zz in range(N):
            im2.set_data(loggamma[:,:,zz])
            im2.zz = zz
            fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event',switch)

fig.tight_layout()
"""

tomo = zTomography(loggamma, zslice)
plt.show()