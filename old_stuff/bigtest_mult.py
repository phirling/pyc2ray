import numpy as np
import c2ray_core as c2r
from parameters import Params
import astropy.units as u
import time
import matplotlib.pyplot as plt
import argparse
import evolve as evo

parser = argparse.ArgumentParser()
parser.add_argument("-N",type=int,default=128)
parser.add_argument("--randomdens",action='store_true')
args = parser.parse_args()

# Number of cells in each dimension
N = int(args.N)
ii = 64

# Whether to use a random or uniform density field
randomdens = args.randomdens

# Import the parameters used in original c2ray
p = Params("parameters.yml")
bh00 = p.bh00
colh0 = p.colh0.value
temph0 = p.temph0.value
albpow = p.albpow
abu_c = p.abu_c
temp0 = 1e4 #p.initial_temperature
sig = p.sigma_HI_at_ion_freq.value # 6.30e-18


# Run Parameters (sizes, etc.)
# dt = 1e4 * u.year.to('s')
t_evol = 139 * u.Myr.to('s')
tsteps = 139
dt = t_evol / tsteps # timestep
#xbox = 5e22 # cm # box size
xbox = 14 * u.kpc.to('cm')
print(xbox)
dxbox = xbox / N
dr = dxbox * np.ones(3) # cell size
#dr = 6.7e20*np.ones(3)
#avgdens = 1.0e-4
#avgdens = 1.981E-04 #1.87e-4 # cm-3
avgdens = 1.0e-3
#xhav = 2.e-4 # initial ionization fraction (uniform)
xhav = 1.2e-3


# Source Parameters
numsrc = 1
srcpos = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
#srcpos[:,0] = np.array([2,N//2,N//2])
srcpos[:,0] = np.array([ii,64,64])
#srcpos[:,1] = np.array([1,80,80])
#srcpos[:,0] = np.array([50,50,50])
py_srcpos = srcpos[:,0] - 1
#srcflux[0] = 1.0e55
#srcflux[0] = 1.0e54
srcflux[0] = 5.0e48
#srcflux[1] = 1.0e48

# Shadow
shadowpos = np.array([1,20,20])
shadow = 0
shadowstrength = 8
shadowr = 10

def add_shadow(shadowpos,ndens,strength,shadowr):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                pos = np.array([i,j,k])
                r = np.sqrt(np.sum((pos-shadowpos)**2))
                if r < shadowr:
                    ndens[i,j,k] *= strength
                #if np.abs(i-pos[0]) < dx and np.abs(j-pos[1]) < dy and np.abs(k-pos[2]) < dz:
                #    ndens[i,j,k] += strength

# Create Density field
print("Creating Density...")
if randomdens:
    #ndens_1 = avgdens * np.random.uniform(size=(N,N,N)) #np.ones((N,N,N))
    ndens_1 = (avgdens / np.random.uniform(0.666,1.5,size=(N,N,N)) )
else:
    ndens_1 = avgdens * np.ones((N,N,N))

if shadow:
    add_shadow(shadowpos,ndens_1,shadowstrength,shadowr)
# Initialize Arrays
ndens_1_f = np.asfortranarray(ndens_1)
xh_f = xhav*np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')
coldens_out_f = np.zeros((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')

xh_new_f = xh_f
# USE EVOLVE3D WRAPPER FUNCTION TO EVOLVE GRID
for t in range(tsteps):
    ct = ((t+1) * dt) * u.s.to('yr')
    print(f"--- Timestep {t+1:n}, tf = {ct : .2e} yrs ---")
    xh_new_f = evo.evolve3D(dt,dr,srcflux,srcpos,temp_f,ndens_1_f,coldens_out_f,
                xh_new_f,phi_ion_f,sig,bh00,albpow,colh0,temph0,abu_c)

# DISPLAY
print("Making Figure...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(12.5,3.7))

# Left: column density
ax1.set_title(f"Column Density, N={N}",fontsize=12)
im1 = ax1.imshow(coldens_out_f[ii,:,:],origin='lower')
c1 = plt.colorbar(im1,ax=ax1)
#ax1.set_xlabel(f"Computation Time: {t_evo : .7f} s",fontsize=12)

# Middle: ionization rate
ax2.set_title(f"Ionization Rate",fontsize=12)
# For some reason this gets mapped wrong with log, do manually:
loggamma = np.log(phi_ion_f[ii,:,:])
im2 = ax2.imshow(loggamma,origin='lower',cmap='inferno')
c2 = plt.colorbar(im2,ax=ax2)
c2.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)

# Right: ionization fraction
#ax3.set_title(f"Mean Ionization Fraction",fontsize=12)
ax3.set_title(f"Neutral Hydrogen Fraction",fontsize=12)
#im3 = ax3.imshow(xh_new_f[ii,:,:],origin='lower',cmap='rainbow',norm='log',vmin=1e-4,vmax=1.0) #cmap='YlGnBu_r'
im3 = ax3.imshow(1.0 - xh_new_f[ii,:,:],origin='lower',cmap='jet',norm='log',vmin=1e-3,vmax=1.0) #cmap='YlGnBu_r'
c3 = plt.colorbar(im3,ax=ax3)
#ax3.set_xlabel(f"Computation Time: {t_chem : .7f} s",fontsize=12)

fig.tight_layout()

plt.show()

import pickle as pkl
with open("xh_final.pkl","wb") as f:
    pkl.dump(xh_new_f,f)
