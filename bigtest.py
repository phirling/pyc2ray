import numpy as np
import RTSC as rtsc
from parameters import Params
import astropy.units as u
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-N",type=int,default=128)
parser.add_argument("--randomdens",action='store_true')
args = parser.parse_args()

# Number of cells in each dimension
N = int(args.N)

# Whether to use a random or uniform density field
randomdens = args.randomdens

# Import the parameters used in original c2ray
p = Params("parameters.yml")
bh00 = p.bh00
colh0 = p.colh0.value
temph0 = p.temph0.value
albpow = p.albpow
abu_c = p.abu_c
temp0 = p.initial_temperature
sig = p.sigma_HI_at_ion_freq.value # 6.30e-18


# Run Parameters (sizes, etc.)
# dt = 1e4 * u.year.to('s')
t_evol = 500 * u.Myr.to('s')
dt = t_evol / 100 # timestep
xbox = 5e22 # cm # box size
dxbox = xbox / N
dr = dxbox * np.ones(3) # cell size
#dr = 6.7e20*np.ones(3)
#avgdens = 1.0e-4
avgdens = 1.981E-04 #1.87e-4 # cm-3
xhav = 2.e-4 # initial ionization fraction (uniform)


# Source Parameters
numsrc = 1
srcpos = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
srcpos[:,0] = np.array([2,N//2,N//2])
#srcpos[:,0] = np.array([50,50,50])
py_srcpos = srcpos[:,0] - 1
srcflux[0] = 1.0e55
#srcflux[0] = 1.0e54
#srcflux[0] = 1.0e57


# For fortran version (non periodic)
last_l = np.ones(3)
last_r = (N) * np.ones(3)


# Create Density field
print("Creating Density...")
if randomdens:
    ndens_1 = avgdens * np.random.uniform(size=(N,N,N)) #np.ones((N,N,N))
else:
    ndens_1 = avgdens * np.ones((N,N,N))

# Initialize Arrays
ndens_1_f = np.asfortranarray(ndens_1)
xh_f = xhav*np.ones((N,N,N),order='F')
xh_av_f = xhav*np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')
coldens_out_f = np.zeros((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')

# RAYTRACE
print("Doing Raytracing/Evolve...")
t1 = time.perf_counter()
rtsc.raytracing_sc.evolve3d(srcflux,srcpos,1,last_l,last_r,coldens_out_f,sig,dr,ndens_1_f,xh_av_f,phi_ion_f)
t2 = time.perf_counter()
t_evo = t2-t1

# DO CHEMISTRY
print("Doing Chemistry...")
t3 = time.perf_counter()
cf = rtsc.chemistry.global_pass(dt,ndens_1_f,temp_f,xh_f,xh_av_f,phi_ion_f,bh00,albpow,colh0,temph0,abu_c)
t4 = time.perf_counter()
t_chem = t4 - t3

print(np.mean(coldens_out_f))

# DISPLAY
print("Making Figure...")
ii = 2
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(12.5,3.7))

# Left: column density
ax1.set_title(f"Column Density, N={N}",fontsize=12)
im1 = ax1.imshow(coldens_out_f[ii,:,:],origin='lower')
c1 = plt.colorbar(im1,ax=ax1)
ax1.set_xlabel(f"Computation Time: {t_evo : .7f} s",fontsize=12)

# Middle: ionization rate
ax2.set_title(f"Ionization Rate",fontsize=12)
# For some reason this gets mapped wrong with log, do manually:
loggamma = np.log(phi_ion_f[ii,:,:])
im2 = ax2.imshow(loggamma,origin='lower',cmap='inferno')
c2 = plt.colorbar(im2,ax=ax2)
c2.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)

# Right: ionization fraction
ax3.set_title(f"Mean Ionization Fraction",fontsize=12)
im3 = ax3.imshow(xh_av_f[ii,:,:],origin='lower',cmap='rainbow',norm='log') #cmap='YlGnBu_r'
c3 = plt.colorbar(im3,ax=ax3)
ax3.set_xlabel(f"Computation Time: {t_chem : .7f} s",fontsize=12)

fig.tight_layout()

import pickle as pkl
aa = phi_ion_f[ii,:,:]
with open("pack.pkl",'wb') as f:
    pkl.dump(aa,f)
plt.show()
