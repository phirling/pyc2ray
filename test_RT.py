import c2ray_core as c2r
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import time

N = 128

zslice = 64


dt = 1.0 * u.Myr.to('s')                # Timestep
boxsize = 14 * u.kpc.to('cm')           # Simulation box size
dxbox = boxsize / N                     # Cell Size (1D)
dr = dxbox * np.ones(3)                 # Cell Size (3D)
avgdens = 1.0e-3                        # Constant Hydrogen number density
xhav = 1.2e-3                           # Initial ionization fraction

# Source Parameters
numsrc = 1                              # Number of sources
srcpos = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
srcpos[:,0] = np.array([64,64,64])      # Position of source
srcflux[0] = 5.0e48                     # Strength of source

# /////////////////////////////////////////////////////////////////////////////////



# C2Ray parameters. These can also be imported from
# the yaml file but for now its simpler like this
temp0 = 1e4
sig = 6.30e-18

# Initialize Arrays
ndens_f = avgdens * np.ones((N,N,N),order='F')
xh_f = xhav*np.ones((N,N,N),order='F')
xh_f2 = xhav*np.ones((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')

last_l = np.ones(3)         # mesh position of left end point for RT
last_r = N * np.ones(3)    # mesh position of right end point for RT

phi_ion = np.zeros((N,N,N),order='F')
phi_ion2 = np.zeros((N,N,N),order='F')
coldensh_out_1 = np.zeros((N,N,N),order='F')
coldensh_out_2 = np.zeros((N,N,N),order='F')

t1 = time.time()
c2r.raytracing.do_source(srcflux,srcpos,1,last_l,last_r,coldensh_out_1,sig,dr,ndens_f,xh_f,phi_ion)
t2 = time.time()
c2r.raytracing.do_source_octa(srcflux,srcpos,1,last_l,last_r,coldensh_out_2,sig,dr,ndens_f,xh_f2,phi_ion2)
t3 = time.time()

score = np.sqrt(((coldensh_out_2 / coldensh_out_1 - 1)**2).sum())
print(f"Full score: {score}")
ma1 = coldensh_out_1.max()
ma2 = coldensh_out_2.max()
max_cdh = max(ma1,ma2)

min1 = coldensh_out_1.min()
min2 = coldensh_out_2.min()
min_cdh = min(min1,min2)

# Display Results
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(12.5,3.7))

# Left: column density
ax1.set_title(f"$\Delta t = {t2-t1:.3f}$ s",fontsize=12)
im1 = ax1.imshow(coldensh_out_1[:,:,zslice],origin='lower',vmin=min_cdh,vmax=max_cdh)
c1 = plt.colorbar(im1,ax=ax1)

ax2.set_title(f"$\Delta t = {t3-t2:.3f}$ s",fontsize=12)
im2 = ax2.imshow(coldensh_out_2[:,:,zslice],origin='lower',vmin=min_cdh,vmax=max_cdh)
c2 = plt.colorbar(im2,ax=ax2)

ax3.set_title("Residual",fontsize=12)
resid = coldensh_out_2[:,:,zslice] / coldensh_out_1[:,:,zslice] - 1
im3 = ax3.imshow(resid,cmap='bwr',origin='lower',vmin=-1,vmax=1)
c3 = plt.colorbar(im3,ax=ax3)

fig.tight_layout()

plt.show()