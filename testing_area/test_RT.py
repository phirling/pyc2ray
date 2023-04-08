import c2ray_core as c2r
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import time
import RTC

N = 300

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
srcpos[:,0] = np.array([65,65,65])      # Position of source
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

phi_ion_f = np.zeros((N,N,N),order='F')
coldensh_out_1 = np.zeros((N,N,N),order='F')

t1 = time.time()
c2r.raytracing.do_source(srcflux,srcpos,1,last_l,last_r,coldensh_out_1,sig,dr,ndens_f,xh_f,phi_ion_f)
t2 = time.time()

# C++ version
cdh1 = np.ravel(np.zeros((N,N,N),dtype='float64'))
cdh2 = np.ravel(np.zeros((N,N,N),dtype='float64'))
srcpos = np.ravel(np.array([[64],[64],[64]],dtype='int32'))
ndens = 1e-3*np.ravel(np.ones((N,N,N),dtype='float64') )
phi_ion = np.ravel(np.zeros((N,N,N),dtype='float64') )
xh_av = 1.2e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )
NumSrc = 1

t3 = time.time()
RTC.octa(srcpos,0,cdh1,sig,dxbox,ndens,xh_av,phi_ion,NumSrc,N)
t4 = time.time()
cdh1 = cdh1.reshape((N,N,N))

# CUDA version
t5 = time.time()
RTC.octa_gpu(srcpos,0,cdh2,sig,dxbox,ndens,xh_av,phi_ion,NumSrc,N)
t6 = time.time()
cdh2 = cdh2.reshape((N,N,N))

#score = np.sqrt(((coldensh_out_2 / coldensh_out_1 - 1)**2).sum())

max_cdh = coldensh_out_1.max()
min_cdh = coldensh_out_1.min()

# Display Results
def residual(A,B):
    return A[:,:,zslice] / B[:,:,zslice] - 1

fig, ax = plt.subplots(2, 3,figsize=(12.5,8))

ax[0,0].set_title(f"Reference (C2Ray), $t = {t2-t1:.3f}$ s",fontsize=12)
im1 = ax[0,0].imshow(coldensh_out_1[:,:,zslice],origin='lower') #,vmin=min_cdh,vmax=max_cdh)
c1 = plt.colorbar(im1,ax=ax[0,0])

ax[1,0].set_title("Residual",fontsize=12)
resid1 = residual(coldensh_out_1, coldensh_out_1)
im3 = ax[1,0].imshow(resid1,cmap='bwr',origin='lower')
c3 = plt.colorbar(im3,ax=ax[1,0])

ax[0,1].set_title(f"OCTA, $t = {t4-t3:.3f}$ s",fontsize=12)
im2 = ax[0,1].imshow(cdh1[:,:,zslice],origin='lower')
c2 = plt.colorbar(im2,ax=ax[0,1])

ax[1,1].set_title("Residual",fontsize=12)
resid2 = residual(cdh1, coldensh_out_1)
im4 = ax[1,1].imshow(resid2,cmap='bwr',origin='lower',vmin=-1,vmax=1)
c4 = plt.colorbar(im4,ax=ax[1,1])

ax[0,2].set_title(f"OCTA GPU, $t = {t6-t5:.3f}$ s",fontsize=12)
im5 = ax[0,2].imshow(cdh2[:,:,zslice],origin='lower')
c5 = plt.colorbar(im5,ax=ax[0,2])

ax[1,2].set_title("Residual",fontsize=12)
resid3 = residual(cdh2, coldensh_out_1)
im6 = ax[1,2].imshow(resid3,cmap='bwr',origin='lower')
c6 = plt.colorbar(im6,ax=ax[1,2])

fig.tight_layout()

plt.show()