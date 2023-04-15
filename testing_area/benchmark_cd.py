import c2ray as c2r                     # Fortran Module (c2ray)
import RTC                              # C++ Module (CUDA, CUDA GPU)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time


""" /////////////////////////////// Main Setup //////////////////////////////////////// """

# Test Parameters
N       = 300       # Grid Size
srcx    = 150       # Source x-position (x=y=z)
rad     = 149       # Radius of Raytracing
numsrc  = 1         # Number of sources
zslice  = 150       # z-coordinate of box to visualize
plot    = True     # Whether or not to plot results

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

""" ////////////////////////////// Metrics Setup /////////////////////////////////////// """

# Mask for accuracy metrics (only compare cells within a radius rad of source)
iv = np.arange(-srcx,N-srcx)
ii,jj,kk = np.meshgrid(iv,iv,iv)
rr = np.sqrt(ii**2 + jj**2 + kk**2)
in_sphere = rr < rad # <-- mask for all cells that are in the sphere to be tested

# Metrics
def residual(A,B):
    A_m = np.where(in_sphere,A,1)
    B_m = np.where(in_sphere,B,1)
    return A_m[:,:,zslice] / B_m[:,:,zslice] - 1

def score(A,B):
    A_m = np.where(in_sphere,A,1)
    B_m = np.where(in_sphere,B,1)
    return np.sqrt(((A_m / B_m - 1)**2).sum())

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

""" ////////////////////////// C++ (OCTA) Version Setup /////////////////////////////// """

# Source Setup
srcpos = np.ravel(np.array([[srcx],[srcx],[srcx]],dtype='int32')) # C++ version uses flattened arrays

# Initialize Arrays
cdh1 = np.ravel(np.zeros((N,N,N),dtype='float64'))
cdh2 = np.ravel(np.zeros((N,N,N),dtype='float64'))
ndens = 1e-3*np.ravel(np.ones((N,N,N),dtype='float64') )
phi_ion = np.ravel(np.zeros((N,N,N),dtype='float64') )
xh_av = 1.2e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )

# Initialize GPU and allocate memory
RTC.device_init(N)

""" ////////////////////////////////// Run Tests ////////////////////////////////////// """

print("Running c2ray...")
t1 = time.time()
c2r.raytracing.do_source(srcflux,srcpos_f,1,rad,coldensh_out_f,sig,dr,ndens_f,xh_f,phi_ion_f)
t2 = time.time()

print("Running OCTA...")
t3 = time.time()
RTC.octa(srcpos,srcflux,0,rad,cdh1,sig,dxbox,ndens,xh_av,phi_ion,numsrc,N)
t4 = time.time()
cdh1 = cdh1.reshape((N,N,N)) # Convert flatened array to 3D

print("Running OCTA GPU...")
t5 = time.time()
RTC.octa_gpu(srcpos,srcflux,0,rad,cdh2,sig,dxbox,ndens,xh_av,phi_ion,numsrc,N)
t6 = time.time()
cdh2 = cdh2.reshape((N,N,N)) # Convert flatened array to 3D

RTC.device_close() # Deallocate GPU memory

""" /////////////////////////////////// Analysis ////////////////////////////////////// """

print("-- Timings --")
print(f"Time (C2Ray):       {t2-t1:.3f} [s]")
print(f"Time (OCTA):        {t4-t3:.3f} [s]")
print(f"Time (OCTA GPU):    {t6-t5:.3f} [s]")
print("\n-- Results --")
print(f"Global error score (C2Ray):       {score(coldensh_out_f,coldensh_out_f):.3e}")
print(f"Global error score (OCTA):        {score(cdh1,coldensh_out_f):.3e}")
print(f"Global error score (OCTA GPU):    {score(cdh2,coldensh_out_f):.3e}")


""" ///////////////////////////////// Visualization /////////////////////////////////// """

if plot:
    fig, ax = plt.subplots(2, 3,figsize=(12.5,8))

    ax[0,0].set_title(f"Reference (C2Ray), $t = {t2-t1:.3f}$ s",fontsize=12)
    im1 = ax[0,0].imshow(coldensh_out_f[:,:,zslice],origin='lower') #,vmin=min_cdh,vmax=max_cdh)
    c1 = plt.colorbar(im1,ax=ax[0,0])
    ax[0,0].add_patch(Circle([srcx,srcx],rad,fill=0,ls='--',ec='white'))

    ax[1,0].set_title("Residual",fontsize=12)
    resid1 = residual(coldensh_out_f, coldensh_out_f)
    im3 = ax[1,0].imshow(resid1,cmap='bwr',origin='lower')
    c3 = plt.colorbar(im3,ax=ax[1,0])

    ax[0,1].set_title(f"OCTA, $t = {t4-t3:.3f}$ s",fontsize=12)
    im2 = ax[0,1].imshow(cdh1[:,:,zslice],origin='lower')
    c2 = plt.colorbar(im2,ax=ax[0,1])
    ax[0,1].add_patch(Circle([srcx,srcx],rad,fill=0,ls='--',ec='white'))

    ax[1,1].set_title("Residual",fontsize=12)
    resid2 = residual(cdh1, coldensh_out_f)
    im4 = ax[1,1].imshow(resid2,cmap='bwr',origin='lower',vmin=-1,vmax=1)
    c4 = plt.colorbar(im4,ax=ax[1,1])

    ax[0,2].set_title(f"OCTA GPU, $t = {t6-t5:.3f}$ s",fontsize=12)
    im5 = ax[0,2].imshow(cdh2[:,:,zslice],origin='lower')
    c5 = plt.colorbar(im5,ax=ax[0,2])
    ax[0,2].add_patch(Circle([srcx,srcx],rad,fill=0,ls='--',ec='white'))

    ax[1,2].set_title("Residual",fontsize=12)
    resid3 = residual(cdh2, coldensh_out_f)
    im6 = ax[1,2].imshow(resid3,cmap='bwr',origin='lower',vmin=-1,vmax=1)
    c6 = plt.colorbar(im6,ax=ax[1,2])

    fig.tight_layout()

    plt.show()