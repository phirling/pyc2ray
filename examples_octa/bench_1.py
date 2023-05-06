import c2ray as c2r                     # Fortran Module (c2ray)
import octa                              # C++ Module (CUDA, CUDA GPU)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import time

"""
ABOUT
Do Raytracing in cubes of increasing size in fixed mesh and measure time.

"""

""" /////////////////////////////// Main Setup //////////////////////////////////////// """

# Test Parameters
N       = 300       # Grid Size
srcx    = 150       # Source x-position (x=y=z)
numsrc  = 1         # Number of sources
zslice  = 150       # z-coordinate of box to visualize
plot_interm    = 0         # Whether or not to plot results
plot_interm_png = 1
plot_final    = 1         # Whether or not to plot results
output_base = "./benchmark_results/"

# Result Arrays
N_rads = 20
rads = np.floor(np.linspace(5,200,N_rads))
timings_c2r = np.empty(N_rads)
timings_octa = np.empty(N_rads)
scores_cd_c2r = np.empty(N_rads)
scores_cd_octa = np.empty(N_rads)
scores_phi_c2r = np.empty(N_rads)
scores_phi_octa = np.empty(N_rads)

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


# Metrics
def score(A,B,rad):
    in_sphere = rr < rad
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
cdh2 = np.ravel(np.zeros((N,N,N),dtype='float64'))
ndens = 1e-3*np.ravel(np.ones((N,N,N),dtype='float64') )
phi_ion2 = np.ravel(np.zeros((N,N,N),dtype='float64') )
xh_av = 1.2e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )

# Initialize GPU and allocate memory
octa.device_init(N)

""" ////////////////////////////////// Run Tests ////////////////////////////////////// """

# Do an empty call of each function because somehow this improves performance ?
c2r.raytracing.do_source(srcflux,srcpos_f,1,1,coldensh_out_f,sig,dr,ndens_f,xh_f,phi_ion_f)
octa.do_source(srcpos,srcflux,0,1,cdh2,sig,dxbox,ndens,xh_av,phi_ion2,numsrc,N)
    
for a, rad in enumerate(rads):
    print(f"Doing radius r = {rad:.2f}")
    print("Running c2ray...")
    phi_ion_f[:,:,:] = 0.0
    t1 = time.time()
    c2r.raytracing.do_source(srcflux,srcpos_f,1,rad,coldensh_out_f,sig,dr,ndens_f,xh_f,phi_ion_f)
    t2 = time.time()

    print("Running OCTA GPU...")
    t5 = time.time()
    octa.do_source(srcpos,srcflux,0,rad,cdh2,sig,dxbox,ndens,xh_av,phi_ion2,numsrc,N)
    t6 = time.time()
    cdh2 = cdh2.reshape((N,N,N))
    phi_ion2 = phi_ion2.reshape((N,N,N))

    timings_c2r[a] = t2-t1
    timings_octa[a] = t6-t5
    scores_cd_c2r[a] = score(coldensh_out_f, coldensh_out_f, rad)
    scores_cd_octa[a] = score(cdh2, coldensh_out_f, rad)
    scores_phi_c2r[a] = score(phi_ion_f, phi_ion_f, rad)
    scores_phi_octa[a] = score(phi_ion2, phi_ion_f, rad)

    if plot_interm:
        plt.imshow(np.log(phi_ion_f[:,:,zslice]),origin='lower')
        plt.show()
    elif plot_interm_png:
        fig_i, (ax_i1,ax_i2) = plt.subplots(1,2,figsize=(10,5))
        fname = output_base + f"bench_r={rad:n}.png"
        ax_i1.imshow(np.log(phi_ion_f[:,:,zslice]),origin='lower')
        ax_i2.imshow(np.log(phi_ion2[:,:,zslice]),origin='lower')
        ax_i1.add_patch(Circle([srcx,srcx],rad,fill=0,ls='--',ec='red',label=f"r = {rad:n}"))
        ax_i2.add_patch(Circle([srcx,srcx],rad,fill=0,ls='--',ec='red',label=f"r = {rad:n}"))
        ax_i1.legend()
        ax_i2.legend()
        ax_i1.set_title("C2Ray")
        ax_i2.set_title("OCTA GPU")
        fig_i.savefig(fname,bbox_inches='tight')
        plt.close(fig_i)
octa.device_close() # Deallocate GPU memory

""" /////////////////////////////////// Analysis ////////////////////////////////////// """

print("Timings (c2ray): \n",timings_c2r)
print("Timings (octa): \n",timings_octa)
print("CD score (c2ray): \n",scores_cd_c2r)
print("CD score (octa): \n",scores_cd_octa)
print("PHI score (c2ray): \n",scores_phi_c2r)
print("PHI score (octa): \n",scores_phi_octa)


""" ///////////////////////////////// Visualization /////////////////////////////////// """

if plot_final:
    fig, ax = plt.subplots(figsize=(5,5))

    ax.plot(rads,timings_c2r,'o--',label="C2Ray")
    ax.plot(rads,timings_octa,'s-',label="OCTA GPU")

    ax.set_xlabel("Raytracing Radius [Mesh Coordinates]")
    ax.set_ylabel("Execution Time [s]")
    ax.legend()

    fig.tight_layout()

    fig.savefig(output_base + "bench_timings.png",bbox_inches='tight')
    #plt.show()