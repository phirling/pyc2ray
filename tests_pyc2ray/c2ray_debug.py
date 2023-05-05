import numpy as np
import astropy.units as u
import time
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("../")
import pyc2ray as pc2r
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("-N",type=int,default=128)
args = parser.parse_args()

# /////////////////////////////////////////////////////////////////////////////////

# Number of cells in each dimension
N = int(args.N)

# Display options
display = False                          # Display results at the end of run
zslice = 64                             # z-slice of the box to visualize

# Output settings
res_basename = "./results_debug/"             # Directory to store pickled results
delta_results = 10                      # Number of timesteps between results

# Run Parameters (sizes, etc.)
tsim = 10                              # Simulation Time (in Myrs)
t_evol = tsim * u.Myr.to('s')           # Simulation Time (in seconds)
tsteps = 1                            # Number of timesteps
#dt = t_evol / tsteps                    # Timestep
dt = 315576009522439.61 # 10 Myr # C2Ray value  t_evol / tsteps                    # Timestep
boxsize = 14 * u.kpc.to('cm')           # Simulation box size
dxbox = 3.3753127248391602E+020 #boxsize / N                     # Cell Size (1D)
dr = dxbox * np.ones(3)                 # Cell Size (3D)
print("dr = ",dr)
# 3.3753127248391602E+020
# 3.37495985e+20
avgdens = 1.0e-3                        # Constant Hydrogen number density
xhav = 1.2e-3                           # Initial ionization fraction

# Source Parameters
numsrc = 1                              # Number of sources
srcpos = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
srcpos[:,0] = np.array([64,64,64])      # Position of source
srcflux[0] = 5.0e48                     # Strength of source
max_subbox = 1000                               # Raytracing box size
subboxsize = 5
# /////////////////////////////////////////////////////////////////////////////////



# C2Ray parameters. These can also be imported from
# the yaml file but for now its simpler like this
eth0=13.598
bh00=2.59e-13
ev2k=1.0/8.617e-05
temph0=eth0*ev2k
temp0 = 1e4
sig = 6.30e-18
fh0=0.83
xih0=1.0
albpow=-0.7
colh0=1.3e-8*fh0*xih0/(eth0*eth0)
abu_c=7.1e-7

# Initialize Arrays
ndens_f = avgdens * np.ones((N,N,N),order='F')
xh_f = xhav*np.ones((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')
# Initialize next step
xh_new_f = xh_f

# Count time
tinit = time.time()

print(f"Running on {numsrc:n} source(s) on {N:n}^3 grid.")
print(f"Constant density: {avgdens:.2e} cm^-3, Temperature: {temp0:.1e} K, initial ionized fraction: {xhav:.2e}")
print(f"Simulation time is {tsim:.2f} Myr(s), using timestep {tsim/tsteps:.2f} Myr(s).")
print("Starting main loop...")

# ===================================== Main loop =====================================
outputn = 0
for t in range(tsteps):
    ct = ((t+1) * dt) * u.s.to('yr')
    if t % delta_results == 0:
        out = res_basename + f"xfrac_{outputn:04}.pkl"
        out_rates = res_basename + f"irate_{outputn:04}.pkl"
        outputn += 1
        with open(out,'wb') as f:
            pkl.dump(xh_new_f,f)
        with open(out_rates,'wb') as f:
            pkl.dump(phi_ion_f,f)
    tnow = time.time()
    print(f"\n --- Timestep {t+1:n}, tf = {ct : .2e} yrs. Wall clock time: {tnow - tinit : .3f} seconds --- \n")
    xh_new_f, phi_ion_f, coldens_out_f = pc2r.evolve3D(dt,dr,srcflux,srcpos,max_subbox,subboxsize,temp_f,ndens_f,
                xh_new_f,sig,bh00,albpow,colh0,temph0,abu_c)
# =====================================================================================

# Final output
with open(res_basename + f"xfrac_{tsteps:04n}.pkl",'wb') as f:
            pkl.dump(xh_new_f,f)
with open(res_basename + f"irate_{tsteps:04n}.pkl",'wb') as f:
            pkl.dump(phi_ion_f,f)

# Display Results
if display:
    print("Making Figure...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(12.5,3.7))

    # Left: column density
    ax1.set_title(f"Column Density, N={N}",fontsize=12)
    im1 = ax1.imshow(coldens_out_f[:,:,zslice],origin='lower')
    c1 = plt.colorbar(im1,ax=ax1)

    # Middle: ionization rate
    ax2.set_title(f"Ionization Rate",fontsize=12)
    # For some reason this gets mapped wrong with log, do manually:
    loggamma = np.log(phi_ion_f[:,:,zslice])
    im2 = ax2.imshow(loggamma,origin='lower',cmap='inferno')
    c2 = plt.colorbar(im2,ax=ax2)
    c2.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)

    # Right: ionization fraction
    ax3.set_title(f"Neutral Hydrogen Fraction",fontsize=12)
    im3 = ax3.imshow(1.0 - xh_new_f[:,:,zslice],origin='lower',cmap='jet',norm='log',vmin=1e-3,vmax=1.0) #cmap='YlGnBu_r'
    c3 = plt.colorbar(im3,ax=ax3)

    fig.tight_layout()

    plt.show()
