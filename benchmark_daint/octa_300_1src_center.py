import sys
sys.path.append("../")

import numpy as np
import astropy.units as u
import astropy.constants as ac
import time
import matplotlib.pyplot as plt
import pickle as pkl
import pyc2ray as pc2r

# /////////////////////////////////////////////////////////////////////////////////

# Number of cells in each dimension
N = 300

# Display options
display = False                          # Display results at the end of run
zslice = 64                             # z-slice of the box to visualize

# Output settings
res_basename = "/scratch/snx3000/phirling/octa_debug/300_1src_center/"             # Directory to store pickled results
delta_results = 5                      # Number of timesteps between results
logfile = res_basename + "pyC2Ray.log"
quiet = True
with open(logfile,"w") as f: f.write("Log file for pyC2Ray. \n\n")

# Run Parameters (sizes, etc.)
tsim = 20                              # Simulation Time (in Myrs)
t_evol = tsim * u.Myr.to('s')           # Simulation Time (in seconds)
tsteps = 20                            # Number of timesteps
dt = t_evol / tsteps                    # Timestep
boxsize_kpc = 50                    # Simulation box size in kpc
avgdens = 1e-3
xhav = 1.2e-3
numsrc = 1                              # Number of sources
r_RT = 200                               # Raytracing radius

# Conversion
boxsize = boxsize_kpc * u.kpc.to('cm') #100 * u.Mpc.to('cm')           
dxbox = boxsize / N                     # Cell Size (1D)
dr = dxbox * np.ones(3)                 # Cell Size (3D)


# Source Parameters
srcpos = np.ravel(np.array([[149],[149],[149]],dtype='int32')) # C++ version uses flattened arrays
srcflux = np.empty(numsrc)
srcflux[0] = 5.0e49

# Allocate GPU memory
pc2r.device_init(N)

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

# Initialize Arrays (will be flattened internally for OCTA)
ndens_f = avgdens * np.ones((N,N,N),order='F')
xh_f = xhav*np.ones((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')

# Initialize next step
xh_new_f = xh_f

# Count time
tinit = time.time()

pc2r.printlog("\n ============================================================================================== \n",logfile,quiet)

pc2r.printlog(f"Box size is {boxsize_kpc:.2f} kpc, on a grid of size {N:n}^3",logfile,quiet)
pc2r.printlog(f"Running on {numsrc:n} source(s), total ionizing flux: {srcflux.sum():.2e} s^-1",logfile,quiet)
pc2r.printlog(f"Constant density: {avgdens:.2e} cm^-3, Temperature: {temp0:.1e} K, initial ionized fraction: {xhav:.2e}",logfile,quiet)
pc2r.printlog(f"Simulation time is {tsim:.2f} Myr(s), using timestep {tsim/tsteps:.2f} Myr(s).",logfile,quiet)
pc2r.printlog(f"Using OCTA Raytracing with r = {r_RT:n} ( --> octahedron height = {np.ceil(np.sqrt(3)*r_RT):n} ).", logfile,quiet)
pc2r.printlog("Starting main loop...",logfile,quiet)
pc2r.printlog("\n ============================================================================================== \n",logfile,quiet)

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
    #print(f"\n --- Timestep {t+1:n}, tf = {ct : .2e} yrs. Wall clock time: {tnow - tinit : .3f} seconds --- \n")
    pc2r.printlog(f"\n --- Timestep {t+1:n}, tf = {ct : .2e} yrs. Wall clock time: {tnow - tinit : .3f} seconds --- \n",logfile,quiet)
    xh_new_f, phi_ion_f = pc2r.evolve3D_octa(dt,dxbox,srcflux,srcpos,r_RT,temp_f,ndens_f,
                xh_new_f,sig,bh00,albpow,colh0,temph0,abu_c,N,logfile=logfile)
# =====================================================================================

# Deallocate GPU memory
pc2r.device_close()

# Print final wall time
pc2r.printlog("\n ============================================================================================== \n",logfile,quiet)
tnow = time.time()
pc2r.printlog(f"done. Final Time: {tnow - tinit : .3f} seconds",logfile,quiet)

# Final output
with open(res_basename + f"xfrac_{tsteps:04n}.pkl",'wb') as f:
            pkl.dump(xh_new_f,f)
with open(res_basename + f"irate_{tsteps:04n}.pkl",'wb') as f:
            pkl.dump(phi_ion_f,f)