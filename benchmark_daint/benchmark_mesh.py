import sys
sys.path.append("../")

import pyc2ray as pc2r
from pyc2ray.raytracing import do_all_sources
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from pyc2ray.visualization import zTomography_rates
import time

# ==========================================================
# Script to measure the performance of pyC2Ray's raytracing
# (without chemistry)
# ==========================================================

# Setup
numsrc = 50
max_subbox = 1000
subboxsize = 50
sourcefile = "100_src_5e49_N300.txt"
mesh_range = [100,150,200,250,300,350,400] #2**np.arange(6,10) #range(1,10,1) #[10]
avgdens = 1e-3
xhav = 1.2e-3
boxsize_kpc = 50

boxsize = boxsize_kpc * u.kpc.to('cm')       
sig = 6.30e-18
timings = []

# Read sources
srcpos, srcflux, numsrc = pc2r.read_sources(sourcefile,numsrc,"pyc2ray")

# Run Tests
for N in mesh_range:
    print(f"Doing N_mesh = {N:n}")

    dxbox = boxsize / N
    dr = dxbox * np.ones(3)

    # Initialize Arrays
    ndens = avgdens * np.ones((N,N,N),order='F')
    xh_av = xhav * np.ones((N,N,N),order='F')

    # Raytrace
    t1 = time.time()
    phi_ion, nsubbox, photonloss = do_all_sources(srcflux,srcpos,max_subbox,subboxsize,sig,dr,ndens,xh_av)
    t2 = time.time()
    timings.append(t2-t1)

# Output: plot and text file
name = f"bench_mesh_nsrc={numsrc:n}_sbox={subboxsize:n}"

# Plot
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(mesh_range,timings,'o-',color='forestgreen')
ax.set_xlabel("$N_{mesh}$")
ax.set_ylabel("Computation Time [s]")
ax.set_title(f"$N_{{src}}={numsrc:n}$, $r_{{subbox}}={subboxsize:n}$")
fig.savefig(name+".eps",bbox_inches='tight')

# Text file
output = np.column_stack((mesh_range,timings))
with open(name + ".txt","a") as f:
    np.savetxt(f,output,("%i %.3f"))


# loggamma = np.where(phi_ion != 0.0,np.log(phi_ion),np.nan)
# tom = zTomography_rates(loggamma,60,1)


# plt.show()