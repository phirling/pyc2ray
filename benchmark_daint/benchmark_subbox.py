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
subboxsize_range = np.arange(10,150,10)
sourcefile = "100_src_5e49_N300.txt"
N = 300
avgdens = 1e-3
xhav = 1.2e-3
boxsize_kpc = 50

boxsize = boxsize_kpc * u.kpc.to('cm')       
sig = 6.30e-18
timings = []

# Read sources
srcpos, srcflux, numsrc = pc2r.read_sources(sourcefile,numsrc,"pyc2ray")

# Initialize Arrays
ndens = avgdens * np.ones((N,N,N),order='F')
xh_av = xhav * np.ones((N,N,N),order='F')
dxbox = boxsize / N
dr = dxbox * np.ones(3)

# Run Tests
for subboxsize in subboxsize_range:
    print(f"Doing subboxsize = {subboxsize:n}")

    # Raytrace
    t1 = time.time()
    phi_ion, nsubbox, photonloss = do_all_sources(srcflux,srcpos,max_subbox,subboxsize,sig,dr,ndens,xh_av)
    t2 = time.time()
    timings.append(t2-t1)

# Output: plot and text file
name = f"bench_subbox_N={N:n}_nsrc={numsrc:n}"

# Plot
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(subboxsize_range,timings,'o-',color='steelblue')
ax.set_xlabel("$r_{subbox}$")
ax.set_ylabel("Computation Time [s]")
ax.set_title(f"$N={N:n}$, $N_{{src}}={numsrc:n}$")
fig.savefig(name+".eps",bbox_inches='tight')

# Text file
output = np.column_stack((subboxsize_range,timings))
with open(name + ".txt","a") as f:
    np.savetxt(f,output,("%i %.3f"))


# loggamma = np.where(phi_ion != 0.0,np.log(phi_ion),np.nan)
# tom = zTomography_rates(loggamma,60,1)


# plt.show()