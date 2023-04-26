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
# (without chemistry) when doing a varying number of sources
# on a grid of fixed size. We set the subboxsize parameter
# to a value such that the whole grid is raytraced for each
# source (i.e. the subbox increment/photon loss technique is
# effectively bypassed).
# ==========================================================

# Setup
N = 128
max_subbox = 1000
subboxsize = 64
sourcefile = "100_src_5e49_N300.txt"
numsrc_range = range(1,100,10) #[10]
avgdens = 1e-3
xhav = 1.2e-3
boxsize_kpc = 50


# C2Ray parameters. These can also be imported from
# the yaml file but for now its simpler like this
sig = 6.30e-18
boxsize = boxsize_kpc * u.kpc.to('cm')       
dxbox = boxsize / N
dr = dxbox * np.ones(3)

# Initialize Arrays
ndens = avgdens * np.ones((N,N,N),order='F')
xh_av = xhav * np.ones((N,N,N),order='F')
timings = []

# Run Tests
for numsrc in numsrc_range:

    print(f"Doing N_src = {numsrc:n}")
    # Read sources
    srcpos, srcflux, numsrc = pc2r.read_sources(sourcefile,numsrc,"pyc2ray")

    # Raytrace
    t1 = time.time()
    phi_ion, nsubbox, photonloss = do_all_sources(srcflux,srcpos,max_subbox,subboxsize,sig,dr,ndens,xh_av)
    t2 = time.time()
    timings.append(t2-t1)

# Output: plot and text file
name = f"bench_nsrc_N={N:n}_sbox={subboxsize:n}"

# Plot
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(numsrc_range,timings,'o-',color='orangered')
ax.set_xlabel("$N_{src}$")
ax.set_ylabel("Computation Time [s]")
ax.set_title(f"$N_{{mesh}}={N:n}$, $r_{{subbox}}={subboxsize:n}$")
fig.savefig(name+".eps",bbox_inches='tight')

# Text file
output = np.column_stack((numsrc_range,timings))
with open(name + ".txt","a") as f:
    np.savetxt(f,output,("%i %.3f"))


# loggamma = np.where(phi_ion != 0.0,np.log(phi_ion),np.nan)
# tom = zTomography_rates(loggamma,60,1)


# plt.show()