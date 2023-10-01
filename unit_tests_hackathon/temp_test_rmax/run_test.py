import sys
sys.path.append("../../")
import pyc2ray as pc2r
import tools21cm as t2c
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
args = parser.parse_args()

# Global parameters
num_steps_between_slices = 1
numzred = 2
paramfile = "parameters.yml"
N = 128
use_octa = args.gpu

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
numsrc = 1
srcpos, srcflux = sim.read_sources("src.txt",numsrc)

# Raytracing Parameters
kpc = 3.086e21
Mpc = 1000*kpc
R_max_cMpc = 0.005
drc = sim.dr_c
rmax_cells = R_max_cMpc*Mpc / drc
print("rmax internal: ",R_max_cMpc*Mpc)
print("rmax cells: ",rmax_cells)
r_RT = 100

max_subbox = 1000
#r_RT = 150

# Measure time
tinit = time.time()

# Setup density
avgdens = 1e-3
ndens = avgdens*np.ones((N,N,N))
sim.ndens = ndens

for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k+1]
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)
    
    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Do num_steps_between_slices timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos, r_RT, max_subbox)

pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)

# Compare with reference
xfrac_pyc2ray = sim.xh


boxsize = sim.boxsize_c / kpc
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(xfrac_pyc2ray[:,:,63],norm='log',cmap='jet',extent=(-boxsize/2,boxsize/2,-boxsize/2,boxsize/2))
ax.set_xlabel('x [ckpc]')
ax.set_ylabel('y [ckpc]')
plt.show()