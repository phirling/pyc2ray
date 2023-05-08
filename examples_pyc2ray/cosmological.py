import sys
sys.path.append("../")
import pyc2ray as pc2r
import numpy as np
import time

# Global parameters
num_steps_between_slices = 10
numzred = 15
paramfile = "parameters.yml"
N = 128
avgdens = 1.0e-3
use_octa = False

# Create C2Ray object
sim = pc2r.C2Ray(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
numsrc = 1
srcpos, srcflux, numsrc = sim.read_sources("src.txt",numsrc)
# Raytracing Parameters
max_subbox = 1000
r_RT = 5

# Measure time
tinit = time.time()

for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k+1]
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)
    #dt = 31557600952243.961

    # Write output
    sim.write_output(zi)

    # Set density field (could be an actual cosmological field here)
    sim.set_constant_average_density(avgdens)
    if sim.cosmological:
        sim.scale_density(zi)

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

# Write final output
sim.write_output(zf)