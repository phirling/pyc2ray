import sys
sys.path.append("../../")
import pyc2ray as pc2r
import time
import argparse

# ======================================================================
# Example 2 for pyc2ray: Cosmological simulation from N-body
# ======================================================================

# Global parameters
num_steps_between_slices = 10       # Number of timesteps between redshift slices
paramfile = "parameters.yml"        # Name of the parameter file
N = 250                             # Mesh size
use_octa = True                # Determines which raytracing algorithm to use

# Raytracing Parameters
max_subbox = 100                   # Maximum subbox when using C2Ray raytracing
r_RT = 5                            # When using C2Ray raytracing, sets the subbox size. When using OCTA, sets the octahedron size

# Create C2Ray object
sim = pc2r.C2Ray(paramfile=paramfile, Nmesh=N, use_octa=use_octa)

# Get redshift list (test case)
zred_array = np.loadtxt(sim.inputs_basename+'redsfhits.txt', dtype=float)
zred_density = np.loadtxt(sim.inputs_basename+'redshift_density.txt', dtype=float)
zred_sources = np.loadtxt(sim.inputs_basename+'redshift_sources.txt', dtype=float)

#srcpos, normflux = sim.read_sources("source.txt",1)

# Measure time
tinit = time.time()

# Loop over redshifts
for k in range(len(zred_array)-1):

    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    # Write output
    sim.write_output(zi)

    # Set density field
    sim.read_density(zi)

    # Read sources
    srcpos, normflux = sim.read_sources(zi)

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, normflux, srcpos, r_RT, max_subbox)

# Write final output
sim.write_output(zf)