import sys
sys.path.append("../../")
import pyc2ray as pc2r
import time

# ======================================================================
# t_evol = 500 Myr
# COARSE TIME: dt = t_evol / 10 = 50 Myr
# 1 output per timestep to have 10 points on the plot
# ======================================================================

# Global parameters
numzred = 10                        # Number of redshift slices
num_steps_between_slices = 1        # Number of timesteps between redshift slices
t_evol = 5e8 # years
paramfile = "parameters.yml"        # Name of the parameter file
N = 256                             # Mesh size
use_octa = False                    # Determines which raytracing algorithm to use

# Raytracing Parameters
max_subbox = 1000                   # Maximum subbox when using C2Ray raytracing
r_RT = 128                            # When using C2Ray raytracing, sets the subbox size. When using OCTA, sets the octahedron size

# Create C2Ray object
sim = pc2r.C2Ray(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,t_evol/numzred)

# Read sources
srcpos, srcstrength = sim.read_sources("source.txt",1)

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

    # Set density field (could be an actual cosmological field here)
    # Eventually, we want a general method that sets the density, scales it
    # sets the redshift etc (like density_ini in C2Ray). This method should
    # also be able to read in actual density fields. For now, do all this
    # manually.
    sim.density_init(zi) # when cosmological is false, zi has no effect

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcstrength, srcpos, r_RT, max_subbox)

# Write final output
sim.write_output(zf)
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)