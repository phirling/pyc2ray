import sys
sys.path.append("../")
import pyc2ray as pc2r
import time

# ======================================================================
# Example 1 for pyc2ray: Single test source in homogeneous density field
# ----------------------------------------------------------------------

# USAGE:
# By default, this script will produce 15 outputs at redshifts separated
# by 10 Myr in cosmic time, starting at z = 9. Between two slices, 10
# timesteps are performed. These are the default settings in C2Ray.
# 
# The parameters of the simulation are set in parameters.yml. In
# particular, cosmology can be enabled and disabled using the
# "cosmological" flag in that file.
# 
# By default, the C2Ray raytracing is used. To use the gpu-accelerated
# version, set use_octa = True. A CUDA-compatible GPU must be present
# and the octa library must be compiled & located in the lib/ directory.
# ======================================================================

# Global parameters
numzred = 15                        # Number of redshift slices
num_steps_between_slices = 10       # Number of timesteps between redshift slices
paramfile = "parameters.yml"        # Name of the parameter file
N = 128                             # Mesh size
use_octa = False                    # Determines which raytracing algorithm to use

# Raytracing Parameters
max_subbox = 1000                   # Maximum subbox when using C2Ray raytracing
r_RT = 5                            # When using C2Ray raytracing, sets the subbox size. When using OCTA, sets the octahedron size

# Create C2Ray object
sim = pc2r.C2Ray(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
srcpos, srcflux, numsrc = sim.read_sources("src.txt",1)

# Measure time
tinit = time.time()

# Loop over redshifts
for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)
    #dt = 31557600952243.961

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

    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos, r_RT, max_subbox)

# Write final output
sim.write_output(zf)