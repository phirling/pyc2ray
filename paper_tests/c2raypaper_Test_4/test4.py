import sys
sys.path.append("../../")
import pyc2ray as pc2r
import time
import numpy as np
import argparse
import astropy.units as u

parser = argparse.ArgumentParser()
parser.add_argument("paramfile",nargs=1,type=str)
parser.add_argument("-resolution",type=str,default=None)
parser.add_argument("--cpu",action='store_true')
args = parser.parse_args()

if args.resolution is None:
    print("Set time resolution")
    exit

# ======================================================================
# TEST 4: UNIFORM DENSITY IN COSMOLOGICAL CONTEXT
# t_evol = 500 MYR FROM REDSHIFT z=9
# COARSE TIME: dt = t_evol / 10 = 50 Myr
# FINE TIME: dt = t_evol / 100 = 5 Myr
#
# SOURCE STRENGTH: 10^54
# BOX SIZE: 7e25 comoving cm = 22.685 comoving Mpc
# ======================================================================

# Global parameters
if args.resolution == "fine":
    numzred = 101
    delta_time = 5
elif args.resolution == "coarse":
    numzred = 11
    delta_time = 50
else:
    raise ValueError("Unknown resolution")

num_steps_between_slices = 1        # Number of timesteps between redshift slices
paramfile = str(args.paramfile[0])        # Name of the parameter file
N = 128                             # Mesh size
ndens0 = 1.87e-7

if args.cpu:
    use_octa = False
else:
    use_octa = True                    # Determines which raytracing algorithm to use

# Raytracing Parameters
max_subbox = 1000                   # Maximum subbox when using C2Ray raytracing
r_RT = 128                            # When using C2Ray raytracing, sets the subbox size. When using OCTA, sets the octahedron size

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,delta_time*1e6)
print(zred_array)

# Read source
srcpos, srcstrength = sim.read_sources("source.txt",1)

# Measure time
tinit = time.time()

# Loop over redshifts
for k in range(numzred-1):

    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    # Write output
    sim.write_output_numbered(k)

    # Set redshift to current slice redshift
    sim.zred = zi

    sim.set_constant_average_density(ndens0,zi)

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcstrength, srcpos, r_RT, max_subbox)

# Write final output
sim.write_output_numbered(numzred-1)
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)
