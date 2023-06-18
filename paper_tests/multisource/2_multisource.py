import sys
sys.path.append("../../")
import pyc2ray as pc2r
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
parser.add_argument("--plot",action='store_true')
args = parser.parse_args()

# Global parameters
num_steps_between_slices = 10
numzred = 2
paramfile = "parameters.yml"
N = 128
avgdens = 1.0e-6
use_octa = args.gpu

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
numsrc = 5
srcpos, srcflux = sim.read_sources("src_mult.txt",numsrc)

# Raytracing Parameters
max_subbox = 1000
subboxsize = 64
r_RT = 150

# Measure time
tinit = time.time()

# Statistics
mean_xfrac = np.empty(num_steps_between_slices)
mean_ionrate = np.empty(num_steps_between_slices)

for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k+1]
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    # Set density field (could be an actual cosmological field here)
    # TODO: this has to set the comoving density which is then scaled to the
    # correct redshift. In the timesteps, the density is then "diluted" gradually
    sim.set_constant_average_density(avgdens,0) 

    print(f"\n=================================")
    print(f"Doing redshift {zi:.3f} to {zf:.3f}")
    print(f"=================================\n")
    # Do num_steps_between_slices timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        print(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n")

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos, r_RT, max_subbox)

if args.plot:
    import matplotlib.pyplot as plt
    plt.imshow(sim.xh[:,:,64],norm='log',cmap='jet')
    #plt.imshow(sim.phi_ion[:,:,64],norm='log',cmap='inferno')
    plt.colorbar()
    plt.show()
else:
    import pickle as pkl
    with open("res.pkl","wb") as f:
        pkl.dump(sim.xh, f)