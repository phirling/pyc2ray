import sys
sys.path.append("../")
import pyc2ray as pc2r
import numpy as np
import time

# Error checking arrays (referene results)
xfrac_target = np.array([0.0031111776832004065, 0.0050288719965858476,
    0.006937708087429684,0.008834954590842289,
    0.010719516713923549,  0.01259090183079535,
    0.014448558298133043,  0.01629287927315295,
    0.018123784540161208,0.01994113328887406])
ionrate_target = np.array([9.4375137308980798e-16,1.4421484110790809e-15,
    1.7156737823407734e-15,1.9192690063171669e-15,
    2.0846347121045658e-15, 2.2251279701987956e-15,
    2.3478723713803599e-15, 2.4571783262395018e-15,
    2.5558612514606512e-15,2.6459043339278365e-15])

# Global parameters
num_steps_between_slices = 10
numzred = 2
paramfile = "parameters.yml"
N = 128
avgdens = 1.0e-6
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
subboxsize = 64
r_RT = 100

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
        
        # Measure error
        mean_xfrac[t] = np.mean(sim.xh)
        mean_ionrate[t] = np.mean(sim.phi_ion)
        err_xfrac = np.abs(mean_xfrac-xfrac_target)/xfrac_target
        err_ionrate = np.abs(mean_ionrate-ionrate_target)/ionrate_target

    print("\n Relative Error to target:")
    print("-- Ionization Fraction --")
    print(err_xfrac)
    print("-- Ionization Rate --")
    print(err_ionrate)