import sys
sys.path.append("../../")
import tools21cm as t2c
import pyc2ray as pc2r
import time
import astropy.constants as ac
import astropy.units as u
import pickle as pkl
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--daint",action='store_true')
args = parser.parse_args()

# Global parameters
numzred = 5                        # Number of redshift slices
num_steps_between_slices = 10        # Number of timesteps between redshift slices
if args.daint: paramfile = "parameters_daint.yml"
else: paramfile = "parameters.yml"
N = 250                             # Mesh size
t_evol = 5e5
use_octa = True                    # Determines which raytracing algorithm to use
r_RT = 50

sim = pc2r.C2Ray_test(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,t_evol/numzred)

numsrc = 100000

with open("cosmo_sources_sorted.pkl","rb") as f:
    sources_list = pkl.load(f)

fgamma = 250
t_s = 3*u.Myr.to('s')
fact = fgamma*sim.cosmology.Ob0/(sim.cosmology.Om0*t_s*ac.m_p.to('Msun').value)

print(fact)
if numsrc is None: srcpos_raw = sources_list[:,:3]
else: srcpos_raw = sources_list[:numsrc,:3]
print(srcpos_raw)
if numsrc is None: normflux_raw = fact*sources_list[:,3]/1e48
else: normflux_raw = fact*sources_list[:numsrc,3]/1e48
normflux = normflux_raw.astype('float64')
srcpos = np.ravel(np.transpose((srcpos_raw - 1).astype('int32')),order='F')


df = t2c.DensityFile("dens_9.938.dat")
z = 9.938
scaling = (1+z)**3
m_H_cgs = 1.673533927065e-24 # Isotopic mass of hydrogen in grams
ndens = dens = scaling * df.cgs_density / m_H_cgs
sim.ndens = ndens

# Measure time
tinit = time.time()

out_i = 0

# RESTART =====
out_i = 40
with open("xfrac_40.000.pkl","rb") as f:
    sim.xh = pkl.load(f)
    
# Loop over redshifts
pc2r.printlog(f"Running on {len(normflux):n} sources...",sim.logfile)
pc2r.printlog(f"Raytracing radius: {r_RT:n} grid cells (= {sim.dr_c*u.cm.to('Mpc'):.3f} comoving Mpc)",sim.logfile)
for k in range(len(zred_array)-1):
    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)

    

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):

        # Write output at each timestep
        sim.write_output(out_i)
        out_i += 1
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)
        pc2r.printlog(f"Mean density is: {sim.ndens.mean():.3e}, mean ionization fraction: {sim.xh.mean():.3e}",sim.logfile)
        sim.evolve3D(dt, normflux, srcpos, r_RT, 1000)

# Write final output
sim.write_output(zf)
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)
