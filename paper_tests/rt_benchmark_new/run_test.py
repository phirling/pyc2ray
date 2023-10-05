import sys
sys.path.append("../../")
import pyc2ray as pc2r
from pyc2ray.utils.sourceutils import format_sources
import numpy as np
import astropy.units as u
import astropy.constants as ac
import time
import argparse
import tools21cm as t2c
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
parser.add_argument("-numsrc",default=None,type=int,help="Number of sources to read from the test file")
parser.add_argument("-R",default=10,type=int)
parser.add_argument("-numreps",default=3,type=int)
parser.add_argument("-o",default="benchmark_result.pkl",type=str)
args = parser.parse_args()

# Global parameters
paramfile = "parameters.yml"
N = 250                             # Mesh size
use_gpu = args.gpu
fgamma = 0.02
outfn = str(args.o)

sim = pc2r.C2Ray_Test(paramfile, N, use_gpu)

# Set up density
df = t2c.DensityFile("../../unit_tests_hackathon/3_multiple_sources_quick/dens_9.938.dat")
z = 9.938
scaling = (1+z)**3
m_H_cgs = 1.673533927065e-24 # Isotopic mass of hydrogen in grams
ndens = dens = scaling * df.cgs_density / m_H_cgs
sim.ndens = ndens

max_subbox = 1000
r_RT = args.R
xh_av = np.copy(sim.xh)

print(f"Rmax = {r_RT:n} cells \n\n")

if args.numsrc is not None:
    nsrc_range = np.array([int(args.numsrc)])
else:
    nsrc_range = np.array([1,10,100,1000,10000,100000]) #100,1000,10000,100000]

timings = np.empty(len(nsrc_range))

for k,nsrc in enumerate(nsrc_range):
    print(f"Doing benchmark for {nsrc:n} sources...")

    # Read sources and convert to flux
    with open("../../unit_tests_hackathon/3_multiple_sources_quick/cosmo_sources_sorted.pkl","rb") as f:
        sources_list = pkl.load(f)
    t_s = 3*u.Myr.to('s')
    fact = fgamma*sim.cosmology.Ob0/(sim.cosmology.Om0*t_s*ac.m_p.to('Msun').value)
    srcpos = sources_list[:nsrc,:3].T
    normflux = fact*sources_list[:nsrc,3]/1e48

    if use_gpu:
        srcpos_flat, normflux_flat = format_sources(srcpos, normflux)
        # Copy positions & fluxes of sources to the GPU in advance
        pc2r.evolve.libasora.source_data_to_device(srcpos_flat,normflux_flat,nsrc)
        coldensh_out_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        phi_ion_flat = np.ravel(np.zeros((N,N,N),dtype='float64'))
        ndens_flat = np.ravel(ndens).astype('float64',copy=True)
        pc2r.evolve.libasora.density_to_device(ndens_flat,N)
        xh_av_flat = np.ravel(xh_av).astype('float64',copy=True)
    else:
        phi_ion = np.zeros((N,N,N),order='F')
        coldensh_out = np.zeros((N,N,N),order='F')

    t_ave = 0
    nreps = int(args.numreps)

    for i in range(nreps):
        t1 = time.time()
        if use_gpu:
            pc2r.evolve.libasora.do_all_sources(r_RT,coldensh_out_flat,sim.sig,sim.dr,ndens_flat,xh_av_flat,phi_ion_flat,nsrc,N,sim.minlogtau,sim.dlogtau,sim.NumTau)
            pass
        else:
            pc2r.evolve.libc2ray.raytracing.do_all_sources(normflux,srcpos,max_subbox,r_RT,coldensh_out,sim.sig,sim.dr,sim.ndens,xh_av,phi_ion,sim.loss_fraction,sim.photo_thin_table,sim.minlogtau,sim.dlogtau,r_RT)
        t2 = time.time()
        t_ave += t2-t1
    t_ave /= nreps
    print(f"Raytracing took {t_ave:.5f} seconds (averaged over {nreps:n} runs).")
    timings[k] = t_ave

if use_gpu:
    asora = "yes"
else:
    asora = "no"

result = {
    "Rmax" : r_RT,
    "nreps" : nreps,
    "ASORA" : asora,
    "numsrc" : nsrc_range,
    "timings" : timings
}

print("Saving Result in " + outfn + "...")
print(result)
with open(outfn,"wb") as f:
    pkl.dump(result,f)