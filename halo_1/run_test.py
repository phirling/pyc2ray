import sys
sys.path.append("../")
import pyc2ray as pc2r
import numpy as np
import time
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
parser.add_argument("-o",type=str,default=None)
args = parser.parse_args()

# Global parameters
paramfile = "parameters.yml"
N = 64
use_octa = args.gpu

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Read sources
numsrc = 1
srcpos, srcflux = sim.read_sources("src.txt",numsrc)

# Measure time
tinit = time.time()

# Setup density
with open("./ndens_nfw.pkl","rb") as f:
    sim.ndens = pkl.load(f)

# Setup Temperature
with open("./temp_nfw.pkl","rb") as f:
    sim.temp = pkl.load(f)

# Timestep
unit_time = 3.085678E16
tdyn = 0.4 * unit_time
dt = 1e-6*tdyn
print(f"dt          = {dt:.4e}")
print(f"dt (Abel)   = {sim.dr/pc2r.radiation.c:.4e}")
num_timesteps = 5


h1ndens0 = (1.0 - sim.xh) * sim.ndens
k_src = N//2-1

slices_irate = np.empty((num_timesteps,N,N))
slices_xfrac = np.empty((num_timesteps,N,N))

for k in range(num_timesteps):
    tnow = time.time()
    pc2r.printlog(f"\n --- Timestep {k:n}, Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

    # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
    sim.evolve3D(dt, srcflux, srcpos)
    print(f"dt (Bolton) = {np.min(1.0 / sim.phi_ion):.4e}")
    slices_xfrac[k] = sim.xh[:,:,k_src]
    slices_irate[k] = sim.phi_ion[:,:,k_src]
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)

if args.o is not None:
    res = {
        "slices_xfrac" : slices_xfrac,
        "slices_irate" : slices_irate,
        "ndens" : sim.ndens[:,:,k_src]
    }
    with open(str(args.o),"wb") as f:
        pkl.dump(res,f)

fig, ax = plt.subplots(2,2,squeeze=False,figsize=(10,10))
im_phi = ax[0,0].imshow(np.log(sim.phi_ion[:,:,k_src].T),cmap='inferno',origin='lower')
plt.colorbar(im_phi,label="$\log\Gamma$")
ax[0,0].set_title("Ionization Rate [1/s]")

im_xfrac = ax[0,1].imshow(sim.xh[:,:,k_src].T,norm='log',cmap='jet',origin='lower',vmin=2e-4,vmax=1)
plt.colorbar(im_xfrac)
ax[0,1].set_title("Ionized H fraction")

vmin = min(h1ndens0[:,:,k_src].min(), (sim.ndens[:,:,k_src].T * (1.0-sim.xh[:,:,k_src])).min())
vmax = max(h1ndens0[:,:,k_src].max(), (sim.ndens[:,:,k_src].T * (1.0-sim.xh[:,:,k_src])).max())


im_h1ndens0 = ax[1,0].imshow(h1ndens0[:,:,k_src].T,norm='log',cmap='RdYlBu',origin='lower',vmin=vmin,vmax=vmax)
plt.colorbar(im_h1ndens0)
ax[1,0].set_title("Density of Neutral H [atom/cm$^3$] (initial)")

im_h1ndens = ax[1,1].imshow(sim.ndens[:,:,k_src].T * (1.0-sim.xh[:,:,k_src]).T,norm='log',cmap='RdYlBu',origin='lower',vmin=vmin,vmax=vmax)
plt.colorbar(im_h1ndens)
ax[1,1].set_title("Density of Neutral H [atom/cm$^3$] (final)")

plt.show()

#   plt.imshow(np.log(sim.phi_ion[:,:,z_src].T),cmap='inferno',origin='lower')
#   plt.colorbar()
#   plt.show()
#   #plt.imshow(sim.ndens[:,:,z_src].T * (1.0-xfrac_pyc2ray[:,:,z_src]).T,norm='log',cmap='RdYlBu',origin='lower')
#   plt.imshow(xfrac_pyc2ray[:,:,z_src].T,norm='log',cmap='jet',origin='lower')
#   plt.title("Neutral Hydrogen Density [atom/cm$^3$]")
#   #plt.imshow(sim.ndens[:,:,z_src].T * (1.0-xfrac_pyc2ray[:,:,z_src]).T,norm='log',cmap='jet',origin='lower')
#   plt.colorbar()
#   plt.show()