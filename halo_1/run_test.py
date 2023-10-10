import sys
sys.path.append("../")
import pyc2ray as pc2r
import tools21cm as t2c
import numpy as np
import time
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
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
# with open("./temp_nfw_128.pkl","rb") as f:
#     sim.temp = pkl.load(f) / 10000000

# Timestep
unit_time = 3.085678E16
tdyn = 0.4 * unit_time
dt = 1e-2*tdyn
print(f"dt          = {dt:.4e}")
print(f"dt (Abel)   = {sim.dr/pc2r.radiation.c:.4e}")
num_timesteps = 10


h1ndens0 = (1.0 - sim.xh) * sim.ndens

for k in range(num_timesteps):
    tnow = time.time()
    pc2r.printlog(f"\n --- Timestep {k:n}, Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

    # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
    sim.evolve3D(dt, srcflux, srcpos)
    print(f"dt (Bolton) = {np.min(1.0 / sim.phi_ion):.4e}")

    fig, ax = plt.subplots(2,2,squeeze=False,figsize=(10,10))
    im_phi = ax[0,0].imshow(np.log(sim.phi_ion[:,:,N//2-1].T),cmap='inferno',origin='lower')
    plt.colorbar(im_phi,label="$\log\Gamma$")
    ax[0,0].set_title("Ionization Rate [1/s]")

    im_xfrac = ax[0,1].imshow(sim.xh[:,:,N//2-1].T,norm='log',cmap='jet',origin='lower')
    plt.colorbar(im_xfrac)
    ax[0,1].set_title("Ionized H fraction")

    vmin = min(h1ndens0[:,:,N//2-1].min(), (sim.ndens[:,:,N//2-1].T * (1.0-sim.xh[:,:,N//2-1])).min())
    vmax = max(h1ndens0[:,:,N//2-1].max(), (sim.ndens[:,:,N//2-1].T * (1.0-sim.xh[:,:,N//2-1])).max())


    im_h1ndens0 = ax[1,0].imshow(h1ndens0[:,:,N//2-1].T,norm='log',cmap='RdYlBu',origin='lower',vmin=vmin,vmax=vmax)
    plt.colorbar(im_h1ndens0)
    ax[1,0].set_title("Density of Neutral H [atom/cm$^3$] (initial)")

    im_h1ndens = ax[1,1].imshow(sim.ndens[:,:,N//2-1].T * (1.0-sim.xh[:,:,N//2-1]).T,norm='log',cmap='RdYlBu',origin='lower',vmin=vmin,vmax=vmax)
    plt.colorbar(im_h1ndens)
    ax[1,1].set_title("Density of Neutral H [atom/cm$^3$] (final)")

    plt.show()
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)


#   plt.imshow(np.log(sim.phi_ion[:,:,N//2-1].T),cmap='inferno',origin='lower')
#   plt.colorbar()
#   plt.show()
#   #plt.imshow(sim.ndens[:,:,N//2-1].T * (1.0-xfrac_pyc2ray[:,:,N//2-1]).T,norm='log',cmap='RdYlBu',origin='lower')
#   plt.imshow(xfrac_pyc2ray[:,:,N//2-1].T,norm='log',cmap='jet',origin='lower')
#   plt.title("Neutral Hydrogen Density [atom/cm$^3$]")
#   #plt.imshow(sim.ndens[:,:,N//2-1].T * (1.0-xfrac_pyc2ray[:,:,N//2-1]).T,norm='log',cmap='jet',origin='lower')
#   plt.colorbar()
#   plt.show()