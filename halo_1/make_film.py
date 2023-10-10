import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file",nargs=1)
parser.add_argument("-interp",default=None)
parser.add_argument("-o",type=str,default=None)
args = parser.parse_args()

fname = str(args.file[0])
with open(fname,"rb") as f:
    res = pkl.load(f)

slices_irate = res["slices_irate"]
slices_xfrac = res["slices_xfrac"]
ndens = res["ndens"]
h1ndens = (1.0 - slices_xfrac[:]) * ndens
nframes = slices_irate.shape[0]

# Create Plot
fig, ax = plt.subplots(2,2,squeeze=False,figsize=(10,10))
im_phi = ax[0,0].imshow(np.log10(slices_irate[0].T),cmap='inferno',origin='lower',interpolation=args.interp)
plt.colorbar(im_phi,label="$\log\Gamma$")
ax[0,0].set_title("Ionization Rate [1/s]")

im_xfrac = ax[0,1].imshow(slices_xfrac[0].T,norm='log',cmap='jet',origin='lower',vmin=2e-4,vmax=1,interpolation=args.interp)
plt.colorbar(im_xfrac)
ax[0,1].set_title("Ionized H fraction")


vmin = np.min(h1ndens)
vmax = np.max(h1ndens)


im_h1ndens0 = ax[1,0].imshow(h1ndens[0].T,norm='log',cmap='RdYlBu',origin='lower',vmin=vmin,vmax=vmax,interpolation=args.interp)
plt.colorbar(im_h1ndens0)
ax[1,0].set_title("Density of Neutral H [atom/cm$^3$] (initial)")

im_h1ndens = ax[1,1].imshow(h1ndens[0].T,norm='log',cmap='RdYlBu',origin='lower',vmin=vmin,vmax=vmax,interpolation=args.interp)
plt.colorbar(im_h1ndens)
ax[1,1].set_title("Density of Neutral H [atom/cm$^3$] (final)")

def update(i):
    im_phi.set_data(np.log10(slices_irate[i]).T)
    im_xfrac.set_data(slices_xfrac[i].T)
    im_h1ndens.set_data(h1ndens[i].T)

ani = FuncAnimation(fig,update,nframes)

plt.show()

if args.o is not None:
    fname = str(args.o)
    ani.save(fname,writer='ffmpeg',fps=50,bitrate=-1,dpi=300)