import sys
sys.path.append("../")
from pyc2ray.visualization import zTomography_xfrac, xfrac_plot
import matplotlib.pyplot as plt
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("file",nargs=1,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="jet")
parser.add_argument("-interp",type=str,default=None)
parser.add_argument("-boxsize",type=float,default=None)
parser.add_argument("-t",type=float,default=None)
parser.add_argument("--zavg",action='store_true')
parser.add_argument("--ionized",action='store_true')
parser.add_argument("-o",type=str,default=None)

args = parser.parse_args()

with open(args.file[0],"rb") as f:
    if args.ionized:
        grid = pkl.load(f)
    else:
        grid = 1.0 - pkl.load(f)

Nmesh = grid.shape[0]

if not args.zavg:
    zz = int(args.z)
    tomo = zTomography_xfrac(grid,zz,incr=1,xmin=1e-4,cmap=args.cmap)
else:
    x_zavg = grid.mean(axis=2)
    fig, ax = plt.subplots(figsize=(6,6))
    if args.ionized: im, cb = xfrac_plot(x_zavg,ax,cmap=args.cmap,interp=args.interp,boxsize=args.boxsize,time=args.t)
    else: im, cb = xfrac_plot(x_zavg,ax,3e-4,cmap=args.cmap,interp=args.interp,boxsize=args.boxsize,time=args.t)
    ax.set_title(f"z-Averaged Neutral H Fraction, $N_{{mesh}}={Nmesh:n}$")

if args.o is None:
    plt.show()
else:
    fig.savefig(args.o)
