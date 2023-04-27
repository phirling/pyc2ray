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
parser.add_argument("-boxsize",type=float,default=None)
parser.add_argument("-t",type=float,default=None)
parser.add_argument("--zavg",action='store_true')

args = parser.parse_args()

with open(args.file[0],"rb") as f:
    xHII = 1.0 - pkl.load(f)

Nmesh = xHII.shape[0]

if not args.zavg:
    zz = int(args.z)
    tomo = zTomography_xfrac(xHII,zz,incr=1,xmin=1e-4,cmap=args.cmap)
else:
    x_zavg = xHII.mean(axis=2)
    fig, ax = plt.subplots(figsize=(6,6))
    im, cb = xfrac_plot(x_zavg,ax,3e-4,cmap=args.cmap,boxsize=args.boxsize,time=args.t)
    ax.set_title(f"z-Averaged Neutral H Fraction, $N_{{mesh}}={Nmesh:n}$")

plt.show()
