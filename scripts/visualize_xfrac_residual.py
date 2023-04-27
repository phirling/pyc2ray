import sys
sys.path.append("../")
from pyc2ray.visualization import resid_plot
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import tools21cm as t2c

parser = argparse.ArgumentParser()
parser.add_argument("files",nargs=2,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="bwr")
parser.add_argument("-interp",type=str,default=None)
parser.add_argument("-boxsize",type=float,default=None)
parser.add_argument("-t",type=float,default=None)
parser.add_argument("-o",type=str,default=None)
parser.add_argument("-clim",type=float,default=None)

args = parser.parse_args()

xHII_c2ray = 1.0 - t2c.XfracFile(args.files[0]).xi

with open(args.files[1],"rb") as f:
    xHII = 1.0 - pkl.load(f)

Nmesh = xHII.shape[0]

x_zavg = xHII.mean(axis=2)
x_zavg_c2ray = xHII_c2ray.mean(axis=2)
resid = x_zavg / x_zavg_c2ray - 1

fig, ax = plt.subplots(figsize=(6,6))

im, cb = resid_plot(resid,ax,cmap=args.cmap,clim=args.clim,boxsize=args.boxsize,time=args.t)

ax.set_title(f"Residual")

if args.o is None:
    plt.show()
else:
    fig.savefig(args.o)
