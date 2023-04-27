import sys
sys.path.append("../")
from pyc2ray.visualization import zTomography_xfrac, xfrac_plot
import tools21cm as t2c
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file",nargs=1,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="jet")
parser.add_argument("-boxsize",type=float,default=None)
parser.add_argument("--zavg",action='store_true')

args = parser.parse_args()

xHII = 1.0 - t2c.XfracFile(args.file[0]).xi

if not args.zavg:
    zz = int(args.z)
    tomo = zTomography_xfrac(xHII,zz,incr=1,xmin=1e-4,cmap=args.cmap)

else:
    x_zavg = xHII.mean(axis=2)
    fig, ax = plt.subplots(figsize=(6,6))
    im, cb = xfrac_plot(x_zavg,ax,3e-4,cmap=args.cmap,boxsize=args.boxsize)
    ax.set_title("z-Averaged Neutral H Fraction")
plt.show()