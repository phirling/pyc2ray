import sys
sys.path.append("../")
from pyc2ray.visualization import zTomography_3panels_xfrac
import tools21cm as t2c
import matplotlib.pyplot as plt
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("files",nargs=2,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="jet")

args = parser.parse_args()

zz = int(args.z)
xHII_c2ray = 1.0 - t2c.XfracFile(args.files[1]).xi
with open(args.files[0],"rb") as f:
    xHII_pyc2ray = 1.0 - pkl.load(f)

resid = (xHII_pyc2ray) / (xHII_c2ray) - 1.0
tomo = zTomography_3panels_xfrac(xHII_c2ray,xHII_pyc2ray,resid,zz,incr=1,xmin=1e-4,cmap=args.cmap)

plt.show()