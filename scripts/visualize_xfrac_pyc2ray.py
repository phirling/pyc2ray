import sys
sys.path.append("../")
from pyc2ray.visualization import zTomography_xfrac
import matplotlib.pyplot as plt
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("file",nargs=1,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="jet")

args = parser.parse_args()

zz = int(args.z)
with open(args.file[0],"rb") as f:
    xHII = 1.0 - pkl.load(f)

tomo = zTomography_xfrac(xHII,zz,incr=1,xmin=1e-4,cmap=args.cmap)

plt.show()
