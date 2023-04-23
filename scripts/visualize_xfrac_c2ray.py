import sys
sys.path.append("../")
from pyc2ray.visualization import zTomography_xfrac
import tools21cm as t2c
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file",nargs=1,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="jet")

args = parser.parse_args()

zz = int(args.z)
xHII = 1.0 - t2c.XfracFile(args.file[0]).xi

tomo = zTomography_xfrac(xHII,zz,incr=1,xmin=1e-4,cmap=args.cmap)

plt.show()