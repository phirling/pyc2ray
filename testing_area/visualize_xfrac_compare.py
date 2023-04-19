import tools21cm as t2c
import matplotlib.pyplot as plt
import argparse
import numpy as np
import astropy.units as u
from tomography import zTomography_xfrac_3panels
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("files",nargs=2,help="snapshot file to be imaged")
parser.add_argument("-z",default=101)
parser.add_argument("-cmap",type=str,default="jet")

args = parser.parse_args()

zz = int(args.z)
xHI_c2ray = t2c.XfracFile(args.files[1]).xi
with open(args.files[0],"rb") as f:
    xHI_pyc2ray = pkl.load(f)

resid = (1.0-xHI_pyc2ray) / (1.0 - xHI_c2ray) - 1.0
tomo = zTomography_xfrac_3panels(xHI_c2ray,xHI_pyc2ray,resid,zz,incr=1,xmin=1e-4,cmap=args.cmap)

plt.show()