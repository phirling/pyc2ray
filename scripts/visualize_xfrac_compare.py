import sys
sys.path.append("../")
from pyc2ray.visualization import zTomography_3panels_xfrac, zTomography_3panels_rates
import tools21cm as t2c
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("files",nargs=2,help="snapshot file to be imaged")
parser.add_argument("-z",default=None)
parser.add_argument("-cmap",type=str,default="jet")
parser.add_argument("--rates",action='store_true')

args = parser.parse_args()

if not args.rates:
    xHII_c2ray = 1.0 - t2c.XfracFile(args.files[1]).xi
    with open(args.files[0],"rb") as f:
        xHII_pyc2ray = 1.0 - pkl.load(f)
    N = xHII_c2ray.shape[0]
    if args.z is None:
        zz = N // 2
    else:
        zz = int(args.z)

    resid = (xHII_pyc2ray) / (xHII_c2ray) - 1.0

    tomo = zTomography_3panels_xfrac(xHII_c2ray,xHII_pyc2ray,resid,zz,incr=1,xmin=1e-4,cmap=args.cmap)
else:
    gamma_c2ray = t2c.IonRateFile(args.files[1]).irate
    with open(args.files[0],"rb") as f:
        gamma_pyc2ray = pkl.load(f)
    N = gamma_c2ray.shape[0]
    if args.z is None:
        zz = N // 2
    else:
        zz = args.z

    resid = gamma_pyc2ray / gamma_c2ray - 1.0

    loggamma_c2ray = np.where(gamma_c2ray != 0.0,np.log(gamma_c2ray),np.nan)
    loggamma_pyc2ray = np.where(gamma_pyc2ray != 0.0,np.log(gamma_pyc2ray),np.nan)
    tomo = zTomography_3panels_rates(loggamma_c2ray,loggamma_pyc2ray,resid,zz,incr=1)

plt.show()
