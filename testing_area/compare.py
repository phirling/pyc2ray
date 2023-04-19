import pickle as pkl
import numpy as np
from tomography import zTomography_3panels_rates
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("files",nargs=2)

args = parser.parse_args()

f1 = args.files[0] #"./multisource_results/c2ray_100_sources_r=150.pkl"
f2 = args.files[1] #"./multisource_results/octa_100_sources_r=150.pkl"

with open(f1,'rb') as f:
    loggamma1 = pkl.load(f)
with open(f2,'rb') as f:
    loggamma2 = pkl.load(f)

gamma1 = np.exp(loggamma1)
gamma2 = np.exp(loggamma2)

resid = gamma1 / gamma2 - 1

tomo = zTomography_3panels_rates(loggamma1, loggamma2, resid, 150)

plt.show()
