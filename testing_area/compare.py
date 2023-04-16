import pickle as pkl
import numpy as np
from tomography import zTomography_3panels
import matplotlib.pyplot as plt

f1 = "c2ray_50_sources_r=150.pkl"
f2 = "octa_50_sources_r=150.pkl"

with open(f1,'rb') as f:
    loggamma1 = pkl.load(f)
with open(f2,'rb') as f:
    loggamma2 = pkl.load(f)

gamma1 = np.exp(loggamma1)
gamma2 = np.exp(loggamma2)

resid = gamma1 / gamma2 - 1

tomo = zTomography_3panels(loggamma1, loggamma2, resid, 150)

plt.show()