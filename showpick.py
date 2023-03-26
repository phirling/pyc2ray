import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import astropy.units as u
from matplotlib.patches import Circle

with open("xh_final.pkl","rb") as f:
    xh = pkl.load(f)

Ng = 5e48
T = 1e4
alph = 2.59e-13 * (T / 1e4)**(-0.7)
ndens = 1e-3
rstromgren = (3*Ng / (4*np.pi*alph * ndens*ndens))**(1./3) * u.cm.to('kpc')
m1 = xh.shape[0]
boxsize = 14.0

rmesh = rstromgren * m1 / boxsize
print(rmesh)

trec = 1. / (alph * ndens) * u.s
print(trec.to('Myr'))

rI = rstromgren * (1-np.exp(-139/trec.to('Myr').value))**(1./3)
rImesh = rI * m1 / boxsize
print(rImesh)
fig, ax = plt.subplots()
ax.set_title(f"Neutral Hydrogen Fraction",fontsize=12)
im = ax.imshow(1.0-xh[:,:,64],origin='lower',norm='log',vmin=1e-3,vmax=1.0,cmap='jet')
c = plt.colorbar(im,ax=ax)
ax.add_patch(Circle([63,63],rmesh,fill=0,ls='--',ec='white'))
ax.add_patch(Circle([63,63],rImesh,fill=0,ls='--',ec='magenta'))
plt.show()