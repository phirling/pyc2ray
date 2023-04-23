import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import astropy.units as u
from matplotlib.patches import Circle
import argparse


parser = argparse.ArgumentParser("Visualize the ionization fraction from a pickled file")
parser.add_argument("files",nargs='+',help="Output file to be visualized")

args = parser.parse_args()

fn = args.files[0]
with open(fn,"rb") as f:
    xh = pkl.load(f)



# To compute Strömgren radius
Ng = 5e48
T = 1e4
alph = 2.59e-13 * (T / 1e4)**(-0.7)
ndens = 1e-3
rstromgren = (3*Ng / (4*np.pi*alph * ndens*ndens))**(1./3) * u.cm.to('kpc')
m1 = xh.shape[0]
boxsize = 14.0

rmesh = rstromgren * m1 / boxsize
#print(rmesh)

trec = 1. / (alph * ndens) * u.s
#print(trec.to('Myr'))


rI = rstromgren * (1-np.exp(-200/trec.to('Myr').value))**(1./3)
rImesh = rI * m1 / boxsize
print(rstromgren)
print(trec.to('Myr'))
fig, ax = plt.subplots()
#ax.set_title(f"Neutral Hydrogen Fraction",fontsize=12)
ax.set_title("Python Wrapped C$^2$Ray",fontsize=12)
im = ax.imshow(1.0-xh[:,:,63],origin='lower',norm='log',vmin=1e-3,vmax=1.0,cmap='jet')
ax.set_xlabel('$x$',fontsize=12)
ax.set_ylabel('$y$',fontsize=12)
c = plt.colorbar(im,ax=ax)
c.set_label(label=r"Neutral Hydrogen Fraction",size=12)
ax.add_patch(Circle([63,63],rmesh,fill=0,ls='--',ec='white',label='Strömgren Radius'))
ax.legend(frameon=False,labelcolor='white')
#ax.add_patch(Circle([63,63],rImesh,fill=0,ls='--',ec='magenta'))
plt.show()
#fig.savefig('wrapped.svg',bbox_inches='tight')