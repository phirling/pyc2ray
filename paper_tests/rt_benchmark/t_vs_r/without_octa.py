import sys
sys.path.append("../../../")
import pyc2ray as pc2r
import numpy as np
import matplotlib.pyplot as plt
import time

kpc = 3.086e21

# Main parameters
N = 400                         # Mesh size
boxsize = 14 * kpc              # Box size in cgs
dr = boxsize / N                # Cell size in cgs
sig = 6.30e-18                  # HI cross section at its ionzing frequency in cgs

# Example grids
ndens = 1e-3 * np.ones((N,N,N),order='F')       # Number density of Hydrogen in cgs
xh_av = 1.2e-3 * np.ones((N,N,N),order='F')     # Current time-averaged ionized fraction of hydrogen

# Read example sources
src_pos, src_flux = pc2r.read_test_sources("sources_more.txt", 10)

# Set up optical depth table
minlogtau = -20
maxlogtau = 4
NumTau = 2000
dlogtau = (maxlogtau-minlogtau)/NumTau
tau = np.empty(NumTau+1)
tau[0] = 0.0
tau[1:] = 10**(minlogtau + np.arange(NumTau)*dlogtau)

# Set up radiation table
Teff = 5e4                      # Temperature of black body
grey = False                    # Use power-law opacity
freq0 = 3288513124000000.0      # Ionization threshhold frequency of Hydrogen in s^-1
pl_index = 2.8
radsource = pc2r.radiation.BlackBodySource(Teff, grey, freq0, pl_index)
photo_thin_table = radsource.make_photo_table(tau, freq0, 10*freq0, 1e48)

radii = np.linspace(10,200,20)
timings = []
mean_phi = []
max_phi = []

# do empty rt
phi_ion = pc2r.raytracing.do_all_sources_unified(dr,src_flux,src_pos,5,False,6,1e-2,ndens,xh_av,photo_thin_table,minlogtau,dlogtau,sig)

# Raytrace
for r in radii:
    print(f"Doing r = {r:.1f}")
    t1 = time.perf_counter()
    phi_ion = pc2r.raytracing.do_all_sources_unified(dr,src_flux,src_pos,r,False,r+1,1e-2,ndens,xh_av,photo_thin_table,minlogtau,dlogtau,sig)
    t2 = time.perf_counter()
    timings.append(t2-t1)
    mean_phi.append(phi_ion.mean())
    max_phi.append(phi_ion.max())
    print(phi_ion[8,284,79])

timings = np.array(timings)
mean_phi = np.array(mean_phi)
max_phi = np.array(max_phi)

# Plot result
plt.plot(radii,timings)
plt.show()
#plt.imshow(phi_ion[:,:,226],cmap='plasma',norm='log')
#plt.show()

output = np.stack((radii,timings,mean_phi,max_phi),axis=1)
print(output)
np.savetxt("result_no_octa.out",output,("%.2f %.6f %.6e %.6e"))
