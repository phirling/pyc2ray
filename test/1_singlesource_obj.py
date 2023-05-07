import sys
sys.path.append("../")
import pyc2ray as pc2r
import numpy as np
import time

xfrac_target = np.array([0.0031111776832004065, 0.0050288719965858476,
    0.006937708087429684,0.008834954590842289,
    0.010719516713923549,  0.01259090183079535,
    0.014448558298133043,  0.01629287927315295,
    0.018123784540161208,0.01994113328887406])
ionrate_target = np.array([9.4375137308980798e-16,1.4421484110790809e-15,
    1.7156737823407734e-15,1.9192690063171669e-15,
    2.0846347121045658e-15, 2.2251279701987956e-15,
    2.3478723713803599e-15, 2.4571783262395018e-15,
    2.5558612514606512e-15,2.6459043339278365e-15])


tsteps = 10
dt = 31557600952243.961
paramfile = "parameters.yml"
N = 128
avgdens = 1.0e-3
xhav = 1.2e-3
temp0 = 1e4
use_octa = False

# Source Parameters
numsrc = 1
srcflux = np.empty(numsrc)
srcflux[0] = 5.0e48
if use_octa:
    srcpos = np.ravel(np.array([[63],[63],[63]],dtype='int32')) # C++ version uses flattened arrays
else:
    srcpos = np.empty((3,numsrc),dtype='int')
    srcpos[:,0] = np.array([64,64,64])

# Raytracing Parameters
max_subbox = 1000
subboxsize = 64
r_RT = 100

# Create Arrays
ndens_f = avgdens * np.ones((N,N,N),order='F')
xh_f = xhav*np.ones((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')

# Create C2Ray object
sim = pc2r.C2Ray(paramfile, N, use_octa)



tinit = time.time()

# Initialize next step
xh_new_f = xh_f
mean_xfrac = np.empty(tsteps)
mean_ionrate = np.empty(tsteps)

for t in range(tsteps):
    tnow = time.time()
    print(f"\n --- Timestep {t+1:n}. Wall clock time: {tnow - tinit : .3f} seconds --- \n")

    xh_new_f, phi_ion_f = sim.evolve3D(dt, srcflux, srcpos, r_RT, temp_f, ndens_f, xh_new_f, max_subbox)
    
    mean_xfrac[t] = np.mean(xh_new_f)
    mean_ionrate[t] = np.mean(phi_ion_f)

    err_xfrac = np.abs(mean_xfrac-xfrac_target)/xfrac_target
    err_ionrate = np.abs(mean_ionrate-ionrate_target)/ionrate_target

print("\n Relative Error to target:")
print("-- Ionization Fraction --")
print(err_xfrac)
print("-- Ionization Rate --")
print(err_ionrate)