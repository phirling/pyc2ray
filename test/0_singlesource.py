import numpy as np
np.set_printoptions(precision=20)
import time
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("../")
import pyc2ray as pc2r

# ========================================================================
# Test 0: Expansion around single source
# 
# Source at the center of the grid. Checks the error of the mean
# ionization fraction and ionization rate over 10 timesteps, relative to
# benchmarked values (May 6, 2023).
#
# A full test (with and without octa) should take about 2 minutes to
# complete.
# ========================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--pyc2ray",action='store_true')
parser.add_argument("--octa",action='store_true')
parser.add_argument("--plot",action='store_true')
args = parser.parse_args()

# Test Parameters (hard coded values for stability)
N = 128
tsteps = 10                            # Number of timesteps
dt = 31557600952243.961 # C2Ray value (Hard coded) # Timestep
dr = 3.3753127248391602E+020
avgdens = 1.0e-3                        # Constant Hydrogen number density
xhav = 1.2e-3                           # Initial ionization fraction

# C2Ray parameters. These can also be imported from
# the yaml file but for now its simpler like this
eth0=13.598
bh00=2.59e-13
ev2k=1.0/8.617e-05
temph0=eth0*ev2k
temp0 = 1e4
sig = 6.30e-18
fh0=0.83
xih0=1.0
albpow=-0.7
colh0=1.3e-8*fh0*xih0/(eth0*eth0)
abu_c=7.1e-7

# Source Parameters
numsrc = 1
srcflux = np.empty(numsrc)
srcflux[0] = 5.0e48

# Raytracing Parameters
max_subbox = 1000
subboxsize = 64
r_RT = 100

# For C2Ray
srcpos = np.empty((3,numsrc),dtype='int')
srcpos[:,0] = np.array([64,64,64])

# For OCTA
srcpos_octa = np.ravel(np.array([[63],[63],[63]],dtype='int32')) # C++ version uses flattened arrays

# Initialize Arrays
ndens_f = avgdens * np.ones((N,N,N),order='F')
xh_f = xhav*np.ones((N,N,N),order='F')
temp_f = temp0 * np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')



# Count time
tinit = time.time()

# Target Arrays
# Without Subboxing
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

# With Subboxing
xfrac_target_sub = np.array([0.0031047134536380766, 0.005022279326016418,
    0.00692855383634331, 0.00882576612974846,
    0.010710299732946115,0.012581507961373608,
    0.014437884192349188,  0.016274331746158846,
    0.01810526577059046, 0.01992265707945536])
ionrate_target_sub = np.array([9.4354327177890989e-16,1.4416986086671406e-15,
    1.7148465725719927e-15,1.9186357265605648e-15,
    2.0840267041583267e-15, 2.2245200628847907e-15,
    2.3472744840312425e-15, 2.4564180134200139e-15,
    2.5553718743648408e-15, 2.6453861610921453e-15])

print(f"Running on {numsrc:n} source(s) on {N:n}^3 grid.")
print(f"Constant density: {avgdens:.2e} cm^-3, Temperature: {temp0:.1e} K, initial ionized fraction: {xhav:.2e}")

if not (args.pyc2ray or args.octa):
    print("No flags passed, nothing to be done")

else:
    if args.pyc2ray:

        # Initialize next step
        xh_new_f = xh_f

        mean_xfrac = np.empty(tsteps)
        mean_ionrate = np.empty(tsteps)
        print("\n ==================================== Running pyc2ray... ==================================== \n")

        for t in range(tsteps):
            tnow = time.time()
            print(f"\n --- Timestep {t+1:n}. Wall clock time: {tnow - tinit : .3f} seconds --- \n")
            xh_new_f, phi_ion_f = pc2r.evolve3D(dt,dr,srcflux,srcpos,max_subbox,subboxsize,temp_f,ndens_f,
                        xh_new_f,sig,bh00,albpow,colh0,temph0,abu_c)
            
            mean_xfrac[t] = np.mean(xh_new_f)
            mean_ionrate[t] = np.mean(phi_ion_f)

        err_xfrac = np.abs(mean_xfrac-xfrac_target)/xfrac_target
        err_ionrate = np.abs(mean_ionrate-ionrate_target)/ionrate_target

        print("\n Relative Error to target (pyc2ray):")
        print("-- Ionization Fraction --")
        print(err_xfrac)
        print("-- Ionization Rate --")
        print(err_ionrate)


    if args.octa:
        # Initialize next step
        xh_new_f = xh_f

        pc2r.device_init(N)
        mean_xfrac_octa = np.empty(tsteps)
        mean_ionrate_octa = np.empty(tsteps)
        print("\n ==================================== Running pyc2ray+OCTA... ==================================== \n")

        for t in range(tsteps):
            tnow = time.time()
            print(f"\n --- Timestep {t+1:n}. Wall clock time: {tnow - tinit : .3f} seconds --- \n")
            xh_new_f, phi_ion_f = pc2r.evolve3D_octa(dt,dr,srcflux,srcpos,r_RT,temp_f,ndens_f,
                xh_new_f,sig,bh00,albpow,colh0,temph0,abu_c,N)
            
            mean_xfrac_octa[t] = np.mean(xh_new_f)
            mean_ionrate_octa[t] = np.mean(phi_ion_f)

        err_xfrac_octa = np.abs(mean_xfrac_octa-xfrac_target)/xfrac_target
        err_ionrate_octa = np.abs(mean_ionrate_octa-ionrate_target)/ionrate_target

        print("\n Relative Error to target (pyc2ray+OCTA):")
        print("-- Ionization Fraction --")
        print(err_xfrac_octa)
        print("-- Ionization Rate --")
        print(err_ionrate_octa)