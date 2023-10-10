import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm
import argparse
import pickle as pkl

parser = argparse.ArgumentParser("Generate density & temperature array for NFW halo centered in cartesian grid")
parser.add_argument("-N",type=int,default=64)
parser.add_argument("--plot",action='store_true')
parser.add_argument("--temperature",action='store_true')
args = parser.parse_args()

N = int(args.N)

# Internal Unit System
UnitLength_in_cm         = 3.085678e21    # kpc
UnitMass_in_g            = 1.989e43       # 1e10 solar mass
UnitVelocity_in_cm_per_s = 1e5            # km/sec 
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs        = UnitMass_in_g * UnitVelocity_in_cm_per_s**2
UnitDensity_in_cgs       = UnitMass_in_g / UnitLength_in_cm**3

# Constants
gamma = 5/3.                    # Adiabatic index
mu    = 0.58822635              # Mean gas weight (assuming full ionization and H (76%) He (24%) mixture)
mh_cgs    = 1.6726e-24          # Hydrogen mass (CGS)
kb_cgs    = 1.3806e-16          # Boltzmann constant (CGS)
G_cgs     = 6.672e-8            # Gravitational constant (CGS)
HUBBLE= 3.2407789e-18           # Hubble constant in h/sec (= 100km/s/Mpc)
HubbleParam = 0.72              # Reduced hubble parameter (small h)
m_p = 1.672661e-24
abu_h =  0.926
abu_he = 0.074
mean_molecular = abu_h + 4.0*abu_he

# Convert constants to internal units
kb = kb_cgs / UnitEnergy_in_cgs
mh = mh_cgs / UnitMass_in_g
G  = G_cgs  / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2)

# Critical density of the universe
rho_c  = pow(HubbleParam*HUBBLE*UnitTime_in_s,2)*3/(8*np.pi*G)

# Parameters of the halo
fb = 0.15                       # Baryonic fraction
M200 = 1e-2                      # Virial mass in internal units (10^10 Msol) 10^8 9.5
c =  5 #17                          # NFW concentration
eps = 1                      # Gravitational softening

# Derived NFW parameters
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * rho_c
r_s = 1/c * (M200/(4/3.*np.pi*rho_c*200))**(1/3.)
r200 = c*r_s

# Numerical parameters
rmax = 1 *r200                  # Integration limit
L = 10
dr = L / N

print("r_s [kpc]  = ",r_s)
print("r200 [kpc] = ",r200)
print("dr [kpc]   = ",dr)

'''
Recall:
Virial radius r200 == radius s.t. mean dens inside r200 is 200*rho_c
Virial mass M200 == total mass inside r200 == 4/3πr200^3 * 200 * rho_c
'''

# Mass Density
def Density(r):
    return fb*rho_0/(((r + eps)/r_s)*(1+(r + eps)/r_s)**2)

def integrand(r):
    return Density(r) * G * Mr(r) /  r**2
# Total mass inside the radius r
def Mr(r):
    return 4*np.pi*rho_0*r_s**3 * ( np.log(1+r/r_s) - (r/r_s)/(1+r/r_s) )
# Pressure
def P(r,rmax):
    Pr = quad(integrand, r, rmax, args=())
    return Pr[0]

# Specific energy
def U(P,r):
    u = P/(gamma-1)/Density(r)
    return u

# Temperature
def T(P,r):
    u = U(P,r)
    T = (gamma-1)*mu*mh/kb *u
    return T

if __name__ == "__main__":
    xi = np.arange(0,N)
    halopos = np.array([N//2-1,N//2-1,N//2-1])
    X,Y,Z = np.meshgrid(xi,xi,xi)
    R = dr * np.sqrt((X - halopos[0])**2 + (Y - halopos[1])**2 + (Z - halopos[2])**2)

    print("Generating Density...")
    ndens = Density(R) * UnitDensity_in_cgs / (mean_molecular * m_p)

    if args.temperature:
        print("Generating Pressure...")
        Pressure = np.empty((N,N,N))
        for i in tqdm(range(N)):
            for j in range(N):
                for k in range(N):
                    Pressure[i,j,k] = P(R[i,j,k],rmax)

        print("Generating Temperature...")
        Temp = T(Pressure,R)

    print("Writing files...")
    with open("ndens_nfw.pkl","wb") as f:
        pkl.dump(ndens,f)
    if args.temperature:
        with open("temp_nfw.pkl","wb") as f:
            pkl.dump(Temp,f)

    if args.plot:
        fig1,ax1 = plt.subplots()
        im1 = ax1.imshow(ndens[:,:,N//2-1],norm='log')
        plt.colorbar(im1)
        ax1.set_title("Density [atom/cm$^3$]")

        fig2,ax2 = plt.subplots()
        im2 = ax2.imshow(Temp[:,:,N//2-1],cmap='hot')
        plt.colorbar(im2)
        ax2.set_title("Temperature [K]")

        plt.show()