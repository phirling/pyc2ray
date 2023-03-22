import numpy as np
import matplotlib.pyplot as plt
import time
import c2ray_core as c2r

# Mesh sizes used
N = 100

# Fake quantities for benchmark
ns = 1
sig = 6.30e-18
dr = 6.7e20*np.ones(3)
avgdens = 1.0e-4
xhav = 2.e-4

# Create one test source
numsrc = 2
srcpos = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
srcpos[:,0] = np.array([2,2,2])
py_srcpos = srcpos[:,0] - 1
srcflux[0] = 1.0e55

# Slice to visualize
ii = 2

# Test second source
srcpos[:,1] = np.array([2,30,30])
srcflux[1] = 1.0e55
do_second_src = 0

# Shadow
def add_shadow(shadowpos,ndens,strength,shadowr):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                pos = np.array([i,j,k])
                r = np.sqrt(np.sum((pos-shadowpos)**2))
                if r < shadowr:
                    ndens[i,j,k] += strength
                #if np.abs(i-pos[0]) < dx and np.abs(j-pos[1]) < dy and np.abs(k-pos[2]) < dz:
                #    ndens[i,j,k] += strength

shadowpos = np.array([2,30,30])
shadow = 0
shadowstrength = 10
shadowr = 10

# For fortran version (non periodic)
last_l = np.ones(3)
last_r = (N) * np.ones(3)

# Create Density field
print("Creating Density...")
ndens_1 = avgdens * np.random.uniform(size=(N,N,N)) #np.ones((N,N,N))

# Add shadow
if shadow:
    add_shadow(shadowpos,ndens_1,shadowstrength,shadowr)

ndens_1_f = np.asfortranarray(ndens_1)
xh_av_f = xhav*np.ones((N,N,N),order='F')
phi_ion_f = np.zeros((N,N,N),order='F')

# Create empty coldens out
coldens_out = np.zeros((N,N,N))
coldens_out_f = np.zeros((N,N,N),order='F')

# Run Evolve3D
print("Evolve Fortran...")
t1 = time.perf_counter()
c2r.raytracing.do_source(srcflux,srcpos,1,last_l,last_r,coldens_out_f,sig,dr,ndens_1_f,xh_av_f,phi_ion_f)
if do_second_src:
    print("doing second source...")
    c2r.raytracing.do_source(srcflux,srcpos,2,last_l,last_r,coldens_out_f,sig,dr,ndens_1_f,xh_av_f,phi_ion_f)
t2 = time.perf_counter()

t_f = t2-t1
# Display
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(13,5))
ax1.set_title(f"Column Density (f2py), N={N}",fontsize=15)
im1 = ax1.imshow(coldens_out_f[ii,:,:],origin='lower')
c1 = plt.colorbar(im1,ax=ax1)
ax1.set_xlabel(f"Computation Time: {t_f : .7f} s",fontsize=15)
ax1.scatter(py_srcpos[1],py_srcpos[2],s=80,marker='*',c='red')
if shadow: ax1.scatter(shadowpos[1],shadowpos[2],s=50,marker='o',c='blue')

ax2.set_title(f"Ionization Rate (log scale)",fontsize=15)
im2 = ax2.imshow(phi_ion_f[ii,:,:],origin='lower',norm='log',cmap='inferno')
c2 = plt.colorbar(im2,ax=ax2)
c2.set_label(label=r"$\Gamma$ [s$^{-1}$]",size=15)
ax2.scatter(py_srcpos[1],py_srcpos[2],s=80,marker='*',c='red')
if shadow: ax2.scatter(shadowpos[1],shadowpos[2],s=50,marker='o',c='blue')

fig.suptitle("Single source, cubic mesh, random density field",fontsize=20)

plt.show()