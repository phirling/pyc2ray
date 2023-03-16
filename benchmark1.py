import numpy as np
import matplotlib.pyplot as plt
import time
import RTSC as rtsc
import pyRTSC as pyrtsc

# Mesh sizes used
Ns = [50]

# Fake quantities for benchmark
dt = 1.0
ns = 1
niter = 1
sig = 1.0
dr = np.ones(3)

# Create one test source
srcpos = np.empty((3,1),dtype='int')
srcpos[:,0] = np.array([2,2,2])
py_srcpos = srcpos[:,0] - 1

# Slice to visualize
ii = 2

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
shadow = False
shadowstrength = 10
shadowr = 10

for N in Ns:
    # For fortran version (non periodic)
    last_l = np.ones(3)
    last_r = (N) * np.ones(3)

    # Create Density field
    print("Creating Density...")
    ndens_1 = np.ones((N,N,N))

    # Add shadow
    if shadow:
        add_shadow(shadowpos,ndens_1,shadowstrength,shadowr)
    
    ndens_1_f = np.asfortranarray(ndens_1)

    # Create empty coldens out
    coldens_out = np.zeros((N,N,N))
    coldens_out_f = np.zeros((N,N,N),order='F')

    # Run Evolve3D
    print("Evolve Python...")
    t1 = time.perf_counter()
    pyrtsc.evolve3D(py_srcpos,coldens_out,sig,dr,ndens_1)
    t2 = time.perf_counter()
    print("Evolve Fortran...")
    t3 = time.perf_counter()
    rtsc.raytracing_sc.evolve3d(dt,srcpos,ns,niter,last_l,last_r,coldens_out_f,sig,dr,ndens_1_f)
    t4 = time.perf_counter()

    t_py = t2-t1
    t_f = t4-t3
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(13,5))
    ax1.set_title(f"Python, N={N}",fontsize=15)
    im1 = ax1.imshow(coldens_out[ii,:,:],origin='lower')
    c1 = plt.colorbar(im1,ax=ax1)
    ax1.set_xlabel(f"Time: {t_py : .7f} s",fontsize=15)
    ax1.scatter(py_srcpos[1],py_srcpos[2],s=80,marker='*',c='red')
    if shadow: ax1.scatter(shadowpos[1],shadowpos[2],s=50,marker='o',c='blue')

    ax2.set_title(f"Fortran (F2PY), N={N}",fontsize=15)
    im2 = ax2.imshow(coldens_out_f[ii,:,:],origin='lower')
    c2 = plt.colorbar(im2,ax=ax2)
    ax2.set_xlabel(f"Time: {t_f : .7f} s",fontsize=15)
    ax2.scatter(py_srcpos[1],py_srcpos[2],s=80,marker='*',c='red')
    if shadow: ax2.scatter(shadowpos[1],shadowpos[2],s=50,marker='o',c='blue')

    fig.suptitle("HI Column Density, single source, cubic mesh, spherical overdensity",fontsize=20)
    plt.show()