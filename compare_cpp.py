import c2ray_core as c2r
import numpy as np

N = 128
numsrc = 1                              # Number of sources
srcpos = np.empty((3,numsrc),dtype='int')
srcflux = np.empty(numsrc)
srcpos[:,0] = np.array([64,64,64])      # Position of source
srcflux[0] = 5.0e48                     # Strength of source

coldensh_out = np.zeros((N,N,N),order='F')
phi_ion = np.zeros((N,N,N),order='F')
ndens = np.ones((N,N,N),order='F')
xh_av = 1.0e-3 * np.ones((N,N,N),order='F')

ns = 1
NumSrc = 1


sig = 1.0
dr = 1.0

# rtpos = np.array([64,64,64])
# c2r.raytracing.evolve0d(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion)
# rtpos = np.array([65,65,65])
# c2r.raytracing.evolve0d(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion)

c2r.raytracing.do_source_octa(srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion)

print(coldensh_out.mean())