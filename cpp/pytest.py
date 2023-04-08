import numpy as np
import RTC
import matplotlib.pyplot as plt
import time

N = 128
cdh = np.ravel(np.zeros((N,N,N),dtype='float64'))
sig = 1.0
dr = 1.0
srcpos = np.ravel(np.array([[64],[64],[64]],dtype='int32'))
ndens = np.ravel(np.ones((N,N,N),dtype='float64') )
phi_ion = np.ravel(np.zeros((N,N,N),dtype='float64') )
xh_av = 1e-3 * np.ravel(np.ones((N,N,N),dtype='float64') )
NumSrc = 1

t1 = time.time()
RTC.octa(srcpos,0,cdh,1.0,1.0,ndens,xh_av,phi_ion,1,N)
t2 = time.time()
cdh[:] = 0.0
t3 = time.time()
RTC.octa_gpu(srcpos,0,cdh,1.0,1.0,ndens,xh_av,phi_ion,1,N)
t4 =time.time()

print(f"Time (CPU): {t2-t1 : .3f} [s]")
print(f"Time (GPU): {t4-t3 : .3f} [s]")

cdhim = cdh.reshape((N,N,N),order='C')
plt.imshow(cdhim[:,:,90])
plt.show()