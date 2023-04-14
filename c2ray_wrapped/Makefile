FC=gfortran
F2PY=f2py
F2PYFLAGS+=--f2cmap f2c.f2py_f2cmap

all: c2ray_RT

c2ray_RT: photorates.f90 raytracing.f90 chemistry.f90
	$(F2PY) -c photorates.f90 raytracing.f90 chemistry.f90 -m c2ray $(F2PYFLAGS)

clean:
	rm *.so *.o