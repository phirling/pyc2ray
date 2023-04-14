# gfortran -c photorates.f90
# gfortran -c chemistry.f90
# gfortran -c raytracing.f90
f2py -c raytracing_simplified.f90 -m c2ray --f2cmap f2c.f2py_f2cmap