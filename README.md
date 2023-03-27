### Requirements
- Numpy
- Matplotlib
- Pickle
- Astropy
- gfortran compiler

### How to use
Run `compile_extensions.sh` to compile the Fortran subroutines using f2py. Then, to execute a simple test run using a grid of size 128 with a single source at the center, run `c2ray_test.py`. The final state of the box will be displayed and, by default, the ionization fractions of the grid are written to Pickle files every 10 timesteps. 
