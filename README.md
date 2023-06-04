# pyc2ray: A Python-wrapped and GPU-accelerated Update of C2Ray
`pyc2ray` is a Python package that wraps [C2Ray](https://github.com/garrelt/C2-Ray3Dm/tree/factorization) [(G. Mellema, I.T. Illiev, A. Alvarez and P.R. Shapiro)](https://ui.adsabs.harvard.edu/link_gateway/2006NewA...11..374M/doi:10.48550/arXiv.astro-ph/0508416), a radiative transfer code widely used for cosmic epoch of reionization simulations, and makes its usage accessible and modular through python.

The core subroutines of C2Ray are implemented in compiled Fortran 90 and accessed as an extension module
built using `f2py`. The other aspects of the RT simulation, such as configuration, I/O, cosmology, source modelling, etc.
are implemented in pure python and are thus easily tweakable for a specific purpose.

In addition, the computationally most expensive step of the RT simulation, namely the raytracing,
has been modified to be GPU-parallelizable. This updated algorithm, called _OCTA_, is written
in C++/CUDA and can be used as a python extension module, both in a standalone way and for C2Ray simulations.

## Get Started
### Requirements
* `f2py`, provided by Numpy
* `gfortran` compiler
* `CUDA` and the `nvcc` compiler (Optional, for the accelerated raytracing library)
### Build Instructions
At a later time, a build system (like `meson` or `setuptools`) may be added. For now, the extension modules
required by the package must be compiled manually.
1. **Compile Fortran Extension Module**

First you need to compile the wrapped subroutines from C2Ray using `f2py`. To do this, run
```
cd src/c2ray/
make
```

If successful, this should create a `libc2ray.*.so` file, where the * is platform-dependent. Copy this file to
`pyc2ray/lib/` (create the directory if it doesn't exist).

2. **Compile CUDA Extension Module (optional)**

If you wish to use the OCTA raytracing library, you also need to compile the C++/CUDA extension.
Head to `src/octa/` and edit the makefile using the appropriate include paths. To find the include path
for numpy, open a python interpreter and run `np.get_include()`.
Then, run `make` and, assuming the build is successful, copy the `libocta.so` file to `pyc2ray/lib/`.

The `pyc2ray` package can then be used, assuming the pyc2ray/ directory is in the python path.

### Usage
A detailed documentation isn't yet available, but example scripts can be found in the `example/` directory
of this repository.

The general usage principle of pyc2ray is that:
1. A full C2Ray simulation can be configured and run using the `pyc2ray.C2Ray()` class along with a parameter file.
2. The individual components can also be used in a standalone way as module functions.

Bear in mind that the object-oriented usage (no. 1) is the reference implementation and that
missing documentation and possibly even errors may happen when using the modules individually, at least during development.

## Directories Guide
* `pyc2ray`: Python package directory
* `examples`: Contains example scripts that showcase the usage of pyc2ray
* `src`: Sources for the extension modules:
    * `c2ray`: Fortran sources adapted from the original C2Ray source code
    * `octa`: C++/CUDA sources for the octa library
* `test`: Scripts to be run during development to check that the code produces the same output as a given benchmark

All other directories are debugging/development files and are not part of the final project.
