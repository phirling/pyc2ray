<a name="logo"/>
<div align="left">
<img src="banner_test.lpng" width="600"></img>
</a>
</div>

# pyc2ray: A Python-wrapped and GPU-accelerated Update of C2Ray
`pyc2ray` is a Python package that wraps [C2Ray](https://github.com/garrelt/C2-Ray3Dm/tree/factorization) [(G. Mellema, I.T. Illiev, A. Alvarez and P.R. Shapiro)](https://ui.adsabs.harvard.edu/link_gateway/2006NewA...11..374M/doi:10.48550/arXiv.astro-ph/0508416), a radiative transfer code widely used for cosmic epoch of reionization simulations, and makes its usage accessible and modular through python.

The core subroutines of C2Ray are implemented in compiled Fortran 90 and accessed as an extension module
built using `f2py`. The other aspects of the RT simulation, such as configuration, I/O, cosmology, source modelling, etc.
are implemented in pure python and are thus easily tweakable for a specific purpose.

In addition, the computationally most expensive step of the RT simulation, namely the raytracing,
has been modified to be GPU-parallelizable. This updated algorithm, called _ASORA_, is written
in CUDA C++ as a python extension module. When no GPU is available, a CPU raytracing method is available as a fallback option.

## Installation
Since the automatic build system isn't fully working yet, the extension modules must be compiled and placed in correct directories manually. After cloning the repository, create the `/lib` directory inside `/pyc2ray/` (from the root of the repository).

**Requirements**:
- C Compiler
- `gfortran` Fortran Compiler
- `nvcc` CUDA compiler
- `f2py` $\geq$ 1.24.4, provided by `numpy`

Additionally, once built, `pyc2ray` requires the `astropy` and `tools21cm` python packages to work.

### 1. Build Fortran extension module (C2Ray)

The tool to build the module is `f2py`, provided by the `numpy` package. The build requires version 1.24.4 or higher, to check run `f2py` without any options. If the version is too old or the command doesn't exist, install the latest numpy version in your current virtual environment. To build the extension module, run
```
cd src/c2ray/
make
```
and move the resulting shared library file `libc2ray.*.so` to the previously created `/pyc2ray/lib/` directory.

### 2. Build CUDA extension module (Asora)
```
cd ../asora/
```
Edit the Makefile and add the correct include paths at lines 3 and 4. To find the correct python include path (line 3), run
```
python -c "import sysconfig; print(sysconfig.get_path(name='include'))"
```
and to find the correct numpy include path (line 4), run
```
python -c "import numpy as np; print(np.get_include())"
```
Then, build the extension module by running `make`, and again move the file `libasora.so` to `/pyc2ray/lib/`.

### 3. Test the Install
If the build was successful, go to `/unit_tests_hackathon/1_single_source` and run
```
mkdir results
python run_example.py --gpu
```
This performs a RT simulation with a single source in a uniform volume, and checks for errors.

## Usage
The general usage principle of pyc2ray is that:
1. A full C2Ray simulation can be configured and run using subclasses of the `pyc2ray.C2Ray()` class along with a parameter file.
2. The individual components can also be used in a standalone way as module functions.


## Directories Guide
* `pyc2ray`: Python package directory
* `examples`: Contains example scripts that showcase the usage of pyc2ray
* `src`: Sources for the extension modules:
    * `c2ray`: Fortran sources adapted from the original C2Ray source code
    * `asora`: C++/CUDA sources for the ASORA raytracing library