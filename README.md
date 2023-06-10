# pyc2ray: A Python-wrapped and GPU-accelerated Update of C2Ray
`pyc2ray` is a Python package that wraps [C2Ray](https://github.com/garrelt/C2-Ray3Dm/tree/factorization) [(G. Mellema, I.T. Illiev, A. Alvarez and P.R. Shapiro)](https://ui.adsabs.harvard.edu/link_gateway/2006NewA...11..374M/doi:10.48550/arXiv.astro-ph/0508416), a radiative transfer code widely used for cosmic epoch of reionization simulations, and makes its usage accessible and modular through python.

The core subroutines of C2Ray are implemented in compiled Fortran 90 and accessed as an extension module
built using `f2py`. The other aspects of the RT simulation, such as configuration, I/O, cosmology, source modelling, etc.
are implemented in pure python and are thus easily tweakable for a specific purpose.

In addition, the computationally most expensive step of the RT simulation, namely the raytracing,
has been modified to be GPU-parallelizable. This updated algorithm, called _ASORA_, is written
in CUDA C++ as a python extension module. When no GPU is available, a CPU raytracing method is available as a fallback option.

## Installation
### Requirements
* `f2py`, provided by Numpy
* `gfortran` compiler
* `CUDA` and the `nvcc` compiler (Optional, for the accelerated raytracing library)
### Build Instructions
To build the required extension modules, `pyc2ray` uses the [Meson](https://mesonbuild.com/) build system, which
automatically identifies the platform-specific settings to compile source files.
The recommended method to install `pyc2ray` is by using `pip` directly on the repo, simply run
```
pip install git+https://github.com/phirling/PC2R.git -v
```
The `-v` flag is optional but prints additional information on the build process, and helps in identifying errors.
Pip will install all the necessary build dependencies (including `meson`) and run `meson compile` on all the
required targets, finally installing the package at the correct location. We recommend installing `pyc2ray`
inside a virtual environment.

`meson` will automatically detect if a CUDA compiler is available on the system and will only build the ASORA raytracing
extension module if this is the case. When ASORA cannot be built, `pyc2ray` can still be used but only using the CPU RT
methods. In this situation, a warning will be printed upon importing the package to inform the user that ASORA is unavailable.

### Configure Build
Developers may find it useful to tweak the build process. In particular, setting the appropriate build flags for the
CUDA extension module can yield significant performance improvements. As an example, its possible to set the
`gpu-architecture` flag of the `nvcc` compiler directly by appending the following to the above pip command:
```
pip install git+https://github.com/phirling/PC2R.git -v --config-settings=setup-args="-Dgpu-architecture=sm_60"
```
where in this example we set the architecture to `sm_60` (the default value, appropriate for e.g. a P100 GPU).
To get a list of all available `meson` settings, clone the repository and then run `meson configure` without
any arguments. To learn how to set these options via `pip`, please refer to the following docs of meson-python:
[Use build config settings](https://meson-python.readthedocs.io/en/latest/how-to-guides/config-settings.html)
and [Passing arguments to Meson](https://meson-python.readthedocs.io/en/latest/how-to-guides/meson-args.html#how-to-guides-meson-args).

### Run a Test
To see if the package works as expected, a series of tests are available in the `test` directory. In particular, we recommend
running
```
python 2_multisource.py --gpu --plot
```
Where the `--gpu` flag can be omitted to use the CPU (serial) raytracing module. After 10 timesteps (which should take
less than a minute to run), you should see a dice-like pattern of 5 ionized regions.

### Usage
A detailed documentation isn't yet available, but example scripts can be found in the `example/` directory
of this repository.

The general usage principle of pyc2ray is that:
1. A full C2Ray simulation can be configured and run using subclasses of the `pyc2ray.C2Ray()` class along with a parameter file.
2. The individual components can also be used in a standalone way as module functions.


## Directories Guide
* `pyc2ray`: Python package directory
* `examples`: Contains example scripts that showcase the usage of pyc2ray
* `src`: Sources for the extension modules:
    * `c2ray`: Fortran sources adapted from the original C2Ray source code
    * `octa`: C++/CUDA sources for the ASORA library
* `test`: Scripts to be run during development to check that the code produces the same output as a given benchmark