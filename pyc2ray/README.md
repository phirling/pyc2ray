The pyc2ray package is structured as follows.

It relies on 2 compiled extension modules:
- libc2ray: Fortran subroutines from C2Ray, adapted and interfaced with python using f2py. ../src/pyc2ray/
- libocta: CUDA C++ implementation of the OCTA raytracing method, compiled with nvcc. ../src/octa/

Rather than giving access to these modules directly to the user, they are managed internally and the user
only interacts with the following python modules:

- evolve.py
Implements the evolve3D subroutine that evolves the whole grid for one timestep, by iterating between
raytracing and chemistry to reach convergence. The raytracing step can be done either using the standard
C2Ray subroutines or the OCTA implementation.
- raytracing.py
Gives access to the raytracing part on its own, again either the C2Ray or the OCTA version.
- chemistry.py
Gives access to the chemistry part on its own, i.e. the doric algorithm from C2Ray.

In the future, a cleaner approach using e.g. setuptools may be added, but for now the extension libraries
need to be compiled manually and copied to the package root directory, as explained in the Build section.

In addition, the submodule utils includes:
- sourceutils.py: Read/Write C2Ray source files
- logutils.py: Write log files
- paramutils.py: Read in YAML parameter files for clean scripts