## Directories Explanations
* c2ray_wrapped: Core routines of the Fortran C2Ray code wrapped using `f2py` for Python. Includes raytracing, simple photoionization rate computation and chemsitry equation solver.
* fortran_RT: Same as above but includes only the raytracing part for comparison with the *OCTA* version
* cpp_RT: Raytracing implemented with the OCTA algorithm, written in C++ and C++ CUDA
* testing_area: Scripts to compare the above two implementations