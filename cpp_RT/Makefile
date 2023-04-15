CXX=g++
NVCC=nvcc
# Include Paths for both local and piz daint
INC=-I/usr/include/python3.10 -I/home/phirling/miniconda3/envs/astro/lib/python3.10/site-packages/numpy/core/include -I/usr/include/python3.6m -I/users/phirling/miniconda3/envs/ion/lib/python3.10/site-packages/numpy/core/include
CXXFLAGS+=-std=c++14 -O2 -pedantic -fPIC -D PERIODIC -D RATES #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
CUDAFLAGS+=-std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused

all: RTC.so

RTC.so: raytracing.o raytracing_gpu.o rtmodule.o
	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o RTC.so raytracing.o raytracing_gpu.o rtmodule.o
raytracing.o: raytracing.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c raytracing.cpp

raytracing_gpu.o: raytracing_gpu.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing_gpu.cu

rtmodule.o: rtmodule.cpp
	$(NVCC) $(CUDAFLAGS) $(INC) -c rtmodule.cpp

clean:
	rm *.so *.o
