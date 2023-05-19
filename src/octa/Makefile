CXX=g++
NVCC=nvcc
PYTHONINC+=-I/project/c31/codes/miniconda3/include/python3.9
NUMPYINC+=-I/users/mibianco/codes/pyC2Ray/pyc2ray-env/lib/python3.9/site-packages/numpy/core/include

# Include Paths for both local and piz daint
INC+=$(PYTHONINC) $(NUMPYINC)
CXXFLAGS+=-std=c++14 -O2 -pedantic -fPIC -D PERIODIC -D RATES #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
CUDAFLAGS+=-std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES --gpu-architecture=sm_60 #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused

all: libocta.so

libocta.so: raytracing_gpu.o rtmodule.o
	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o libocta.so raytracing_gpu.o rtmodule.o

raytracing_gpu.o: raytracing_gpu.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing_gpu.cu

rtmodule.o: rtmodule.cpp
	$(NVCC) $(CUDAFLAGS) $(INC) -c rtmodule.cpp

clean:
	rm *.so *.o
