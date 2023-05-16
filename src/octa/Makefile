CXX=g++
NVCC=nvcc
PYTHONINC+=-I/insert_here_path_to_python_include
NUMPYINC+=-I/insert_here_path_to_numpy_include

# Include Paths for both local and piz daint
INC+=$(PYTHONINC) $(NUMPYINC)
CXXFLAGS+=-std=c++14 -O2 -pedantic -fPIC -D PERIODIC -D RATES #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
CUDAFLAGS+=-std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused

all: libocta.so

libocta.so: raytracing_gpu.o rtmodule.o
	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o libocta.so raytracing_gpu.o rtmodule.o

raytracing_gpu.o: raytracing_gpu.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing_gpu.cu

rtmodule.o: rtmodule.cpp
	$(NVCC) $(CUDAFLAGS) $(INC) -c rtmodule.cpp

clean:
	rm *.so *.o
