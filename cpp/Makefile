CXX=g++
LD=${CXX}
CXXFLAGS+=-std=c++11 -O3 -pedantic #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
NVCCFLAGS+=-std=c++11 #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
LDFLAGS+= -L/usr/local/lib -lm $(CXXFLAGS) -lgsl -lcblas -v

all: test

gpu: test_gpu

test: test.o raytracing.o
	$(LD) $(LDFLAGS) -o test test.o raytracing.o
test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c test.cpp
raytracing.o: raytracing.cpp
	$(CXX) $(CXXFLAGS) -c raytracing.cpp raytracing.hh

test_gpu: test_gpu.o raytracing_gpu.o raytracing.o
	nvcc $(NVCCFLAGS) -o test_gpu test_gpu.o raytracing.o raytracing_gpu.o
test_gpu.o: test_gpu.cu
	nvcc $(NVCCFLAGS) -c test_gpu.cu
raytracing_gpu.o: raytracing_gpu.cu
	nvcc $(NVCCFLAGS) -c raytracing_gpu.cu raytracing_gpu.cuh
clean:
	rm -f test *.o *~