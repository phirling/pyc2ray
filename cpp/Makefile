CXX=g++
LD=${CXX}
CXXFLAGS+=-std=c++11 -O3
LDFLAGS+= -L/usr/local/lib -lm $(CXXFLAGS) -lgsl -lcblas -v

all: test

test: test.o raytracing.o
	$(LD) $(LDFLAGS) -o test test.o raytracing.o
test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c test.cpp
raytracing.o: raytracing.cpp
	$(CXX) $(CXXFLAGS) -c raytracing.cpp raytracing.hh

clean:
	rm -f test *.o *~