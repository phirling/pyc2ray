#include <cuda.h>
#include <vector>

inline __device__ int modulo_gpu(const int & a,const int & b);

inline __device__ int sign_gpu(const double & x);

inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N);


__device__ void cinterp_gpu(
    const int i,
    const int j,
    const int k,
    const int i0,
    const int j0,
    const int k0,
    double & cdensi,
    double & path,
    double* coldensh_out,
    const double sigma_HI_at_ion_freq,
    const int & m1);

__global__ void evolve0D_gpu(
    const int r,
    const int i0,
    const int j0,
    const int k0,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1,
    const int d1,
    const int d2,
    const int d3);

void do_source_octa_gpu(const std::vector<std::vector<int> > & srcpos,
    const int & ns,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    const std::vector<std::vector<std::vector<double> > > & ndens,
    const std::vector<std::vector<std::vector<double> > > & xh_av,
    std::vector<std::vector<std::vector<double> > > & phi_ion,
    const int & NumSrc,
    const int & m1);