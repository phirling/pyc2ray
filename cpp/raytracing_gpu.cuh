#include <cuda.h>
#include <vector>

inline __device__ int modulo_gpu(const int & a,const int & b);

inline __device__ int sign_gpu(const double & x);

inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N);

inline int mem_offst(const int & i,const int & j,const int & k,const int & N)
{   
    return N*N*i + N*j + k;
}

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

void do_source_octa_gpu(const std::vector<std::vector<int> > & srcpos,      // Position of all sources
    const int & ns,                                                     // Source number
    double* coldensh_out,     // Outgoing column density
    const double & sig,                                                 // Cross section
    const double & dr,                                                  // Cell size
    const std::vector<std::vector<std::vector<double> > > & ndens,      // Hydrogen number density
    const std::vector<std::vector<std::vector<double> > > & xh_av,      // Time-average ionization fraction
    std::vector<std::vector<std::vector<double> > > & phi_ion,          // Ionization Rates
    const int & NumSrc,                                                 // Number of sources
    const int & m1);