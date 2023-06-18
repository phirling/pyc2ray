#pragma once
#include <cuda_runtime.h>
#include <vector>

// ========================================================================
// Header file for OCTA raytracing library.
// Functions defined and documented in raytracing_gpu.cu
// ========================================================================

// Modulo function with Fortran convention
inline int modulo(const int & a,const int & b);
inline __device__ int modulo_gpu(const int & a,const int & b);

// Device sign function
inline __device__ int sign_gpu(const double & x);

// Flat array index from i,j,k coordinates
inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N);

// Photoionization rate from tables
__device__ double photoion_rates_gpu(const double & strength,const double & coldens_in,const double & coldens_out,
    const double & Vfact,const double & sig,const double* table,const double & minlogtau,const double & dlogtau,const int& NumTau);

// Table interpolation lookup function
__device__ double photo_lookuptable(const double*,const double &,const double &,const double &,const int &);

// Photoionization rates from analytical expression (grey-opacity)
__device__ double photoion_rates_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig);

// Allocate grid memory
void device_init(const int &);

// Deallocate grid memory
void device_close();

// Copy density grid to device memory
void density_to_device(double*,const int &);

// Copy radiation tables to device memory
void photo_table_to_device(double*,const int &);

// Pointers to device memory
extern double * cdh_dev;
extern double * n_dev;
extern double * x_dev;
extern double * phi_dev;
extern double * photo_thin_table_dev;

// Raytrace all sources and compute photoionization rates
void do_all_sources_octa_gpu(
    int* srcpos,
    double* srcstrength,
    const double & R,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    double* ndens,
    double* xh_av,
    double* phi_ion,
    const int & NumSrc,
    const int & m1,
    const double & minlogtau,
    const double & dlogtau,
    const int & NumTau);

// Raytracing kernel, called by do_all_sources
__global__ void evolve0D_gpu_new(
    const int q,
    const int i0,
    const int j0,
    const int k0,
    const double strength,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1,
    const double* photo_table,
    const double minlogtau,
    const double dlogtau,
    const int NumTau,
    const int last_l,
    const int last_r
);

// Short-characteristics interpolation function from C2Ray
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


// Check if point is in domain (deprecated)
inline __device__ bool in_box_gpu(const int & i,const int & j,const int & k,const int & N)
{
    return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
}