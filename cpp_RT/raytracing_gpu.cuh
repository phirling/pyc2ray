#pragma once
#include <cuda_runtime.h>
#include <vector>

inline __device__ int modulo_gpu(const int & a,const int & b);

inline __device__ int sign_gpu(const double & x);

inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N);

inline __device__ bool in_box_gpu(const int & i,const int & j,const int & k,const int & N)
{
    return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
}

__device__ double photoion_rate_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & nHI,const double & sig);


void device_init(const int &);

void device_close();

//extern unsigned long meshsizze;

extern double * cdh_dev;
extern double * n_dev;
extern double * x_dev;
extern double * phi_dev;

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
    const int d1,
    const int d2,
    const int d3);

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
    const int m1
);

void do_source_octa_gpu(
    int* srcpos,
    double* srcstrength,
    const int & ns,
    const double & R,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    double* ndens,
    double* xh_av,
    double* phi_ion,
    const int & NumSrc,
    const int & m1);