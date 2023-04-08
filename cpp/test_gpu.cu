#include <iostream>
#include "raytracing_gpu.cuh"
#include "raytracing.hh"
#include <vector>
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

int main()
{   
    int dev_id = 0;

    cudaDeviceProp device_prop;
    cudaGetDevice(&dev_id);
    cudaGetDeviceProperties(&device_prop, dev_id);
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                    "threads can use ::cudaSetDevice()"
                << std::endl;
        return -1;
    }

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties returned error code " << error
                << ", line(" << __LINE__ << ")" << std::endl;
        return error;
    } else {
        std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
                << "\" with compute capability " << device_prop.major << "."
                << device_prop.minor << std::endl;
    }

    int N = 128;
    //std::vector<std::vector<int>> srcpos(3,std::vector<int>(1));
    //std::vector<std::vector<std::vector<double> > > coldensh_out(N,std::vector<std::vector<double> >(N,std::vector<double>(N)));
    // std::vector<std::vector<std::vector<double> > > ndens(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1.0)));
    // std::vector<std::vector<std::vector<double> > > phi_ion(N,std::vector<std::vector<double> >(N,std::vector<double>(N,0.0)));
    // std::vector<std::vector<std::vector<double> > > xh_av(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1e-3)));
    //double* coldensh_out = (double*)malloc(N*N*N*sizeof(double));
    //double* coldensh_out = (double*)calloc(N*N*N,sizeof(double));
    //double* coldensh_out = &(cdh[0][0][N]);

    double* coldensh_out = (double*)calloc(N*N*N,sizeof(double));
    double* ndens = (double*)calloc(N*N*N,sizeof(double));
    double* phi_ion = (double*)calloc(N*N*N,sizeof(double));
    double* xh_av = (double*)calloc(N*N*N,sizeof(double));

    std::fill(coldensh_out,coldensh_out + N*N*N,0.0);
    std::fill(ndens,ndens + N*N*N,1.0);
    std::fill(phi_ion,phi_ion+ N*N*N,0.0);
    std::fill(xh_av,xh_av+ N*N*N,1e-3);

    int NumSrc = 1;
    int ns=0;
    int* srcpos = (int*)malloc(3*NumSrc*sizeof(int));
    srcpos[0]                = 64;      //srcpos[0][ns];
    srcpos[NumSrc + ns]      = 64;   //srcpos[1][ns];
    srcpos[2*NumSrc + ns]    = 64; //srcpos[2][ns];
    // srcpos[0][0] = 1;
    // srcpos[1][0] = 1;
    // srcpos[2][0] = 1;


    double sig = 1.0;
    double dr = 1.0;
    int test_idx = mem_offst(14,24,19,N);
    double mean;
    
    auto t1 = clk::now();
    do_source_octa_gpu(srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);
    second elapsed = clk::now() - t1;
    std::cout << "Time for RT (GPU) = " << elapsed.count() << " [s]\n";

    mean = coldensh_out[test_idx];
    std::cout << "Test value: " << mean << std::endl;
    
    std::fill(coldensh_out,coldensh_out + N*N*N,0.0);
    t1 = clk::now();
    do_source_octa(srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);
    elapsed = clk::now() - t1;
    std::cout << "Time for RT (CPU) = " << elapsed.count() << " [s]\n";

    mean = coldensh_out[test_idx];
    std::cout << "Test value: " << mean << std::endl;

    return 0;
}