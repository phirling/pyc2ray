#include <iostream>
#include "raytracing_gpu.cuh"
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
    std::vector<std::vector<int>> srcpos(3,std::vector<int>(1));
    //std::vector<std::vector<std::vector<double> > > coldensh_out(N,std::vector<std::vector<double> >(N,std::vector<double>(N)));
    std::vector<std::vector<std::vector<double> > > ndens(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1.0)));
    std::vector<std::vector<std::vector<double> > > phi_ion(N,std::vector<std::vector<double> >(N,std::vector<double>(N,0.0)));
    std::vector<std::vector<std::vector<double> > > xh_av(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1e-3)));
    //double* coldensh_out = (double*)malloc(N*N*N*sizeof(double));
    double* coldensh_out = (double*)calloc(N*N*N,sizeof(double));
    //double* coldensh_out = &(cdh[0][0][N]);

    srcpos[0][0] = 64;
    srcpos[1][0] = 64;
    srcpos[2][0] = 64;

    int ns=0;
    int NumSrc = 1;

    std::vector<int> rtpos = {64,64,64};

    double sig = 1.0;
    double dr = 1.0;

    // evolve0D(rtpos,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);
    // rtpos = {65,65,65};
    // evolve0D(rtpos,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);
    // std::cout << coldensh_out[65][65][65] << std::endl;
    
    auto t1 = clk::now();
    do_source_octa_gpu(srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);
    second elapsed = clk::now() - t1;

    std::cout << "Time for RT  = " << elapsed.count() << " [s]\n";
    double mean;
    //double* cdh = &coldensh_out[0][0][0]; //coldensh_out.data()->data()->data();

    //mean = gsl_stats_mean(cdh,1,N*N);
    mean = coldensh_out[mem_offst(80,80,65,N)];
    //mean = coldensh_out[mem_offst(2,2,2,N)];

    //std::cout << coldensh_out[0][0][0] << " " << coldensh_out[1][1][1] << " " << coldensh_out[2][2][2] << " " << std::endl;
    std::cout << std::endl << mean << std::endl;
    return 0;
}