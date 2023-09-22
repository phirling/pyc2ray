#include "memory.cuh"
#include <iostream>

// ========================================================================
// Global variables. Pointers to GPU memory to store grid data
//
// To avoid uneccessary memory movement between host and device, we
// allocate dedicated memory on the device via a call to device_init at the
// beginning of the program. Data is copied to and from the host memory
// (typically numpy arrays) only when it changes and is required. For example:
//
// * The density field is copied to the device only when it
// actually changes, i.e. at the beginning of a timestep.
// * The photoionization rates for each source are computed and summed
// directly on the device and are copied to the host only when all sources
// have been passed.
// * The column density is NEVER copied back to the host, since it is only
// accessed on the device when computing ionization rates.
// ========================================================================
double* cdh_dev;                // Outgoing column density of the cells
double* n_dev;                  // Density
double* x_dev;                  // Time-averaged ionized fraction
double* phi_dev;                // Photoionization rates
double* photo_thin_table_dev;   // Radiation table

// ========================================================================
// Initialization function to allocate device memory (pointers above)
// ========================================================================
void device_init(const int & N)
{
    int dev_id = 0;

    cudaDeviceProp device_prop;
    cudaGetDevice(&dev_id);
    cudaGetDeviceProperties(&device_prop, dev_id);
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                    "threads can use ::cudaSetDevice()"
                << std::endl;
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties returned error code " << error
                << ", line(" << __LINE__ << ")" << std::endl;
    } else {
        std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
                << "\" with compute capability " << device_prop.major << "."
                << device_prop.minor << std::endl;
    }

    // Byte-size of grid data
    int bytesize = N*N*N*sizeof(double);

    // Allocate memory
    cudaMalloc(&cdh_dev,bytesize);
    cudaMalloc(&n_dev,bytesize);
    cudaMalloc(&x_dev,bytesize);
    cudaMalloc(&phi_dev,bytesize);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Couldn't allocate memory" << std::endl;
    }
    else {
        std::cout << "Succesfully allocated " << 4*bytesize/1e6 << " Mb of device memory for grid of size N = " << N << std::endl;
    }
}

// ========================================================================
// Utility functions to copy data to device
// ========================================================================
void density_to_device(double* ndens,const int & N)
{
    cudaMemcpy(n_dev,ndens,N*N*N*sizeof(double),cudaMemcpyHostToDevice);
}

void photo_table_to_device(double* table,const int & NumTau)
{
    cudaMalloc(&photo_thin_table_dev,NumTau*sizeof(double));
    cudaMemcpy(photo_thin_table_dev,table,NumTau*sizeof(double),cudaMemcpyHostToDevice);
}

// ========================================================================
// Deallocate device memory at the end of a run
// ========================================================================
void device_close()
{   
    printf("Deallocating device memory...\n");
    cudaFree(&cdh_dev);
    cudaFree(&n_dev);
    cudaFree(&x_dev);
    cudaFree(&phi_dev);
    cudaFree(&photo_thin_table_dev);
}
