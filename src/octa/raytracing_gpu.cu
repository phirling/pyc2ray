#include "raytracing_gpu.cuh"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <exception>
#include <string>
#include <iostream>

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define FOURPI 12.566370614359172463991853874177    // 4π
#define INV4PI 0.079577471545947672804111050482     // 1/4π
#define TAU_PHOTO_LIMIT 1.0e-7                      // Limit to consider a cell "optically thin/thick"
#define MAX_COLDENSH 2e30                           // Column density limit (rates are set to zero above this)
#define S_STAR_REF 1e48                             // Reference ionizing flux (strength of source is given in this unit)

// ========================================================================
// Utility Device Functions
// ========================================================================

// Fortran-type modulo function (C modulo is signed)
inline __device__ int modulo_gpu(const int & a,const int & b) { return (a%b+b)%b; }

// Sign function on the device
inline __device__ int sign_gpu(const double & x) { if (x>=0) return 1; else return -1;}

// Flat-array index from 3D (i,j,k) indices
inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N) { return N*N*i + N*j + k;}

// Weight function for C2Ray interpolation function (see cinterp_gpu below)
__device__ inline double weightf_gpu(const double & cd, const double & sig) { return 1.0/max(0.6,cd*sig);}

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

    auto error = cudaGetLastError();
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

// ========================================================================
// Main function: raytrace all sources and add up ionization rates
// ========================================================================
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
    const int & NumTau)
    {   
        // Byte-size of grid data
        auto meshsize = m1*m1*m1*sizeof(double);

        // Determine how large the octahedron should be, based on the raytracing radius. Currently,
        // this is set s.t. the radius equals the distance from the source to the middle of the faces
        // of the octahedron. To raytrace the whole box, the octahedron bust be 1.5*N in size
        int max_q = std::ceil(1.73205080757 * R); //std::ceil(1.5 * m1);

        // Grid size is initialized to (1,1) but will be adapted as the octahedron grows in size.
        // The z-dimension is always of size 2, it indexes the upper and lower part of the octahedron
        dim3 gs(1,1,2);

        // Block size. Set to 8x8 but this can be used for performance tuning on different GPUs
        int bl = 8;
        dim3 bs(bl,bl);

        // To efficiently fill large arrays with 0, we use the thrust library, which needs a custom pointer
        // object to the data to work
        thrust::device_ptr<double> cdh(cdh_dev);
        thrust::device_ptr<double> ion(phi_dev);

        // Here we fill the ionization rate array with zero before raytracing all sources. The LOCALRATES flag
        // is for debugging purposes and will be removed later on
        #if defined(LOCALRATES) || defined(RATES)
        thrust::fill(ion,ion + m1*m1*m1,0.0);
        #endif

        // Copy current ionization fraction to the device
        // cudaMemcpy(n_dev,ndens,meshsize,cudaMemcpyHostToDevice);  < --- !! density array is not modified, octa assumes that it has been copied to the device before
        cudaMemcpy(x_dev,xh_av,meshsize,cudaMemcpyHostToDevice);

        // Source position & strength variables
        int i0,j0,k0;
        double strength;

        for (int ns = 0; ns < NumSrc; ns++)
        {   
            // Set source position & strength
            // For compatibility with c2ray, source position is stored as: (dim0: coordinate, dim1: src number)
            i0 = srcpos[3*ns + 0];
            j0 = srcpos[3*ns + 1];
            k0 = srcpos[3*ns + 2];
            strength = srcstrength[ns];

            // std::cout << "Doing source at " << i0 << " " << j0 << " " << k0 << ", strength = " << strength << std::endl;

            // Set column density to zero for each source
            thrust::fill(cdh,cdh + m1*m1*m1,0.0);
            
            // OCTA loop: raytrace in octahedral shells of increasing size
            for (int q=0 ; q <= max_q; q++)
            {   
                // Grid size for the current shell and block size
                int grl = (2*q + 1) / bl + 1;
                gs.x = grl;
                gs.y = grl;

                // Raytracing kernel: see below
                evolve0D_gpu_new<<<gs,bs>>>(q,i0,j0,k0,strength,cdh_dev,sig,dr,n_dev,x_dev,phi_dev,m1,photo_thin_table_dev,minlogtau,dlogtau,NumTau);

                // Synchronize GPU
                cudaDeviceSynchronize();

                // Check for errors. TODO: make this better
                auto error = cudaGetLastError();
                if(error != cudaSuccess) {
                    std::cout << "error at q=" << q << std::endl;
                    throw std::runtime_error("Error Launching Kernel: "
                                            + std::string(cudaGetErrorName(error)) + " - "
                                            + std::string(cudaGetErrorString(error)));
                }
            }
        }

        // Copy the accumulated ionization fraction back to the host and check for errors
        #if defined(LOCALRATES) || defined(RATES)
        auto error = cudaMemcpy(phi_ion,phi_dev,meshsize,cudaMemcpyDeviceToHost);
        #endif
        //TODO: check for errors
    }


// ========================================================================
// Raytracing kernel, adapted from C2Ray. Calculates in/out column density
// to the current cell and finds the photoionization rate
// ========================================================================
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
    const int NumTau
)
{
    // x and y coordinates are cartesian
    int i = - q + blockIdx.x * blockDim.x + threadIdx.x;
    int j = - q + blockIdx.y * blockDim.y + threadIdx.y;

    int sgn, mq;

    // Determine whether we are in the upper or lower pyramid of the octahedron
    if (blockIdx.z == 0)
        {sgn = 1; mq = q;}
    else
        {sgn = -1; mq = q-1;}

    // Only proceed if the point is on the octahedron (see Fig. ?? in paper (TODO))
    if (abs(i) + abs(j) <= mq)
    {
        int k = k0 + sgn*q - sgn*(abs(i) + abs(j));

        // Center to source
        i += i0;
        j += j0;

        int pos[3];
        double path;
        double coldensh_in;                                // Column density to the cell
        double nHI_p;                                      // Local density of neutral hydrogen in the cell
        double xh_av_p;                                    // Local ionization fraction of cell

        #if defined(LOCALRATES)
        double xs, ys, zs;
        double dist2;
        double vol_ph;
        #endif

        // When not in periodic mode, only treat cell if its in the grid
        #if !defined(PERIODIC)
        if (in_box_gpu(i,j,k,m1))
        #endif
        {   
            // Map to periodic grid
            pos[0] = modulo_gpu(i,m1);
            pos[1] = modulo_gpu(j,m1);
            pos[2] = modulo_gpu(k,m1);

            // Get local ionization fraction & Hydrogen density
            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);

            // Only treat cell if it hasn't been done before
            if (coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] == 0.0)
            {   
                // If its the source cell, just find path (no incoming column density)
                if (i == i0 &&
                    j == j0 &&
                    k == k0)
                {
                    coldensh_in = 0.0;
                    path = 0.5*dr;
                    #if defined(LOCALRATES)
                    // vol_ph = dr*dr*dr / (4*M_PI);
                    vol_ph = dr*dr*dr;
                    #endif
                }

                // If its another cell, do interpolation to find incoming column density
                else
                {
                    cinterp_gpu(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out,sig,m1);
                    path *= dr;
                    #if defined(LOCALRATES)
                    // Find the distance to the source
                    xs = dr*(i-i0);
                    ys = dr*(j-j0);
                    zs = dr*(k-k0);
                    dist2=xs*xs+ys*ys+zs*zs;
                    // vol_ph = dist2 * path;
                    vol_ph = dist2 * path * FOURPI;
                    #endif
                }

                // Add to column density array. TODO: is this really necessary ?
                coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] = coldensh_in + nHI_p * path;
                
                // Compute photoionization rates from column density. WARNING: for now this is limited to the grey-opacity test case source
                #if defined(LOCALRATES)
                if (coldensh_in <= MAX_COLDENSH)
                {
                    #if defined(GREY_NOTABLES)
                    double phi = photoion_rates_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig);
                    #else
                    double phi = photoion_rates_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig,photo_table,minlogtau,dlogtau,NumTau);
                    #endif
                    // Divide the photo-ionization rates by the appropriate neutral density
                    // (part of the photon-conserving rate prescription)
                    phi /= nHI_p;

                    phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi;
                }
                #endif
            }
        }
    }
}

// ========================================================================
// Compute photoionization rate from in/out column density by looking up
// values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables. These
// tables are assumed to have been copied to device memory in advance using
// photo_table_to_device()
// ========================================================================
__device__ double photoion_rates_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig,
    const double* table,const double & minlogtau,const double & dlogtau,const int& NumTau)
{
    // Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
    double tau_in = coldens_in * sig;
    double tau_out = coldens_out * sig;

    double prefact = strength / Vfact;

    double phi_photo_in = prefact * photo_lookuptable(table,tau_in,minlogtau,dlogtau,NumTau);
    double phi_photo_out = prefact * photo_lookuptable(table,tau_out,minlogtau,dlogtau,NumTau);
    return phi_photo_in - phi_photo_out;
}

// ========================================================================
// Grey-opacity test case photoionization rate, computed from analytical
// expression rather than using tables. To use this version, compile
// with the -DGREY_NOTABLES flag
// ========================================================================
__device__ double photoion_rates_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig)
{
    // Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
    double tau_in = coldens_in * sig;
    double tau_out = coldens_out * sig;

    

    // If cell is optically thick
    if (fabs(tau_out - tau_in) > TAU_PHOTO_LIMIT)
        // return strength * INV4PI / (Vfact * nHI) * (exp(-tau_in) - exp(-tau_out));
        return (strength*S_STAR_REF / (Vfact)) * (exp(-tau_in) - exp(-tau_out));
    // If cell is optically thin
    else
        // return strength * INV4PI * sig * (tau_out - tau_in) / (Vfact) * exp(-tau_in);
        return (strength*S_STAR_REF / (Vfact)) * (tau_out - tau_in) * exp(-tau_in);
}

// ========================================================================
// Utility function to look up the integral value corresponding to an
// optical depth τ by doing linear interpolation.
// ========================================================================
__device__ double photo_lookuptable(const double* photo_thin_table,const double & tau,const double & minlogtau,const double & dlogtau,const int & NumTau)
{
    double logtau;
    double real_i, residual;
    int i0, i1;
    // Find table index and do linear interpolation
    // Recall that tau(0) = 0 and tau(1:NumTau) ~ logspace(minlogtau,maxlogtau) (so in reality the table has size NumTau+1)
    logtau = log10(max(1.0e-20,tau));
    real_i = min(float(NumTau),max(0.0,1.0+(logtau-minlogtau)/dlogtau));
    i0 = int( real_i );
    i1 = min(NumTau, i0+1);
    residual = real_i - double(i0);
    return photo_thin_table[i0] + residual*(photo_thin_table[i1] - photo_thin_table[i0]);
}


// ========================================================================
// Short-characteristics interpolation function, adapted from C2Ray
// ========================================================================
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
    const int & m1)
{
    int idel,jdel,kdel;
    int idela,jdela,kdela;
    int im,jm,km;
    unsigned int ip,imp,jp,jmp,kp,kmp;
    int sgni,sgnj,sgnk;
    double alam,xc,yc,zc,dx,dy,dz,s1,s2,s3,s4;
    double c1,c2,c3,c4;
    double w1,w2,w3,w4;
    double di,dj,dk;

    // calculate the distance between the source point (i0,j0,k0) and 
    // the destination point (i,j,k)
    idel=i-i0;
    jdel=j-j0;
    kdel=k-k0;
    idela=abs(idel);
    jdela=abs(jdel);
    kdela=abs(kdel);
    
    // Find coordinates of points closer to source
    sgni=sign_gpu(idel);
    sgnj=sign_gpu(jdel);
    sgnk=sign_gpu(kdel);
    im=i-sgni;
    jm=j-sgnj;
    km=k-sgnk;
    di=double(idel);
    dj=double(jdel);
    dk=double(kdel);

    // Z plane (bottom and top face) crossing
    // we find the central (c) point (xc,xy) where the ray crosses 
    // the z-plane below or above the destination (d) point, find the 
    // column density there through interpolation, and add the contribution
    // of the neutral material between the c-point and the destination
    // point.
    if (kdela >= jdela && kdela >= idela) {
        // alam is the parameter which expresses distance along the line s to d
        // add 0.5 to get to the interface of the d cell.
        alam=(double(km-k0)+sgnk*0.5)/dk;
            
        xc=alam*di+double(i0); // x of crossing point on z-plane 
        yc=alam*dj+double(j0); // y of crossing point on z-plane
        
        dx=2.0*abs(xc-(double(im)+0.5*sgni)); // distances from c-point to
        dy=2.0*abs(yc-(double(jm)+0.5*sgnj)); // the corners.
        
        s1=(1.-dx)*(1.-dy);    // interpolation weights of
        s2=(1.-dy)*dx;         // corner points to c-point
        s3=(1.-dx)*dy;
        s4=dx*dy;

        ip  = modulo_gpu(i  ,m1);
        imp = modulo_gpu(im ,m1);
        jp  = modulo_gpu(j  ,m1);
        jmp = modulo_gpu(jm ,m1);
        kmp = modulo_gpu(km ,m1);
        
        c1=     coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        c2=     coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
        c3=     coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
        c4=     coldensh_out[mem_offst_gpu(ip,jp,kmp,m1)];

        // extra weights for better fit to analytical solution
        w1=   s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2=   s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3=   s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4=   s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
        
        // column density at the crossing point
        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (kdela == 1 && (idela == 1||jdela == 1))
        {
            if (idela == 1 && jdela == 1)
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }

        // Path length from c through d to other side cell.
        path=sqrt((di*di+dj*dj)/(dk*dk)+1.0);
    }
    else if (jdela >= idela && jdela >= kdela)
    {
        alam=(double(jm-j0)+sgnj*0.5)/dj;
        zc=alam*dk+double(k0);
        xc=alam*di+double(i0);
        dz=2.0*abs(zc-(double(km)+0.5*sgnk));
        dx=2.0*abs(xc-(double(im)+0.5*sgni));
        s1=(1.-dx)*(1.-dz);
        s2=(1.-dz)*dx;
        s3=(1.-dx)*dz;
        s4=dx*dz;

        ip  = modulo_gpu(i,m1);
        imp = modulo_gpu(im,m1);
        jmp = modulo_gpu(jm,m1);
        kp  = modulo_gpu(k,m1);
        kmp = modulo_gpu(km,m1);

        c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        c2=  coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
        c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
        c4=  coldensh_out[mem_offst_gpu(ip,jmp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1=s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2=s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3=s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4=s4*weightf_gpu(c4,sigma_HI_at_ion_freq);

        cdensi=   (c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (jdela == 1 && (idela == 1||kdela == 1))
        {
            if (idela == 1 && kdela == 1)
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }
        path=sqrt((di*di+dk*dk)/(dj*dj)+1.0);
    }
    else
    {
        alam=(double(im-i0)+sgni*0.5)/di;
        zc=alam*dk+double(k0);
        yc=alam*dj+double(j0);
        dz=2.0*abs(zc-(double(km)+0.5*sgnk));
        dy=2.0*abs(yc-(double(jm)+0.5*sgnj));
        s1=(1.-dz)*(1.-dy);
        s2=(1.-dz)*dy;
        s3=(1.-dy)*dz;
        s4=dy*dz;

        imp=modulo_gpu(im ,m1);
        jp= modulo_gpu(j  ,m1);
        jmp=modulo_gpu(jm ,m1);
        kp= modulo_gpu(k  ,m1);
        kmp=modulo_gpu(km ,m1);

        c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        c2=  coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
        c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
        c4=  coldensh_out[mem_offst_gpu(imp,jp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1   =s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2   =s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3   =s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4   =s4*weightf_gpu(c4,sigma_HI_at_ion_freq);

        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        if ( idela == 1  &&  ( jdela == 1 || kdela == 1 ) )
        {
            if ( jdela == 1  &&  kdela == 1 )
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }
        path=sqrt(1.0+(dj*dj+dk*dk)/(di*di));
    }
}





















// ==========================================================================================================
// OLD OR EXPERIMENTAL CODE. KEPT AS REFERENCE BUT UNUSED
// ==========================================================================================================



// WIP: compute rates in a separate step ? This would require having a path and coldensh_in grid (replace cdh_out by path)
__global__ void do_rates(
    const int rad,
    const int i0,
    const int j0,
    const int k0,
    const double strength,
    double* coldensh_in,
    double* path,
    double* ndens,
    double* phi_ion,
    const double sig,
    const double dr,
    const int m1
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (abs(i) + abs(j) + abs(k) <= rad)
    {
        double vol_ph;
        int pos[3];
        pos[0] = modulo_gpu(i,m1);
        pos[1] = modulo_gpu(j,m1);
        pos[2] = modulo_gpu(k,m1);
        double cdh_in = coldensh_in[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
        double nHI = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
        double cdh_out;
        double phi;
        if (i == i0 && j == j0 && k == k0)
        {
            vol_ph = dr*dr*dr / (4*M_PI);
        }
        else
        {
            double xs = dr*(i-i0);
            double ys = dr*(j-j0);
            double zs = dr*(k-k0);
            double dist2=xs*xs+ys*ys+zs*zs;
            vol_ph = dist2 * path[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
        }

        cdh_out = cdh_in + path[mem_offst_gpu(pos[0],pos[1],pos[2],m1)]*nHI;
        phi = photoion_rates_test_gpu(strength,cdh_in,cdh_out,vol_ph,sig);
        phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi / nHI;

    } 
}