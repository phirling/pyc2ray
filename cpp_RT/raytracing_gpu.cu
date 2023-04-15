#include "raytracing_gpu.cuh"
#include "raytracing.hh"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <exception>
#include <string>
#include <iostream>

#define INV4PI 0.079577471545947672804111050482
#define TAU_PHOTO_LIMIT 1.0e-7
#define MAX_COLDENSH 2e30

inline __device__ int modulo_gpu(const int & a,const int & b) { return (a%b+b)%b; }

inline __device__ int sign_gpu(const double & x) { if (x>=0) return 1; else return -1;}

inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N)
{   
    return N*N*i + N*j + k;
}

__device__ inline double weightf_gpu(const double & cd, const double & sig)
{
    return 1.0/fmax(0.6,cd*sig);
}


unsigned long meshsizze;
double* cdh_dev;
double* n_dev;
double* x_dev;
double* phi_dev;


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
    const int & m1)
    {   
        int i0 = srcpos[ns];            //srcpos[0][ns];
        int j0 = srcpos[NumSrc + ns];   //srcpos[1][ns];
        int k0 = srcpos[2*NumSrc + ns]; //srcpos[2][ns];
        double strength = srcstrength[ns];
        // Source position
        //std::vector<int> srcpos_p = {srcpos[0][ns], srcpos[1][ns], srcpos[2][ns]};

        auto meshsize = m1*m1*m1*sizeof(double);
        // First, do the source cell
        //std::vector<int> rtpos = {srcpos_p[0],srcpos_p[1],srcpos_p[2]};
        //evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);
        double path = 0.5*dr;
        double src_cell_val = ndens[mem_offst(i0,j0,k0,m1)] * path * (1.0 - xh_av[mem_offst(i0,j0,k0,m1)]);

        //std::cout << coldensh_out[mem_offst(i0,j0,k0,m1)] << std::endl;
        // Sweep the grid by treating the faces of octahedra of increasing size.
        int max_q = std::ceil(1.73205080757 * R); //std::ceil(1.5 * m1);

        // double* coldensh_out_dev;
        // double* ndens_dev;
        // double* xh_av_dev;
        // double* phi_ion_dev;

        // cudaMalloc(&coldensh_out_dev,meshsize);
        // cudaMalloc(&ndens_dev,meshsize);
        // cudaMalloc(&xh_av_dev,meshsize);
        // cudaMalloc(&phi_ion_dev,meshsize);

        thrust::device_ptr<double> cdh(cdh_dev);
        thrust::device_ptr<double> ion(phi_dev);
        thrust::fill(cdh,cdh + m1*m1*m1,0.0);
        #if defined(RATES)
        thrust::fill(ion,ion + m1*m1*m1,0.0);
        #endif
        thrust::fill(cdh + mem_offst(i0,j0,k0,m1), cdh + mem_offst(i0,j0,k0,m1) +1,src_cell_val);

        // cudaMemcpy(coldensh_out_dev,coldensh_out,meshsize,cudaMemcpyHostToDevice);
        cudaMemcpy(n_dev,ndens,meshsize,cudaMemcpyHostToDevice);
        cudaMemcpy(x_dev,xh_av,meshsize,cudaMemcpyHostToDevice);
        //cudaMemcpy(phi_dev,phi_ion,meshsize,cudaMemcpyHostToDevice);
        cudaStream_t stream[8];
        for (int a = 0; a < 8 ; a++)
        {
            cudaStreamCreate(&stream[a]);
        }
        int bl = 4;
        dim3 gs(1,1,2);
        dim3 bs(bl,bl);
        for (int q=1 ; q <= max_q; q++)
        {   
            int grl = (2*q + 1) / bl + 1;
            gs.x = grl;
            gs.y = grl;
            evolve0D_gpu_new<<<gs,bs>>>(q,i0,j0,k0,strength,cdh_dev,sig,dr,n_dev,x_dev,phi_dev,m1);
            cudaDeviceSynchronize();

            auto error = cudaGetLastError();
            if(error != cudaSuccess) {
                std::cout << "error at q=" << q << std::endl;
                throw std::runtime_error("Error Launching Kernel: "
                                        + std::string(cudaGetErrorName(error)) + " - "
                                        + std::string(cudaGetErrorString(error)));
            }
        }

        for (int a = 0; a < 8 ; a++)
        {
            cudaStreamDestroy(stream[a]);
        }

        auto error = cudaMemcpy(coldensh_out,cdh_dev,meshsize,cudaMemcpyDeviceToHost);
        #if defined(RATES)
        error = cudaMemcpy(phi_ion,phi_dev,meshsize,cudaMemcpyDeviceToHost);
        #endif
    }

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
)
{
    int i = - q + blockIdx.x * blockDim.x + threadIdx.x;
    int j = - q + blockIdx.y * blockDim.y + threadIdx.y;

    int sgn, mq;
    if (blockIdx.z == 0)
        {sgn = 1; mq = q;}
    else
        {sgn = -1; mq = q-1;}

    if (abs(i) + abs(j) <= mq)
    {
        int k = k0 + sgn*q - sgn*(abs(i) + abs(j));
        i += i0;
        j += j0;

        int pos[3];
        double path;
        double coldensh_in;                                // Column density to the cell
        double nHI_p;                                      // Local density of neutral hydrogen in the cell
        double xh_av_p;                                    // Local ionization fraction of cell

        #if defined(RATES)
        double xs, ys, zs;
        double dist2;
        double vol_ph;
        #endif

        if (in_box_gpu(i,j,k,m1))
        {
            pos[0] = modulo_gpu(i,m1);
            pos[1] = modulo_gpu(j,m1);
            pos[2] = modulo_gpu(k,m1);

            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);

            if (coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] == 0.0)
            {
                if (i == i0 &&
                    j == j0 &&
                    k == k0)
                {
                    coldensh_in = 0.0;
                    path = 0.5*dr;
                    #if defined(RATES)
                    vol_ph = dr*dr*dr / (4*M_PI);
                    #endif
                }
                else
                {
                    cinterp_gpu(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out,sig,m1);
                    path *= dr;
                    #if defined(RATES)
                    // Find the distance to the source
                    xs = dr*(i-i0);
                    ys = dr*(j-j0);
                    zs = dr*(k-k0);
                    dist2=xs*xs+ys*ys+zs*zs;
                    vol_ph = dist2 * path;
                    #endif
                }
                coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] = coldensh_in + nHI_p * path;
                
                #if defined(RATES)
                if (coldensh_in <= MAX_COLDENSH)
                {
                    double phi = photoion_rate_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,nHI_p,sig);
                    phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi;
                }
                #endif
            }
        }
    }
}

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
    double* phi_dev,
    const int m1,
    const int d1,
    const int d2,
    const int d3)
{
    if (blockIdx.x <= q && threadIdx.x <= blockIdx.x)
    {   
        // integer :: nx,nd,idim                                         // loop counters (used in LLS)
        //std::vector<int> pos(3);                                     // RT position modulo periodicity
        int pos[3];
        //double xs,ys,zs;                                   // Distances between source and cell
        //double dist2,
        double path;
        //double vol_ph;                          // Distance parameters
        double coldensh_in;                                // Column density to the cell
        // bool stop_rad_transfer;                                    // Flag to stop column density when above max column density
        double nHI_p;                                      // Local density of neutral hydrogen in the cell
        double xh_av_p;                                    // Local ionization fraction of cell
        //double phi_ion_p;                                  // Local photoionization rate of cell (to be computed)
        //stop_rad_transfer = false;
        #if defined(RATES)
        double xs, ys, zs;
        double dist2;
        double vol_ph;
        #endif

        int k = k0 + d1*(q-blockIdx.x);
        int i = i0 + d2*(blockIdx.x - threadIdx.x);
        int j = j0 + d3*(blockIdx.x - (blockIdx.x - threadIdx.x));

        if (in_box_gpu(i,j,k,m1))
        {
            pos[0] = modulo_gpu(i,m1);
            pos[1] = modulo_gpu(j,m1);
            pos[2] = modulo_gpu(k,m1);

            //srcpos_p[0] = srcpos[0][ns];
            //srcpos_p[1] = srcpos[1][ns];
            //srcpos_p[2] = srcpos[2][ns];

            // xh_av_p = 1e-3;
            // nHI_p = (1.0 - xh_av_p);

            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);

            if (coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] == 0.0)
            {
                if (i == i0 &&
                    j == j0 &&
                    k == k0)
                {
                    coldensh_in = 0.0;
                    path = 0.5*dr;
                    #if defined(RATES)
                    vol_ph = dr*dr*dr / (4*M_PI);
                    #endif
                }
                else
                {
                    cinterp_gpu(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out,sig,m1);
                    path *= dr;
                    #if defined(RATES)
                    // Find the distance to the source
                    xs = dr*(i-i0);
                    ys = dr*(j-j0);
                    zs = dr*(k-k0);
                    dist2=xs*xs+ys*ys+zs*zs;
                    vol_ph = dist2 * path;
                    #endif
                }
                coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] = coldensh_in + nHI_p * path;
                
                #if defined(RATES)
                if (coldensh_in <= MAX_COLDENSH)
                {
                    double phi = photoion_rate_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,nHI_p,sig);
                    phi_dev[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi;
                }
                #endif
            }
        }
    }
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
//      if (idel == 0) sgni=0
    sgnj=sign_gpu(jdel);
//      if (jdel == 0) sgnj=0
    sgnk=sign_gpu(kdel);
//      if (kdel == 0) sgnk=0
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
        
        // ip =(i-1)  % m1 + 1;
        // imp=(im-1) % m1 + 1;
        // jp =(j-1)  % m1 + 1;
        // jmp=(jm-1) % m1 + 1;
        // kmp=(km-1) % m1 + 1;

        // ip =(i)  % m1;
        // imp=(im) % m1;
        // jp =(j)  % m1;
        // jmp=(jm) % m1;
        // kmp=(km) % m1;

        ip  = modulo_gpu(i  ,m1);
        imp = modulo_gpu(im ,m1);
        jp  = modulo_gpu(j  ,m1);
        jmp = modulo_gpu(jm ,m1);
        kmp = modulo_gpu(km ,m1);
        
        // std::cout << ip << " " << imp << " " <<  jp << " " <<  jmp << " " <<  kmp << " " <<  std::endl;
        c1=     coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)]; //coldensh_out[imp][jmp][kmp];    //# column densities at the
        c2=     coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)]; //coldensh_out[ip][jmp][kmp];     //# four corners
        c3=     coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)]; //coldensh_out[imp][jp][kmp];
        c4=     coldensh_out[mem_offst_gpu(ip,jp,kmp,m1)]; //coldensh_out[ip][jp][kmp];

        // extra weights for better fit to analytical solution
        w1=   s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2=   s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3=   s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4=   s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
        
        // column density at the crossing point
        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);
        // Take care of diagonals
        // if (kdela == idela||kdela == jdela) then
        // if (kdela == idela && kdela == jdela) then
        if (kdela == 1 && (idela == 1||jdela == 1)) {
        if (idela == 1 && jdela == 1) {
            cdensi=   1.73205080757*cdensi;
        }
        else{
            cdensi=   1.41421356237*cdensi;
        }
        }

        // Path length from c through d to other side cell.
        path=sqrt((di*di+dj*dj)/(dk*dk)+1.0); // pathlength from c to d point  

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

        // ip=(i-1)   % m1 +1;
        // imp=(im-1) % m1 +1;
        // jmp=(jm-1) % m1 +1;
        // kp=(k-1)   % m1 +1;
        // kmp=(km-1) % m1 +1;

        // ip=(i)   % m1;
        // imp=(im) % m1;
        // jmp=(jm) % m1;
        // kp=(k)   % m1;
        // kmp=(km) % m1;

        ip  = modulo_gpu(i,m1);
        imp = modulo_gpu(im,m1);
        jmp = modulo_gpu(jm,m1);
        kp  = modulo_gpu(k,m1);
        kmp = modulo_gpu(km,m1);
    

        //c1=  coldensh_out[imp][jmp][kmp];
        //c2=  coldensh_out[ip][jmp][kmp];
        //c3=  coldensh_out[imp][jmp][kp];
        //c4=  coldensh_out[ip][jmp][kp];

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
        if (jdela == 1 && (idela == 1||kdela == 1)) {
        if (idela == 1 && kdela == 1) {
            //write(logf,*) 'error',i,j,k
            cdensi=   1.73205080757*cdensi;
        }
        else{
            //write(logf,*) 'diagonal',i,j,k
            cdensi=   1.41421356237*cdensi;
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

        // imp=(im-1) % m1 +1;
        // jp= (j-1)  % m1 +1;
        // jmp=(jm-1) % m1 +1;
        // kp= (k-1)  % m1 +1;
        // kmp=(km-1) % m1 +1;

        // imp=(im) % m1;
        // jp= (j)  % m1;
        // jmp=(jm) % m1;
        // kp= (k)  % m1;
        // kmp=(km) % m1;

        imp=modulo_gpu(im ,m1);
        jp= modulo_gpu(j  ,m1);
        jmp=modulo_gpu(jm ,m1);
        kp= modulo_gpu(k  ,m1);
        kmp=modulo_gpu(km ,m1);

        //c1=  coldensh_out[imp][jmp][kmp];
        //c2=  coldensh_out[imp][jp][kmp];
        //c3=  coldensh_out[imp][jmp][kp];
        //c4=  coldensh_out[imp][jp][kp];

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

        if ( idela == 1  &&  ( jdela == 1 || kdela == 1 ) ) {
        if ( jdela == 1  &&  kdela == 1 ) {
            cdensi=   1.73205080757*cdensi;
        }
        else{
            cdensi   =1.41421356237*cdensi;
        }
        }
        path=sqrt(1.0+(dj*dj+dk*dk)/(di*di));
    }
}

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

    cudaMalloc(&cdh_dev,N*N*N*sizeof(double));
    cudaMalloc(&n_dev,N*N*N*sizeof(double));
    cudaMalloc(&x_dev,N*N*N*sizeof(double));
    cudaMalloc(&phi_dev,N*N*N*sizeof(double));
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Couldn't allocate memory" << std::endl;
    }
    else {
        std::cout << "Succesfully allocated device memory for grid of size N = " << N << std::endl;
    }
}

void device_close()
{   
    printf("Freeing device pointers...\n");
    cudaFree(&cdh_dev);
    cudaFree(&n_dev);
    cudaFree(&x_dev);
    cudaFree(&phi_dev);
}

__device__ double photoion_rate_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & nHI,const double & sig)
{
    // Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
    double tau_in = coldens_in * sig;
    double tau_out = coldens_out * sig;

    // If cell is optically thick
    if (fabs(tau_out - tau_in) > TAU_PHOTO_LIMIT)
        return strength * INV4PI / (Vfact * nHI) * (exp(-tau_in) - exp(-tau_out));
    // If cell is optically thin
    else
        return strength * INV4PI * sig * (tau_out - tau_in) / (Vfact) * exp(-tau_in);
}