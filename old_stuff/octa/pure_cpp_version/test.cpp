#include <iostream>
#include "raytracing.hh"
#include <vector>
#include <gsl/gsl_statistics_double.h>

int main()
{   
    int N = 3;
    //std::vector<std::vector<int>> srcpos(3,std::vector<int>(1));
    // std::vector<std::vector<std::vector<double> > > coldensh_out(N,std::vector<std::vector<double> >(N,std::vector<double>(N)));
    // std::vector<std::vector<std::vector<double> > > ndens(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1.0)));
    // std::vector<std::vector<std::vector<double> > > phi_ion(N,std::vector<std::vector<double> >(N,std::vector<double>(N,0.0)));
    // std::vector<std::vector<std::vector<double> > > xh_av(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1e-3)));
    int NumSrc = 1;
    int* srcpos = (int*)malloc(3*NumSrc*sizeof(int));
    double* coldensh_out = (double*)calloc(N*N*N,sizeof(double));
    double* ndens = (double*)calloc(N*N*N,sizeof(double));
    double* phi_ion = (double*)calloc(N*N*N,sizeof(double));
    double* xh_av = (double*)calloc(N*N*N,sizeof(double));

    std::fill(coldensh_out,coldensh_out + N*N*N,0.0);
    std::fill(ndens,ndens + N*N*N,1.0);
    std::fill(phi_ion,phi_ion+ N*N*N,0.0);
    std::fill(xh_av,xh_av+ N*N*N,1e-3);

    std::cout << coldensh_out[26] << std::endl;

    int ns=0;
    srcpos[0]                = 1;      //srcpos[0][ns];
    srcpos[NumSrc + ns]      = 1;   //srcpos[1][ns];
    srcpos[2*NumSrc + ns]    = 1; //srcpos[2][ns];


    double sig = 1.0;
    double dr = 1.0;

    
    do_source_octa(srcpos,ns,1.8*N,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);

    double mean;

    mean = coldensh_out[mem_offst(2,2,1,N)];

    std::cout << mean << std::endl;
    return 0;
}