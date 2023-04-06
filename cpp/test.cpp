#include <iostream>
#include "raytracing.hh"
#include <vector>
#include <gsl/gsl_statistics_double.h>

int main()
{   
    int N = 128;
    std::vector<std::vector<int>> srcpos(3,std::vector<int>(1));
    std::vector<std::vector<std::vector<double> > > coldensh_out(N,std::vector<std::vector<double> >(N,std::vector<double>(N)));
    std::vector<std::vector<std::vector<double> > > ndens(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1.0)));
    std::vector<std::vector<std::vector<double> > > phi_ion(N,std::vector<std::vector<double> >(N,std::vector<double>(N,0.0)));
    std::vector<std::vector<std::vector<double> > > xh_av(N,std::vector<std::vector<double> >(N,std::vector<double>(N,1e-3)));

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
    
    do_source_octa(srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);

    double mean;
    double* cdh = &coldensh_out[0][0][0]; //coldensh_out.data()->data()->data();

    //mean = gsl_stats_mean(cdh,1,N*N);
    //mean = coldensh_out[70][70][70];
    mean = cdh[878057];

    //std::cout << coldensh_out[0][0][0] << " " << coldensh_out[1][1][1] << " " << coldensh_out[2][2][2] << " " << std::endl;
    std::cout << mean << std::endl;
    return 0;
}