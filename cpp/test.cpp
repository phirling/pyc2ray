#include <iostream>
#include "raytracing.hh"
#include <vector>

int main()
{   
    int N = 128;
    std::vector<std::vector<int>> srcpos(3,std::vector<int>(1));
    std::vector<std::vector<std::vector<double>>> coldensh_out(N,std::vector<std::vector<double>>(N,std::vector<double>(N)));
    std::vector<std::vector<std::vector<double>>> ndens(N,std::vector<std::vector<double>>(N,std::vector<double>(N,1.0)));
    std::vector<std::vector<std::vector<double>>> phi_ion(N,std::vector<std::vector<double>>(N,std::vector<double>(N,0.0)));
    std::vector<std::vector<std::vector<double>>> xh_av(N,std::vector<std::vector<double>>(N,std::vector<double>(N,1e-3)));

    srcpos[0][0] = 63;
    srcpos[1][0] = 63;
    srcpos[2][0] = 63;

    int ns=1;
    int NumSrc = 1;

    std::vector<int> rtpos = {63,63,63};

    double sig = 1.0;
    double dr = 1.0;

    evolve0D(rtpos,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,N);

    std::cout << coldensh_out[63][63][63] << std::endl;
    
    return 0;
}