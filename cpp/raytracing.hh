#include <vector>

inline int sign(const double & x) { if (x>=0) return 1; else return -1;}

inline int modulo(int a, int b) { return (a%b+b)%b; }

inline int mem_offst(const int & i,const int & j,const int & k,const int & N)
{   
    return N*N*i + N*j + k;
}

void cinterp(
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

void evolve0D(
    const int i,
    const int j,
    const int k,
    const int i0,
    const int j0,
    const int k0,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1);

    
void do_source_octa(
    int* srcpos,
    const int & ns,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    double* ndens,
    double* xh_av,
    double* phi_ion,
    const int & NumSrc,
    const int & m1);