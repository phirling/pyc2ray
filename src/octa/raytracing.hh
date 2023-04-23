#include <vector>

inline int sign(const double & x) { if (x>=0) return 1; else return -1;}

inline int modulo(const int & a,const int & b) { return (a%b+b)%b; }

inline int mem_offst(const int & i,const int & j,const int & k,const int & N)
{   
    return N*N*i + N*j + k;
}

inline bool in_box(const int & i,const int & j,const int & k,const int & N)
{
    return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
}

double photoion_rate_test(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & nHI,const double & sig);

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
    const double & strength,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1);

    
void do_source_octa(
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

void all_sources_octa(
    int* srcpos,      // Position of all sources
    double* srcstrength,                                                     // Source number
    const double & R,
    double* coldensh_out,     // Outgoing column density
    const double & sig,                                                 // Cross section
    const double & dr,                                                  // Cell size
    double* ndens,      // Hydrogen number density
    double* xh_av,      // Time-average ionization fraction
    double* phi_ion,          // Ionization Rates
    const int & NumSrc,                                                 // Number of sources
    const int & m1);