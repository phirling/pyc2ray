#include <vector>

inline int sign(const double & x) { if (x>=0) return 1; else return -1;}

inline int modulo(int a, int b) { return (a%b+b)%b; }
// inline long int modulo(long int a, long int b) { return (a%b+b)%b; }

void cinterp(const std::vector<int> & pos,const std::vector<int> & srcpos,
    double & cdensi,double & path,std::vector<std::vector<std::vector<double> > > & coldensh_out,
    const double & sigma_HI_at_ion_freq,const int & m1);

inline double weightf(const double &, const double &);


void evolve0D(
    const std::vector<int> & rtpos,
    const std::vector<int> & srcpos,
    const int & ns,
    std::vector<std::vector<std::vector<double> > > & coldensh_out,
    const double & sig,
    const double & dr,
    const std::vector<std::vector<std::vector<double> > > & ndens,
    const std::vector<std::vector<std::vector<double> > > & xh_av,
    std::vector<std::vector<std::vector<double> > > & phi_ion,
    const int & NumSrc,
    const int & m1);

    
void do_source_octa(const std::vector<std::vector<int> > & srcpos,
    const int & ns,
    std::vector<std::vector<std::vector<double> > > & coldensh_out,
    const double & sig,
    const double & dr,
    const std::vector<std::vector<std::vector<double> > > & ndens,
    const std::vector<std::vector<std::vector<double> > > & xh_av,
    std::vector<std::vector<std::vector<double> > > & phi_ion,
    const int & NumSrc,
    const int & m1);