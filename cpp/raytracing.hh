#include <vector>

void cinterp(const std::vector<int> & pos,const std::vector<int> & srcpos,
    double & cdensi,double & path,std::vector<std::vector<std::vector<double>>> & coldensh_out,
    const double & sigma_HI_at_ion_freq,const int & m1);

inline double weightf(const double &, const double &);

inline int sign(const double & x) { if (x>=0) return 1; else return -1;}

void evolve0D(
    const std::vector<int> & rtpos,
    const std::vector<std::vector<int>> & srcpos,
    const int & ns,
    std::vector<std::vector<std::vector<double>>> & coldensh_out,
    const double & sig,
    const double & dr,
    std::vector<std::vector<std::vector<double>>> & ndens,
    const std::vector<std::vector<std::vector<double>>> & xh_av,
    std::vector<std::vector<std::vector<double>>> & phi_ion,
    const int & NumSrc,
    const int & m1);