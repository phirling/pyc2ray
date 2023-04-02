#include <iostream>
#include <cmath>
#include "raytracing.hh"

static const double sqrt2 = std::sqrt(2.0);
static const double sqrt3 = std::sqrt(3.0);
static const double minweight = 1.0/0.6;

void do_source_octa(const std::vector<std::vector<int> > & srcpos,      // Position of all sources
    const int & ns,                                                     // Source number
    std::vector<std::vector<std::vector<double> > > & coldensh_out,     // Outgoing column density
    const double & sig,                                                 // Cross section
    const double & dr,                                                  // Cell size
    const std::vector<std::vector<std::vector<double> > > & ndens,      // Hydrogen number density
    const std::vector<std::vector<std::vector<double> > > & xh_av,      // Time-average ionization fraction
    std::vector<std::vector<std::vector<double> > > & phi_ion,          // Ionization Rates
    const int & NumSrc,                                                 // Number of sources
    const int & m1)                                                     // Mesh size
    {   
        // Source position
        std::vector<int> srcpos_p = {srcpos[0][ns], srcpos[1][ns], srcpos[2][ns]};

        // First, do the source cell
        std::vector<int> rtpos = {srcpos_p[0],srcpos_p[1],srcpos_p[2]};
        evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

        // Sweep the grid by treating the faces of octahedra of increasing size.
        int max_r = std::ceil(1.5 * m1);
        for (int r=1 ; r <= max_r; r++)
        {
            for (int k = 0; k <= r; k++)
            {   
                //std::cout << "k=" << k << std::endl;
                for (int j = 0; j <= k; j++)
                {   
                    rtpos[2] = srcpos_p[2] + (r-k);
                    rtpos[0] = srcpos_p[0] + (k-j);
                    rtpos[1] = srcpos_p[1] + (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] + (r-k);
                    rtpos[0] = srcpos_p[0] - (k-j);
                    rtpos[1] = srcpos_p[1] + (k-(k-j));
                    //std::cout << rtpos[0] << " " << rtpos[1] << " " << rtpos[2] << std::endl;
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] + (r-k);
                    rtpos[0] = srcpos_p[0] + (k-j);
                    rtpos[1] = srcpos_p[1] - (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] + (r-k);
                    rtpos[0] = srcpos_p[0] - (k-j);
                    rtpos[1] = srcpos_p[1] - (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] - (r-k);
                    rtpos[0] = srcpos_p[0] + (k-j);
                    rtpos[1] = srcpos_p[1] + (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] - (r-k);
                    rtpos[0] = srcpos_p[0] - (k-j);
                    rtpos[1] = srcpos_p[1] + (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] - (r-k);
                    rtpos[0] = srcpos_p[0] + (k-j);
                    rtpos[1] = srcpos_p[1] - (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);

                    rtpos[2] = srcpos_p[2] - (r-k);
                    rtpos[0] = srcpos_p[0] - (k-j);
                    rtpos[1] = srcpos_p[1] - (k-(k-j));
                    evolve0D(rtpos,srcpos_p,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1);
                }
            }
        }
    }
    
void evolve0D(
    const std::vector<int> & rtpos,
    const std::vector<int> & srcpos_p,
    const int & ns,
    std::vector<std::vector<std::vector<double> > > & coldensh_out,
    const double & sig,
    const double & dr,
    const std::vector<std::vector<std::vector<double> > > & ndens,
    const std::vector<std::vector<std::vector<double> > > & xh_av,
    std::vector<std::vector<std::vector<double> > > & phi_ion,
    const int & NumSrc,
    const int & m1)
{
    // integer :: nx,nd,idim                                         // loop counters (used in LLS)
    //std::vector<int> pos(3);                                     // RT position modulo periodicity
    int pos[3];
    double xs,ys,zs;                                   // Distances between source and cell
    double dist2,path,vol_ph;                          // Distance parameters
    double coldensh_in;                                // Column density to the cell
    bool stop_rad_transfer;                                    // Flag to stop column density when above max column density
    double nHI_p;                                      // Local density of neutral hydrogen in the cell
    double xh_av_p;                                    // Local ionization fraction of cell
    double phi_ion_p;                                  // Local photoionization rate of cell (to be computed)
    stop_rad_transfer = false;

    // Map pos to mesh pos, assuming a periodic mesh
    // pos[0] = (rtpos[0] -1) % m1 + 1;
    // pos[1] = (rtpos[1] -1) % m1 + 1;
    // pos[2] = (rtpos[2] -1) % m1 + 1;

    // pos[0] = (rtpos[0]) % m1;
    // pos[1] = (rtpos[1]) % m1;
    // pos[2] = (rtpos[2]) % m1;

    pos[0] = modulo(rtpos[0],m1);
    pos[1] = modulo(rtpos[1],m1);
    pos[2] = modulo(rtpos[2],m1);

    //srcpos_p[0] = srcpos[0][ns];
    //srcpos_p[1] = srcpos[1][ns];
    //srcpos_p[2] = srcpos[2][ns];

    xh_av_p = xh_av[pos[0]][pos[1]][pos[2]];
    nHI_p = ndens[pos[0]][pos[1]][pos[2]] * (1.0 - xh_av_p);

    if (coldensh_out[pos[0]][pos[1]][pos[2]] == 0.0)
    {
        if (rtpos[0] == srcpos_p[0] &&
            rtpos[1] == srcpos_p[1] &&
            rtpos[2] == srcpos_p[2])
        {
            coldensh_in = 0.0;
            path = 0.5*dr;
            // std::cout << path << std::endl;
            vol_ph = dr*dr*dr / (4*M_PI);
        }
        else
        {
            cinterp(rtpos,srcpos_p,coldensh_in,path,coldensh_out,sig,m1);
            path *= dr;
        }
        // std::cout << coldensh_in << "    " << path << std::endl;
        coldensh_out[pos[0]][pos[1]][pos[2]] = coldensh_in + nHI_p * path;
    }
}

void cinterp(const std::vector<int> & pos,const std::vector<int> & srcpos,
    double & cdensi,double & path,std::vector<std::vector<std::vector<double> > > & coldensh_out,
    const double & sigma_HI_at_ion_freq,const int & m1)
{
    int i,j,k,i0,j0,k0;

    int idel,jdel,kdel;
    int idela,jdela,kdela;
    int im,jm,km;
    unsigned int ip,imp,jp,jmp,kp,kmp;
    int sgni,sgnj,sgnk;
    double alam,xc,yc,zc,dx,dy,dz,s1,s2,s3,s4;
    double c1,c2,c3,c4;
    double w1,w2,w3,w4;
    double di,dj,dk;


    // map to local variables (should be pointers ;)
    i=pos[0];        // + 1 if using with python, zero-indexing
    j=pos[1];        // + 1 if using with python, zero-indexing
    k=pos[2];        // + 1 if using with python, zero-indexing
    i0=srcpos[0];    // + 1 if using with python, zero-indexing
    j0=srcpos[1];    // + 1 if using with python, zero-indexing
    k0=srcpos[2];    // + 1 if using with python, zero-indexing

    // calculate the distance between the source point (i0,j0,k0) and 
    // the destination point (i,j,k)
    idel=i-i0;
    jdel=j-j0;
    kdel=k-k0;
    idela=abs(idel);
    jdela=abs(jdel);
    kdela=abs(kdel);
    
    // Find coordinates of points closer to source
    sgni=sign(idel);
//      if (idel == 0) sgni=0
    sgnj=sign(jdel);
//      if (jdel == 0) sgnj=0
    sgnk=sign(kdel);
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

        ip  = modulo(i  ,m1);
        imp = modulo(im ,m1);
        jp  = modulo(j  ,m1);
        jmp = modulo(jm ,m1);
        kmp = modulo(km ,m1);
        
        // std::cout << ip << " " << imp << " " <<  jp << " " <<  jmp << " " <<  kmp << " " <<  std::endl;
        c1=     coldensh_out[imp][jmp][kmp];    //# column densities at the
        c2=     coldensh_out[ip][jmp][kmp];     //# four corners
        c3=     coldensh_out[imp][jp][kmp];
        c4=     coldensh_out[ip][jp][kmp];

        // extra weights for better fit to analytical solution
        w1=   s1*weightf(c1,sigma_HI_at_ion_freq);
        w2=   s2*weightf(c2,sigma_HI_at_ion_freq);
        w3=   s3*weightf(c3,sigma_HI_at_ion_freq);
        w4=   s4*weightf(c4,sigma_HI_at_ion_freq);
        
        // column density at the crossing point
        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);
        // Take care of diagonals
        // if (kdela == idela||kdela == jdela) then
        // if (kdela == idela && kdela == jdela) then
        if (kdela == 1 && (idela == 1||jdela == 1)) {
        if (idela == 1 && jdela == 1) {
            cdensi=   sqrt3*cdensi;
        }
        else{
            cdensi=   sqrt2*cdensi;
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

        ip  = modulo(i,m1);
        imp = modulo(im,m1);
        jmp = modulo(jm,m1);
        kp  = modulo(k,m1);
        kmp = modulo(km,m1);
    

        c1=  coldensh_out[imp][jmp][kmp];
        c2=  coldensh_out[ip][jmp][kmp];
        c3=  coldensh_out[imp][jmp][kp];
        c4=  coldensh_out[ip][jmp][kp];

        // extra weights for better fit to analytical solution
        w1=s1*weightf(c1,sigma_HI_at_ion_freq);
        w2=s2*weightf(c2,sigma_HI_at_ion_freq);
        w3=s3*weightf(c3,sigma_HI_at_ion_freq);
        w4=s4*weightf(c4,sigma_HI_at_ion_freq);

        cdensi=   (c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (jdela == 1 && (idela == 1||kdela == 1)) {
        if (idela == 1 && kdela == 1) {
            //write(logf,*) 'error',i,j,k
            cdensi=   sqrt3*cdensi;
        }
        else{
            //write(logf,*) 'diagonal',i,j,k
            cdensi=   sqrt2*cdensi;
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

        imp=modulo(im ,m1);
        jp= modulo(j  ,m1);
        jmp=modulo(jm ,m1);
        kp= modulo(k  ,m1);
        kmp=modulo(km ,m1);

        c1=  coldensh_out[imp][jmp][kmp];
        c2=  coldensh_out[imp][jp][kmp];
        c3=  coldensh_out[imp][jmp][kp];
        c4=  coldensh_out[imp][jp][kp];

        // extra weights for better fit to analytical solution
        w1   =s1*weightf(c1,sigma_HI_at_ion_freq);
        w2   =s2*weightf(c2,sigma_HI_at_ion_freq);
        w3   =s3*weightf(c3,sigma_HI_at_ion_freq);
        w4   =s4*weightf(c4,sigma_HI_at_ion_freq);

        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        if ( idela == 1  &&  ( jdela == 1 || kdela == 1 ) ) {
        if ( jdela == 1  &&  kdela == 1 ) {
            cdensi=   sqrt3*cdensi;
        }
        else{
            cdensi   =sqrt2*cdensi;
        }
        }
        path=sqrt(1.0+(dj*dj+dk*dk)/(di*di));
    }
}

inline double weightf(const double & cd, const double & sig)
{
    return 1.0/std::max(0.6,cd*sig);
}