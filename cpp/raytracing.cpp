#include <iostream>
#include <cmath>
#include "raytracing.hh"

static const double sqrt2 = std::sqrt(2.0);
static const double sqrt3 = std::sqrt(3.0);
static const double minweight = 1.0/0.6;

inline double weightf(const double & cd, const double & sig)
{
    return 1.0/fmax(0.6,cd*sig);
}












void do_source_octa(
    int* srcpos,      // Position of all sources
    const int & ns,                                                     // Source number
    const double & R,
    double* coldensh_out,     // Outgoing column density
    const double & sig,                                                 // Cross section
    const double & dr,                                                  // Cell size
    double* ndens,      // Hydrogen number density
    double* xh_av,      // Time-average ionization fraction
    double* phi_ion,          // Ionization Rates
    const int & NumSrc,                                                 // Number of sources
    const int & m1)                                                     // Mesh size
    {   
        // First, do the source cell
        int i0 = srcpos[ns];            //srcpos[0][ns];
        int j0 = srcpos[NumSrc + ns];   //srcpos[1][ns];
        int k0 = srcpos[2*NumSrc + ns]; //srcpos[2][ns];
        int i = i0;
        int j = j0;
        int k = k0;

        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);
        // Sweep the grid by treating the faces of octahedra of increasing size.
        int max_q =  std::ceil(sqrt3 * R); // std::ceil(1.5 * m1); //
        for (int q=1 ; q <= max_q; q++)
        {   
            //printf("r = %i \n",r);
            for (int s = 0; s <= q; s++)
            {   
                for (int t = 0; t <= s; t++)
                {   
                    //std::cout << i0 << j0 << k0 << std::endl;
                    k = k0 + (q-s);
                    i = i0 + (s-t);
                    j = j0 + (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);

                    k = k0 + (q-s);
                    i = i0 - (s-t);
                    j = j0 + (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);

                    k = k0 + (q-s);
                    i = i0 + (s-t);
                    j = j0 - (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);

                    k = k0 + (q-s);
                    i = i0 - (s-t);
                    j = j0 - (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);


                    k = k0 - (q-s);
                    i = i0 + (s-t);
                    j = j0 + (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);

                    k = k0 - (q-s);
                    i = i0 - (s-t);
                    j = j0 + (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);

                    k = k0 - (q-s);
                    i = i0 + (s-t);
                    j = j0 - (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);

                    k = k0 - (q-s);
                    i = i0 - (s-t);
                    j = j0 - (s-(s-t));
                    if (in_box(i,j,k,m1))
                        evolve0D(i,j,k,i0,j0,k0,coldensh_out,sig,dr,ndens,xh_av,phi_ion,m1);
                }   
            }
        }
    }
    
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
    const int m1)
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

    pos[0] = modulo(i,m1);
    pos[1] = modulo(j,m1);
    pos[2] = modulo(k,m1);

    //srcpos_p[0] = srcpos[0][ns];
    //srcpos_p[1] = srcpos[1][ns];
    //srcpos_p[2] = srcpos[2][ns];

    //xh_av_p = 1e-3;
    //nHI_p = (1.0 - xh_av_p);

    xh_av_p = xh_av[mem_offst(pos[0],pos[1],pos[2],m1)];
    nHI_p = ndens[mem_offst(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);
    

    if (coldensh_out[mem_offst(pos[0],pos[1],pos[2],m1)] == 0.0)
    {
        if (i == i0 &&
            j == j0 &&
            k == k0)
        {
            coldensh_in = 0.0;
            path = 0.5*dr;
            // std::cout << path << std::endl;
            //vol_ph = dr*dr*dr / (4*M_PI);
        }
        else
        {
            //printf("%i %i %i %i %i %i %f %f %f %i\n",i,j,k,i0,j0,k0,coldensh_in,path,sig,m1);
            //printf("mod=%f \n",weightf(mem_offst(i,j,k,m1),sig));
            cinterp(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out,sig,m1);
            //printf("%f \n",path);
            path *= dr;
        }
        // std::cout << coldensh_in << "    " << path << std::endl
        coldensh_out[mem_offst(pos[0],pos[1],pos[2],m1)] = coldensh_in + nHI_p * path;
        //printf("%i ",mem_offst(pos[0],pos[1],pos[2],m1));
        //coldensh_out[mem_offst(pos[0],pos[1],pos[2],m1)] = coldensh_in;
        //phi_ion[0] = 1.0;
    }
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
    idela=std::abs(idel);
    jdela=std::abs(jdel);
    kdela=std::abs(kdel);
    
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
        
        dx=2.0*std::abs(xc-(double(im)+0.5*sgni)); // distances from c-point to
        dy=2.0*std::abs(yc-(double(jm)+0.5*sgnj)); // the corners.
        
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
        c1=     coldensh_out[mem_offst(imp,jmp,kmp,m1)]; //coldensh_out[imp][jmp][kmp];    //# column densities at the
        c2=     coldensh_out[mem_offst(ip,jmp,kmp,m1)]; //coldensh_out[ip][jmp][kmp];     //# four corners
        c3=     coldensh_out[mem_offst(imp,jp,kmp,m1)]; //coldensh_out[imp][jp][kmp];
        c4=     coldensh_out[mem_offst(ip,jp,kmp,m1)]; //coldensh_out[ip][jp][kmp];

        // extra weights for better fit to analytical solution
        w1=   s1*weightf(c1,sigma_HI_at_ion_freq);
        w2=   s2*weightf(c2,sigma_HI_at_ion_freq);
        w3=   s3*weightf(c3,sigma_HI_at_ion_freq);
        w4=   s4*weightf(c4,sigma_HI_at_ion_freq);
        
        // column density at the crossing point
        //printf("%i %i %i \n",imp,jmp,kmp);
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
        dz=2.0*std::abs(zc-(double(km)+0.5*sgnk));
        dx=2.0*std::abs(xc-(double(im)+0.5*sgni));
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
    

        //c1=  coldensh_out[imp][jmp][kmp];
        //c2=  coldensh_out[ip][jmp][kmp];
        //c3=  coldensh_out[imp][jmp][kp];
        //c4=  coldensh_out[ip][jmp][kp];

        c1=  coldensh_out[mem_offst(imp,jmp,kmp,m1)];
        c2=  coldensh_out[mem_offst(ip,jmp,kmp,m1)];
        c3=  coldensh_out[mem_offst(imp,jmp,kp,m1)];
        c4=  coldensh_out[mem_offst(ip,jmp,kp,m1)];

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
        dz=2.0*std::abs(zc-(double(km)+0.5*sgnk));
        dy=2.0*std::abs(yc-(double(jm)+0.5*sgnj));
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

        //c1=  coldensh_out[imp][jmp][kmp];
        //c2=  coldensh_out[imp][jp][kmp];
        //c3=  coldensh_out[imp][jmp][kp];
        //c4=  coldensh_out[imp][jp][kp];

        c1=  coldensh_out[mem_offst(imp,jmp,kmp,m1)];
        c2=  coldensh_out[mem_offst(imp,jp,kmp,m1)];
        c3=  coldensh_out[mem_offst(imp,jmp,kp,m1)];
        c4=  coldensh_out[mem_offst(imp,jp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1   =s1*weightf(c1,sigma_HI_at_ion_freq);
        w2   =s2*weightf(c2,sigma_HI_at_ion_freq);
        w3   =s3*weightf(c3,sigma_HI_at_ion_freq);
        w4   =s4*weightf(c4,sigma_HI_at_ion_freq);

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