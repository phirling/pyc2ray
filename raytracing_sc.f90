! =================================== Note ==============================================
! In original c2ray, "phi" is a custom structure containing fields for
! the different rates. Here, simply an array (to work with f2py).
! Same goes for "ion", which in original has 3 fields: current, av, old
!
! These are the elements of "phi":
! 1.  photo_cell_HI          ! HI photoionization rate of the cell    
! 2.  heat_cell_HI           ! HI heating rate of the cell       
! 3.  photo_in_HI            ! HI photoionization rate incoming to the cell    
! 4.  heat_in_HI             ! HI heating rate incoming to the cell
! 5.  photo_out_HI           ! HI photoionization rate outgoing from the cell
! 6.  heat_out_HI            ! HI heating rate outgoing from the cell
! 7.  heat                   ! Total heating rate of the cell
! 8.  photo_in               ! Total photoionization rate incoming to the cell
! 9.  photo_out               ! Total photoionization rate incoming to the cell
!
!
! For "ion" CAUTION: 0-indexation is used for some reason.
! 0 indexes the fraction of neutral hydrogen "y",
! 1 indexes the fraction of ionized hydrogen "x"
! ( see ionfractions_module.f90 )
!
! Here, we'll simply use x, the ionized fraction, and compute y = 1-x whenever necessary
! =======================================================================================

module raytracing_sc

    !! Fortran Extension module for c2ray with f2py
    !! Based on column_density.f90, evolve_source.f90, evolve_point.f90 of original c2ray.
    !! Authors: G. Mellema, I. Illiev, ca 2013
    !! This version: adapted for test case only and usage with f2py.
    !! Author: P. Hirling, 2023

    use, intrinsic :: iso_fortran_env, only: real64 ! <-- This replaces the "dp" parameter in original c2ray (unpractical to use)
    implicit none

    real(kind=real64), parameter :: pi = 3.14159265358979323846264338_real64 ! Double precision pi
    real(kind=real64), parameter :: epsilon=1e-14_real64 ! Double precision very small number

    contains

    ! ===============================================================================================
    !! This subroutine computes the column density and ionization rate on the whole
    !! grid, for one source. The global rates of all sources are then added and applied
    !! using other subroutines that remain to be translated.
    ! ===============================================================================================
    subroutine evolve3D(dt,srcpos,ns,niter,last_l,last_r,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3)
        ! subroutine arguments
        real(kind=real64),intent(in) :: dt                              ! time step
        integer, intent(in) :: NumSrc                                   ! Number of sources
        integer,intent(in)      :: ns                                   ! source number 
        integer,intent(in)      :: niter                                ! global iteration number
        integer,intent(in) :: srcpos(3,NumSrc)                          ! positions of ALL sources (mesh)
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)                ! Hydrogen Density Field
        real(kind=real64), dimension(3), intent(in) :: dr               ! Cell size
        real(kind=real64),intent(inout) :: coldensh_out(m1,m2,m3)       ! Outgoing column density of the cells
        real(kind=real64),intent(in):: sig                              ! Hydrogen ionization cross section (sigma_HI_at_ion_freq)
        integer, intent(in) :: m1                                       ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                       ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                       ! mesh size z (hidden by f2py)
         ! TODO: deal with last_l and last_r
        integer,dimension(3), intent(in) :: last_l                      ! mesh position of left end point for RT
        integer,dimension(3), intent(in) :: last_r                      ! mesh position of right end point for RT        

        ! TODO: add OpenMP parallelization at the Fortran level.
        ! This should work with f2py, see https://stackoverflow.com/questions/46505778/f2py-with-openmp-gives-import-error-in-python

        ! If no OpenMP, traverse mesh plane by plane in the z direction up/down from the source

        integer :: k  ! z-coord of plane
        ! 1. transfer in the upper part of the grid (above srcpos(3))
        do k=srcpos(3,ns),last_r(3)
            call evolve2D(dt,k,srcpos,ns,niter,last_l,last_r,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3)
        end do

        ! 2. transfer in the lower part of the grid (below srcpos(3))
        do k=srcpos(3,ns)-1,last_l(3),-1
            call evolve2D(dt,k,srcpos,ns,niter,last_l,last_r,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3)
        end do

    end subroutine evolve3D


    ! ===============================================================================================
    !! This subroutine does the short characteristics for a whole plane at constant z
    !! (specified by argument k). This of course assumes that the previous plane has
    !! already been done.
    ! ===============================================================================================
    subroutine evolve2D(dt,k,srcpos,ns,niter,last_l,last_r,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3)
        ! subroutine arguments
        real(kind=real64),intent(in) :: dt                              ! time step
        integer, intent(in) :: NumSrc                                   ! Number of sources
        integer,intent(in)      :: ns                                   ! source number 
        integer,intent(in)      :: niter                                ! global iteration number
        integer, intent(in) :: k                                        ! z-coord of plane
        integer,intent(in) :: srcpos(3,NumSrc)                          ! positions of ALL sources (mesh)
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)                ! Hydrogen Density Field
        real(kind=real64), dimension(3), intent(in) :: dr               ! Cell size
        real(kind=real64),intent(inout) :: coldensh_out(m1,m2,m3)       ! Outgoing column density of the cells
        real(kind=real64),intent(in):: sig                              ! Hydrogen ionization cross section (sigma_HI_at_ion_freq)
        integer, intent(in) :: m1                                       ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                       ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                       ! mesh size z (hidden by f2py)
         ! TODO: deal with last_l and last_r
        integer,dimension(3), intent(in) :: last_l                      ! mesh position of left end point for RT
        integer,dimension(3), intent(in) :: last_r                      ! mesh position of right end point for RT


        integer,dimension(3) :: rtpos    ! cell position (for RT)
        integer :: i,j                   ! mesh positions

        rtpos(3) = k
        ! sweep in `positive' j direction
        do j=srcpos(2,ns),last_r(2)
            rtpos(2)=j
            do i=srcpos(1,ns),last_r(1)
                rtpos(1)=i
                call evolve0D(dt,rtpos,srcpos,ns,niter,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3) ! `positive' i
            end do
            do i=srcpos(1,ns)-1,last_l(1),-1
                rtpos(1)=i
                call evolve0D(dt,rtpos,srcpos,ns,niter,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3) ! `negative' i
            end do
        end do

        ! sweep in `negative' j direction
        do j=srcpos(2,ns)-1,last_l(2),-1
            rtpos(2)=j
            do i=srcpos(1,ns),last_r(1)
                rtpos(1)=i
                call evolve0D(dt,rtpos,srcpos,ns,niter,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3) ! `positive' i
            end do
            do i=srcpos(1,ns)-1,last_l(1),-1
                rtpos(1)=i
                call evolve0D(dt,rtpos,srcpos,ns,niter,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3) ! `negative' i
            end do
        end do
    end subroutine evolve2D


    ! ===============================================================================================
    !! Does the short characteristics for one cell and a single source. Has to be called in the correct
    !! order by the parent routines evolve2D and evolve3D.
    ! ===============================================================================================
    subroutine evolve0D(dt,rtpos,srcpos,ns,niter,coldensh_out,sig,dr,ndens,NumSrc,m1,m2,m3)
    
        ! This version (2023) modified for use with f2py (P. Hirling)

        ! Original: pass in number of source and fetch data (position, strength,etc)
        ! in global arrays using source number as index.

        ! Note for multiple sources
        ! We call this routine for every grid point and for every source (ns).
        ! The photo-ionization rates for each grid point are found and added
        ! to phih_grid, but the ionization fractions are not updated.

        ! TODO: deal with rates (phi), LLS
        use, intrinsic :: iso_fortran_env, only: real64
        ! use cinterp_sc, only: cinterp
        ! column density for stopping chemistry !***how should this criterion be for including he and more than one freq bands?
        ! for the moment, leave it as it is, it's probably ok. 
        real(kind=real64),parameter :: max_coldensh=2e19 !2e29!2.0e22_real64!2e19_real64 

        ! subroutine arguments
        real(kind=real64),intent(in) :: dt                              ! time step
        integer, intent(in) :: NumSrc                                   ! Number of sources
        integer,dimension(3),intent(in) :: rtpos                        ! cell position (for RT)
        integer,intent(in)      :: ns                                   ! source number 
        integer,intent(in)      :: niter                                ! global iteration number
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)                ! Hydrogen Density Field
        integer,intent(in) :: srcpos(3,NumSrc)                          ! positions of ALL sources (mesh)
        real(kind=real64), dimension(3), intent(in) :: dr               ! Cell size
        real(kind=real64),intent(inout) :: coldensh_out(m1,m2,m3)       ! Outgoing column density of the cells
        real(kind=real64),intent(in):: sig                              ! Hydrogen ionization cross section (sigma_HI_at_ion_freq)
        integer, intent(in) :: m1                                       ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                       ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                       ! mesh size z (hidden by f2py)
        ! real(kind=real64),intent(inout) :: xh_av(m1,m2,m3) ! Time-averaged HI fraction

        integer :: nx,nd,idim ! loop counters
        integer,dimension(3) :: pos
        real(kind=real64) :: dist2,path,vol_ph
        real(kind=real64) :: xs,ys,zs
        real(kind=real64) :: coldensh_in
        logical :: stop_rad_transfer
        real(kind=real64) :: ndens_p    ! Local density of cell
        

        ! type(photrates) :: phi, dummiphi  ----> Replace by array
        ! type(ionstates) :: ion            ----> Not actually used here since chemistry is done at a later time (see line 203 of evolve_point.f90)
        real(kind=real64), dimension(9) :: phi ! Ionization & Heating rates
        

        ! Reset check on radiative transfer
        stop_rad_transfer=.false.

        ! Map pos to mesh pos, assuming a periodic mesh
        pos(1) = modulo(rtpos(1) -1,m1) + 1
        pos(2) = modulo(rtpos(2) -1,m2) + 1
        pos(3) = modulo(rtpos(3) -1,m3) + 1

        ndens_p = ndens(pos(1),pos(2),pos(3))

        ! If coldensh_out is zero, we have not done this point
        ! yet, so do it. Otherwise do nothing. (grid is set to 0 for every source)
        if (coldensh_out(pos(1),pos(2),pos(3)) == 0.0) then
            ! Find the column density at the entrance point of the cell (short
            ! characteristics)
            if ( all( rtpos(:) == srcpos(:,ns) ) ) then
                ! Do not call cinterp for the source point.
                ! Set coldensh and path by hand
                coldensh_in=0.0
                path=0.5*dr(1)

                ! Find the distance to the source (average?)
                !dist2=0.5*dr(1) !NOT NEEDED         ! this makes vol=dx*dy*dz
                !vol_ph=4.0/3.0*pi*dist2**3
                vol_ph=dr(1)*dr(2)*dr(3)
                
            else
                ! For all other points call cinterp to find the column density
                call cinterp(rtpos,srcpos(:,ns),coldensh_in,path,coldensh_out,sig,m1,m2,m3)

                path=path*dr(1)

                ! Find the distance to the source
                xs=dr(1)*real(rtpos(1)-srcpos(1,ns))
                ys=dr(2)*real(rtpos(2)-srcpos(2,ns))
                zs=dr(3)*real(rtpos(3)-srcpos(3,ns))
                dist2=xs*xs+ys*ys+zs*zs

                ! Find the volume of the shell this cell is part of 
                ! (dilution factor).
                vol_ph=4.0*pi*dist2*path

                ! Add LLS opacity TODO
                ! Initialize local LLS (if type of LLS is appropriate)
                ! --> if (use_LLS) then
                ! -->     if (type_of_LLS == 3) then
                ! -->         ! Set the flag for stopping radiative transfer if the
                ! -->         ! distance is larger than the maximum distance set by
                ! -->         ! the LLS model.
                ! -->         if (dist2 > R_max_LLS*R_max_LLS) stop_rad_transfer=.true.
                ! -->         else
                ! -->         if (type_of_LLS == 2) call LLS_point (pos(1),pos(2),pos(3))
                ! -->         coldensh_in = coldensh_in + coldensh_LLS * path/dr(1)
                ! -->         endif
                ! -->     endif          
                ! --> endif

                ! Set the flag for no further radiative transfer if the ingoing column
                ! density is above the maximum allowed value
                if (coldensh_in > max_coldensh) stop_rad_transfer=.true.
                ! Only ray trace and exit. Do not touch the ionization
                ! fractions. They are updated using phih_grid in evolve0d_global.
            endif

            ! Add the (time averaged) column density of this cell
            ! to the total column density (for this source)
            ! and add the LLS column density to this.
            ! GM/110224: No! This messes up phi since phi is based
            !  upon the difference between the in and out column density.
            !  Instead add the LLS to coldensh_in, see above
            coldensh_out(pos(1),pos(2),pos(3))=coldensh_in + ndens_p*path ! coldens(path,1.0_real64,ndens_p) ! TODO: replace 1.0 by actual xh_av

            ! Calculate (photon-conserving) photo-ionization rate from the
            ! column densities.

            ! Limit the calculation to a certain maximum column density (hydrogen)
            ! --> if (.not.stop_rad_transfer) then 
            ! -->     ! photoion_rates the structure of rates (photo and heating)
            ! -->     phi=photoion_rates(coldensh_in,coldensh_out(pos(1),pos(2),pos(3)), &
            ! -->         vol_ph,ns,ion%h_av(1))
            ! -->     ! Divide the photo-ionization rates by the appropriate neutral density
            ! -->     ! (part of the photon-conserving rate prescription)
            ! -->     phi%photo_cell_HI=phi%photo_cell_HI/(ion%h_av(0)*ndens_p)
            ! -->     
            ! -->     ! Calculate the losses due to LLSs. TODO
            ! -->     ! -->  if (use_LLS) call total_LLS_loss(phi%photo_in_HI*vol/vol_ph, &
            ! -->     ! -->      coldensh_LLS * path/dr(1))          
            ! --> else
            ! -->     ! If the H0 column density is above the maximum or the R_max
            ! -->     ! condition from the LLS model is triggered, set rates to zero
            ! -->     phi%photo_cell_HI = 0.0_real64
            ! -->     phi%photo_out_HI = 0.0_real64
            ! -->     phi%heat = 0.0_real64
            ! -->     phi%photo_in = 0.0_real64
            ! -->     phi%photo_out = 0.0_real64
            ! --> endif

            ! Add photo-ionization rate to the global array 
            ! (this array is applied in evolve0D_global)
            ! --> phih_grid(pos(1),pos(2),pos(3))= &
            ! -->     phih_grid(pos(1),pos(2),pos(3))+phi%photo_cell_HI
            ! --> if (.not. isothermal) &
            ! -->     phiheat(pos(1),pos(2),pos(3))=phiheat(pos(1),pos(2),pos(3))+phi%heat
            ! --> ! Photon statistics: register number of photons leaving the grid
            ! --> ! Note: This is only the H0 photo-ionization rate
            ! --> if ( (any(rtpos(:) == last_l(:))) .or. &
            ! -->     (any(rtpos(:) == last_r(:))) ) then
            ! -->     photon_loss_src_thread(tn)=photon_loss_src_thread(tn) + &
            ! -->         phi%photo_out*vol/vol_ph
            ! -->     !photon_loss_src(1,tn)=photon_loss_src(1,tn) + phi%h_out*vol/vol_ph
            ! --> endif
        endif ! end of coldens test
        
    end subroutine evolve0D
    

    ! ===============================================================================================
    !! Finds the column density at pos as seen from the source point srcpos
    !! through interpolation. The interpolation
    !! depends on the orientation of the ray. The ray crosses either
    !! a z-plane, a y-plane or an x-plane.
    ! ===============================================================================================
    subroutine cinterp (pos,srcpos,cdensi,path,coldensh_out,sigma_HI_at_ion_freq,m1,m2,m3)

        ! Author: Garrelt Mellema
        ! Date: 21-Mar-2006 (06-Aug-2004)
        ! History:
        ! Original routine written by Alex Raga, Garrelt Mellema, Jane Arthur
        ! and Wolfgang Steffen in 1999.
        ! Modified for use with a grid based approach.
        ! Better handling of the diagonals.
        ! Fortran90
        ! This version (2023) modified for use with f2py (P. Hirling)

        ! does the interpolation to find the column density at pos
        ! as seen from the source point srcpos. the interpolation
        ! depends on the orientation of the ray. The ray crosses either
        ! a z-plane, a y-plane or an x-plane.
        implicit none
            !! phirling:
            !! Some global variables in original c2ray (e.g. coldensh_out)
            !! have been replaced by arguments for f2py-usage
            !! commented bits of code have been deleted
            integer,dimension(3),intent(in) :: pos !< cell position (mesh)
            integer,dimension(3),intent(in) :: srcpos !< source position (mesh)
            real(kind=real64),intent(out) :: cdensi !< column density to cell
            real(kind=real64),intent(out) :: path !< path length over cell
            real(kind=real64),intent(inout) :: coldensh_out(m1,m2,m3) !< Outgoing column density of the cells
            real(kind=real64),intent(in):: sigma_HI_at_ion_freq !< Hydrogen ionization cross section
            integer, intent(in) :: m1 ! < mesh size x (hidden by f2py)
            integer, intent(in) :: m2 ! < mesh size y (hidden by f2py)
            integer, intent(in) :: m3 ! < mesh size z (hidden by f2py)
            real(kind=real64),parameter :: sqrt3=sqrt(3.0)
            real(kind=real64),parameter :: sqrt2=sqrt(2.0)
            
            integer :: i,j,k,i0,j0,k0

            integer :: idel,jdel,kdel
            integer :: idela,jdela,kdela
            integer :: im,jm,km
            integer :: ip,imp,jp,jmp,kp,kmp
            integer :: sgni,sgnj,sgnk
            real(kind=real64) :: alam,xc,yc,zc,dx,dy,dz,s1,s2,s3,s4
            real(kind=real64) :: c1,c2,c3,c4
            real(kind=real64) :: w1,w2,w3,w4
            real(kind=real64) :: di,dj,dk


            !!!!!!!DEC$ ATTRIBUTES FORCEINLINE :: weightf
            ! map to local variables (should be pointers ;)
            i=pos(1)        ! + 1 if using with python, zero-indexing
            j=pos(2)        ! + 1 if using with python, zero-indexing
            k=pos(3)        ! + 1 if using with python, zero-indexing
            i0=srcpos(1)    ! + 1 if using with python, zero-indexing
            j0=srcpos(2)    ! + 1 if using with python, zero-indexing
            k0=srcpos(3)    ! + 1 if using with python, zero-indexing

            ! calculate the distance between the source point (i0,j0,k0) and 
            ! the destination point (i,j,k)
            idel=i-i0
            jdel=j-j0
            kdel=k-k0
            idela=abs(idel)
            jdela=abs(jdel)
            kdela=abs(kdel)
            
            ! Find coordinates of points closer to source
            sgni=sign(1,idel)
        !      if (idel == 0) sgni=0
            sgnj=sign(1,jdel)
        !      if (jdel == 0) sgnj=0
            sgnk=sign(1,kdel)
        !      if (kdel == 0) sgnk=0
            im=i-sgni
            jm=j-sgnj
            km=k-sgnk
            di=real(idel)
            dj=real(jdel)
            dk=real(kdel)

            ! Z plane (bottom and top face) crossing
            ! we find the central (c) point (xc,xy) where the ray crosses 
            ! the z-plane below or above the destination (d) point, find the 
            ! column density there through interpolation, and add the contribution
            ! of the neutral material between the c-point and the destination
            ! point.
            if (kdela >= jdela.and.kdela >= idela) then
                ! alam is the parameter which expresses distance along the line s to d
                ! add 0.5 to get to the interface of the d cell.
                alam=(real(km-k0)+sgnk*0.5)/dk
                    
                xc=alam*di+real(i0) ! x of crossing point on z-plane 
                yc=alam*dj+real(j0) ! y of crossing point on z-plane
                
                dx=2.0*abs(xc-(real(im)+0.5*sgni)) ! distances from c-point to
                dy=2.0*abs(yc-(real(jm)+0.5*sgnj)) ! the corners.
                
                s1=(1.-dx)*(1.-dy)    ! interpolation weights of
                s2=(1.-dy)*dx         ! corner points to c-point
                s3=(1.-dx)*dy
                s4=dx*dy
                
                ip =modulo(i-1, m1)+1
                imp=modulo(im-1,m1)+1
                jp =modulo(j-1, m2)+1
                jmp=modulo(jm-1,m2)+1
                kmp=modulo(km-1,m3)+1
                c1=     coldensh_out(imp,jmp,kmp)    !# column densities at the
                c2=     coldensh_out(ip,jmp,kmp)     !# four corners
                c3=     coldensh_out(imp,jp,kmp)
                c4=     coldensh_out(ip,jp,kmp)

                ! extra weights for better fit to analytical solution
                w1=   s1*weightf(c1)
                w2=   s2*weightf(c2)
                w3=   s3*weightf(c3)
                w4=   s4*weightf(c4)
                ! column density at the crossing point
                cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4)
                ! Take care of diagonals
                ! if (kdela == idela.or.kdela == jdela) then
                ! if (kdela == idela.and.kdela == jdela) then
                if (kdela == 1.and.(idela == 1.or.jdela == 1)) then
                if (idela == 1.and.jdela == 1) then
                    cdensi=   sqrt3*cdensi
                else
                    cdensi=   sqrt2*cdensi
                endif
                endif

                ! Path length from c through d to other side cell.
                path=sqrt((di*di+dj*dj)/(dk*dk)+1.0) ! pathlength from c to d point  


                ! y plane (left and right face) crossing
                ! (similar approach as for the z plane, see comments there)
            elseif (jdela >= idela.and.jdela >= kdela) then
                alam=(real(jm-j0)+sgnj*0.5)/dj
                zc=alam*dk+real(k0)
                xc=alam*di+real(i0)
                dz=2.0*abs(zc-(real(km)+0.5*sgnk))
                dx=2.0*abs(xc-(real(im)+0.5*sgni))
                s1=(1.-dx)*(1.-dz)
                s2=(1.-dz)*dx
                s3=(1.-dx)*dz
                s4=dx*dz
                ip=modulo(i-1,m1)+1
                imp=modulo(im-1,m1)+1
                jmp=modulo(jm-1,m2)+1
                kp=modulo(k-1,m3)+1
                kmp=modulo(km-1,m3)+1

                c1=  coldensh_out(imp,jmp,kmp)

                c2=  coldensh_out(ip,jmp,kmp)

                c3=  coldensh_out(imp,jmp,kp)

                c4=  coldensh_out(ip,jmp,kp)

                ! extra weights for better fit to analytical solution
                w1=s1*weightf(c1)
                w2=s2*weightf(c2)
                w3=s3*weightf(c3)
                w4=s4*weightf(c4)

                cdensi=   (c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4)

                ! Take care of diagonals
                if (jdela == 1.and.(idela == 1.or.kdela == 1)) then
                if (idela == 1.and.kdela == 1) then
                    !write(logf,*) 'error',i,j,k
                    cdensi=   sqrt3*cdensi
                else
                    !write(logf,*) 'diagonal',i,j,k
                    cdensi=   sqrt2*cdensi
                endif
                endif

                path=sqrt((di*di+dk*dk)/(dj*dj)+1.0)
                

                ! x plane (front and back face) crossing
                ! (similar approach as with z plane, see comments there)

            elseif(idela >= jdela.and.idela >= kdela) then
                alam=(real(im-i0)+sgni*0.5)/di
                zc=alam*dk+real(k0)
                yc=alam*dj+real(j0)
                dz=2.0*abs(zc-(real(km)+0.5*sgnk))
                dy=2.0*abs(yc-(real(jm)+0.5*sgnj))
                s1=(1.-dz)*(1.-dy)
                s2=(1.-dz)*dy
                s3=(1.-dy)*dz
                s4=dy*dz

                imp=modulo(im-1,m1)+1
                jp= modulo(j-1,m2)+1
                jmp=modulo(jm-1,m2)+1
                kp= modulo(k-1,m3)+1
                kmp=modulo(km-1,m3)+1
                c1=  coldensh_out(imp,jmp,kmp)
                c2=  coldensh_out(imp,jp,kmp)
                c3=  coldensh_out(imp,jmp,kp)
                c4=  coldensh_out(imp,jp,kp)

                ! extra weights for better fit to analytical solution
                w1   =s1*weightf(c1)
                w2   =s2*weightf(c2)
                w3   =s3*weightf(c3)
                w4   =s4*weightf(c4)

                cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4)

                if ( idela == 1 .and. ( jdela == 1 .or. kdela == 1 ) ) then
                if ( jdela == 1 .and. kdela == 1 ) then
                    cdensi=   sqrt3*cdensi
                else
                    cdensi   =sqrt2*cdensi
                endif
                endif
                path=sqrt(1.0+(dj*dj+dk*dk)/(di*di))
                
            end if

            contains
                !> Weight function for interpolation in cinterp. This is used only
                ! by the parent subroutine and can hence be "contained" in it.
                real(kind=real64) function weightf(cd)
                    real(kind=real64):: sig
                    real(kind=real64),intent(in) :: cd
                    real(kind=real64),parameter :: minweight=1.0_real64/0.6_real64
                    sig=sigma_HI_at_ion_freq
                    weightf=1.0/max(0.6_real64,cd*sigma_HI_at_ion_freq)
                end function weightf

    end subroutine cinterp

end module raytracing_sc