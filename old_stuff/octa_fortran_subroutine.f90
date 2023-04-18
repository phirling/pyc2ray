! ===============================================================================================
!! Alternative version of do_source that sweeps the grid in growing octahedral faces. Its
!! intended use is as a reference for future parallelization on GPU. Using it as it is
!! on the CPU will result in a factor ~2 performance decrease.
!! 
!! The argument list of evolve0D has been changed (for subbox stuff) so this subroutine as it is
!! won't compile
! ===============================================================================================
subroutine do_source_octa(srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1,m2,m3)

    use, intrinsic :: iso_fortran_env, only: real64          ! This replaces the "dp" parameter in original c2ray (unpractical to use)

    ! subroutine arguments
    integer, intent(in) :: NumSrc                                   !> Number of sources
    integer,intent(in)      :: ns                                   !> source number 
    real(kind=real64),intent(in) :: srcflux(NumSrc)                 !> Strength of source. TODO: this is specific to the test case, need more general input
    integer,intent(in) :: srcpos(3,NumSrc)                          !> positions of ALL sources (mesh)
    real(kind=real64), intent(in) :: ndens(m1,m2,m3)                !> Hydrogen Density Field
    real(kind=real64), intent(in) :: dr               !> Cell size
    real(kind=real64),intent(inout) :: coldensh_out(m1,m2,m3)       !> Outgoing column density of the cells
    real(kind=real64),intent(inout) :: xh_av(m1,m2,m3)              !> Time-averaged HI ionization fractions of the cells (--> density of ionized H is xh_av * ndens)
    real(kind=real64),intent(inout) :: phi_ion(m1,m2,m3)            !> H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
    real(kind=real64),intent(in):: sig                              !> Hydrogen ionization cross section (sigma_HI_at_ion_freq)
    integer, intent(in) :: m1                                       !> mesh size x (hidden by f2py)
    integer, intent(in) :: m2                                       !> mesh size y (hidden by f2py)
    integer, intent(in) :: m3                                       !> mesh size z (hidden by f2py)
    ! integer,dimension(3), intent(in) :: last_l                      !> mesh position of left end point for RT
    ! integer,dimension(3), intent(in) :: last_r                      !> mesh position of right end point for RT  

    integer,dimension(3) :: rtpos
    integer :: r,k,j
    integer :: max_r
    coldensh_out(:,:,:) = 0.0

    ! Conservative estimate for max_r
    max_r = ceiling(1.5 * m1)

    ! First, do the source point
    rtpos(:) = srcpos(:,ns)
    call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
        last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)

    ! Sweep the grid by treating the faces of octahedra of increasing size.
    do r=1,max_r
        do k=0,r
            do j=0,k
                ! -- Top of the octahedron --
                ! Face QI
                rtpos(3) = srcpos(3,ns) + (r-k)
                rtpos(1) = srcpos(1,ns) + (k-j)
                rtpos(2) = srcpos(2,ns) + (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
                
                ! Face QII
                rtpos(3) = srcpos(3,ns) + (r-k)
                rtpos(1) = srcpos(1,ns) - (k-j)
                rtpos(2) = srcpos(2,ns) + (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
                
                ! Face QIII
                rtpos(3) = srcpos(3,ns) + (r-k)
                rtpos(1) = srcpos(1,ns) + (k-j)
                rtpos(2) = srcpos(2,ns) - (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
                
                ! Face QIV
                rtpos(3) = srcpos(3,ns) + (r-k)
                rtpos(1) = srcpos(1,ns) - (k-j)
                rtpos(2) = srcpos(2,ns) - (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)

                ! -- Bottom of the octahedron --
                ! Face QI
                rtpos(3) = srcpos(3,ns) - (r-k)
                rtpos(1) = srcpos(1,ns) + (k-j)
                rtpos(2) = srcpos(2,ns) + (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
                
                ! Face QII
                rtpos(3) = srcpos(3,ns) - (r-k)
                rtpos(1) = srcpos(1,ns) - (k-j)
                rtpos(2) = srcpos(2,ns) + (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
                
                ! Face QIII
                rtpos(3) = srcpos(3,ns) - (r-k)
                rtpos(1) = srcpos(1,ns) + (k-j)
                rtpos(2) = srcpos(2,ns) - (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
                
                ! Face QIV
                rtpos(3) = srcpos(3,ns) - (r-k)
                rtpos(1) = srcpos(1,ns) - (k-j)
                rtpos(2) = srcpos(2,ns) - (k-(k-j))
                call evolve0D(rtpos,srcflux,srcpos,ns,coldensh_out,sig,dr,ndens,xh_av,phi_ion, &
                    last_l, last_r, photon_loss_src,NumSrc,m1,m2,m3)
            enddo
        enddo
    enddo
    
end subroutine do_source_octa