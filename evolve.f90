module evolve
    !! Module to evolve the whole grid for (possibly multiple) sources

    use, intrinsic :: iso_fortran_env, only: real64
    use raytracing, only: do_source
    use chemistry, only: global_pass

    implicit none

    real(kind=real64), parameter :: epsilon=1e-14_real64                        ! Double precision very small number

    contains

    subroutine evolve3D(dt,dr,srcflux,srcpos,temp,ndens,coldensh_out,xh,xh_av,phi_ion, &
                        sig,bh00,albpow,colh0,temph0,abu_c,NumSrc,m1,m2,m3)

        ! Subroutine Arguments
        real(kind=real64),intent(in) :: dt                              !> time step
        real(kind=real64), dimension(3), intent(in) :: dr               !> Cell size
        real(kind=real64),intent(in) :: srcflux(NumSrc)                 !> Strength of source. TODO: this is specific to the test case, need more general input
        integer,intent(in) :: srcpos(3,NumSrc)                          !> positions of ALL sources (mesh)
        real(kind=real64), intent(in) :: temp(m1,m2,m3)                 !> Temperature field
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)                !> Hydrogen Density Field
        real(kind=real64),intent(inout) :: coldensh_out(m1,m2,m3)       !> Outgoing column density of the cells
        real(kind=real64),intent(inout) :: xh(m1,m2,m3)                 !> HI ionization fractions of the cells
        real(kind=real64),intent(inout) :: xh_av(m1,m2,m3)              !> Time-averaged HI ionization fractions of the cells (--> density of ionized H is xh_av * ndens)
        real(kind=real64),intent(inout) :: phi_ion(m1,m2,m3)            !> H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)

        ! Physical Constants (passed as arguments to the Fortran routine)
        real(kind=real64),intent(in):: sig                              !> Hydrogen ionization cross section (sigma_HI_at_ion_freq)
        real(kind=real64),intent(in) :: bh00                            !> Hydrogen recombination parameter (value at 10^4 K)
        real(kind=real64),intent(in) :: albpow                          !> Hydrogen recombination parameter (power law index)
        real(kind=real64),intent(in) :: colh0                           !> Hydrogen collisional ionization parameter
        real(kind=real64),intent(in) :: temph0                          !> Hydrogen ionization energy expressed in K
        real(kind=real64),intent(in) :: abu_c                           !> Carbon abundance
        integer, intent(in) :: NumSrc                                   !> Number of sources
        integer, intent(in) :: m1                                       !> mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                       !> mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                       !> mesh size z (hidden by f2py)


        integer :: conv_flag
        integer,dimension(3) :: last_l                                  !> mesh position of left end point for RT
        integer,dimension(3) :: last_r                                  !> mesh position of right end point for RT

        integer :: ns

        ! In c2ray, evolution around a source happens in subboxes of increasing sizes. For now, here, always do the whole grid.
        last_l(:) = 1
        last_r(:) = m1

        ! TODO: add MPI distribution of sources. For now, use "static serial" mode

        ! Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        do ns=1,NumSrc
            call do_source(srcflux,srcpos,ns,last_l,last_r,coldensh_out,sig,dr,ndens,xh_av,phi_ion,NumSrc,m1,m2,m3)
        enddo

        ! Now, apply these rates to compute the updated ionization fraction
        call global_pass(dt,ndens,temp,xh,xh_av,phi_ion,bh00,albpow,colh0,temph0,abu_c,conv_flag,m1,m2,m3)

        write(*,*) "Number of non-converged points: ",conv_flag, " of ", m1*m2*m3

    end subroutine evolve3D

end module evolve