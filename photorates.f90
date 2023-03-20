module photorates
    use, intrinsic :: iso_fortran_env, only: real64 ! <-- This replaces the "dp" parameter in original c2ray (unpractical to use)
    implicit none
    real(kind=real64), parameter :: inv4pi = 0.079577471545947672804111050482_real64

    contains
    !! Subroutine to compute the photoionization rate of a test source,
    !! that is, a source with a known number of ionizing photons per
    !! unit time and assuming a frequency-independent cross section (sig).
    subroutine photoion_rates_test(strength,coldens_in,coldens_out,Vfact,nHI,sig,phi_photoion)

        ! Optical depth below which we should use the optically thin tables
        real(kind=real64),parameter :: tau_photo_limit = 1.0e-7 

        ! Subroutine Arguments
        real(kind=real64), intent(in) :: strength
        real(kind=real64), intent(in) :: coldens_in
        real(kind=real64), intent(in) :: coldens_out
        real(kind=real64), intent(in) :: Vfact
        real(kind=real64), intent(in) :: nHI
        real(kind=real64), intent(in) :: sig
        real(kind=real64), intent(out) :: phi_photoion

        real(kind=real64) :: tau_in
        real(kind=real64) :: tau_out

        tau_in = coldens_in * sig
        tau_out = coldens_out * sig

        ! If cell is optically thick
        if (abs(tau_out - tau_in) > tau_photo_limit) then
            phi_photoion = strength * inv4pi / (Vfact * nHI) * (exp(-tau_in) - exp(-tau_out))
        ! If cell is optically thin
        else
            phi_photoion = strength * inv4pi * sig * (tau_out - tau_in) / (Vfact) * exp(-tau_in)
        endif

    end subroutine photoion_rates_test
    
end module photorates