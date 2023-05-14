module photorates
    !! Module to compute photoionization rates (experimental)
    !! Based on c2ray (G. Mellema et al)
    use, intrinsic :: iso_fortran_env, only: real64 ! <-- This replaces the "dp" parameter in original c2ray (unpractical to use)
    implicit none
    real(kind=real64), parameter :: inv4pi = 0.079577471545947672804111050482_real64    ! 1/4π
    real(kind=real64), parameter :: S_star = 1.0e48_real64                              ! Reference strength of sources (normalization)

    contains
    !! Subroutine to compute the photoionization rate of a test source,
    !! that is, a source with a known number of ionizing photons per
    !! unit time and assuming a frequency-independent cross section (sig).
    subroutine photoion_rates_test(normflux,coldens_in,coldens_out,Vfact,nHI,sig,phi_photo_cell,phi_photo_out)

        ! Optical depth below which we should use the optically thin tables
        real(kind=real64),parameter :: tau_photo_limit = 1.0e-7 

        ! Subroutine Arguments
        real(kind=real64), intent(in) :: normflux               ! Strength of the test source, in s^-1
        real(kind=real64), intent(in) :: coldens_in             ! Column density into to cell
        real(kind=real64), intent(in) :: coldens_out            ! Column density at cell exit
        real(kind=real64), intent(in) :: Vfact                  ! Volume factor (dilution, cell volume, etc) see evolve0D TODO: figure out correct form
        real(kind=real64), intent(in) :: nHI                    ! Density of neutral Hydrogen
        real(kind=real64), intent(in) :: sig                    ! Hydrogen photoionization cross section (constant here)
        real(kind=real64), intent(out) :: phi_photo_cell        ! Photoionization rate of the cell Gamma, in s^-1
        real(kind=real64), intent(out) :: phi_photo_out         ! Photoionization rate at the output of the cell (radiation that leaves the cell), in s^-1

        real(kind=real64) :: tau_in                             ! Optical Depth to cell
        real(kind=real64) :: tau_out                            ! Optical Depth at cell exit
        real(kind=real64) :: phi_photo_in                       ! Photoionization rate at input of cell (radiation that enters the cell)
        real(kind=real64) :: prefact

        ! Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
        tau_in = coldens_in * sig
        tau_out = coldens_out * sig

        ! Compute incoming photoionization rate
        !prefact = strength * inv4pi / (Vfact * nHI)
        !prefact = strength / (Vfact * nHI)
        prefact = normflux * S_star / (Vfact)
        phi_photo_in = prefact * (exp(-tau_in))

        ! If cell is optically thick
        if (abs(tau_out - tau_in) > tau_photo_limit) then
            phi_photo_out = prefact * (exp(-tau_out))
            phi_photo_cell = phi_photo_in - phi_photo_out
            ! phi_photo_cell = strength * inv4pi / (Vfact * nHI) * (exp(-tau_in) - exp(-tau_out))
            
        ! If cell is optically thin
        else
            !phi_photo_cell = strength * inv4pi * sig * (tau_out - tau_in) / (Vfact) * exp(-tau_in)
            !phi_photo_cell = strength * sig * (tau_out - tau_in) / (Vfact) * exp(-tau_in)
            phi_photo_cell = prefact * (tau_out - tau_in) * exp(-tau_in)
            phi_photo_out = phi_photo_in - phi_photo_cell
        endif

    end subroutine photoion_rates_test
    

    !! IN DEVELOPMENT
    !! Photoionization rates from precalculated tables
    subroutine photoion_rates(normflux,coldens_in,coldens_out,Vfact,sig,phi_photo_cell, &
            phi_photo_out,photo_thin_table,minlogtau,dlogtau,NumTau)

        ! Subroutine Arguments
        real(kind=real64), intent(in) :: normflux               ! 
        real(kind=real64), intent(in) :: coldens_in             ! Column density into to cell
        real(kind=real64), intent(in) :: coldens_out            ! Column density at cell exit
        real(kind=real64), intent(in) :: Vfact                  ! Volume factor (dilution, cell volume, etc) see evolve0D TODO: figure out correct form
        real(kind=real64), intent(in) :: sig                    ! Hydrogen photoionization cross section (constant here)
        

        real(kind=real64), intent(out) :: phi_photo_cell        ! Photoionization rate of the cell Gamma, in s^-1
        real(kind=real64), intent(out) :: phi_photo_out         ! Photoionization rate at the output of the cell (radiation that leaves the cell), in s^-1

        real(kind=real64),intent(in) :: photo_thin_table(NumTau)
        integer, intent(in) :: NumTau
        real(kind=real64), intent(in) :: minlogtau
        real(kind=real64), intent(in) :: dlogtau

        real(kind=real64) :: tau_in                             ! Optical Depth to cell
        real(kind=real64) :: tau_out                            ! Optical Depth at cell exit
        real(kind=real64) :: logtau_in                             ! Optical Depth to cell
        real(kind=real64) :: logtau_out                            ! Optical Depth at cell exit
        real(kind=real64) :: phi_photo_in                       ! Photoionization rate at input of cell (radiation that enters the cell)
        real(kind=real64) :: prefact
        integer :: table_idx_in
        integer :: table_idx_out
        
        ! Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
        tau_in = coldens_in * sig
        tau_out = coldens_out * sig

        prefact = normflux / Vfact

        phi_photo_in = prefact * photo_lookuptable(tau_in)
        phi_photo_out = prefact * photo_lookuptable(tau_out)
        phi_photo_cell = phi_photo_in - phi_photo_out

        contains
        real(kind=real64) function photo_lookuptable(tau)
            real(kind=real64),intent(in) :: tau
            real(kind=real64) :: logtau
            real(kind=real64) :: real_i, residual
            integer :: i0, i1
            
            ! Find table index and do linear interpolation
            ! Recall that tau(0) = 0 and tau(1:NumTau) ~ logspace(minlogtau,maxlogtau)
            logtau = log10(max(1.0e-20_real64,tau))
            real_i = min(real(NumTau),max(0.0_real64,1.0+(logtau-minlogtau)/dlogtau))
            i0 = int( real_i )
            i1 = min(NumTau, i0+1)
            residual = real_i - real(i0)
            photo_lookuptable = photo_thin_table(i0) + residual*(photo_thin_table(i1) - photo_thin_table(i0))
        end function photo_lookuptable

    end subroutine photoion_rates

end module photorates