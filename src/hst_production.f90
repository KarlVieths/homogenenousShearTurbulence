! Production filter (doc.production/production_filter.md).
!
! No mode is removed or damped; the mean shear produces energy only at the
! smaller scales of the x-z plane.  The production term -v (S, 0, S2) of
! the momentum equation is multiplied, column by column, by
!
!   G(k_h) = smoothstep((k_h/prod_kc - (1 - prod_edge))/(2 prod_edge)),  k_h = sqrt(kx^2 + kz^2)
!
! (prod_edge = 0: a step; columns on k_h = prod_kc within 1e-6 are forced),
! which is isotropic in the plane, so nothing depends on the direction of
! the shear (S, S2(t)).  The mean-shear advection (the tilting of the
! wavevectors, with its share of the pressure) acts on every mode and
! conserves energy; the columns with G < 1 are fed by the nonlinear
! transfer only.  In the equations (hst_equations) the lift-up term of eta
! is multiplied by G and d2v gets -(1 - G) (i alfa S + i beta S2) D1 v.
! gprod = G; 1 everywhere when the filter is off (prod_kc = 0).
!
! Diagnostics (hst_stats, statistics/production_runtime.dat): the
! variances and the dissipation of the unforced range (weight 1 - G) and
! the G-weighted shear stresses, which are the production.
module hst_production

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params

  implicit none
  private
  public :: production_active, production_setup, ncol_unforced, ncol_lu, ncol_orr

  integer(C_INT), save :: ncol_unforced = 0      ! this rank's columns with G < 1 (the (0,0) column not counted)
  integer(C_INT), save :: ncol_lu = 0            ! ... with glu < 1 (lift-up weakened)
  integer(C_INT), save :: ncol_orr = 0           ! ... with gorr < 1 (Orr transfer weakened)

contains

  logical function production_active()
    production_active = (prod_kc > 0.0d0)
  end function production_active

  ! Allocates gprod (always: the equations read it) and builds it; G does
  ! not depend on time.
  subroutine production_setup()
    integer(C_INT) :: ix, iz
    integer :: cnt(3), ierr
    real(C_DOUBLE) :: r, x
    allocate (gprod(-nz:nz, nx0:nxN), glu(-nz:nz, nx0:nxN), gorr(-nz:nz, nx0:nxN))
    gprod = 1.0d0
    ncol_unforced = 0
    if (production_active()) then
      do ix = nx0, nxN
        do iz = -nz, nz
          r = sqrt((alfa0*ix)**2 + (beta0*iz)**2)/prod_kc
          if (prod_edge > 0.0d0) then
            x = min(1.0d0, max(0.0d0, (r - (1.0d0 - prod_edge))/(2.0d0*prod_edge)))
            gprod(iz, ix) = x*x*(3.0d0 - 2.0d0*x)
          else if (r < 1.0d0 - 1.0d-6) then
            gprod(iz, ix) = 0.0d0
          end if
          if (gprod(iz, ix) < 1.0d0 .and. .not. (ix == 0 .and. iz == 0)) ncol_unforced = ncol_unforced + 1
        end do
      end do
    end if
    ! the weights of the two mechanisms: one place, one array each, read by one thread per column
    glu = prod_lu*gprod
    gorr = prod_orr*gprod
    ncol_lu = 0; ncol_orr = 0
    do ix = nx0, nxN
      do iz = -nz, nz
        if (ix == 0 .and. iz == 0) cycle
        if (glu(iz, ix) < 1.0d0) ncol_lu = ncol_lu + 1
        if (gorr(iz, ix) < 1.0d0) ncol_orr = ncol_orr + 1
      end do
    end do
    !$omp target enter data map(to: gprod, glu, gorr)
    if (has_terminal .and. production_active()) write (*, '(A,I0)') &
      '   production filter: columns with G < 1 on rank 0: ', ncol_unforced
    ! the run header: the weights and the columns they weaken, over all ranks ((0,0) not counted)
    cnt = [ncol_lu, ncol_orr, (2*nz + 1)*(nxN - nx0 + 1)]
    call MPI_Allreduce(MPI_IN_PLACE, cnt, 3, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (has_terminal .and. (prod_lu /= 1.0d0 .or. prod_orr /= 1.0d0)) write (*, '(A,F8.4,A,I0,A,F8.4,A,I0,A,I0,A)') &
      '   production weights: lift-up', prod_lu, ': columns with weight < 1: ', cnt(1), ';  Orr transfer', prod_orr, &
      ': columns with weight < 1: ', cnt(2), '  (of ', cnt(3) - 1, ')'
  end subroutine production_setup

end module hst_production
