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
! The weights glu = prod_lu G and gorr = prod_orr G of the two mechanisms of
! the production (doc.production/selective_lu_orr.md) take effect at the times
! prod_lu_start and prod_orr_start: before it the mechanism has the weight G
! alone (the production filter as it is, 1 without it), from it prod_lu G
! (prod_orr G).  glu and gorr hold the weights in effect: update_production_weights(t)
! rebuilds them (and copies them to the device) when a switch time has been
! reached, once per step at its start (timestep, outstats), so that a step has
! one set of weights; production_clip_dt shortens the next step so that it
! ends on a switch time, which makes the switch a step boundary (the RK
! substeps never see a jump, the order in time is kept).  The default
! start time, -huge, is 'from the start of the run'.
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
  public :: production_active, production_setup, update_production_weights, production_clip_dt
  public :: ncol_unforced, ncol_lu, ncol_orr

  integer(C_INT), save :: ncol_unforced = 0      ! this rank's columns with G < 1 (the (0,0) column not counted)
  integer(C_INT), save :: ncol_lu = 0            ! ... with glu < 1 (lift-up weakened)
  integer(C_INT), save :: ncol_orr = 0           ! ... with gorr < 1 (Orr transfer weakened)
  logical, save :: lu_in = .false., orr_in = .false.    ! the state of glu, gorr: prod_lu (prod_orr) in effect

contains

  logical function production_active()
    production_active = (prod_kc > 0.0d0)
  end function production_active

  ! .true. when the weight that starts at ts is in effect at time t.  A step that was cut to end on ts
  ! (production_clip_dt) ends within round-off of it, so a tolerance of 1e-9 (relative to max(1, |ts|)) decides.
  logical function in_effect(t, ts)
    real(C_DOUBLE), intent(in) :: t, ts
    if (ts <= -0.5d0*huge(1.0d0)) then
      in_effect = .true.                 ! the default: from the start
    else
      in_effect = t >= ts - 1.0d-9*max(1.0d0, abs(ts))
    end if
  end function in_effect

  ! glu and gorr on the host for the state (lu_in, orr_in): prod_lu G or G alone, prod_orr G or G alone.
  subroutine fill_weights()
    glu = merge(prod_lu, 1.0d0, lu_in)*gprod
    gorr = merge(prod_orr, 1.0d0, orr_in)*gprod
  end subroutine fill_weights

  ! The weights in effect at time t (the start of a step, or the time of a statistics row): rebuilt on the
  ! host and copied to the device when a switch time has been reached; nothing otherwise.  Idempotent and
  ! not collective (every rank has the same time).
  subroutine update_production_weights(t)
    real(C_DOUBLE), intent(in) :: t
    logical :: lu_new, orr_new
    lu_new = in_effect(t, prod_lu_start)
    orr_new = in_effect(t, prod_orr_start)
    if ((lu_new .eqv. lu_in) .and. (orr_new .eqv. orr_in)) return
    lu_in = lu_new; orr_in = orr_new
    call fill_weights()
    !$omp target update to(glu, gorr)
    if (has_terminal) write (*, '(A,F16.9,A,F8.4,A,F8.4,A)') '   production weights from time', t, ': lift-up', &
      merge(prod_lu, 1.0d0, lu_in), ',  Orr transfer', merge(prod_orr, 1.0d0, orr_in), '  (times the filter G)'
  end subroutine update_production_weights

  ! The step that follows a step ended at t, of length dt, cut so that it ends on the next switch time
  ! that lies within it: the weights of a step are then those of its start (update_production_weights)
  ! and the switch is exact.  A switch time already reached (within in_effect's tolerance) cuts nothing.
  real(C_DOUBLE) function production_clip_dt(t, dt)
    real(C_DOUBLE), intent(in) :: t, dt
    real(C_DOUBLE) :: d
    d = dt
    d = clipped(prod_lu_start)
    d = clipped(prod_orr_start)
    production_clip_dt = d
  contains
    real(C_DOUBLE) function clipped(ts)
      real(C_DOUBLE), intent(in) :: ts
      clipped = d
      if (in_effect(t, ts)) return
      if (t + d > ts - 1.0d-9*max(1.0d0, abs(ts))) clipped = ts - t
    end function clipped
  end function production_clip_dt

  ! Allocates gprod (always: the equations read it) and builds it; G does
  ! not depend on time.  glu and gorr are filled for the time of the deck (time); update_production_weights
  ! follows the clock from there (hst calls it again once a restart has set the time).
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
    lu_in = in_effect(time, prod_lu_start)
    orr_in = in_effect(time, prod_orr_start)
    call fill_weights()
    ncol_lu = 0; ncol_orr = 0                  ! the columns the deck's weights weaken (once they are in effect)
    do ix = nx0, nxN
      do iz = -nz, nz
        if (ix == 0 .and. iz == 0) cycle
        if (prod_lu*gprod(iz, ix) < 1.0d0) ncol_lu = ncol_lu + 1
        if (prod_orr*gprod(iz, ix) < 1.0d0) ncol_orr = ncol_orr + 1
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
