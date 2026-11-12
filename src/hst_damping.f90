! Large-scale damping (DAMPING.md).
!
! A body force -g(kx, kz) u on the Fourier columns inside a region of the
! x-z wavenumber plane defined in the shear frame: with e_par the direction
! of the shear vector (S, S2(t)) (or a fixed angle), k_par = k . e_par,
! k_perp = k . e_perp and
!
!   r = ( (|k_par|/damp_kpar)^p + (|k_perp|/damp_kperp)^p )^(1/p),   p = damp_norm,
!
! the columns with r < 1 are damped (r = 2: an ellipse with the semi-axes
! damp_kpar, damp_kperp; p large: the rectangle).  Modes on the boundary
! (r = 1 within 1e-6) are kept, so with the semi-axes 2 pi/Lx, 2 pi/Lz of a
! small box the small box's own modes are the first ones kept.
!
!   soft (damp_rate > 0): g = damp_rate inside, falling to zero through a
!     smoothstep over the last damp_edge of r (damp_edge = 0: a step);
!     treated explicitly with the other explicit terms (RK3), so
!     damp_rate * deltat must stay below about 1;
!   hard (damp_hard): the columns are zeroed every substep (their right-hand
!     sides are dropped before the implicit solve), the limit damp_rate ->
!     infinity: a Galerkin truncation, which removes no power.
!
! Wide-strip mask (hard mask with damp_rw > 1; DAMPING.md section 5): the
! removed set is enlarged to
!
!   { (kx, kz) /= (0, 0) :  |k_perp| < damp_kperp  and  r < damp_rw },
!
! the ellipse r < damp_rw (the semi-axes scaled by damp_rw) clipped to the
! strip |k_perp| < damp_kperp, i.e. every column whose wavelength across the
! shear exceeds the small box's Lz and whose scale is above 1/damp_rw of the
! small box's.  r < 1 implies |k_perp| < damp_kperp, so the ellipse is
! contained and damp_rw = 1 is the plain ellipse.  With damp_delta > 0 the
! strip removal leaves the sliver |k_perp| < damp_delta*damp_kperp in place
! (the ellipse is removed in any case).  Columns on a boundary of the strip
! (|k_perp| = damp_kperp) or of the ellipse r = damp_rw are kept, as for the
! plain ellipse, within the same tolerance 1e-6.
!
! The (0,0) column (the mean profiles) is damped only with damp_mean.  With
! damp_follow the region turns with the shear direction atan2(S2(t), S)
! and gdamp is rebuilt every substep in which the direction changed;
! otherwise it is built once.  A hard mask that is built or rebuilt also
! zeroes the state in the masked columns (damping_zero_state), so that a
! field read from a restart file, or generated, with energy there does not
! feed the other columns through the nonlinear term of the first substep.
!
! Diagnostics (hst_stats, damping_runtime.dat): the energy in the damped
! columns and the power 2 g <u.u> they lose.  damping_report: the start-up
! summary of the mask.
module hst_damping

  use, intrinsic :: iso_c_binding
  use hst_params
  use hst_derivatives, only: s2_of

  implicit none
  private
  public :: damping_active, damping_setup, damping_update, damping_theta, damping_mask, damping_soft_factor, ncol_damped
  public :: damping_zero_state, damping_report, damping_radius, damping_rate_of

  real(C_DOUBLE), parameter :: PI = 3.141592653589793d0
  real(C_DOUBLE), parameter :: BTOL = 1.0d-6            ! a column within BTOL of a boundary of the region is kept
  real(C_DOUBLE), save :: theta_last = -huge(1.0d0)
  integer(C_INT), save :: ncol_damped = 0        ! this rank's damped columns

contains

  logical function damping_active()
    damping_active = (damp_kpar > 0.0d0 .and. damp_kperp > 0.0d0 .and. (damp_hard .or. damp_rate > 0.0d0))
  end function damping_active

  ! 1 when the explicit damping term is in use, 0 for the hard mask or no damping.
  real(C_DOUBLE) function damping_soft_factor()
    damping_soft_factor = 0.0d0
    if (damping_active() .and. .not. damp_hard) damping_soft_factor = 1.0d0
  end function damping_soft_factor

  ! Direction of e_par (radians from x towards z) at time t.
  real(C_DOUBLE) function damping_theta(t)
    real(C_DOUBLE), intent(in) :: t
    if (damp_follow) then
      damping_theta = atan2(s2_of(t), S)
    else
      damping_theta = damp_angle*PI/180.0d0
    end if
  end function damping_theta

  ! The radius r of a column with the shear-frame wavenumbers (kpar, kperp).
  pure real(C_DOUBLE) function damping_radius(kpar, kperp)
    real(C_DOUBLE), intent(in) :: kpar, kperp
    damping_radius = ((abs(kpar)/damp_kpar)**damp_norm + (abs(kperp)/damp_kperp)**damp_norm)**(1.0d0/damp_norm)
  end function damping_radius

  ! Wide-strip part of the hard mask: the column at the radius r with the
  ! cross-shear wavenumber kperp is in the strip |k_perp| < damp_kperp, inside
  ! r < damp_rw, and outside the sliver of the half-width damp_delta*damp_kperp.
  ! Never true for damp_rw = 1 (the plain ellipse).
  pure logical function in_wide_strip(kperp, r)
    real(C_DOUBLE), intent(in) :: kperp, r
    in_wide_strip = .false.
    if (damp_rw <= 1.0d0) return
    in_wide_strip = abs(kperp) < damp_kperp*(1.0d0 - BTOL) .and. r < damp_rw*(1.0d0 - BTOL) &
                    .and. abs(kperp) >= damp_delta*damp_kperp*(1.0d0 - BTOL)
  end function in_wide_strip

  ! The damping rate of the column with the shear-frame wavenumbers (kpar,
  ! kperp): 0 outside the region, damp_rate (soft; smoothstep over damp_edge)
  ! or 1 (hard: masked) inside.  Not the (0,0) rule, which is the caller's.
  pure real(C_DOUBLE) function damping_rate_of(kpar, kperp) result(g)
    real(C_DOUBLE), intent(in) :: kpar, kperp
    real(C_DOUBLE) :: r, x
    r = damping_radius(kpar, kperp)
    g = 0.0d0
    if (r < 1.0d0 - BTOL) then
      if (damp_hard) then
        g = 1.0d0
      else if (r <= 1.0d0 - damp_edge) then
        g = damp_rate
      else
        x = (1.0d0 - r)/damp_edge
        g = damp_rate*x*x*(3.0d0 - 2.0d0*x)
      end if
    else if (damp_hard .and. in_wide_strip(kperp, r)) then
      g = 1.0d0
    end if
  end function damping_rate_of

  ! Allocates gdamp (always: the equations read it) and builds it for the
  ! start time.
  subroutine damping_setup()
    allocate (gdamp(-nz:nz, nx0:nxN))
    gdamp = 0.0d0
    !$omp target enter data map(to: gdamp)
    if (damping_active()) call damping_update(time)
    if (has_terminal .and. damping_active()) write (*, '(A,F8.3,A,L1,A,F8.3,A,I0)') &
      '   damping: direction at start ', damping_theta(time)*180.0d0/PI, ' deg,  hard = ', damp_hard, &
      '  rate = ', damp_rate, '  damped columns on rank 0: ', ncol_damped
  end subroutine damping_setup

  ! gdamp for the shear direction at time t; nothing to do when the
  ! direction has not changed since the last call (unless force).  changed
  ! tells whether gdamp was rebuilt.
  subroutine damping_update(t, changed, force)
    real(C_DOUBLE), intent(in) :: t
    logical, intent(out), optional :: changed
    logical, intent(in), optional :: force
    integer(C_INT) :: ix, iz, n
    real(C_DOUBLE) :: theta, c, s, kx, kz, kpar, kperp, g

    if (present(changed)) changed = .false.
    theta = damping_theta(t)
    if (present(force)) then
      if (force) theta_last = -huge(1.0d0)
    end if
    if (theta == theta_last) return
    theta_last = theta
    if (present(changed)) changed = .true.
    c = cos(theta); s = sin(theta)
    n = 0
    do ix = nx0, nxN
      kx = alfa0*ix
      do iz = -nz, nz
        kz = beta0*iz
        kpar = kx*c + kz*s
        kperp = -kx*s + kz*c
        g = damping_rate_of(kpar, kperp)
        if (ix == 0 .and. iz == 0 .and. .not. damp_mean) g = 0.0d0
        gdamp(iz, ix) = g
        if (g > 0.0d0) n = n + 1
      end do
    end do
    ncol_damped = n
    !$omp target update to(gdamp)
  end subroutine damping_update

  ! Hard mask: drops the right-hand sides of the masked columns before the
  ! implicit solve, so the solve returns zero for them.
  subroutine damping_mask()
    integer(C_INT) :: ix, iy, iz
    !$omp target teams distribute parallel do collapse(3) default(none) &
    !$omp shared(rhs, oldrhs, gdamp, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz)
    do ix = nx0, nxN
      do iz = -nz, nz
        do iy = ny0, nyN
          if (gdamp(iz, ix) > 0.0d0) then
            rhs(iy, iz, ix, 1) = 0.0d0; rhs(iy, iz, ix, 2) = 0.0d0
            oldrhs(iy, iz, ix, 1) = 0.0d0; oldrhs(iy, iz, ix, 2) = 0.0d0
          end if
        end do
      end do
    end do
  end subroutine damping_mask

  ! Hard mask: sets the velocity V (ghost rows included) and the carried
  ! explicit terms of the masked columns to zero.  After the initial field
  ! or a restart is read, and after every rebuild of the mask.  On the
  ! device (V, oldrhs, gdamp are mapped there).
  subroutine damping_zero_state()
    integer(C_INT) :: ix, iy, iz, m
    !$omp target teams distribute parallel do collapse(3) default(none) &
    !$omp shared(V, oldrhs, gdamp, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, m)
    do ix = nx0, nxN
      do iz = -nz, nz
        do iy = ny0 - 2, nyN + 2
          if (gdamp(iz, ix) > 0.0d0) then
            do m = 1, 3
              V(iy, iz, ix, m) = 0.0d0
            end do
            if (iy >= ny0 .and. iy <= nyN) then
              oldrhs(iy, iz, ix, 1) = 0.0d0; oldrhs(iy, iz, ix, 2) = 0.0d0
            end if
          end if
        end do
      end do
    end do
  end subroutine damping_zero_state

  ! Start-up summary of the mask for the shear direction of the last
  ! rebuild: the number of masked columns (the half plane stored, kx >= 0,
  ! and the whole plane, both signs; ellipse and wide strip separately), the
  ! masked columns of the small box when the semi-axes are whole multiples
  ! of the box's wavenumbers, and the smallest kept horizontal wavenumbers.
  ! Host only, on the whole lattice (does not depend on the rank layout);
  ! call it on one rank.
  subroutine damping_report()
    integer(C_INT) :: ix, iz, nxr, nzr, ia, ib, nlist
    integer(C_INT) :: n_ell, n_strip, n_half, n_full, ix_kept, iz_kept, ix_strip, iz_strip
    real(C_DOUBLE) :: theta, c, s, kx, kz, kpar, kperp, g, r, kh, kh_kept, kh_strip, xr, zr
    logical :: masked, in_ellipse, sb_ok
    integer(C_INT), parameter :: MAXLIST = 24

    if (.not. damping_active()) return
    theta = damping_theta(time)
    c = cos(theta); s = sin(theta)
    n_ell = 0; n_strip = 0; n_half = 0
    kh_kept = huge(1.0d0); kh_strip = huge(1.0d0)
    ix_kept = 0; iz_kept = 0; ix_strip = 0; iz_strip = 0
    do ix = 0, nx
      kx = alfa0*ix
      do iz = -nz, nz
        kz = beta0*iz
        kpar = kx*c + kz*s
        kperp = -kx*s + kz*c
        g = damping_rate_of(kpar, kperp)
        if (ix == 0 .and. iz == 0 .and. .not. damp_mean) g = 0.0d0
        masked = (g > 0.0d0)
        r = damping_radius(kpar, kperp)
        in_ellipse = (r < 1.0d0 - BTOL)
        if (masked) then
          n_half = n_half + 1
          ! full plane: the columns with ix > 0 stand for themselves and their conjugates
          if (ix > 0) then
            if (in_ellipse) then; n_ell = n_ell + 2; else; n_strip = n_strip + 2; end if
          else
            if (in_ellipse) then; n_ell = n_ell + 1; else; n_strip = n_strip + 1; end if
          end if
        else if (ix /= 0 .or. iz /= 0) then
          kh = sqrt(kx*kx + kz*kz)
          if (kh < kh_kept) then
            kh_kept = kh; ix_kept = ix; iz_kept = iz
          end if
          if (abs(kperp) < damp_kperp*(1.0d0 - BTOL) .and. kh < kh_strip) then
            kh_strip = kh; ix_strip = ix; iz_strip = iz
          end if
        end if
      end do
    end do
    n_full = n_ell + n_strip
    write (*, '(A)') '   damping mask (shear-frame region, summary at the start direction):'
    write (*, '(A,I0,A,I0,A,I0,A,I0,A)') '     masked columns: ', n_full, ' in the whole plane (', n_ell, ' ellipse + ', &
      n_strip, ' wide strip), ', n_half, ' stored (kx >= 0)'
    if (damp_rw > 1.0d0) write (*, '(A,F8.4,A,F8.4)') '     wide strip: |k_perp| <', damp_kperp, ' and r <', damp_rw
    if (damp_rw > 1.0d0 .and. damp_delta > 0.0d0) write (*, '(A,F8.4,A,F8.4,A)') &
      '     sliver kept: |k_perp| <', damp_delta*damp_kperp, '  (damp_delta =', damp_delta, ')'
    ! the small box's own columns: indices that are multiples of n = semi-axis/fundamental
    xr = damp_kpar/alfa0; zr = damp_kperp/beta0
    nxr = nint(xr); nzr = nint(zr)
    sb_ok = (nxr >= 1 .and. nzr >= 1 .and. abs(xr - nxr) < 1.0d-6*xr .and. abs(zr - nzr) < 1.0d-6*zr)
    if (sb_ok) then
      write (*, '(A,I0,A,I0,A)', advance='no') '     small-box columns (kx, kz in units of the semi-axes; n = ', nxr, ', ', nzr, &
        ') masked:'
      nlist = 0
      do ia = 0, nx/nxr
        do ib = -nz/nzr, nz/nzr
          if (ia == 0 .and. ib <= 0) cycle              ! (0,0), and the conjugates of the ib > 0 columns
          kx = alfa0*ia*nxr; kz = beta0*ib*nzr
          if (damping_rate_of(kx*c + kz*s, -kx*s + kz*c) > 0.0d0) then
            nlist = nlist + 1
            if (nlist <= MAXLIST) write (*, '(A,I0,A,I0,A)', advance='no') ' (', ia, ',', ib, ')'
          end if
        end do
      end do
      if (nlist == 0) write (*, '(A)', advance='no') ' none'
      if (nlist > MAXLIST) write (*, '(A,I0,A)', advance='no') ' ... (', nlist, ' in all)'
      write (*, '(A)') ''
    end if
    if (kh_kept < huge(1.0d0)) write (*, '(A,F9.4,A,F7.4,A,I0,A,I0,A)') &
      '     smallest |k_h| kept: ', kh_kept, ' (|k_h| Lz/2pi = ', kh_kept/damp_kperp, ') at (ix, iz) = (', ix_kept, ', ', iz_kept, ')'
    if (kh_strip < huge(1.0d0)) write (*, '(A,F9.4,A,F7.4,A,I0,A,I0,A)') &
      '     smallest |k_h| kept in the strip |k_perp| < kperp: ', kh_strip, ' (|k_h| Lz/2pi = ', kh_strip/damp_kperp, &
      ') at (ix, iz) = (', ix_strip, ', ', iz_strip, ')'
  end subroutine damping_report

end module hst_damping
