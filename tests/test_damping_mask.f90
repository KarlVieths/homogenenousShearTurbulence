! The hard damping mask: geometry of the wide-strip variant and the state it leaves.
!
!   mpirun -np 1 build-cpu/test_damping_mask tests/decks/mask_strip.in
!
! The deck is the 3:2:1 box enlarged three times (alfa0 = 2 pi/9, beta0 = 2 pi/3)
! with the hard mask of the semi-axes (2 pi/3, 2 pi), damp_rw = 2.5.  Two parts.
!
! State.  A random field has energy in the masked columns; damping_zero_state
! removes it, and 15 nonlinear steps keep it at exactly zero while the free
! columns stay alive.  The shear S2 is switched on at t = 0.01 (deck: s2_start
! is set here), so the mask rotates from 0 to atan2(0.5, 1) in the middle of
! the run and the rebuild of the mask has to zero the state again.
!
! Geometry (the lattice is rescaled to n = 3, 5, 10 and the angle varied; the
! mask is the table of DAMPING.md section 5):
!   T1  damp_rw = 1 is the plain ellipse: gdamp equals the original predicate
!       (r < 1 - 1e-6) at every angle, and the ellipse counts 24, 68, 304
!       columns (the whole plane without (0,0)) at theta = 0.
!   T2  the lattice points of option A: removed in the strip outside the
!       ellipse, kept on the strip edge |k_perp| = kb, kept at r = rw, kept
!       in the small-scale strip, (0,0) kept (removed only with damp_mean).
!       Wide strip counts at theta = 0: 74, 220, 918 (rw = 2.5); option B
!       (delta = 1/6) keeps the sliver.
!   T3  symmetry: mask(k) = mask(-k) on the kx = 0 plane; the mask at theta is
!       the z-mirror of the mask at -theta.
!   T5  masked counts against the angle, theta = 0 .. 45 in 0.25 steps.
program test_damping_mask

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
  use test_common
  use hst_input
  use hst_mpi
  use hst_fft
  use hst_setup
  use hst_derivatives
  use hst_linsolve
  use hst_equations
  use hst_initial
  use hst_damping

  implicit none

  real(C_DOUBLE), parameter :: PI = 3.141592653589793d0
  logical :: ok = .true.
  real(C_DOUBLE) :: ka, kb

  call test_start()
  if (nproc /= 1) error stop 'test_damping_mask runs on one rank'
  s2_mean = 0.5d0; s2_period = 0.0d0; s2_start = 0.01d0      ! the mask turns at t = 0.01
  call test_setup()
  ka = damp_kpar; kb = damp_kperp
  if (.not. (damp_hard .and. damp_rw > 1.0d0)) error stop 'test_damping_mask needs the wide-strip deck (mask_strip.in)'

  call state_part()
  call geometry_part()

  call test_finish(ok)

contains

  subroutine check(cond, msg)
    logical, intent(in) :: cond
    character(len=*), intent(in) :: msg
    if (.not. cond) then
      ok = .false.
      if (has_terminal) print *, '   FAILED: '//msg
    end if
  end subroutine check

  ! energy sum |V|^2 (ghost rows included) in the masked and in the free columns
  subroutine masked_energy(e_masked, e_free)
    real(C_DOUBLE), intent(out) :: e_masked, e_free
    integer :: ix, iz, iy, m
    !$omp target update from(V)
    e_masked = 0.0d0; e_free = 0.0d0
    do ix = nx0, nxN
      do iz = -nz, nz
        do iy = ny0 - 2, nyN + 2
          do m = 1, 3
            if (gdamp(iz, ix) > 0.0d0) then
              e_masked = e_masked + abs(V(iy, iz, ix, m))**2
            else
              e_free = e_free + abs(V(iy, iz, ix, m))**2
            end if
          end do
        end do
      end do
    end do
  end subroutine masked_energy

  subroutine state_part()
    real(C_DOUBLE) :: e_masked, e_free, theta_end
    integer :: m, istep
    logical :: changed
    time = 0.0d0; deltat = 2.0d-3; oldrhs = 0
    !$omp target update to(oldrhs)
    call generate_initial_field()
    !$omp target update to(V)
    do m = 1, 3
      call fill_ghosts(m)
    end do
    call masked_energy(e_masked, e_free)
    call check(e_masked > 0.0d0, 'the random field has no energy in the masked columns (nothing to test)')
    call damping_zero_state()
    call masked_energy(e_masked, e_free)
    call check(e_masked == 0.0d0, 'damping_zero_state leaves energy in the masked columns')
    call check(e_free > 0.0d0, 'damping_zero_state removed the free columns too')
    if (has_terminal) write (*, '(A,I0,A,ES10.3)') '   masked columns on the rank: ', ncol_damped, '   energy of the free columns: ', e_free
    do istep = 1, 15
      call timestep()
      call masked_energy(e_masked, e_free)
      if (e_masked /= 0.0d0) then
        call check(.false., 'masked columns not exactly zero after a step')
        exit
      end if
      if (.not. (e_free > 0.0d0 .and. e_free < huge(1.0d0))) then
        call check(.false., 'free columns died or blew up')
        exit
      end if
    end do
    ! the rebuild flag: false for the direction of the last build, true when forced
    call damping_update(time, changed)
    call check(.not. changed, 'damping_update reports a rebuild for an unchanged direction')
    call damping_update(time, changed, force=.true.)
    call check(changed, 'damping_update(force) does not rebuild')
    theta_end = damping_theta(time)
    call check(abs(theta_end - atan2(0.5d0, 1.0d0)) < 1.0d-12, 'the mask did not turn with S2')
    if (has_terminal) write (*, '(A,F8.3,A,I0,A,ES10.3,A,ES9.2)') '   15 nonlinear steps to t = ', time, ', mask turned to ', &
      nint(theta_end*180.0d0/PI), ' deg, free energy ', e_free, ', masked energy ', e_masked
  end subroutine state_part

  ! number of masked columns of the whole plane (without (0,0) unless damp_mean): the columns with
  ! kx > 0 stand for themselves and their conjugates
  integer function count_full()
    integer :: ix, iz
    count_full = 0
    do ix = nx0, nxN
      do iz = -nz, nz
        if (gdamp(iz, ix) > 0.0d0) count_full = count_full + merge(1, 2, ix == 0)
      end do
    end do
  end function count_full

  ! gdamp at the angle (deg) and the current damp_rw, damp_delta, alfa0, beta0
  subroutine build(angle_deg)
    real(C_DOUBLE), intent(in) :: angle_deg
    damp_follow = .false.; damp_angle = angle_deg
    call damping_update(0.0d0, force=.true.)
  end subroutine build

  logical function masked_at(ix, iz)
    integer, intent(in) :: ix, iz
    masked_at = (gdamp(iz, ix) > 0.0d0)
  end function masked_at

  subroutine geometry_part()
    integer, parameter :: NS(3) = [3, 5, 10]
    integer, parameter :: N_ELL(3) = [24, 68, 304], N_A(3) = [74, 220, 918]
    real(C_DOUBLE), parameter :: ANGLES(5) = [0.0d0, 7.5d0, 15.0d0, 22.5d0, 45.0d0]
    integer :: in, n, ia, ix, iz, cnt, cmin(2), cmax(2), k
    real(C_DOUBLE) :: kx, kz, c, s, kpar, kperp, r, ang
    logical :: old, same
    real(C_DOUBLE), allocatable :: gplus(:, :)

    allocate (gplus(-nz:nz, nx0:nxN))
    do in = 1, 3
      n = NS(in)
      alfa0 = ka/n; beta0 = kb/n

      ! T1: damp_rw = 1 is the plain ellipse, at every angle
      damp_rw = 1.0d0; damp_delta = 0.0d0
      do ia = 1, size(ANGLES)
        call build(ANGLES(ia))
        c = cos(ANGLES(ia)*PI/180.0d0); s = sin(ANGLES(ia)*PI/180.0d0)
        same = .true.
        do ix = nx0, nxN
          do iz = -nz, nz
            kx = alfa0*ix; kz = beta0*iz
            kpar = kx*c + kz*s; kperp = -kx*s + kz*c
            r = ((abs(kpar)/damp_kpar)**damp_norm + (abs(kperp)/damp_kperp)**damp_norm)**(1.0d0/damp_norm)
            old = (r < 1.0d0 - 1.0d-6)
            if (ix == 0 .and. iz == 0) old = .false.
            if (old .neqv. masked_at(ix, iz)) same = .false.
          end do
        end do
        call check(same, 'damp_rw = 1 differs from the plain ellipse')
      end do
      call build(0.0d0)
      call check(count_full() == N_ELL(in), 'ellipse count at theta = 0')
      if (has_terminal) write (*, '(A,I3,A,I5,A,I5,A)') '   n =', n, ': ellipse columns ', count_full(), ' (expected ', N_ELL(in), ')'
      ! the wide strip: counts of DAMPING.md, section 3.3
      damp_rw = 2.5d0
      call build(0.0d0)
      call check(count_full() == N_A(in), 'option A count at theta = 0')
      if (has_terminal) write (*, '(A,I3,A,I5,A,I5,A)') '   n =', n, ': option A (rw = 2.5) ', count_full(), ' (expected ', N_A(in), ')'

      ! T3: mask(k) = mask(-k) on kx = 0, and the z-mirror of the mask at -theta
      do ia = 1, size(ANGLES)
        call build(ANGLES(ia)); gplus = gdamp
        same = .true.
        do iz = -nz, nz
          if ((gplus(iz, 0) > 0.0d0) .neqv. (gplus(-iz, 0) > 0.0d0)) same = .false.
        end do
        call check(same, 'mask(k) /= mask(-k) on kx = 0')
        call build(-ANGLES(ia))
        do ix = nx0, nxN
          do iz = -nz, nz
            if ((gplus(iz, ix) > 0.0d0) .neqv. (gdamp(-iz, ix) > 0.0d0)) same = .false.
          end do
        end do
        call check(same, 'mask at theta is not the z-mirror of the mask at -theta')
      end do
    end do

    ! T2: lattice points of the n = 3 box at theta = 0 (X = kx/ka = ix/3, Z = kz/kb = iz/3)
    alfa0 = ka/3; beta0 = kb/3
    damp_rw = 2.5d0; damp_delta = 0.0d0; damp_mean = .false.
    call build(0.0d0)
    call check(masked_at(4, 1), 'T2a: (4/3, 1/3), in the strip outside the ellipse, must be removed')
    call check(masked_at(3, 0) .and. masked_at(6, 0), 'T2: the small box (1,0), (2,0) must be removed')
    call check(.not. masked_at(9, 0) .and. .not. masked_at(9, 1), 'T2c: r > rw in the strip must be kept')
    call check(.not. masked_at(0, 3) .and. .not. masked_at(0, -3) .and. .not. masked_at(1, 3), 'T2b: the strip edge |Z| = 1 must be kept')
    call check(.not. masked_at(3, 3) .and. .not. masked_at(2, 4), 'T2: narrow columns must be kept')
    call check(.not. masked_at(0, 0), 'T2d: (0,0) must be kept')
    damp_mean = .true.; call build(0.0d0)
    call check(masked_at(0, 0), 'T2d: (0,0) is removed with damp_mean')
    damp_mean = .false.
    damp_rw = 2.0d0; call build(0.0d0)
    call check(masked_at(3, 0) .and. .not. masked_at(6, 0), 'rw = 2: (1,0) removed, (2,0) on r = rw kept')
    ! option B (delta = 1/6): the sliver |Z| < 1/6 survives outside the ellipse, the rest of the strip goes
    damp_rw = 2.5d0; damp_delta = 1.0d0/6.0d0; call build(0.0d0)
    call check(.not. masked_at(3, 0) .and. .not. masked_at(6, 0), 'option B: the sliver (1,0), (2,0) must be kept')
    call check(masked_at(4, 1) .and. masked_at(1, 0), 'option B: strip outside the sliver removed, ellipse removed')
    call check(count_full() == 64, 'option B count at n = 3 (regression number)')
    damp_delta = 0.0d0

    ! T5: the count against the angle, n = 3 and 5
    do in = 1, 2
      n = NS(in); alfa0 = ka/n; beta0 = kb/n
      do k = 1, 2
        damp_rw = merge(1.0d0, 2.5d0, k == 1)
        cmin(k) = huge(1); cmax(k) = 0
        do ia = 0, 180
          ang = 0.25d0*ia
          call build(ang)
          cnt = count_full()
          cmin(k) = min(cmin(k), cnt); cmax(k) = max(cmax(k), cnt)
        end do
      end do
      if (has_terminal) write (*, '(A,I3,A,I4,A,I4,A,I4,A,I4)') '   T5 n =', n, ': masked columns over theta = 0 .. 45 (0.25 steps): ellipse ', &
        cmin(1), ' ..', cmax(1), ',  option A rw = 2.5 ', cmin(2), ' ..', cmax(2)
    end do

    ! the start-up summary on the n = 5 lattice, theta = 0
    n = 5; alfa0 = ka/n; beta0 = kb/n; damp_rw = 2.5d0; damp_delta = 0.0d0
    call build(0.0d0)
    if (has_terminal) call damping_report()
    deallocate (gplus)
  end subroutine geometry_part

end program test_damping_mask
