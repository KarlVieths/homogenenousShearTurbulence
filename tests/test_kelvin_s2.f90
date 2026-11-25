! Kelvin mode in an oscillating spanwise shear, column by column, as a time
! series (tests/KELVIN_S2.md).
!
!   mpirun -np 1 build-cpu/test_kelvin_s2 deck.in      (linear = .true. is forced)
!
! One Fourier column (kx, kz) holds one plane wave, in the shear U = (S y, 0,
! S2(t) y) and with the weights of its column, gl = glu (the lift-up, the
! source of eta) and go = gorr (the Orr energy transfer); the production filter
! G of the column is gl = go = G, and prod_lu, prod_orr weight the two apart
! (doc.production/selective_lu_orr.md).  The weights are computed here from the
! deck, gl = prod_lu G(k_h) and go = prod_orr G(k_h) with G the smoothstep of
! doc.production/production_filter.md, and not read from the solver: the
! solver's arrays glu, gorr must equal them (else the test fails), or a wrong
! construction of the weights would also be in the reference.
!
!   ky(t)  = ky0 - kx S t - kz gamma_z(t),            gamma_z = int S2 dt
!   v(t)   = v0 (k0^2/k(t)^2)^((1 + go)/2) E(t),      E = exp(-ni int k^2 dt)
!   eta(t) = E(t) [eta0 - i gl v0 int (S kz - S2 kx) (k0^2/k^2)^((1 + go)/2) dt]
!   |u|^2  = (k^2 |v|^2 + |eta|^2)/kh^2   (= |u0|^2 E^2 for gl = go = 0)
!
! for any S2(t).  The program integrates the mode with the production time
! step, and at every output time compares, with the analytic ky(t):
!   - the projections of v and eta on exp(i ky y)  (amplitude errors)
!   - the projection residual  |v - vhat exp(i ky y)|/|v|  (a wrong phase or
!     wavenrgber in the shear-periodic shift shows up here)
!   - the column energy, and for gl = go = 0 against |u0|^2 E^2 (no gain)
!   - gamma_y of the code against its closed form
! Amplitude errors are relative to the largest reference amplitude of the
! run.  With refine = r the run is repeated at deltat/2, ..., deltat/2^r and
! the observed order (in v, or in eta with order_qty = 'eta') is printed;
! order_min > 0 makes it a pass criterion for every pair of levels, and
! order_fine_min > 0 for the finest pair alone (the others may still be
! outside the asymptotic range).
!
! Switch times (prod_lu_start, prod_orr_start): before them the weight of a mechanism is G alone, from
! them prod_lu G (prod_orr G), so that g_lu(t) and g_orr(t) are step functions and the mode is
! integrated piecewise: v = v0 E (k0^2/k_s^2)^p1 (k_s^2/k^2)^p2 after a switch of the Orr weight at
! t_s (p = (1 + g_orr)/2 before and after, k_s = k(t_s)), eta = E [eta0 - i v0 int g_lu(t) b(t) F(t) dt]
! with F = v/(v0 E).  The switch times must lie on the time grid (a multiple of the coarsest step from
! the start; the program stops otherwise): the solver cuts its steps to end on them (hst), this test
! does not.  After every step the weights of the solver (glu, gorr on the device) are compared with the
! weights the deck gives for that step, taken at its mid-point.
!
! The mode is set by the &kelvin group of the deck (read here, the solver
! does not know it): ix_mode, iz_mode, iy_mode (ky0 = 2 pi iy_mode/ly at t =
! 0), v0, eta0 as (re, im) pairs, dt_out, refine, order_min, order_fine_min,
! order_qty, tol, csv.  With
! csv = 'file' the finest run is written as a table that kelvin_ref.py
! --compare checks against its own, independent, reference.
program test_kelvin_s2

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
  use test_common
  use hst_input
  use hst_mpi
  use hst_setup
  use hst_derivatives
  use hst_equations

  implicit none

  real(C_DOUBLE), parameter :: PI = 3.141592653589793d0
  complex(C_DOUBLE_COMPLEX), parameter :: I = (0.0d0, 1.0d0)
  ! 5-point Gauss-Legendre rule on [-1, 1]
  real(C_DOUBLE), parameter :: GLX(5) = (/-0.9061798459386640d0, -0.5384693101056831d0, 0.0d0, &
                                          0.5384693101056831d0, 0.9061798459386640d0/)
  real(C_DOUBLE), parameter :: GLW(5) = (/0.2369268850561891d0, 0.4786286704993665d0, 0.5688888888888889d0, &
                                          0.4786286704993665d0, 0.2369268850561891d0/)

  ! the &kelvin group
  integer :: ix_mode, iz_mode, iy_mode, refine
  real(C_DOUBLE) :: v0re, v0im, eta0re, eta0im, dt_out, order_min, order_fine_min, tol
  character(len=256) :: csv
  character(len=8) :: order_qty
  namelist /kelvin/ ix_mode, iz_mode, iy_mode, v0re, v0im, eta0re, eta0im, dt_out, refine, order_min, order_fine_min, order_qty, &
    tol, csv

  integer(C_INT) :: ix, iz
  real(C_DOUBLE) :: kx, kz, kh2, ky0q, k02, g_lu, g_orr, g_lu0, g_orr0, pa, pb, tsw, ks2, t0, dt0, fq
  integer :: nbad
  complex(C_DOUBLE_COMPLEX) :: v0, eta0
  integer :: nout, nper0, lev, ierr
  real(C_DOUBLE) :: err_prev, order, finest(6)
  logical :: ok, failed, weights_ok
  real(C_DOUBLE), allocatable :: tt(:), kyv(:), gcode(:), gex(:), s2v(:)
  complex(C_DOUBLE_COMPLEX), allocatable :: vref(:), etaref(:), vnum(:), etanum(:)
  real(C_DOUBLE), allocatable :: eref(:), enrg(:), e0e2(:), resv(:), reseta(:)

  call test_start()
  linear = .true.
  if (nproc /= 1) error stop 'test_kelvin_s2 runs on one rank'
  call read_kelvin_group()
  call test_setup()
  if (ystretch /= 0.0d0) error stop 'test_kelvin_s2 needs a uniform y grid'

  ix = ix_mode; iz = iz_mode
  kx = alfa0*ix; kz = beta0*iz
  kh2 = kx*kx + kz*kz
  if (kh2 == 0.0d0) error stop 'the (0,0) column is not a Kelvin mode'
  ky0q = 2.0d0*PI*iy_mode/ly
  ! the weights of the column from the deck: after the switch times prod_lu G, prod_orr G, before G alone
  g_lu0 = filter_g(sqrt(kh2)); g_orr0 = filter_g(sqrt(kh2))
  g_lu = prod_lu*g_lu0; g_orr = prod_orr*g_orr0
  pa = 0.5d0*(1.0d0 + g_orr); pb = 0.5d0*(1.0d0 + g_orr0)
  weights_ok = .true.; nbad = 0
  if (trim(order_qty) /= 'v' .and. trim(order_qty) /= 'eta') error stop "order_qty must be 'v' or 'eta'"
  v0 = cmplx(v0re, v0im, C_DOUBLE_COMPLEX); eta0 = cmplx(eta0re, eta0im, C_DOUBLE_COMPLEX)
  t0 = time
  k02 = kh2 + ky_of(t0)**2
  tsw = max(prod_orr_start, t0)            ! the Orr weight changes here (t0: it is in effect from the start)
  ks2 = kh2 + ky_of(tsw)**2
  if (dt_fixed <= 0.0d0) dt_fixed = 1.0d-3
  dt0 = dt_fixed
  if (dt_out <= 0.0d0) then
    ! 8 outputs per S2 period: for kx = 0 the mode is back at ky0 after every
    ! period, where v = v0 E for any G, so period ends alone cannot tell the
    ! filter from no filter; the gain is largest half a period in
    dt_out = t_max/10.0d0
    if (s2_period > 0.0d0) dt_out = s2_period/8.0d0
  end if
  nout = max(1, nint(t_max/dt_out))
  nper0 = max(1, nint(dt_out/dt0))
  dt0 = dt_out/nper0
  call check_switch_on_grid(prod_lu_start, 'prod_lu_start')
  call check_switch_on_grid(prod_orr_start, 'prod_orr_start')
  allocate (tt(0:nout), kyv(0:nout), gcode(0:nout), gex(0:nout), s2v(0:nout))
  allocate (vref(0:nout), etaref(0:nout), vnum(0:nout), etanum(0:nout))
  allocate (eref(0:nout), enrg(0:nout), e0e2(0:nout), resv(0:nout), reseta(0:nout))
  call make_reference()

  if (has_terminal) then
    write (*, '(A,I0,A,I0,A,I0,A,F8.4,A,F8.4,A,F6.3,A,F6.3)') '   column (', ix, ',', iz, '), ky0 index ', iy_mode, &
      ', g_LU = ', g_lu, ', g_Orr = ', g_orr, ', S = ', S, ', |S2| <= ', abs(s2_amplitude) + abs(s2_mean)
    write (*, '(A,F8.4,A,I0,A,F8.4,A,ES9.2)') '   to t = ', t0 + nout*dt_out, ' in ', nout, ' outputs of ', dt_out, &
      ', S2 period ', s2_period
    if (prod_lu_start > t0) write (*, '(A,F9.4,A,F8.4,A,F8.4)') '   lift-up weight from t = ', prod_lu_start, ': g_LU ', g_lu0, ' -> ', g_lu
    if (prod_orr_start > t0) write (*, '(A,F9.4,A,F8.4,A,F8.4)') '   Orr weight from t = ', prod_orr_start, ': g_Orr ', g_orr0, ' -> ', g_orr
  end if

  failed = .false.
  err_prev = -1.0d0
  do lev = 0, refine
    call run_level(nper0*2**lev, dt0/2**lev, finest, lev == refine)
    fq = finest(1)
    if (trim(order_qty) == 'eta') fq = finest(2)
    order = 0.0d0
    if (lev > 0 .and. err_prev > 1.0d-11 .and. fq > 0.0d0) order = log(err_prev/fq)/log(2.0d0)
    ok = finest(1) <= tol .and. finest(2) <= tol .and. finest(3) <= tol .and. finest(4) <= tol &
         .and. finest(5) <= tol .and. finest(6) <= 1.0d-10
    if (lev > 0 .and. order_min > 0.0d0 .and. err_prev > 1.0d-11 .and. order < order_min) ok = .false.
    if (lev == refine .and. lev > 0 .and. order_fine_min > 0.0d0) then
      if (err_prev <= 1.0d-11) then
        ok = .false.            ! the order is meaningless at round-off: a deck that asks for it must be resolved
        if (has_terminal) write (*, '(A)') '   order_fine_min: the error of the coarser level is at round-off, no order to judge'
      else if (order < order_fine_min) then
        ok = .false.
      end if
    end if
    if (has_terminal) then
      write (*, '(A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2)') '   dt = ', dt0/2**lev, &
        '  err v ', finest(1), '  eta ', finest(2), '  energy ', finest(3), '  proj ', finest(4), '  G=0 gain ', finest(5)
      if (lev > 0 .and. order /= 0.0d0) write (*, '(A,A,A,F6.2)') '        observed order in ', trim(order_qty), ': ', order
    end if
    if (.not. weights_ok) then
      ok = .false.
      if (has_terminal) write (*, '(A,I0,A)') '   the weights of the solver differ from the deck in ', nbad, ' steps'
    end if
    if (.not. ok) failed = .true.
    err_prev = fq
  end do
  if (has_terminal) write (*, '(A,ES10.2)') '   max |gamma_y(code) - gamma_z(closed form)| = ', finest(6)
  call test_finish(.not. failed)

contains

  ! The &kelvin group of the deck (all optional).
  subroutine read_kelvin_group()
    character(len=256) :: deck
    integer :: unit, ios
    ix_mode = 1; iz_mode = 1; iy_mode = 1
    v0re = 1.0d0; v0im = 0.3d0; eta0re = 0.4d0; eta0im = -0.2d0
    dt_out = 0.0d0; refine = 0; order_min = 0.0d0; order_fine_min = 0.0d0; order_qty = 'v'; tol = 1.0d-6; csv = ''
    deck = 'hst.in'
    if (command_argument_count() >= 1) call get_command_argument(1, deck)
    open (newunit=unit, file=trim(deck), status='old', action='read', iostat=ios)
    if (ios /= 0) error stop 'cannot open the deck'
    read (unit, nml=kelvin, iostat=ios)
    if (ios > 0) error stop 'cannot read the &kelvin group'
    close (unit)
  end subroutine read_kelvin_group

  !------------------------------------------------------ closed forms -----
  ! The production filter of a column, from the definition (hst_production):
  ! G = smoothstep((k_h/prod_kc - (1 - prod_edge))/(2 prod_edge)), a step for prod_edge = 0
  ! (columns on k_h = prod_kc within 1e-6 are forced), 1 for prod_kc = 0.
  real(C_DOUBLE) function filter_g(kh)
    real(C_DOUBLE), intent(in) :: kh
    real(C_DOUBLE) :: r, x
    filter_g = 1.0d0
    if (prod_kc <= 0.0d0) return
    r = kh/prod_kc
    if (prod_edge > 0.0d0) then
      x = min(1.0d0, max(0.0d0, (r - (1.0d0 - prod_edge))/(2.0d0*prod_edge)))
      filter_g = x*x*(3.0d0 - 2.0d0*x)
    else if (r < 1.0d0 - 1.0d-6) then
      filter_g = 0.0d0
    end if
  end function filter_g

  ! S2(t) and gamma_z(t) = int_0^t S2, written from the definition of the
  ! deck (zero before s2_start; s2_mean + s2_amplitude sin(omega (t - s2_start))
  ! or, for s2_period = 0, the constant s2_mean + s2_amplitude).
  real(C_DOUBLE) function s2_ex(t)
    real(C_DOUBLE), intent(in) :: t
    s2_ex = 0.0d0
    if (t < s2_start) return
    if (s2_period > 0.0d0) then
      s2_ex = s2_mean + s2_amplitude*sin(2.0d0*PI*(t - s2_start)/s2_period)
    else
      s2_ex = s2_mean + s2_amplitude
    end if
  end function s2_ex

  real(C_DOUBLE) function gamma_ex(t)
    real(C_DOUBLE), intent(in) :: t
    real(C_DOUBLE) :: om
    gamma_ex = 0.0d0
    if (t <= s2_start) return
    if (s2_period > 0.0d0) then
      om = 2.0d0*PI/s2_period
      gamma_ex = s2_mean*(t - s2_start) + s2_amplitude/om*(1.0d0 - cos(om*(t - s2_start)))
    else
      gamma_ex = (s2_mean + s2_amplitude)*(t - s2_start)
    end if
  end function gamma_ex

  real(C_DOUBLE) function ky_of(t)
    real(C_DOUBLE), intent(in) :: t
    ky_of = ky0q - kx*S*t - kz*gamma_ex(t)
  end function ky_of

  ! The weights of the column at time t: G alone before the switch time, prod_lu G (prod_orr G) from it.
  real(C_DOUBLE) function glu_of(t)
    real(C_DOUBLE), intent(in) :: t
    glu_of = g_lu0
    if (t >= prod_lu_start) glu_of = g_lu
  end function glu_of

  real(C_DOUBLE) function gorr_of(t)
    real(C_DOUBLE), intent(in) :: t
    gorr_of = g_orr0
    if (t >= prod_orr_start) gorr_of = g_orr
  end function gorr_of

  ! v/(v0 E) = prod of (k_a^2/k_b^2)^p over the stretches of constant Orr weight, p = (1 + go)/2:
  ! (k0^2/k^2)^pb up to the switch, (k0^2/ks^2)^pb (ks^2/k^2)^pa after it (tsw = t0 when the
  ! weight is in effect from the start).
  real(C_DOUBLE) function vfac(t)
    real(C_DOUBLE), intent(in) :: t
    real(C_DOUBLE) :: k2
    k2 = kh2 + ky_of(t)**2
    if (t < tsw) then
      vfac = (k02/k2)**pb
    else
      vfac = (k02/ks2)**pb*(ks2/k2)**pa
    end if
  end function vfac

  ! A switch time inside the run must be a multiple of the coarsest step from t0 (the test steps on that
  ! grid, so that the switch is a step boundary, which is what the solver does by cutting its steps).
  subroutine check_switch_on_grid(ts, name)
    real(C_DOUBLE), intent(in) :: ts
    character(len=*), intent(in) :: name
    real(C_DOUBLE) :: q
    if (ts <= t0 .or. ts >= t0 + nout*dt_out) return
    q = (ts - t0)/dt0
    if (abs(q - nint(q)) > 1.0d-9*max(1.0d0, abs(q))) then
      if (has_terminal) write (*, '(A,A,A,ES12.5,A,ES12.5,A)') '   ', name, ' = ', ts, ' is not a multiple of the time step ', dt0, ' from t0'
      error stop 'a switch time must lie on the time grid'
    end if
  end subroutine check_switch_on_grid

  ! Adds the integrals over [ta, tb] of k^2 (viscous decay) and of the lift-up
  ! source g_lu(t) (S kz - S2 kx) v/(v0 E) to aint and bint; the interval is cut at the
  ! switch times, where the integrand jumps.
  subroutine add_integrals(ta, tb, aint, bint)
    real(C_DOUBLE), intent(in) :: ta, tb
    real(C_DOUBLE), intent(inout) :: aint, bint
    real(C_DOUBLE) :: bp(4), tmp
    integer :: nb, ib
    nb = 1; bp(1) = ta
    if (prod_lu_start > ta .and. prod_lu_start < tb) then
      nb = nb + 1; bp(nb) = prod_lu_start
    end if
    if (prod_orr_start > ta .and. prod_orr_start < tb) then
      nb = nb + 1; bp(nb) = prod_orr_start
    end if
    nb = nb + 1; bp(nb) = tb
    if (nb == 4 .and. bp(2) > bp(3)) then
      tmp = bp(2); bp(2) = bp(3); bp(3) = tmp
    end if
    do ib = 1, nb - 1
      call add_panels(bp(ib), bp(ib + 1), aint, bint)
    end do
  end subroutine add_integrals

  subroutine add_panels(ta, tb, aint, bint)
    real(C_DOUBLE), intent(in) :: ta, tb
    real(C_DOUBLE), intent(inout) :: aint, bint
    integer :: npan, ip, ig
    real(C_DOUBLE) :: h, tm, tau, k2
    npan = max(1, ceiling((tb - ta)/0.02d0))
    h = (tb - ta)/npan
    do ip = 1, npan
      tm = ta + (ip - 0.5d0)*h
      do ig = 1, 5
        tau = tm + 0.5d0*h*GLX(ig)
        k2 = kh2 + ky_of(tau)**2
        aint = aint + 0.5d0*h*GLW(ig)*k2
        bint = bint + 0.5d0*h*GLW(ig)*glu_of(tau)*(S*kz - s2_ex(tau)*kx)*vfac(tau)
      end do
    end do
  end subroutine add_panels

  ! The reference at the output times, and what does not depend on the run.
  subroutine make_reference()
    integer :: m
    real(C_DOUBLE) :: aint, bint, k2, visc, e0
    aint = 0.0d0; bint = 0.0d0
    e0 = 0.5d0*((k02/kh2)*abs(v0)**2 + abs(eta0)**2/kh2)
    do m = 0, nout
      tt(m) = t0 + m*dt_out
      if (m > 0) call add_integrals(tt(m - 1), tt(m), aint, bint)
      kyv(m) = ky_of(tt(m))
      k2 = kh2 + kyv(m)**2
      visc = exp(-ni*aint)
      vref(m) = v0*vfac(tt(m))*visc
      etaref(m) = visc*(eta0 - I*v0*bint)
      eref(m) = 0.5d0*((k2/kh2)*abs(vref(m))**2 + abs(etaref(m))**2/kh2)
      e0e2(m) = e0*visc*visc
      gex(m) = gamma_ex(tt(m))
      s2v(m) = s2_ex(tt(m))
    end do
  end subroutine make_reference

  !------------------------------------------------------------- the run ----
  ! The plane wave at t0, in the interior rows and the ghost rows (and the
  ! conjugate column for kx = 0).
  subroutine set_initial()
    integer :: iy
    real(C_DOUBLE) :: ky
    complex(C_DOUBLE_COMPLEX) :: u0, w0, ph
    ky = ky_of(t0)
    u0 = (-kx*ky*v0 - I*kz*eta0)/kh2
    w0 = (-kz*ky*v0 + I*kx*eta0)/kh2
    V = 0; oldrhs = 0; rhs = 0
    do iy = ny0 - 2, nyN + 2
      ph = exp(I*ky*y(iy))
      V(iy, iz, ix, 1) = u0*ph
      V(iy, iz, ix, 2) = v0*ph
      V(iy, iz, ix, 3) = w0*ph
      if (ix == 0) V(iy, -iz, 0, :) = conjg(V(iy, iz, 0, :))
    end do
    !$omp target update to(V, oldrhs, rhs)
  end subroutine set_initial

  ! The solver's weights of the column (on the device) against the deck's for the step that started at
  ! tstart: the deck's value at the mid-point of the step (a switch time is on the grid, so the mid-point
  ! is unambiguous).  A weight updated on the host only, late, early or for the wrong mechanism fails here.
  subroutine check_weights(tstart, dt)
    real(C_DOUBLE), intent(in) :: tstart, dt
    real(C_DOUBLE) :: tm, el, eo
    !$omp target update from(glu, gorr)
    tm = tstart + 0.5d0*dt
    el = abs(glu(iz, ix) - glu_of(tm)); eo = abs(gorr(iz, ix) - gorr_of(tm))
    if (el > 1.0d-14 .or. eo > 1.0d-14) then
      weights_ok = .false.
      nbad = nbad + 1
      if (has_terminal .and. nbad <= 3) write (*, '(A,F10.5,A,2ES14.6,A,2ES14.6)') '   step from t = ', tstart, &
        ': solver weights (glu, gorr) =', glu(iz, ix), gorr(iz, ix), ', deck:', glu_of(tm), gorr_of(tm)
    end if
  end subroutine check_weights

  ! Amplitude of v, eta and the energy of the column, and the projection
  ! residuals, at the current V.
  subroutine measure(m)
    integer, intent(in) :: m
    integer :: iy, n
    real(C_DOUBLE) :: ky, nv, nrv, ne, nre, en
    complex(C_DOUBLE_COMPLEX) :: vh, eh, ph, eta
    !$omp target update from(V)
    ky = kyv(m)
    n = nyN - ny0 + 1
    vh = 0; eh = 0; en = 0
    do iy = ny0, nyN
      ph = exp(-I*ky*y(iy))
      eta = I*kz*V(iy, iz, ix, 1) - I*kx*V(iy, iz, ix, 3)
      vh = vh + V(iy, iz, ix, 2)*ph
      eh = eh + eta*ph
      en = en + abs(V(iy, iz, ix, 1))**2 + abs(V(iy, iz, ix, 2))**2 + abs(V(iy, iz, ix, 3))**2
    end do
    vh = vh/n; eh = eh/n
    nv = 0; nrv = 0; ne = 0; nre = 0
    do iy = ny0, nyN
      ph = exp(I*ky*y(iy))
      eta = I*kz*V(iy, iz, ix, 1) - I*kx*V(iy, iz, ix, 3)
      nv = nv + abs(V(iy, iz, ix, 2))**2
      nrv = nrv + abs(V(iy, iz, ix, 2) - vh*ph)**2
      ne = ne + abs(eta)**2
      nre = nre + abs(eta - eh*ph)**2
    end do
    vnum(m) = vh; etanum(m) = eh
    enrg(m) = 0.5d0*en/n
    resv(m) = sqrt(nrv/nv)
    reseta(m) = sqrt(nre/max(ne, tiny(1.0d0)))
    gcode(m) = gamma_y_of(time)
  end subroutine measure

  ! One run at nper steps per output interval; fills the numerics and returns
  !   f(1) v, f(2) eta, f(3) energy: max error / max reference amplitude
  !   f(4) projection residual of v and eta (max over the run)
  !   f(5) energy against |u0|^2 E^2 (G = 0 only, else 0)
  !   f(6) max |gamma_y(code) - closed form|
  subroutine run_level(nper, dt, f, last)
    integer, intent(in) :: nper
    logical, intent(in) :: last
    real(C_DOUBLE), intent(in) :: dt
    real(C_DOUBLE), intent(out) :: f(6)
    integer :: m, n
    real(C_DOUBLE) :: pv, pe, pn, tstart
    time = t0
    deltat = dt
    call set_initial()
    call measure(0)
    do m = 1, nout
      do n = 1, nper
        tstart = time
        call timestep()
        call check_weights(tstart, dt)
      end do
      call measure(m)
    end do
    pv = maxval(abs(vref)); pe = maxval(abs(etaref)); pn = maxval(eref)
    f(1) = maxval(abs(vnum - vref))/pv
    f(2) = maxval(abs(etanum - etaref))/pe
    f(3) = maxval(abs(enrg - eref))/pn
    f(4) = max(maxval(resv), maxval(reseta))
    f(5) = 0.0d0
    ! no gain: both weights zero for the whole run (after the switch, and before it when it is in the run)
    if (g_lu == 0.0d0 .and. g_orr == 0.0d0 .and. (g_lu0 == 0.0d0 .or. prod_lu_start <= t0) &
        .and. (g_orr0 == 0.0d0 .or. prod_orr_start <= t0)) f(5) = maxval(abs(enrg - e0e2))/maxval(e0e2)
    f(6) = maxval(abs(gcode - gex))
    if (last .and. len_trim(csv) > 0) call write_csv(dt)
  end subroutine run_level

  subroutine write_csv(dt)
    real(C_DOUBLE), intent(in) :: dt
    integer :: unit, m
    open (newunit=unit, file=trim(csv), status='replace', action='write')
    write (unit, '(A)') '# test_kelvin_s2: numerical amplitudes of one Kelvin column (kelvin_ref.py --compare reads this header)'
    write (unit, '(A,ES24.16)') '# kx = ', kx
    write (unit, '(A,ES24.16)') '# kz = ', kz
    write (unit, '(A,ES24.16)') '# ky0 = ', ky_of(t0)
    write (unit, '(A,ES24.16)') '# g_lu = ', g_lu
    write (unit, '(A,ES24.16)') '# g_orr = ', g_orr
    write (unit, '(A,ES24.16)') '# g_lu_before = ', g_lu0
    write (unit, '(A,ES24.16)') '# g_orr_before = ', g_orr0
    write (unit, '(A,ES24.16E3)') '# t_lu = ', max(prod_lu_start, -1.0d300)       ! -huge: 'from the start' (-1e300 in the table)
    write (unit, '(A,ES24.16E3)') '# t_orr = ', max(prod_orr_start, -1.0d300)
    write (unit, '(A,ES24.16)') '# S = ', S
    write (unit, '(A,ES24.16)') '# s2_mean = ', s2_mean
    write (unit, '(A,ES24.16)') '# s2_amplitude = ', s2_amplitude
    write (unit, '(A,ES24.16)') '# s2_period = ', s2_period
    write (unit, '(A,ES24.16)') '# s2_start = ', s2_start
    write (unit, '(A,ES24.16)') '# nu = ', ni
    write (unit, '(A,ES24.16)') '# t0 = ', t0
    write (unit, '(A,2ES24.16)') '# v0 = ', v0
    write (unit, '(A,2ES24.16)') '# eta0 = ', eta0
    write (unit, '(A)') 't,S2,gamma_code,ky,Re_v,Im_v,Re_eta,Im_eta,energy,res_v,res_eta,dt'
    do m = 0, nout
      write (unit, '(ES24.16,11(",",ES24.16))') tt(m), s2v(m), gcode(m), kyv(m), real(vnum(m)), aimag(vnum(m)), &
        real(etanum(m)), aimag(etanum(m)), enrg(m), resv(m), reseta(m), dt
    end do
    close (unit)
  end subroutine write_csv

end program test_kelvin_s2
