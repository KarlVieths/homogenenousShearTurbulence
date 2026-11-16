! Kelvin mode in an oscillating spanwise shear, column by column, as a time
! series (tests/KELVIN_S2.md).
!
!   mpirun -np 1 build-cpu/test_kelvin_s2 deck.in      (linear = .true. is forced)
!
! One Fourier column (kx, kz) holds one plane wave, in the shear U = (S y, 0,
! S2(t) y) and with the production filter G of the column:
!
!   ky(t)  = ky0 - kx S t - kz gamma_z(t),            gamma_z = int S2 dt
!   v(t)   = v0 (k0^2/k(t)^2)^((1 + G)/2) E(t),       E = exp(-ni int k^2 dt)
!   eta(t) = E(t) [eta0 - i G v0 int (S kz - S2 kx) (k0^2/k^2)^((1 + G)/2) dt]
!   |u|^2  = (k^2 |v|^2 + |eta|^2)/kh^2   (= |u0|^2 E^2 for G = 0)
!
! for any S2(t).  The program integrates the mode with the production time
! step, and at every output time compares, with the analytic ky(t):
!   - the projections of v and eta on exp(i ky y)  (amplitude errors)
!   - the projection residual  |v - vhat exp(i ky y)|/|v|  (a wrong phase or
!     wavenrgber in the shear-periodic shift shows up here)
!   - the column energy, and for G = 0 against |u0|^2 E^2 (no gain)
!   - gamma_y of the code against its closed form
! Amplitude errors are relative to the largest reference amplitude of the
! run.  With refine = r the run is repeated at deltat/2, ..., deltat/2^r and
! the observed order is printed; order_min > 0 makes it a pass criterion.
!
! The mode is set by the &kelvin group of the deck (read here, the solver
! does not know it): ix_mode, iz_mode, iy_mode (ky0 = 2 pi iy_mode/ly at t =
! 0), v0, eta0 as (re, im) pairs, dt_out, refine, order_min, tol, csv.  With
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
  real(C_DOUBLE) :: v0re, v0im, eta0re, eta0im, dt_out, order_min, tol
  character(len=256) :: csv
  namelist /kelvin/ ix_mode, iz_mode, iy_mode, v0re, v0im, eta0re, eta0im, dt_out, refine, order_min, tol, csv

  integer(C_INT) :: ix, iz
  real(C_DOUBLE) :: kx, kz, kh2, ky0q, k02, gp, pexp, t0, dt0
  complex(C_DOUBLE_COMPLEX) :: v0, eta0
  integer :: nout, nper0, lev, ierr
  real(C_DOUBLE) :: err_prev, order, finest(6)
  logical :: ok, failed
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
  gp = gprod(iz, ix)
  pexp = 0.5d0*(1.0d0 + gp)
  v0 = cmplx(v0re, v0im, C_DOUBLE_COMPLEX); eta0 = cmplx(eta0re, eta0im, C_DOUBLE_COMPLEX)
  t0 = time
  k02 = kh2 + ky_of(t0)**2
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
  allocate (tt(0:nout), kyv(0:nout), gcode(0:nout), gex(0:nout), s2v(0:nout))
  allocate (vref(0:nout), etaref(0:nout), vnum(0:nout), etanum(0:nout))
  allocate (eref(0:nout), enrg(0:nout), e0e2(0:nout), resv(0:nout), reseta(0:nout))
  call make_reference()

  if (has_terminal) then
    write (*, '(A,I0,A,I0,A,I0,A,F8.4,A,F6.3,A,F6.3)') '   column (', ix, ',', iz, '), ky0 index ', iy_mode, &
      ', G = ', gp, ', S = ', S, ', |S2| <= ', abs(s2_amplitude) + abs(s2_mean)
    write (*, '(A,F8.4,A,I0,A,F8.4,A,ES9.2)') '   to t = ', t0 + nout*dt_out, ' in ', nout, ' outputs of ', dt_out, &
      ', S2 period ', s2_period
  end if

  failed = .false.
  err_prev = -1.0d0
  do lev = 0, refine
    call run_level(nper0*2**lev, dt0/2**lev, finest, lev == refine)
    order = 0.0d0
    if (lev > 0 .and. err_prev > 1.0d-11 .and. finest(1) > 0.0d0) order = log(err_prev/finest(1))/log(2.0d0)
    ok = finest(1) <= tol .and. finest(2) <= tol .and. finest(3) <= tol .and. finest(4) <= tol &
         .and. finest(5) <= tol .and. finest(6) <= 1.0d-10
    if (lev > 0 .and. order_min > 0.0d0 .and. err_prev > 1.0d-11 .and. order < order_min) ok = .false.
    if (has_terminal) then
      write (*, '(A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2)') '   dt = ', dt0/2**lev, &
        '  err v ', finest(1), '  eta ', finest(2), '  energy ', finest(3), '  proj ', finest(4), '  G=0 gain ', finest(5)
      if (lev > 0 .and. order /= 0.0d0) write (*, '(A,F6.2)') '        observed order in v: ', order
    end if
    if (.not. ok) failed = .true.
    err_prev = finest(1)
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
    dt_out = 0.0d0; refine = 0; order_min = 0.0d0; tol = 1.0d-6; csv = ''
    deck = 'hst.in'
    if (command_argument_count() >= 1) call get_command_argument(1, deck)
    open (newunit=unit, file=trim(deck), status='old', action='read', iostat=ios)
    if (ios /= 0) error stop 'cannot open the deck'
    read (unit, nml=kelvin, iostat=ios)
    if (ios > 0) error stop 'cannot read the &kelvin group'
    close (unit)
  end subroutine read_kelvin_group

  !------------------------------------------------------ closed forms -----
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

  ! Adds the integrals over [ta, tb] of k^2 (viscous decay) and of the lift-up
  ! weight (S kz - S2 kx)(k0^2/k^2)^p to aint and bint.
  subroutine add_integrals(ta, tb, aint, bint)
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
        bint = bint + 0.5d0*h*GLW(ig)*(S*kz - s2_ex(tau)*kx)*(k02/k2)**pexp
      end do
    end do
  end subroutine add_integrals

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
      vref(m) = v0*(k02/k2)**pexp*visc
      etaref(m) = visc*(eta0 - I*gp*v0*bint)
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
    real(C_DOUBLE) :: pv, pe, pn
    time = t0
    deltat = dt
    call set_initial()
    call measure(0)
    do m = 1, nout
      do n = 1, nper
        call timestep()
      end do
      call measure(m)
    end do
    pv = maxval(abs(vref)); pe = maxval(abs(etaref)); pn = maxval(eref)
    f(1) = maxval(abs(vnum - vref))/pv
    f(2) = maxval(abs(etanum - etaref))/pe
    f(3) = maxval(abs(enrg - eref))/pn
    f(4) = max(maxval(resv), maxval(reseta))
    f(5) = 0.0d0
    if (gp == 0.0d0) f(5) = maxval(abs(enrg - e0e2))/maxval(e0e2)
    f(6) = maxval(abs(gcode - gex))
    if (last .and. len_trim(csv) > 0) call write_csv()
  end subroutine run_level

  subroutine write_csv()
    integer :: unit, m
    open (newunit=unit, file=trim(csv), status='replace', action='write')
    write (unit, '(A)') '# test_kelvin_s2: numerical amplitudes of one Kelvin column (kelvin_ref.py --compare reads this header)'
    write (unit, '(A,ES24.16)') '# kx = ', kx
    write (unit, '(A,ES24.16)') '# kz = ', kz
    write (unit, '(A,ES24.16)') '# ky0 = ', ky_of(t0)
    write (unit, '(A,ES24.16)') '# G = ', gp
    write (unit, '(A,ES24.16)') '# S = ', S
    write (unit, '(A,ES24.16)') '# s2_mean = ', s2_mean
    write (unit, '(A,ES24.16)') '# s2_amplitude = ', s2_amplitude
    write (unit, '(A,ES24.16)') '# s2_period = ', s2_period
    write (unit, '(A,ES24.16)') '# s2_start = ', s2_start
    write (unit, '(A,ES24.16)') '# nu = ', ni
    write (unit, '(A,ES24.16)') '# t0 = ', t0
    write (unit, '(A,2ES24.16)') '# v0 = ', v0
    write (unit, '(A,2ES24.16)') '# eta0 = ', eta0
    write (unit, '(A)') 't,S2,gamma_code,ky,Re_v,Im_v,Re_eta,Im_eta,energy,res_v,res_eta'
    do m = 0, nout
      write (unit, '(ES24.16,10(",",ES24.16))') tt(m), s2v(m), gcode(m), kyv(m), real(vnum(m)), aimag(vnum(m)), &
        real(etanum(m)), aimag(etanum(m)), enrg(m), resv(m), reseta(m)
    end do
    close (unit)
  end subroutine write_csv

end program test_kelvin_s2
