! Runtime statistics in the layout of the CPL code (io.cpl, spanwShear
! variant), reduced on the device and written by the terminal rank.  With
! a Stokes layer, stokes_runtime.dat adds  time  energy_out  energy_in
! diss_out  diss_in  (region averages of <u_i u_i>/2 and nu <grad u : grad u>,
! outside and inside |y - ly/2| < 8 delta, as the StokesLayer variant of
! io.cpl).
!
! All statistics are *y-averaged* (box-mean) quantities: the fluctuation
! moments are formed by Parseval's theorem along the spectral directions
! (x streamwise, z spanwise) and then averaged over the shear direction y,
! so every column has the units of a spatial mean per unit volume and is
! ready to use without further rescaling.  The kinematic viscosity nu = 1/re
! is folded into the dissipation.
!
! The files are in statistics/ (made by make_output_dirs, hst_io), opened with
! position='append': they are never truncated, a restart continues them, and
! a new run in a directory that has them adds its lines after the old ones.
!
! Runtimedata columns:
!   time  meanflowx  meanflowy  S  S2  gamma_x  gamma_y  deltat  cfl*deltat
!   energy  diss  uv  vw
! variances_runtime.dat columns:
!   time  uu  vv  ww  uw
! damping_runtime.dat (hst_damping active):
!   time  theta_deg  ncol  energy_damped  power_damped
!   (ly/2 <u_i u_i> over the damped columns and ly/2 <2 g u_i u_i>, the
!   power the damping removes; zero for the hard mask)
! damping_families.dat (hst_damping active; DAMPING.md section 5): the second moments of
!   six overlapping families of columns, in the shear frame of the mask
!   (e_par at the angle theta of damping_theta), as fractions of the fluctuation
!   and per unit volume like variances_runtime.dat:
!     time  theta_deg  share15  and per family  share  <u_par^2>  <v^2>  <u_perp^2>  -<u_par v>
!   share15 = energy fraction at |k_h| < 1.5 damp_kperp (|k| Lz/2pi < 1.5 of the small box);
!   the families (tol = half the smaller lattice spacing, kb = damp_kperp, ka = damp_kpar):
!     1 mean         the (0,0) column (about the box mean)
!     2 span_uniform |k_perp| < tol  (spanwise uniform: the Orr / burst columns)
!     3 stream_unif. |k_par| < tol   (streamwise uniform)
!     4 narrow       |k_perp| >= kb  (wavelength across the shear <= small-box Lz)
!     5 strip_small  |k_perp| < kb and r >= damp_rw (the surviving small-scale strip)
!     6 streaks      narrow and |k_par| < tol (the streaks of every length)
!   Families 2, 3 overlap 4, 5 and 6: the shares do not add up to one.
!   -<u_par v> = -(<u v> cos theta + <v w> sin theta) of the family's columns.
! statistics/spectra/exz<i>.dat (with every field snapshot): the column energies
!   ix  iz  kx  kz  E  with E = ly/2 sum_y |u|^2 of the column (modes with
!   ix > 0 counted twice, as in energy), averaged over the dt_stat samples
!   since the previous snapshot.
! all in the uniform (streamwise, shearwise, spanwise) = (u, v, w) naming of
! this solver: x streamwise with the Fourier wavenumber alfa0, z spanwise
! with beta0, y the compact-FD / shear direction (U = S*y) carrying v.
! energy = <u_i u_i>/2 (the turbulent kinetic energy k),
! diss = nu <du_i/dx_j du_i/dx_j> (dissipation, the pseudo-dissipation
! nu<|grad u|^2>; here from the compact derivatives, in CPL from centred
! differences), uv = <u v> and vw = <v w> are the Reynolds shear stresses
! (no factor 1/2), and the variances uu = <u u> etc. are the mean normal
! Reynolds stresses.  In variances_runtime.dat the last column is the
! Reynolds shear stress uw = <u w>: the cross-correlation of the streamwise
! and spanwise fluctuations (not <u v>).  meanflowx/y are
! the integrals of the mean profiles (kept as integrals, as in CPL).
! Averaging operator.  The fluctuation is the field minus its box mean, i.e.
! the (0,0,0) mode only: the (0,0) column keeps its y-dependent part
! (u00(y) - <u00>_y, w00(y) - <w00>_y; v00 is identically zero), which is part
! of the fluctuation like any other mode.  This is the spatial operator of
! the post-processor (post/cpl, mod_rey_decomp.cpl), so a live statistic and
! the one reconstructed from the snapshots differ only in their temporal
! part (instantaneous here; time or phase average there).  The box mean is
! constant in time (v00 = 0 and the y-average of u00, w00 is conserved), so
! removing it instant by instant equals removing its time or phase mean.
! It needs no extra pass: the box-height integrals mfx, mfz of the (0,0)
! column are already reduced, and with them the sum of squares about the mean
! follows from  sum dyl (f - fbar)^2 = sum dyl f^2 - (sum dyl f)^2/sum dyl.
! The column energies exz of the spectrum use it the same way.
! This operator presupposes statistical homogeneity in y (a constant mean).
! A Stokes layer (stokes_active()) makes the flow inhomogeneous in y: the
! spatial average is then possible along x and z only, and the statistics
! are profiles (one-point, one-time) or y-dependent quantities, not the
! y-averaged numbers written here.  That case is NOT covered by the
! definition above: for it the whole (0,0) column (the imposed, phase-locked
! mean flow) is simply excluded, as before the change (inc00 = 0), a stopgap
! and future work (y-resolved statistics, mean profile removed plane by plane).
! Modes with ix > 0 count twice.
module hst_stats

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
  use hst_linsolve, only: line_solve, KIND_DY
  use hst_derivatives, only: s2_of, gamma_y_of
  use hst_stokes, only: stokes_active
  use hst_damping, only: damping_active, damping_theta, damping_soft_factor, damping_radius, ncol_damped

  implicit none
  private
  public :: outstats, open_runtimedata, close_runtimedata, spectrum_write

  integer, parameter :: unit_rt = 121, unit_var = 122, unit_sl = 123, unit_dp = 124, unit_df = 125
  integer, parameter :: NFAM = 6                      ! families of columns of damping_families.dat
  real(C_DOUBLE), parameter :: FTOL = 1.0d-6          ! a column within FTOL of the strip edge / r = rw counts as on the boundary
  real(C_DOUBLE), allocatable, save :: fcol(:, :, :)  ! per-column second moments (uu, vv, ww, uv, vw, uw): device scratch
  real(C_DOUBLE), allocatable, save :: exz(:, :), exz_acc(:, :)   ! column energies: device scratch, host sum
  integer, save :: nacc = 0
  real(C_DOUBLE), save :: t_acc0 = 0.0d0
  real(C_DOUBLE), save :: mean_acc = 0.0d0   ! sum over the samples of (box mean u)^2 + (box mean w)^2 times ly, for E(0,0)

contains

  ! the directory statistics/ must exist (make_output_dirs)
  subroutine open_runtimedata()
    if (has_terminal) open (unit=unit_rt, file='statistics/Runtimedata', action='write', position='append')
    if (has_terminal) open (unit=unit_var, file='statistics/variances_runtime.dat', action='write', position='append')
    if (has_terminal .and. stokes_active()) open (unit=unit_sl, file='statistics/stokes_runtime.dat', action='write', position='append')
    if (has_terminal .and. damping_active()) open (unit=unit_dp, file='statistics/damping_runtime.dat', action='write', position='append')
    if (has_terminal .and. damping_active()) call open_families()
    if (damping_active()) then
      allocate (fcol(-nz:nz, nx0:nxN, 6))
      fcol = 0.0d0
      !$omp target enter data map(to: fcol)
    end if
    allocate (exz(-nz:nz, nx0:nxN), exz_acc(-nz:nz, nx0:nxN))
    exz = 0.0d0; exz_acc = 0.0d0; nacc = 0; t_acc0 = time; mean_acc = 0.0d0
    !$omp target enter data map(to: exz)
  end subroutine open_runtimedata

  subroutine close_runtimedata()
    if (has_terminal) close (unit_rt)
    if (has_terminal) close (unit_var)
    if (has_terminal .and. stokes_active()) close (unit_sl)
    if (has_terminal .and. damping_active()) close (unit_dp)
    if (has_terminal .and. damping_active()) close (unit_df)
    if (allocated(fcol)) then
      !$omp target exit data map(delete: fcol)
      deallocate (fcol)
    end if
    !$omp target exit data map(delete: exz)
    deallocate (exz, exz_acc)
  end subroutine close_runtimedata

  ! damping_families.dat: appended to; the header line only for a new file
  subroutine open_families()
    integer(C_SIZE_T) :: fsize
    open (unit=unit_df, file='statistics/damping_families.dat', action='write', position='append')
    inquire (unit=unit_df, size=fsize)
    if (fsize <= 0) then
      write (unit_df, '(A)') '# time theta_deg share15 | per family: share <u_par^2> <v^2> <u_perp^2> -<u_par v> | families: '// &
        'mean span_uniform stream_uniform narrow strip_small streaks (DAMPING.md section 5, hst_stats.f90)'
    end if
  end subroutine open_families

  ! Second moments of the six families (sums over the columns with the
  ! y weights of outstats, not yet divided by ly): fam(:, f) = (uu, vv, ww,
  ! uv, vw, uw) of the family f, and sh15 = the sum <u_i u_i> of the columns
  ! with |k_h| < 1.5 damp_kperp.  Collective.  mfx, mfz: the box-height
  ! integrals of the (0,0) column, whose box-mean part comes off family 1.
  subroutine family_stats(inc00, mfx, mfz, fam, sh15)
    integer(C_INT), intent(in) :: inc00
    real(C_DOUBLE), intent(in) :: mfx, mfz
    real(C_DOUBLE), intent(out) :: fam(6, NFAM), sh15
    integer(C_INT) :: ix, iz, iy, f
    real(C_DOUBLE) :: w, a(6), a1, a2, a3, a4, a5, a6, kx, kz, kpar, kperp, r, c, sn, tolk, ly_w, buf(6*NFAM + 1)
    complex(C_DOUBLE_COMPLEX) :: cu, cv, cw
    logical :: narrow, member(NFAM)
    integer :: ierr

    ! per-column second moments on the device
    !$omp target teams distribute parallel do collapse(2) default(none) &
    !$omp shared(V, dyl, fcol, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, w, cu, cv, cw, a1, a2, a3, a4, a5, a6)
    do ix = nx0, nxN
      do iz = -nz, nz
        a1 = 0.0d0; a2 = 0.0d0; a3 = 0.0d0; a4 = 0.0d0; a5 = 0.0d0; a6 = 0.0d0
        do iy = ny0, nyN
          cu = V(iy, iz, ix, 1); cv = V(iy, iz, ix, 2); cw = V(iy, iz, ix, 3)
          w = 2.0d0*dyl(iy)
          if (ix == 0) w = dyl(iy)
          a1 = a1 + w*dreal(cu*conjg(cu)); a2 = a2 + w*dreal(cv*conjg(cv)); a3 = a3 + w*dreal(cw*conjg(cw))
          a4 = a4 + w*dreal(cu*conjg(cv)); a5 = a5 + w*dreal(cw*conjg(cv)); a6 = a6 + w*dreal(cu*conjg(cw))
        end do
        fcol(iz, ix, 1) = a1; fcol(iz, ix, 2) = a2; fcol(iz, ix, 3) = a3
        fcol(iz, ix, 4) = a4; fcol(iz, ix, 5) = a5; fcol(iz, ix, 6) = a6
      end do
    end do
    !$omp target update from(fcol)

    c = cos(damping_theta(time)); sn = sin(damping_theta(time))
    tolk = 0.5d0*min(alfa0, beta0)
    fam = 0.0d0; sh15 = 0.0d0
    do ix = nx0, nxN
      kx = alfa0*ix
      do iz = -nz, nz
        kz = beta0*iz
        a = fcol(iz, ix, :)
        if (ix == 0 .and. iz == 0) then
          fam(:, 1) = fam(:, 1) + a                 ! raw; the box mean comes off after the reduction
          cycle
        end if
        kpar = kx*c + kz*sn; kperp = -kx*sn + kz*c
        r = damping_radius(kpar, kperp)
        narrow = abs(kperp) >= damp_kperp*(1.0d0 - FTOL)
        member = .false.
        member(2) = abs(kperp) < tolk
        member(3) = abs(kpar) < tolk
        member(4) = narrow
        member(5) = (.not. narrow) .and. r >= damp_rw*(1.0d0 - FTOL)
        member(6) = narrow .and. abs(kpar) < tolk
        do f = 2, NFAM
          if (member(f)) fam(:, f) = fam(:, f) + a
        end do
        if (sqrt(kx*kx + kz*kz) < 1.5d0*damp_kperp) sh15 = sh15 + a(1) + a(2) + a(3)
      end do
    end do
    buf(1:6*NFAM) = reshape(fam, [6*NFAM]); buf(6*NFAM + 1) = sh15
    call MPI_Allreduce(MPI_IN_PLACE, buf, 6*NFAM + 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    fam = reshape(buf(1:6*NFAM), [6, NFAM]); sh15 = buf(6*NFAM + 1)
    if (inc00 == 1) then
      ! the (0,0) column enters as the fluctuation about its box mean (as in outstats)
      ly_w = sum(dyl)
      fam(1, 1) = fam(1, 1) - mfx*mfx/ly_w
      fam(3, 1) = fam(3, 1) - mfz*mfz/ly_w
      fam(6, 1) = fam(6, 1) - mfx*mfz/ly_w
    else
      fam(:, 1) = 0.0d0                              ! Stokes layer: the (0,0) column is the imposed flow
    end if
    sh15 = sh15 + fam(1, 1) + fam(2, 1) + fam(3, 1)
  end subroutine family_stats

  ! Writes the column energies averaged since the previous call (or the
  ! start) and restarts the average.
  subroutine spectrum_write(filename)
    character(len=*), intent(in) :: filename
    real(C_DOUBLE), allocatable :: full(:, :)
    integer(C_INT) :: ix, iz
    integer :: ierr, unit
    allocate (full(-nz:nz, 0:nx)); full = 0.0d0
    if (nacc > 0) full(:, nx0:nxN) = exz_acc/nacc
    call MPI_Allreduce(MPI_IN_PLACE, full, size(full), MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    ! the (0,0) column enters as the fluctuation about the box mean: the
    ! sum of the squared box means (identical on every rank) comes off
    if (nacc > 0) full(0, 0) = full(0, 0) - mean_acc/nacc
    if (has_terminal) then
      open (newunit=unit, file=filename, action='write')
      write (unit, '(A,F12.5,A,F12.5,A,I0,A)') '# column energies ly/2 sum_y |u|^2 averaged over t = ', t_acc0, ' .. ', time, &
        ' (', nacc, ' samples);  ix  iz  kx  kz  E'
      do ix = 0, nx
        do iz = -nz, nz
          write (unit, '(I5,1X,I5,1X,ES15.7,1X,ES15.7,1X,ES15.7)') ix, iz, alfa0*ix, beta0*iz, 0.5d0*full(iz, ix)
        end do
      end do
      close (unit)
    end if
    deallocate (full)
    exz_acc = 0.0d0; nacc = 0; t_acc0 = time; mean_acc = 0.0d0
  end subroutine spectrum_write

  subroutine outstats()
    integer(C_INT) :: ix, iy, iz, c
    real(C_DOUBLE) :: w, eps, uv, uu, vv, ww, vw, uw, grad, mfx, mfz
    real(C_DOUBLE) :: q_in, q_out, e_in, e_out, g_in, g_out, l_in, l_out
    real(C_DOUBLE) :: sums(16), q2, ed, pd, gsoft, qc, inv_ly, ly_w
    integer(C_INT) :: inc00         ! 1: the (0,0) column enters the fluctuation (about its box mean)
    complex(C_DOUBLE_COMPLEX) :: cu, cv, cw, dq
    real(C_DOUBLE) :: fam(6, NFAM), sh15, th, ct, st, qf
    integer :: ierr, f

    ! sums over modes and rows, each row weighted by its spacing dyl: the
    ! spectral (Parseval) sums over x and z give the horizontal mean via
    ! the mode weight w, and integrating in y gives a box-height integral.
    eps = 0; uv = 0; uu = 0; vv = 0; ww = 0; vw = 0; uw = 0; mfx = 0; mfz = 0
    q_in = 0; q_out = 0; e_in = 0; e_out = 0; ed = 0; pd = 0
    gsoft = damping_soft_factor()
    inc00 = 1
    if (stokes_active()) inc00 = 0         ! the column is the imposed Stokes layer: mean flow
    !$omp target teams distribute parallel do collapse(3) default(none) &
    !$omp shared(V, k2, dyl, inlayer, gdamp, gsoft, inc00, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, w, cu, cv, cw, qc) &
    !$omp reduction(+:eps, uv, uu, vv, ww, vw, uw, mfx, mfz, q_in, q_out, e_in, e_out, ed, pd)
    do ix = nx0, nxN
      do iz = -nz, nz
        do iy = ny0, nyN
          cu = V(iy, iz, ix, 1); cv = V(iy, iz, ix, 2); cw = V(iy, iz, ix, 3)
          if (ix == 0 .and. iz == 0) then
            mfx = mfx + dreal(cu)*dyl(iy)
            mfz = mfz + dreal(cw)*dyl(iy)
            if (gdamp(iz, ix) > 0.0d0) then      ! damped mean profiles (damp_mean)
              qc = dyl(iy)*dreal(cu*conjg(cu) + cw*conjg(cw))
              ed = ed + qc; pd = pd + 2.0d0*gsoft*gdamp(iz, ix)*qc
            end if
            ! second moments of the profiles about their box means: the raw sums here,
            ! the squared box means come off after the reduction (v00 = 0 exactly)
            if (inc00 == 1) then
              uu = uu + dyl(iy)*dreal(cu*conjg(cu))
              ww = ww + dyl(iy)*dreal(cw*conjg(cw))
              uw = uw + dyl(iy)*dreal(cu*conjg(cw))
            end if
            cycle
          end if
          w = 2.0d0*dyl(iy)
          if (ix == 0) w = dyl(iy)
          if (gdamp(iz, ix) > 0.0d0) then
            qc = w*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
            ed = ed + qc; pd = pd + 2.0d0*gsoft*gdamp(iz, ix)*qc
          end if
          uu = uu + w*dreal(cu*conjg(cu))
          vv = vv + w*dreal(cv*conjg(cv))
          ww = ww + w*dreal(cw*conjg(cw))
          uv = uv + w*dreal(cu*conjg(cv))
          vw = vw + w*dreal(cw*conjg(cv))
          uw = uw + w*dreal(cu*conjg(cw))
          eps = eps + w*k2(iz, ix)*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
          if (inlayer(iy) == 1) then
            q_in = q_in + w*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
            e_in = e_in + w*k2(iz, ix)*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
          else
            q_out = q_out + w*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
            e_out = e_out + w*k2(iz, ix)*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
          end if
        end do
      end do
    end do
    ! y derivatives of the three components, one at a time, into scratch
    do c = 1, 3
      call line_solve(KIND_DY, 0.0d0, V(:, :, :, c), rhs(:, :, :, 1))
      grad = 0; g_in = 0; g_out = 0
      !$omp target teams distribute parallel do collapse(3) default(none) &
      !$omp shared(rhs, dyl, inlayer, inc00, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, w, dq) reduction(+:grad, g_in, g_out)
      do ix = nx0, nxN
        do iz = -nz, nz
          do iy = ny0, nyN
            ! the derivative of a constant is zero, so the (0,0) column's slope is that of its fluctuation
            if (ix == 0 .and. iz == 0 .and. inc00 == 0) cycle
            w = 2.0d0*dyl(iy)
            if (ix == 0) w = dyl(iy)
            dq = rhs(iy, iz, ix, 1)
            grad = grad + w*dreal(dq*conjg(dq))
            if (inlayer(iy) == 1) then
              g_in = g_in + w*dreal(dq*conjg(dq))
            else
              g_out = g_out + w*dreal(dq*conjg(dq))
            end if
          end do
        end do
      end do
      eps = eps + grad; e_in = e_in + g_in; e_out = e_out + g_out
    end do
    ! column energies (the 2-D spectrum), accumulated on the host
    !$omp target teams distribute parallel do collapse(2) default(none) &
    !$omp shared(V, dyl, exz, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, w, cu, cv, cw, qc)
    do ix = nx0, nxN
      do iz = -nz, nz
        qc = 0.0d0
        do iy = ny0, nyN
          cu = V(iy, iz, ix, 1); cv = V(iy, iz, ix, 2); cw = V(iy, iz, ix, 3)
          w = 2.0d0*dyl(iy)
          if (ix == 0) w = dyl(iy)
          qc = qc + w*dreal(cu*conjg(cu) + cv*conjg(cv) + cw*conjg(cw))
        end do
        exz(iz, ix) = qc
      end do
    end do
    !$omp target update from(exz)
    exz_acc = exz_acc + exz; nacc = nacc + 1
    ! over the ranks, then back into the named quantities
    sums = [eps, uv, uu, vv, ww, vw, uw, mfx, mfz, q_in, q_out, e_in, e_out, ed, pd, dble(ncol_damped)]
    call MPI_Allreduce(MPI_IN_PLACE, sums, 16, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    eps = sums(1); uv = sums(2); uu = sums(3); vv = sums(4); ww = sums(5); vw = sums(6); uw = sums(7)
    mfx = sums(8); mfz = sums(9); q_in = sums(10); q_out = sums(11); e_in = sums(12); e_out = sums(13)
    ed = sums(14); pd = sums(15)
    if (inc00 == 1) then
      ! squared box means off the (0,0) column's raw sums (sum dyl = ly up to round-off)
      ly_w = sum(dyl)
      uu = uu - mfx*mfx/ly_w
      ww = ww - mfz*mfz/ly_w
      uw = uw - mfx*mfz/ly_w
      mean_acc = mean_acc + (mfx*mfx + mfz*mfz)/ly_w
    end if
    q2 = uu + vv + ww
    if (damping_active()) call family_stats(inc00, mfx, mfz, fam, sh15)
    if (has_terminal) then
      ! Convert the box-height integrals to y-averaged (box-mean) quantities
      ! with physically-meaningful units (mean per unit volume), so nothing
      ! downstream needs to remember the ly, 1/2 or nu factors:
      !   energy = q2/(2 ly)          = <u_i u_i>/2            (TKE k)
      !   diss   = ni*eps/ly          = nu <grad u : grad u>   (dissipation)
      !   uv, vw = uv/ly, vw/ly       = <u v>, <v w>           (Reynolds shear)
      !   uu..uw = uu/ly .. uw/ly     = <u u>, ...             (normal stresses)
      ! with the kinematic viscosity nu = 1/re = ni.
      inv_ly = 1.0d0/ly
      write (*, '(F12.5,2X,ES11.4,2X,F8.4,4(2X,ES13.6))') time, deltat, cfl*deltat, &
        0.5d0*q2*inv_ly, ni*eps*inv_ly, uv*inv_ly, vw*inv_ly
      write (unit_rt, '(13(ES23.15,1X))') time, mfx, mfz, S, s2_of(time), S*time, gamma_y_of(time), deltat, cfl*deltat, &
        0.5d0*q2*inv_ly, ni*eps*inv_ly, uv*inv_ly, vw*inv_ly
      write (unit_var, '(5(ES23.15,1X))') time, uu*inv_ly, vv*inv_ly, ww*inv_ly, uw*inv_ly
      flush (unit_rt); flush (unit_var)
      if (stokes_active()) then
        ! region averages inside and outside |y - ly/2| < 8 delta (io.cpl):
        ! energy_out/in = <u_i u_i>/2 and diss_out/in = nu <grad u : grad u>
        l_in = sum(dyl, mask=(inlayer == 1)); l_out = sum(dyl, mask=(inlayer == 0))
        write (unit_sl, '(5(ES23.15,1X))') time, 0.5d0*q_out/l_out, 0.5d0*q_in/l_in, ni*e_out/l_out, ni*e_in/l_in
        flush (unit_sl)
      end if
      if (damping_active()) then
        write (unit_dp, '(2(ES23.15,1X),I8,2(1X,ES23.15))') time, damping_theta(time)*180.0d0/3.141592653589793d0, &
          nint(sums(16)), 0.5d0*ed, 0.5d0*pd
        flush (unit_dp)
        ! families: the shear frame of the mask, shares of the fluctuation, stresses per unit volume
        th = damping_theta(time); ct = cos(th); st = sin(th)
        write (unit_df, '(2(ES23.15,1X))', advance='no') time, th*180.0d0/3.141592653589793d0
        write (unit_df, '(ES23.15,1X)', advance='no') sh15/max(q2, tiny(1.0d0))
        do f = 1, NFAM
          qf = fam(1, f) + fam(2, f) + fam(3, f)
          write (unit_df, '(5(ES23.15,1X))', advance='no') qf/max(q2, tiny(1.0d0)), &
            (ct*ct*fam(1, f) + st*st*fam(3, f) + 2.0d0*ct*st*fam(6, f))*inv_ly, fam(2, f)*inv_ly, &
            (st*st*fam(1, f) + ct*ct*fam(3, f) - 2.0d0*ct*st*fam(6, f))*inv_ly, -(ct*fam(4, f) + st*fam(5, f))*inv_ly
        end do
        write (unit_df, '(A)') ''
        flush (unit_df)
      end if
    end if
  end subroutine outstats

end module hst_stats
