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
! Runtimedata columns:
!   time  meanflowx  meanflowy  S  S2  gamma_x  gamma_y  deltat  cfl*deltat
!   energy  diss  uv  vw
! variances_runtime.dat columns:
!   time  uu  vv  ww  uw
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
! The (0,0) mode is excluded from the fluctuation statistics; modes with
! ix > 0 count twice.
module hst_stats

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
  use hst_linsolve, only: line_solve, KIND_DY
  use hst_derivatives, only: s2_of, gamma_y_of
  use hst_stokes, only: stokes_active

  implicit none
  private
  public :: outstats, open_runtimedata, close_runtimedata

  integer, parameter :: unit_rt = 121, unit_var = 122, unit_sl = 123

contains

  subroutine open_runtimedata()
    if (has_terminal) open (unit=unit_rt, file='Runtimedata', action='write', position='append')
    if (has_terminal) open (unit=unit_var, file='variances_runtime.dat', action='write', position='append')
    if (has_terminal .and. stokes_active()) open (unit=unit_sl, file='stokes_runtime.dat', action='write', position='append')
  end subroutine open_runtimedata

  subroutine close_runtimedata()
    if (has_terminal) close (unit_rt)
    if (has_terminal) close (unit_var)
    if (has_terminal .and. stokes_active()) close (unit_sl)
  end subroutine close_runtimedata

  subroutine outstats()
    integer(C_INT) :: ix, iy, iz, c
    real(C_DOUBLE) :: w, eps, uv, uu, vv, ww, vw, uw, grad, mfx, mfz
    real(C_DOUBLE) :: q_in, q_out, e_in, e_out, g_in, g_out, l_in, l_out
    real(C_DOUBLE) :: sums(13), q2, inv_ly
    complex(C_DOUBLE_COMPLEX) :: cu, cv, cw, dq
    integer :: ierr

    ! sums over modes and rows, each row weighted by its spacing dyl: the
    ! spectral (Parseval) sums over x and z give the horizontal mean via
    ! the mode weight w, and integrating in y gives a box-height integral.
    eps = 0; uv = 0; uu = 0; vv = 0; ww = 0; vw = 0; uw = 0; mfx = 0; mfz = 0
    q_in = 0; q_out = 0; e_in = 0; e_out = 0
    !$omp target teams distribute parallel do collapse(3) default(none) &
    !$omp shared(V, k2, dyl, inlayer, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, w, cu, cv, cw) &
    !$omp reduction(+:eps, uv, uu, vv, ww, vw, uw, mfx, mfz, q_in, q_out, e_in, e_out)
    do ix = nx0, nxN
      do iz = -nz, nz
        do iy = ny0, nyN
          cu = V(iy, iz, ix, 1); cv = V(iy, iz, ix, 2); cw = V(iy, iz, ix, 3)
          if (ix == 0 .and. iz == 0) then
            mfx = mfx + dreal(cu)*dyl(iy)
            mfz = mfz + dreal(cw)*dyl(iy)
            cycle
          end if
          w = 2.0d0*dyl(iy)
          if (ix == 0) w = dyl(iy)
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
      !$omp shared(rhs, dyl, inlayer, nx0, nxN, nz, ny0, nyN) private(ix, iy, iz, w, dq) reduction(+:grad, g_in, g_out)
      do ix = nx0, nxN
        do iz = -nz, nz
          do iy = ny0, nyN
            if (ix == 0 .and. iz == 0) cycle
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
    ! over the ranks, then back into the named quantities
    sums = [eps, uv, uu, vv, ww, vw, uw, mfx, mfz, q_in, q_out, e_in, e_out]
    call MPI_Allreduce(MPI_IN_PLACE, sums, 13, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    eps = sums(1); uv = sums(2); uu = sums(3); vv = sums(4); ww = sums(5); vw = sums(6); uw = sums(7)
    mfx = sums(8); mfz = sums(9); q_in = sums(10); q_out = sums(11); e_in = sums(12); e_out = sums(13)
    q2 = uu + vv + ww
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
    end if
  end subroutine outstats

end module hst_stats
