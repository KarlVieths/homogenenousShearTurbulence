! statistics/split_runtime.dat (hst_stats, selective_lu_orr.md section 4) against a reference summed
! the other way round.
!
!   mpirun -np N build-cpu/test_split_stats [hst.in]
!
! A random field with profiles in the (0,0) column (a large box mean), the weights prod_lu = 0.5 and
! prod_orr = 0.3 times a smooth production filter, outstats once with S2 = 0 and once with a constant
! S2 /= 0.  The code under test sums on the device from the field as it is, the (0,0) column's energy
! as raw sum minus the squared box means.  The reference (here, on the host, all the field on one
! rank) subtracts the box mean from the field first and then sums every column with plain loops:
!   a = S alfa + S2 beta,  b = S beta - S2 alfa,  kh2 = alfa^2 + beta^2
!   P_Orr = -(a/kh2) Re <(alfa u + beta w) v*>,  P_LU = -(b/kh2) Re <(beta u - alfa w) v*>
!   E_tor = <|beta u - alfa w|^2>/(2 kh2),  E_pol = <|u|^2 + |v|^2 + |w|^2>/2 - E_tor
! (the (0,0) column's horizontal velocity, with its box mean off, is E_tor).  Compared: the ten columns
! of the file, including the sums over the columns with k_par = 0 (ix = 0 for S2 = 0) and k_perp = 0
! (iz = 0), where only one mechanism acts (and which must not be zero for S2 = 0, or the test would not
! see them).  It also checks the identities E_tor + E_pol = energy and P_LU + P_Orr = -S uv - S2 vw of
! Runtimedata, diss against Runtimedata, and that the weights matter (the weighted and the unweighted
! sums differ).
program test_split_stats

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
  use test_common
  use hst_mpi
  use hst_derivatives
  use hst_initial
  use hst_io, only: make_output_dirs
  use hst_stats

  implicit none

  real(C_DOUBLE), parameter :: PI = 3.141592653589793d0
  integer :: ierr, m, iy, icase
  real(C_DOUBLE) :: lyw, errmax, errcase, s2
  real(C_DOUBLE) :: ref(11), got(11), rt(13), scale
  logical :: passed, case_ok

  call test_start()
  prod_kc = 6.0d0; prod_edge = 0.3d0; prod_lu = 0.5d0; prod_orr = 0.3d0     ! G from 0 to 1 over k_h = 4.2 .. 7.8
  S = 1.0d0
  call test_setup()
  time = 0.0d0
  call generate_initial_field()
  if (has_average) then
    do iy = ny0 - 2, nyN + 2
      V(iy, 0, 0, 1) = dcmplx(0.30d0 + 0.20d0*sin(2*PI*y(iy)/ly) + 0.10d0*cos(4*PI*y(iy)/ly), 0.0d0)
      V(iy, 0, 0, 3) = dcmplx(-0.15d0 + 0.25d0*cos(2*PI*y(iy)/ly) + 0.05d0*sin(6*PI*y(iy)/ly), 0.0d0)
      V(iy, 0, 0, 2) = 0.0d0
    end do
  end if
  !$omp target update to(V)
  do m = 1, 3
    call fill_ghosts(m)
  end do
  call make_output_dirs()

  passed = .true.
  errmax = 0.0d0
  do icase = 1, 2
    ! S2 = 0 (k_par = ix = 0, k_perp = iz = 0 are columns), then a constant S2
    s2_amplitude = 0.0d0; s2_period = 0.0d0; s2_start = 0.0d0
    if (icase == 1) s2_mean = 0.0d0
    if (icase == 2) s2_mean = 0.3d0
    s2 = s2_of(time)
    call open_runtimedata()
    call outstats()
    call close_runtimedata()
    case_ok = .true.
    if (has_terminal) call read_last_rows(got, rt)
    call reference(ref, s2)
    if (has_terminal) then
      scale = maxval(abs(ref(3:6)))             ! the production sums set the scale of the others
      errcase = 0.0d0
      call cmp('E_tor        ', 1, ref(1), got(1), maxval(abs(ref(1:2))))
      call cmp('E_pol        ', 2, ref(2), got(2), maxval(abs(ref(1:2))))
      call cmp('P_LU         ', 3, ref(3), got(3), scale)
      call cmp('P_Orr        ', 4, ref(4), got(4), scale)
      call cmp('g_LU P_LU    ', 5, ref(5), got(5), scale)
      call cmp('g_Orr P_Orr  ', 6, ref(6), got(6), scale)
      call cmp('P_LU(k_par=0)', 7, ref(7), got(7), scale)
      call cmp('P_Orr(k_perp=0)', 8, ref(8), got(8), scale)
      call cmp('E_00         ', 9, ref(9), got(9), maxval(abs(ref(1:2))))
      ! Runtimedata: energy (10), diss (11), uv (12), vw (13) -> the identities
      write (*, '(A,ES10.3,A,ES10.3)') '    E_tor + E_pol - energy ', abs(got(1) + got(2) - rt(10)), &
        '   P_LU + P_Orr + S uv + S2 vw ', abs(got(3) + got(4) + S*rt(12) + s2*rt(13))
      errcase = max(errcase, abs(got(1) + got(2) - rt(10))/rt(10), abs(got(3) + got(4) + S*rt(12) + s2*rt(13))/scale)
      if (abs(got(10) - rt(11)) > 1.0d-14*abs(rt(11))) errcase = max(errcase, 1.0d0)
      ! the mechanism that cannot act in the columns of the other is zero there
      if (icase == 1) then
        if (abs(ref(7)) < 1.0d-3*scale .or. abs(ref(8)) < 1.0d-3*scale) then
          write (*, '(A)') '    the k_par = 0 / k_perp = 0 sums are zero: the test does not see them'
          errcase = max(errcase, 1.0d0)
        end if
      end if
      ! the weights act: the weighted sums are not the unweighted ones
      if (abs(got(5) - got(3)) < 1.0d-2*scale .or. abs(got(6) - got(4)) < 1.0d-3*scale) then
        write (*, '(A)') '    the weighted and the unweighted sums agree: the test does not see the weights'
        errcase = max(errcase, 1.0d0)
      end if
      write (*, '(A,I0,A,F5.2,A,ES10.3)') '  case ', icase, ' (S2 = ', s2, '): max relative error ', errcase
      errmax = max(errmax, errcase)
      case_ok = errcase < 1.0d-10
    end if
    passed = passed .and. case_ok
    ! the rows of the first case stay in the files (appended); the last line is the one read
  end do
  call MPI_Bcast(passed, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
  call test_finish(passed)

contains

  ! the last lines of split_runtime.dat (columns 2..10 into g(1:10), 8 = diss) and of Runtimedata
  subroutine read_last_rows(g, r)
    real(C_DOUBLE), intent(out) :: g(11), r(13)
    integer :: unit, ios
    real(C_DOUBLE) :: row(11), rrow(13), tt
    character(len=512) :: line
    g = 0.0d0; r = 0.0d0
    open (newunit=unit, file='statistics/split_runtime.dat', action='read')
    do
      read (unit, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (line(1:1) == '#') cycle
      read (line, *, iostat=ios) tt, row(1:10)
      if (ios == 0) then
        ! columns: E_tor E_pol P_LU P_Orr gP_LU gP_Orr diss P_LU(kpar=0) P_Orr(kperp=0) E_00
        g(1:6) = row(1:6); g(10) = row(7); g(7) = row(8); g(8) = row(9); g(9) = row(10)
      end if
    end do
    close (unit)
    open (newunit=unit, file='statistics/Runtimedata', action='read')
    do
      read (unit, *, iostat=ios) rrow
      if (ios /= 0) exit
      r = rrow
    end do
    close (unit)
  end subroutine read_last_rows

  subroutine cmp(name, idx, expect, gotv, sc)
    character(len=*), intent(in) :: name
    integer, intent(in) :: idx
    real(C_DOUBLE), intent(in) :: expect, gotv, sc
    real(C_DOUBLE) :: e
    e = abs(gotv - expect)/max(sc, 1.0d-300)
    errcase = max(errcase, e)
    write (*, '(2A,2(A,ES15.7),A,ES9.2)') '  ', name, '  live ', gotv, '   reference ', expect, '   error/scale ', e
  end subroutine cmp

  ! The reference, from the field with its box mean subtracted, all modes with plain loops (ref(1..9) as
  ! got, per unit volume).  The field is on the host (never changed on the device here).
  subroutine reference(r, s2v)
    real(C_DOUBLE), intent(out) :: r(11)
    real(C_DOUBLE), intent(in) :: s2v
    integer :: ix, iy, iz
    real(C_DOUBLE) :: al, be, kh2, aa, bb, w, plu, porr, ubar, wbar, acc(9), loc2(2), pt
    complex(C_DOUBLE_COMPLEX) :: cu, cv, cw

    ! box means of the (0,0) column
    loc2 = 0.0d0
    if (has_average) then
      do iy = ny0, nyN
        loc2(1) = loc2(1) + dreal(V(iy, 0, 0, 1))*dyl(iy)
        loc2(2) = loc2(2) + dreal(V(iy, 0, 0, 3))*dyl(iy)
      end do
    end if
    call MPI_Allreduce(MPI_IN_PLACE, loc2, 2, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    lyw = sum(dyl)
    ubar = loc2(1)/lyw; wbar = loc2(2)/lyw
    acc = 0.0d0
    do ix = nx0, nxN
      do iz = -nz, nz
        do iy = ny0, nyN
          cu = V(iy, iz, ix, 1); cv = V(iy, iz, ix, 2); cw = V(iy, iz, ix, 3)
          w = 2.0d0*dyl(iy); if (ix == 0) w = dyl(iy)
          if (ix == 0 .and. iz == 0) then
            ! the profile about its box mean: toroidal (horizontal velocity, no v)
            acc(1) = acc(1) + 0.5d0*w*(abs(cu - ubar)**2 + abs(cw - wbar)**2)
            acc(9) = acc(9) + 0.5d0*w*(abs(cu - ubar)**2 + abs(cw - wbar)**2)
            cycle
          end if
          al = alfa0*ix; be = beta0*iz
          kh2 = al*al + be*be
          aa = S*al + s2v*be; bb = S*be - s2v*al
          acc(1) = acc(1) + 0.5d0*w*abs(be*cu - al*cw)**2/kh2
          acc(2) = acc(2) + 0.5d0*w*(abs(cu)**2 + abs(cv)**2 + abs(cw)**2) - 0.5d0*w*abs(be*cu - al*cw)**2/kh2
          porr = -(aa/kh2)*w*dreal((al*cu + be*cw)*conjg(cv))
          plu = -(bb/kh2)*w*dreal((be*cu - al*cw)*conjg(cv))
          acc(3) = acc(3) + plu; acc(4) = acc(4) + porr
          acc(5) = acc(5) + glu(iz, ix)*plu; acc(6) = acc(6) + gorr(iz, ix)*porr
          ! the columns where one mechanism acts alone: S2 = 0, k_par = alfa, k_perp = beta; otherwise
          ! the columns that are exactly aligned (a = 0 or b = 0 within the solver's tolerance)
          if (s2v == 0.0d0) then
            if (ix == 0) acc(7) = acc(7) + plu
            if (iz == 0) acc(8) = acc(8) + porr
          else
            pt = 1.0d-9*sqrt(S*S + s2v*s2v)*sqrt(kh2)
            if (abs(aa) <= pt) acc(7) = acc(7) + plu
            if (abs(bb) <= pt) acc(8) = acc(8) + porr
          end if
        end do
      end do
    end do
    call MPI_Allreduce(MPI_IN_PLACE, acc, 9, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    r(1:9) = acc/ly
    r(10:11) = 0.0d0
  end subroutine reference

end program test_split_stats
