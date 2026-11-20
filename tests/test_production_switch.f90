! The switch times of the weights of the production (hst_production, selective_lu_orr.md section 9):
! update_production_weights and production_clip_dt on their own, with no time stepping.
!
!   mpirun -np 1 build-cpu/test_production_switch [hst.in]      (one rank: the check that the filter is partial is global)
!
! prod_lu = 0.25 from t = 1, prod_orr = 0.4 from t = 2, a smooth filter G (prod_kc = 6).  Checked:
!   - the weights in effect: G alone before the switch, prod_lu G (prod_orr G) from it, for every
!     column, on the host and on the device (the arrays are copied back before they are compared);
!     idempotent, reversible (time set back, as test_kelvin_s2 does between its levels), unchanged by
!     a time just short of the switch, in effect at the round-off below it (1e-9, the tolerance for a
!     step that was cut to end on the switch time);
!   - production_clip_dt: the step that would cross a switch time ends on it, one that does not is
!     left alone, the earlier of two switch times wins, a switch time already reached cuts nothing;
!   - the loop of hst (clip, update at the start of the step, advance): the steps land on both switch
!     times within round-off, none straddles one, and the weights of every step are those of its start;
!   - the default start time (-huge): the weights are prod_lu G, prod_orr G from the start and no step
!     is cut.
program test_production_switch

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
  use test_common
  use hst_mpi
  use hst_production

  implicit none

  logical :: passed

  call test_start()
  if (nproc /= 1) error stop 'test_production_switch runs on one rank'
  prod_kc = 6.0d0; prod_edge = 0.3d0; prod_lu = 0.25d0; prod_orr = 0.4d0
  prod_lu_start = 1.0d0; prod_orr_start = 2.0d0
  time = 0.0d0
  call test_setup()
  passed = .true.

  ! --- the weights in effect
  call weights('at the set-up (t = 0)', 1.0d0, 1.0d0)
  call update_production_weights(0.5d0);                 call weights('t = 0.5', 1.0d0, 1.0d0)
  call update_production_weights(1.0d0 - 1.0d-6);        call weights('t = 1 - 1e-6 (not reached)', 1.0d0, 1.0d0)
  call update_production_weights(1.0d0 - 1.0d-12);       call weights('t = 1 - 1e-12 (reached to round-off)', prod_lu, 1.0d0)
  call update_production_weights(1.5d0);                 call weights('t = 1.5', prod_lu, 1.0d0)
  call update_production_weights(1.5d0);                 call weights('t = 1.5 again (idempotent)', prod_lu, 1.0d0)
  call update_production_weights(2.0d0);                 call weights('t = 2', prod_lu, prod_orr)
  call update_production_weights(0.0d0);                 call weights('t set back to 0', 1.0d0, 1.0d0)
  call update_production_weights(1.5d0);                 call weights('t = 1.5 (the second pass)', prod_lu, 1.0d0)
  call update_production_weights(7.0d0);                 call weights('t = 7', prod_lu, prod_orr)
  call update_production_weights(0.0d0)
  if (.not. any(gprod < 1.0d0 .and. gprod > 0.0d0) .or. all(gprod < 1.0d0)) call fail('the filter is not partial: the test cannot tell G from 1')

  ! --- the step cut
  call clip('t = 0, dt = 0.5: no switch in the step', 0.0d0, 0.5d0, 0.5d0)
  call clip('t = 0.7, dt = 0.5: ends on t = 1', 0.7d0, 0.5d0, 0.3d0)
  call clip('t = 0.95, dt = 1.5: the earlier of the two', 0.95d0, 1.5d0, 0.05d0)
  call clip('t = 0.9, dt = 1.5: both inside, the first', 0.9d0, 1.5d0, 0.1d0)
  call clip('t = 1, dt = 0.5: lift-up reached, Orr at 2 beyond', 1.0d0, 0.5d0, 0.5d0)
  call clip('t = 1 - 1e-12, dt = 0.1: reached to round-off', 1.0d0 - 1.0d-12, 0.1d0, 0.1d0)
  call clip('t = 1.9, dt = 0.2: ends on t = 2', 1.9d0, 0.2d0, 0.1d0)
  call clip('t = 1.95, dt = 1: ends on t = 2', 1.95d0, 1.0d0, 0.05d0)
  call clip('t = 2, dt = 0.3: both reached', 2.0d0, 0.3d0, 0.3d0)
  call clip('t = 3, dt = 1', 3.0d0, 1.0d0, 1.0d0)

  ! --- the loop of hst
  call loop(0.37d0, 2.6d0)
  call loop(0.05d0, 2.6d0)
  call loop(1.0d0, 2.6d0)

  ! --- the default start time: from the start, nothing cut
  prod_lu_start = -huge(1.0d0); prod_orr_start = -huge(1.0d0)
  call update_production_weights(0.0d0);                 call weights('default start times, t = 0', prod_lu, prod_orr)
  call clip('default start times: no cut', 0.0d0, 1.0d10, 1.0d10)
  call clip('default start times: no cut (t < 0)', -5.0d0, 0.3d0, 0.3d0)

  call test_finish(passed)

contains

  subroutine fail(msg)
    character(len=*), intent(in) :: msg
    passed = .false.
    if (has_terminal) write (*, '(A,A)') '   FAIL: ', msg
  end subroutine fail

  ! glu = fl G and gorr = fo G in every column, after the device copy has been brought back
  subroutine weights(label, fl, fo)
    character(len=*), intent(in) :: label
    real(C_DOUBLE), intent(in) :: fl, fo
    real(C_DOUBLE) :: el, eo
    !$omp target update from(glu, gorr)
    el = maxval(abs(glu - fl*gprod)); eo = maxval(abs(gorr - fo*gprod))
    if (has_terminal) write (*, '(A,A,A,2ES10.2)') '   weights ', label, ': error', el, eo
    if (el > 1.0d-15 .or. eo > 1.0d-15) call fail('weights ('//label//')')
  end subroutine weights

  subroutine clip(label, t, dt, expect)
    character(len=*), intent(in) :: label
    real(C_DOUBLE), intent(in) :: t, dt, expect
    real(C_DOUBLE) :: d
    d = production_clip_dt(t, dt)
    if (has_terminal) write (*, '(A,A,A,ES22.14,A,ES22.14)') '   clip ', label, ': ', d, '  expected', expect
    if (abs(d - expect) > 1.0d-14*max(1.0d0, abs(expect))) call fail('clip ('//label//')')
  end subroutine clip

  ! hst: the step after t is cut, the weights are updated at its start, then the time advances by it
  subroutine loop(dt, tend)
    real(C_DOUBLE), intent(in) :: dt, tend
    real(C_DOUBLE) :: t, d, hit1, hit2, fl, fo
    integer :: n
    prod_lu_start = 1.0d0; prod_orr_start = 2.0d0
    call update_production_weights(0.0d0)
    t = 0.0d0; hit1 = 1.0d9; hit2 = 1.0d9; n = 0
    do while (t < tend)
      d = production_clip_dt(t, dt)
      if (d <= 0.0d0 .or. n > 10000) then       ! a cut to a step of length 0 (or less) would never advance
        call fail('loop: a step of length <= 0 (or no end)')
        exit
      end if
      call update_production_weights(t)
      fl = 1.0d0; fo = 1.0d0
      if (t >= 1.0d0 - 1.0d-9) fl = prod_lu
      if (t >= 2.0d0 - 1.0d-9) fo = prod_orr
      call weights_quiet(fl, fo)
      if (t < 1.0d0 - 1.0d-9 .and. t + d > 1.0d0 + 1.0d-9) call fail('a step straddles t = 1')
      if (t < 2.0d0 - 1.0d-9 .and. t + d > 2.0d0 + 1.0d-9) call fail('a step straddles t = 2')
      t = t + d
      hit1 = min(hit1, abs(t - 1.0d0)); hit2 = min(hit2, abs(t - 2.0d0))
      n = n + 1
    end do
    if (has_terminal) write (*, '(A,F6.3,A,I0,A,ES9.2,A,ES9.2)') '   loop, dt = ', dt, ': ', n, ' steps, landing on t = 1 within', hit1, &
      ', on t = 2 within', hit2
    if (hit1 > 1.0d-12 .or. hit2 > 1.0d-12) call fail('the steps do not land on the switch times')
  end subroutine loop

  subroutine weights_quiet(fl, fo)
    real(C_DOUBLE), intent(in) :: fl, fo
    !$omp target update from(glu, gorr)
    if (maxval(abs(glu - fl*gprod)) > 1.0d-15 .or. maxval(abs(gorr - fo*gprod)) > 1.0d-15) &
      call fail('loop: the weights of a step are not those of its start')
  end subroutine weights_quiet

end program test_production_switch
