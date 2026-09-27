! The wall-normal (y) linear systems: one cyclic pentadiagonal system per
! Fourier mode, all modes solved at once, one GPU thread per mode, each
! line split over the npy y slabs.
!
! The five-point compact stencil on a periodic y gives a pentadiagonal
! matrix whose stencil wraps around the box; the wrapped entries carry the
! shear-periodic phase.  Written with band index j = -2..2 (column = row + j
! modulo ny), row iy of a line is
!   a(j) = coef(sys, iy, j) * wrap,   wrap = ph if iy + j >= ny, conjg(ph) if iy + j < 0, else 1
! with the real stencil combination coef of the system sys (line_solve).
!
! The solve uses bordering, slab by slab: the last two unknowns of slab s,
! its border b_s = (x(nyN-1), x(nyN)), are the interface unknowns.  Rows
! and columns ny0..nyN-2 of the slab form a *plain* pentadiagonal matrix
! P_s (its rows 0 and 1 also point at the border b_{s-1} of the slab
! below, its rows m-2 and m-1 at b_s; with one slab, b_{s-1} is b_s across
! the wrap), so
!   1. P_s is factored, and three right-hand sides are forward-substituted
!      in the same sweep: b, and the two columns that couple to b_{s-1};
!   2. the four rows of the slab that its border rows and the border rows
!      of the slab below reach (0, 1, m-2, m-1) are then known as affine
!      functions of b_{s-1} and b_s.  Substituted into the two border rows
!      of every slab they give a block-cyclic system for the 2 npy border
!      unknowns of the line (the reduced interface system): the record of
!      a slab is these four rows (value and four coefficients each) and
!      the right-hand side of its two border rows, 28 reals per line;
!      every rank of the y column assembles the whole system from an
!      allgather of the records and solves it (Gaussian elimination, the
!      system is diagonally dominant);
!   3. the back substitution of b minus the border columns times b_{s-1}
!      and the seed b_s is the solution.
! With npy = 1 the reduced system is the 2x2 Schur complement of the
! single slab.  (channel/src/linsolve does the same with a multi-level
! tree of reductions; for npy <= 8 one level is enough.)
!
! The rows are generated on the fly inside the sweep, so the matrix is
! never stored.  P_s is *real*: the stencil coefficients are real and the
! wrap phase only enters the border rows and, through the wrapped entries
! of rows 0 and 1 of the first slab, the border columns, where it is one
! common factor conjg(ph) that is applied when the record is used.  So
! the forward sweep works in real arithmetic on the matrix and the two
! border columns (one real division per row) and in complex arithmetic
! only on the right-hand side; it keeps the two previous rows in
! registers, divides each row by its pivot and writes the two upper
! entries U1, U2, the border columns B1, B2 (real) and the substituted
! right-hand side XR (complex).  The entries of the last two rows of P_s
! that point at b_s stay in U1, U2 and act through the seed of the back
! substitution.  The rows 0 and 1 of the slab after the back substitution
! are inner products of the substituted right-hand sides with the first
! two rows of the inverse of the unit upper factor, which is a *forward*
! recurrence (w(i) = -U1(i-1) w(i-1) - U2(i-2) w(i-2)), so they are
! accumulated in the same sweep.  The solve is then two passes over the
! line around one allgather: the forward sweep reads the right-hand side
! from the field and writes the five columns and the record, the backward
! sweep reads them and writes the solution into the field (128 bytes per
! row).  Storage is interleaved, line index first (U1(iline, iy)), so that
! the threads of a kernel read consecutive addresses.  Lines are processed
! in batches of line_chunk x columns (deck parameter; 0 = all columns at
! once on the GPU, 16 on the CPU) to bound that workspace.
!
! (The system kind is called `sys` and the substituted right-hand side
!  `XR` here, where main has `kind` and `X`: with hst_mpi, a CUDA Fortran
!  module, visible in this file, nvfortran 25.9 no longer resolves those
!  two names in the OpenMP data-sharing clauses.)
module hst_linsolve

  use, intrinsic :: iso_c_binding
  use hst_params
  use hst_derivatives, only: shear_shifts

  implicit none
  private
  public :: init_linsolve, free_linsolve, line_solve
  public :: KIND_D2V, KIND_ETA, KIND_POISSON, KIND_D0, KIND_DY

  ! the systems line_solve assembles (see there)
  integer(C_INT), parameter :: KIND_D2V = 1, KIND_ETA = 2, KIND_POISSON = 3, KIND_D0 = 4, KIND_DY = 5
  integer(C_INT), parameter :: NPY_MAX = 8          ! slabs per line: the reduced system is at most 16 x 16
  ! The record of a slab for the reduced system, per line: for each of the
  ! rows 0, 1, m-2, m-1 (in this order, 6 reals each) the value x (2), then
  ! the coefficients p, q of b_{s-1} and t, u of b_s in
  !   x(row) = x - cph (p b_{s-1,1} + q b_{s-1,2}) - (t b_{s,1} + u b_{s,2}),
  ! cph = conjg(ph) for the first slab and 1 otherwise; then the right-hand
  ! sides of the two border rows (4).
  integer(C_INT), parameter :: NREC = 28
  real(C_DOUBLE), allocatable, save :: U1(:, :), U2(:, :), B1(:, :), B2(:, :)
  complex(C_DOUBLE_COMPLEX), allocatable, save :: XR(:, :)
  real(C_DOUBLE), allocatable, save :: record(:, :, :)        ! record(NREC, line, slab)
  integer(C_INT), save :: nlines_max, chunk

contains

  subroutine init_linsolve()
    if (npy > NPY_MAX) then
      if (has_terminal) print *, 'ERROR: npy is at most', NPY_MAX, '(hst_linsolve)'
      error stop 1
    end if
    ! default: every column at once on the GPU (fewest launches); 16 on the
    ! CPU, where the workspace of all columns falls out of the cache
    chunk = 16
#ifdef HAVE_CUDA
    chunk = nxB
#endif
    if (line_chunk > 0) chunk = line_chunk
    chunk = min(chunk, nxB)
    nlines_max = (2*nz + 1)*chunk
    allocate (U1(nlines_max, 0:nyB - 1), U2(nlines_max, 0:nyB - 1), B1(nlines_max, 0:nyB - 1), B2(nlines_max, 0:nyB - 1))
    allocate (XR(nlines_max, 0:nyB - 1), record(NREC, nlines_max, 0:npy - 1))
    U1 = 0; U2 = 0; B1 = 0; B2 = 0; XR = 0; record = 0
    !$omp target enter data map(to: U1, U2, B1, B2, XR, record)
    if (has_terminal) write (*, '(A,I0,A,I0,A,F8.1,A)') '   line solver: batches of ', chunk, ' x columns, ', &
      nlines_max, ' lines, workspace ', (48.0d0*nyB + 8.0d0*NREC*npy)*nlines_max/1024.0d0**2, ' MB per rank'
  end subroutine init_linsolve

  subroutine free_linsolve()
    !$omp target exit data map(delete: U1, U2, B1, B2, XR, record)
    deallocate (U1, U2, B1, B2, XR, record)
  end subroutine free_linsolve

  ! Assemble and solve one system per mode for fields with the layout of a
  ! component of V (pass e.g. V(:, :, :, 2) or rhs(:, :, :, 1)); the
  ! right-hand side comes from src and the solution goes to the interior
  ! rows of dst (a different array):
  !   KIND_D2V     [lambda (D2 - k2 D0) - ni (D4 - 2 k2 D2 + k2^2 D0)] x = src
  !   KIND_ETA     [lambda D0 - ni (D2 - k2 D0)] x = src
  !   KIND_POISSON (D2 - k2 D0) x = src                  (lambda unused)
  !   KIND_D0      D0 x = src         the unweighted quantity behind a D0-weighted sum
  !   KIND_DY      D0 x = D1 src      d/dy, with the ghost rows of src filled
  ! The wrap phase uses the displacements shift_x, shift_z of the upper
  ! image; without them, those of the current time.  The (0,0) mode of
  ! KIND_D2V is singular and is set to zero; that of KIND_POISSON is
  ! singular too and dst is left untouched there for the caller.
  subroutine line_solve(sys, lambda, src, dst, shift_x, shift_z)
    use hst_mpi, only: allgather_y
    integer(C_INT), intent(in) :: sys
    real(C_DOUBLE), intent(in) :: lambda
    complex(C_DOUBLE_COMPLEX), intent(in), contiguous :: src(ny0 - 2:, -nz:, nx0:)
    complex(C_DOUBLE_COMPLEX), intent(inout), contiguous :: dst(ny0 - 2:, -nz:, nx0:)
    real(C_DOUBLE), intent(in), optional :: shift_x, shift_z
    integer(C_INT) :: ix0, ix1, nl, ix, iz, iy, il, ncol, n, m, ld, nslab, islab
    real(C_DOUBLE) :: sx, sz
    complex(C_DOUBLE_COMPLEX) :: ph

    if (present(shift_x)) then
      sx = shift_x; sz = shift_z
    else
      call shear_shifts(time, sx, sz)
    end if
    ncol = 2*nz + 1
    n = ny
    m = nyB
    ld = nlines_max
    nslab = npy
    islab = ipy
    do ix0 = nx0, nxN, chunk
      ix1 = min(ix0 + chunk - 1, nxN)
      nl = (ix1 - ix0 + 1)*ncol
      ! forward sweep, one thread per line: factor the slab's rows and
      ! write its record for the reduced system
      !$omp target teams distribute parallel do default(none) &
      !$omp shared(U1, U2, B1, B2, XR, record, src, der, k2, ni, lambda, sys, ix0, nz, nx0, ncol, n, m, ny0, nl, ld, islab) &
      !$omp private(il, ix, iz)
      do il = 1, nl
        ix = ix0 + (il - 1)/ncol
        iz = mod(il - 1, ncol) - nz
        if (.not. (ix == 0 .and. iz == 0 .and. (sys == KIND_D2V .or. sys == KIND_POISSON))) then
          call penta_forward(sys, lambda, ni, k2(iz, ix), n, m, ny0, ld, il, (iz + nz + 1) + ncol*(ix - nx0), &
                             der, U1, U2, B1, B2, XR, record(:, il, islab), src)
        end if
      end do
      call allgather_y(record, NREC*ld)
      ! reduced system and backward sweep, one thread per line
      !$omp target teams distribute parallel do default(none) &
      !$omp shared(U1, U2, B1, B2, XR, record, dst, der, k2, ni, lambda, sys, ix0, nz, nx0, ncol, alfa0, beta0, sx, sz, n, m, ny0, &
      !$omp        nl, ld, nslab, islab) private(il, ix, iz, iy, ph)
      do il = 1, nl
        ix = ix0 + (il - 1)/ncol
        iz = mod(il - 1, ncol) - nz
        if (ix == 0 .and. iz == 0 .and. sys == KIND_D2V) then
          do iy = 0, m - 1
            dst(ny0 + iy, iz, ix) = 0.0d0
          end do
        else if (.not. (ix == 0 .and. iz == 0 .and. sys == KIND_POISSON)) then
          ph = exp(dcmplx(0.0d0, -(alfa0*ix*sx + beta0*iz*sz)))
          call penta_backward(sys, lambda, ni, k2(iz, ix), ph, n, m, ny0, ld, il, (iz + nz + 1) + ncol*(ix - nx0), nslab, islab, &
                              der, U1, U2, B1, B2, XR, record, dst)
        end if
      end do
    end do
  end subroutine line_solve

  ! Row iy (global), band entry j of the system `sys` before the wrap phase.
  real(C_DOUBLE) function coef(sys, lambda, ni, kk, n, der, iy, j)
    !$omp declare target
    integer(C_INT), intent(in) :: sys, n, iy, j
    real(C_DOUBLE), intent(in) :: lambda, ni, kk
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    select case (sys)
    case (KIND_D2V)
      coef = lambda*(der(iy, 2, j) - kk*der(iy, 0, j)) - &
             ni*(der(iy, 3, j) - 2.0d0*kk*der(iy, 2, j) + kk*kk*der(iy, 0, j))
    case (KIND_ETA)
      coef = lambda*der(iy, 0, j) - ni*(der(iy, 2, j) - kk*der(iy, 0, j))
    case (KIND_POISSON)
      coef = der(iy, 2, j) - kk*der(iy, 0, j)
    case default
      coef = der(iy, 0, j)
    end select
  end function coef

  ! Right-hand side of the slab's row iy (local, 0..m-1) of line jl of src
  ! (the field's lines in memory order, ghost rows included): the field
  ! itself, or its D1 stencil through the ghost rows for KIND_DY.
  complex(C_DOUBLE_COMPLEX) function rhs_row(sys, n, m, ny0, der, src, iy, jl)
    !$omp declare target
    integer(C_INT), intent(in) :: sys, n, m, ny0, iy, jl
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    complex(C_DOUBLE_COMPLEX), intent(in) :: src(-2:m + 1, *)
    integer(C_INT) :: j
    if (sys == KIND_DY) then
      rhs_row = 0.0d0
      do j = -2, 2
        rhs_row = rhs_row + der(ny0 + iy, 1, j)*src(iy + j, jl)
      end do
    else
      rhs_row = src(iy, jl)
    end if
  end function rhs_row

  ! Forward sweep of one line over the slab's interior rows 0..m-3 (global
  ! ny0..nyN-2): generate row i, take the entries of rows 0 and 1 that
  ! point at the border below (their phase, if any, factored out) as the
  ! border columns, eliminate with rows i-2 and i-1, divide by the pivot,
  ! store the upper part, the border columns and the substituted
  ! right-hand side in workspace column il; accumulate the first two rows
  ! of the back substitution; write the slab's record.  Must stay inside
  ! this module: a declare-target procedure called across a module
  ! boundary does not survive nvlink.
  subroutine penta_forward(sys, lambda, ni, kk, n, m, ny0, ld, il, jl, der, U1, U2, B1, B2, XR, rec, src)
    !$omp declare target
    integer(C_INT), intent(in) :: sys, n, m, ny0, ld, il, jl
    real(C_DOUBLE), intent(in) :: lambda, ni, kk
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    real(C_DOUBLE), intent(inout) :: U1(ld, 0:m - 1), U2(ld, 0:m - 1), B1(ld, 0:m - 1), B2(ld, 0:m - 1)
    complex(C_DOUBLE_COMPLEX), intent(inout) :: XR(ld, 0:m - 1)
    real(C_DOUBLE), intent(out) :: rec(NREC)
    complex(C_DOUBLE_COMPLEX), intent(in) :: src(-2:m + 1, *)
    integer(C_INT) :: i, j, mi
    real(C_DOUBLE) :: a(-2:2), l, rp, t1, t2, p, q            ! row i: entries and border columns, scaled by 1/pivot
    real(C_DOUBLE) :: u11, u12, p1, q1, u21, u22, p2, q2       ! rows i-1 and i-2, scaled
    real(C_DOUBLE) :: w, w1, w2, v, v1, v2                     ! rows 0 and 1 of the inverse unit upper factor at i, i-1, i-2
    real(C_DOUBLE) :: sp0, sq0, sp1, sq1                       ! their inner products with the border columns
    complex(C_DOUBLE_COMPLEX) :: bx, x1, x2, sx0, sx1          ! the right-hand side: row i, i-1, i-2, the inner products

    mi = m - 2                                                 ! interior rows of the slab
    u11 = 0.0d0; u12 = 0.0d0; x1 = 0.0d0; p1 = 0.0d0; q1 = 0.0d0
    u21 = 0.0d0; u22 = 0.0d0; x2 = 0.0d0; p2 = 0.0d0; q2 = 0.0d0
    w1 = 0.0d0; w2 = 0.0d0; v1 = 0.0d0; v2 = 0.0d0
    sx0 = 0.0d0; sp0 = 0.0d0; sq0 = 0.0d0; sx1 = 0.0d0; sp1 = 0.0d0; sq1 = 0.0d0
    do i = 0, mi - 1
      do j = -2, 2
        a(j) = coef(sys, lambda, ni, kk, n, der, ny0 + i, j)
      end do
      p = 0.0d0; q = 0.0d0
      if (i == 0) then
        p = a(-2); q = a(-1)
      end if
      if (i == 1) q = a(-2)
      bx = rhs_row(sys, n, m, ny0, der, src, i, jl)
      if (i >= 2) then
        l = a(-2)
        a(-1) = a(-1) - l*u21
        a(0) = a(0) - l*u22
        bx = bx - l*x2; p = p - l*p2; q = q - l*q2
      end if
      if (i >= 1) then
        l = a(-1)
        a(0) = a(0) - l*u11
        a(1) = a(1) - l*u12
        bx = bx - l*x1; p = p - l*p1; q = q - l*q1
      end if
      rp = 1.0d0/a(0)
      t1 = a(1)*rp; t2 = a(2)*rp; bx = bx*rp; p = p*rp; q = q*rp
      U1(il, i) = t1; U2(il, i) = t2; B1(il, i) = p; B2(il, i) = q; XR(il, i) = bx
      ! rows 0 and 1 of the inverse of the unit upper factor: w(0) = 1, v(1) = 1,
      ! w(i) = -U1(i-1) w(i-1) - U2(i-2) w(i-2), the same for v
      w = -(u11*w1 + u22*w2); if (i == 0) w = 1.0d0
      v = -(u11*v1 + u22*v2); if (i == 1) v = 1.0d0
      sx0 = sx0 + w*bx; sp0 = sp0 + w*p; sq0 = sq0 + w*q
      sx1 = sx1 + v*bx; sp1 = sp1 + v*p; sq1 = sq1 + v*q
      w2 = w1; w1 = w; v2 = v1; v1 = v
      u21 = u11; u22 = u12; x2 = x1; p2 = p1; q2 = q1
      u11 = t1; u12 = t2; x1 = bx; p1 = p; q1 = q
    end do
    ! The record: rows 0 and 1 from the sums (w(mi-2), w(mi-1) weight the
    ! entries U2(mi-2), U1(mi-1), U2(mi-1) that point at b_s), rows mi-2
    ! and mi-1 from the registers, and the right-hand side of the border rows.
    rec(1) = dreal(sx0); rec(2) = dimag(sx0); rec(3) = sp0; rec(4) = sq0
    rec(5) = w2*u22 + w1*u11; rec(6) = w1*u12
    rec(7) = dreal(sx1); rec(8) = dimag(sx1); rec(9) = sp1; rec(10) = sq1
    rec(11) = v2*u22 + v1*u11; rec(12) = v1*u12
    bx = x2 - u21*x1
    rec(13) = dreal(bx); rec(14) = dimag(bx); rec(15) = p2 - u21*p1; rec(16) = q2 - u21*q1
    rec(17) = u22 - u21*u11; rec(18) = -u21*u12
    rec(19) = dreal(x1); rec(20) = dimag(x1); rec(21) = p1; rec(22) = q1
    rec(23) = u11; rec(24) = u12
    bx = rhs_row(sys, n, m, ny0, der, src, m - 2, jl)
    rec(25) = dreal(bx); rec(26) = dimag(bx)
    bx = rhs_row(sys, n, m, ny0, der, src, m - 1, jl)
    rec(27) = dreal(bx); rec(28) = dimag(bx)
  end subroutine penta_forward

  ! The reduced interface system of one line from the records of all
  ! slabs, solved for the 2 nslab border unknowns, and the backward sweep
  ! of this slab's rows into dst(:, jl).
  subroutine penta_backward(sys, lambda, ni, kk, ph, n, m, ny0, ld, il, jl, nslab, islab, der, U1, U2, B1, B2, XR, rec, dst)
    !$omp declare target
    integer(C_INT), intent(in) :: sys, n, m, ny0, ld, il, jl, nslab, islab
    real(C_DOUBLE), intent(in) :: lambda, ni, kk
    complex(C_DOUBLE_COMPLEX), intent(in) :: ph
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    real(C_DOUBLE), intent(in) :: U1(ld, 0:m - 1), U2(ld, 0:m - 1), B1(ld, 0:m - 1), B2(ld, 0:m - 1)
    complex(C_DOUBLE_COMPLEX), intent(in) :: XR(ld, 0:m - 1)
    real(C_DOUBLE), intent(in) :: rec(NREC, ld, 0:nslab - 1)
    complex(C_DOUBLE_COMPLEX), intent(inout) :: dst(-2:m + 1, *)
    integer(C_INT) :: s, t, r, c, i, j, k, ns, ga, cm, cs, cp
    real(C_DOUBLE) :: c1m4, c1m3, c10, c2m3, c20, c21, d11, d12, d21, d22
    complex(C_DOUBLE_COMPLEX) :: cph, php, x0, x1, xm2, xm1, f, piv, xb1, xb2, yb1, yb2, xk, xk1, xk2
    complex(C_DOUBLE_COMPLEX) :: A(2*NPY_MAX, 2*NPY_MAX), b(2*NPY_MAX)
    integer(C_INT), parameter :: nb = 4                        ! rows per iteration of the backward sweep
    real(C_DOUBLE) :: u1s(0:nb - 1), u2s(0:nb - 1), b1s(0:nb - 1), b2s(0:nb - 1)
    complex(C_DOUBLE_COMPLEX) :: xs(0:nb - 1)

    ! Block row s: the two border rows of slab s, global rows ga = ny0_s + m - 2
    ! and ga + 1, with the entries at their own rows m-2, m-1 (c1m4, c1m3,
    ! c2m3), at b_s (d11..d22) and at rows 0, 1 of slab s+1 (c10, c20, c21,
    ! with the phase ph across the box edge); the four rows substituted from
    ! the records.  The columns of b_{s-1}, b_s, b_{s+1} coincide when
    ! nslab is 1 or 2, so the blocks are accumulated.
    ns = 2*nslab
    do r = 1, ns
      b(r) = 0.0d0
      do c = 1, ns
        A(r, c) = 0.0d0
      end do
    end do
    do s = 0, nslab - 1
      t = mod(s + 1, nslab)                                    ! the slab above
      cm = 2*mod(s - 1 + nslab, nslab); cs = 2*s; cp = 2*t     ! columns of b_{s-1}, b_s, b_{s+1}, minus one
      cph = 1.0d0; if (s == 0) cph = conjg(ph)                 ! phase of slab s's coupling to b_{s-1}
      php = 1.0d0; if (s == nslab - 1) php = ph                ! phase of slab s's coupling to slab s+1
      ga = s*m + m - 2
      c1m4 = coef(sys, lambda, ni, kk, n, der, ga, -2)
      c1m3 = coef(sys, lambda, ni, kk, n, der, ga, -1)
      d11 = coef(sys, lambda, ni, kk, n, der, ga, 0)
      d12 = coef(sys, lambda, ni, kk, n, der, ga, 1)
      c10 = coef(sys, lambda, ni, kk, n, der, ga, 2)
      c2m3 = coef(sys, lambda, ni, kk, n, der, ga + 1, -2)
      d21 = coef(sys, lambda, ni, kk, n, der, ga + 1, -1)
      d22 = coef(sys, lambda, ni, kk, n, der, ga + 1, 0)
      c20 = coef(sys, lambda, ni, kk, n, der, ga + 1, 1)
      c21 = coef(sys, lambda, ni, kk, n, der, ga + 1, 2)
      ! row 2s+1: c1m4 x_s(m-2) + c1m3 x_s(m-1) + d11 b1 + d12 b2 + c10 php x_t(0) = r1
      r = cs + 1
      xm2 = dcmplx(rec(13, il, s), rec(14, il, s)); xm1 = dcmplx(rec(19, il, s), rec(20, il, s))
      x0 = dcmplx(rec(1, il, t), rec(2, il, t))
      b(r) = dcmplx(rec(25, il, s), rec(26, il, s)) - c1m4*xm2 - c1m3*xm1 - c10*php*x0
      A(r, cm + 1) = A(r, cm + 1) - cph*(c1m4*rec(15, il, s) + c1m3*rec(21, il, s))
      A(r, cm + 2) = A(r, cm + 2) - cph*(c1m4*rec(16, il, s) + c1m3*rec(22, il, s))
      A(r, cs + 1) = A(r, cs + 1) + d11 - (c1m4*rec(17, il, s) + c1m3*rec(23, il, s))
      A(r, cs + 2) = A(r, cs + 2) + d12 - (c1m4*rec(18, il, s) + c1m3*rec(24, il, s))
      f = c10*php
      if (t == 0) f = f*conjg(ph)                              ! slab t's coupling to b_s is across the box edge
      A(r, cs + 1) = A(r, cs + 1) - f*rec(3, il, t)
      A(r, cs + 2) = A(r, cs + 2) - f*rec(4, il, t)
      A(r, cp + 1) = A(r, cp + 1) - c10*php*rec(5, il, t)
      A(r, cp + 2) = A(r, cp + 2) - c10*php*rec(6, il, t)
      ! row 2s+2: c2m3 x_s(m-1) + d21 b1 + d22 b2 + php (c20 x_t(0) + c21 x_t(1)) = r2
      r = cs + 2
      x1 = dcmplx(rec(7, il, t), rec(8, il, t))
      b(r) = dcmplx(rec(27, il, s), rec(28, il, s)) - c2m3*xm1 - php*(c20*x0 + c21*x1)
      A(r, cm + 1) = A(r, cm + 1) - cph*c2m3*rec(21, il, s)
      A(r, cm + 2) = A(r, cm + 2) - cph*c2m3*rec(22, il, s)
      A(r, cs + 1) = A(r, cs + 1) + d21 - c2m3*rec(23, il, s)
      A(r, cs + 2) = A(r, cs + 2) + d22 - c2m3*rec(24, il, s)
      f = php
      if (t == 0) f = f*conjg(ph)
      A(r, cs + 1) = A(r, cs + 1) - f*(c20*rec(3, il, t) + c21*rec(9, il, t))
      A(r, cs + 2) = A(r, cs + 2) - f*(c20*rec(4, il, t) + c21*rec(10, il, t))
      A(r, cp + 1) = A(r, cp + 1) - php*(c20*rec(5, il, t) + c21*rec(11, il, t))
      A(r, cp + 2) = A(r, cp + 2) - php*(c20*rec(6, il, t) + c21*rec(12, il, t))
    end do
    ! Gaussian elimination without pivoting, then back substitution.
    do k = 1, ns - 1
      piv = 1.0d0/A(k, k)
      do i = k + 1, ns
        f = A(i, k)*piv
        do j = k + 1, ns
          A(i, j) = A(i, j) - f*A(k, j)
        end do
        b(i) = b(i) - f*b(k)
      end do
    end do
    do i = ns, 1, -1
      f = b(i)
      do j = i + 1, ns
        f = f - A(i, j)*b(j)
      end do
      b(i) = f/A(i, i)
    end do
    ! This slab's border and the one below it (with the phase for the first slab).
    xb1 = b(2*islab + 1); xb2 = b(2*islab + 2)
    dst(m - 2, jl) = xb1
    dst(m - 1, jl) = xb2
    cm = 2*mod(islab - 1 + nslab, nslab)
    cph = 1.0d0; if (islab == 0) cph = conjg(ph)
    yb1 = cph*b(cm + 1); yb2 = cph*b(cm + 2)
    ! Backward sweep of b minus the border columns times the border below,
    ! seeded with this slab's border (rows m-4, m-3 reach it through U1,
    ! U2).  nb rows per iteration, their loads issued together before any
    ! of them is used: the compiler keeps the loads of a row behind the
    ! stores of the previous one, and one memory latency per row would
    ! bound the sweep.
    xk1 = xb1; xk2 = xb2
    do i = m - 3, nb - 1, -nb
      do k = 0, nb - 1
        xs(k) = XR(il, i - k); b1s(k) = B1(il, i - k); b2s(k) = B2(il, i - k); u1s(k) = U1(il, i - k); u2s(k) = U2(il, i - k)
      end do
      do k = 0, nb - 1
        xk = xs(k) - b1s(k)*yb1 - b2s(k)*yb2 - u1s(k)*xk1 - u2s(k)*xk2
        dst(i - k, jl) = xk
        xk2 = xk1; xk1 = xk
      end do
    end do
    do j = i, 0, -1                                            ! the rows left over
      xk = XR(il, j) - B1(il, j)*yb1 - B2(il, j)*yb2 - U1(il, j)*xk1 - U2(il, j)*xk2
      dst(j, jl) = xk
      xk2 = xk1; xk1 = xk
    end do
  end subroutine penta_backward

end module hst_linsolve
