! The wall-normal (y) linear systems: one cyclic pentadiagonal system per
! Fourier mode, all modes solved at once, one GPU thread per mode.
!
! The five-point compact stencil on a periodic y gives a pentadiagonal
! matrix whose stencil wraps around the box; the wrapped entries carry the
! shear-periodic phase.  Written with band index j = -2..2 (column = row + j
! modulo ny), row iy of a line is
!   a(j) = coef(kind, iy, j) * wrap,   wrap = ph if iy + j >= ny, conjg(ph) if iy + j < 0, else 1
! with the real stencil combination coef of the system kind (line_solve).
!
! The solve uses bordering: the last two unknowns x(ny-2), x(ny-1) are the
! border.  Rows and columns 0..ny-3 form a *plain* pentadiagonal matrix P
! (the wrapped entries of rows 0, 1 point at the border columns, and rows
! ny-2, ny-1 are the border rows), so
!   1. P is factored, and three right-hand sides are forward-substituted in
!      the same sweep: b, and the two columns that couple to the border,
!   2. a 2x2 Schur complement gives the border,
!   3. the back substitution of b minus the border columns times the border
!      is the solution.
! This is the npy = 1 case of a distributed Schur solve.
!
! The rows are generated on the fly inside the sweep, so the matrix is
! never stored.  P is *real*: the stencil coefficients are real and the
! wrap phase only enters the border rows and, through the wrapped entries
! of rows 0 and 1, the border columns, where it is one common factor
! conjg(ph) that is applied in the back substitution.  So the forward
! sweep works in real arithmetic on the matrix and the two border columns
! (one real division per row) and in complex arithmetic only on the
! right-hand side; it keeps the two previous rows in registers, divides
! each row by its pivot and writes the two upper entries U1, U2, the
! border columns B1, B2 (real) and the substituted right-hand side X
! (complex).  The entries of the last two rows of P that point at the
! border columns stay in U1, U2 and act through the seed of the back
! substitution, x(n-2) and x(n-1).  The Schur complement needs the
! back-substituted values at the four rows coupled to the border: the last
! two are the last two rows of the sweep, and the first two are inner
! products of the substituted right-hand sides with the first two rows of
! the inverse of the unit upper factor, which is a *forward* recurrence
! (w(i) = -U1(i-1) w(i-1) - U2(i-2) w(i-2)), so both are accumulated in
! the same sweep and the border is known before the back substitution.
! The solve is then two passes over the line: the forward sweep reads the
! right-hand side from the field and writes the five columns, the backward
! sweep reads them and writes the solution into the field (128 bytes per
! row).  Storage is interleaved, line index first (U1(iline, iy)), so that
! the threads of a kernel read consecutive addresses.  Lines are processed
! in batches of line_chunk x columns (deck parameter; 0 = all columns at
! once on the GPU, 16 on the CPU) to bound that workspace.
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
  real(C_DOUBLE), allocatable, save :: U1(:, :), U2(:, :), B1(:, :), B2(:, :)
  complex(C_DOUBLE_COMPLEX), allocatable, save :: X(:, :)
  integer(C_INT), save :: nlines_max, chunk

contains

  subroutine init_linsolve()
    ! default: every column at once on the GPU (fewest launches); 16 on the
    ! CPU, where the workspace of all columns falls out of the cache
    chunk = 16
#ifdef HAVE_CUDA
    chunk = nxB
#endif
    if (line_chunk > 0) chunk = line_chunk
    chunk = min(chunk, nxB)
    nlines_max = (2*nz + 1)*chunk
    allocate (U1(nlines_max, 0:ny - 1), U2(nlines_max, 0:ny - 1), B1(nlines_max, 0:ny - 1), B2(nlines_max, 0:ny - 1))
    allocate (X(nlines_max, 0:ny - 1))
    U1 = 0; U2 = 0; B1 = 0; B2 = 0; X = 0
    !$omp target enter data map(to: U1, U2, B1, B2, X)
    if (has_terminal) write (*, '(A,I0,A,I0,A,F8.1,A)') '   line solver: batches of ', chunk, ' x columns, ', &
      nlines_max, ' lines, workspace ', 48.0d0*nlines_max*ny/1024.0d0**2, ' MB per rank'
  end subroutine init_linsolve

  subroutine free_linsolve()
    !$omp target exit data map(delete: U1, U2, B1, B2, X)
    deallocate (U1, U2, B1, B2, X)
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
  subroutine line_solve(kind, lambda, src, dst, shift_x, shift_z)
    integer(C_INT), intent(in) :: kind
    real(C_DOUBLE), intent(in) :: lambda
    complex(C_DOUBLE_COMPLEX), intent(in), contiguous :: src(ny0 - 2:, -nz:, nx0:)
    complex(C_DOUBLE_COMPLEX), intent(inout), contiguous :: dst(ny0 - 2:, -nz:, nx0:)
    real(C_DOUBLE), intent(in), optional :: shift_x, shift_z
    integer(C_INT) :: ix0, ix1, nl, ix, iz, iy, il, ncol, n, ld
    real(C_DOUBLE) :: sx, sz
    complex(C_DOUBLE_COMPLEX) :: ph

    if (present(shift_x)) then
      sx = shift_x; sz = shift_z
    else
      call shear_shifts(time, sx, sz)
    end if
    ncol = 2*nz + 1
    n = ny
    ld = nlines_max
    do ix0 = nx0, nxN, chunk
      ix1 = min(ix0 + chunk - 1, nxN)
      nl = (ix1 - ix0 + 1)*ncol
      ! one thread per line: build the rows, factor and solve
      !$omp target teams distribute parallel do default(none) &
      !$omp shared(U1, U2, B1, B2, X, src, dst, der, k2, ni, lambda, kind, ix0, nz, nx0, ncol, alfa0, beta0, sx, sz, n, nl, ld) &
      !$omp private(il, ix, iz, iy, ph)
      do il = 1, nl
        ix = ix0 + (il - 1)/ncol
        iz = mod(il - 1, ncol) - nz
        if (ix == 0 .and. iz == 0 .and. kind == KIND_D2V) then
          do iy = 0, n - 1
            dst(iy, iz, ix) = 0.0d0
          end do
        else if (.not. (ix == 0 .and. iz == 0 .and. kind == KIND_POISSON)) then
          ph = exp(dcmplx(0.0d0, -(alfa0*ix*sx + beta0*iz*sz)))
          call cyclic_penta_solve(kind, lambda, ni, k2(iz, ix), ph, n, ld, il, (iz + nz + 1) + ncol*(ix - nx0), &
                                  der, U1, U2, B1, B2, X, src, dst)
        end if
      end do
    end do
  end subroutine line_solve

  ! Row iy, band entry j of the system `kind` before the wrap phase.
  real(C_DOUBLE) function coef(kind, lambda, ni, kk, n, der, iy, j)
    !$omp declare target
    integer(C_INT), intent(in) :: kind, n, iy, j
    real(C_DOUBLE), intent(in) :: lambda, ni, kk
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    select case (kind)
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

  ! Right-hand side of row iy of line jl of src (the field's lines in memory
  ! order): the field itself, or its D1 stencil through the ghost rows for
  ! KIND_DY.
  complex(C_DOUBLE_COMPLEX) function rhs_row(kind, n, der, src, iy, jl)
    !$omp declare target
    integer(C_INT), intent(in) :: kind, n, iy, jl
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    complex(C_DOUBLE_COMPLEX), intent(in) :: src(-2:n + 1, *)
    integer(C_INT) :: j
    if (kind == KIND_DY) then
      rhs_row = 0.0d0
      do j = -2, 2
        rhs_row = rhs_row + der(iy, 1, j)*src(iy + j, jl)
      end do
    else
      rhs_row = src(iy, jl)
    end if
  end function rhs_row

  ! One line: right-hand side in src(:, jl), solution to dst(:, jl) (line jl
  ! of the field in memory order, ghost rows included), workspace column il.
  ! Must stay inside this module: a declare-target procedure called across
  ! a module boundary does not survive nvlink.
  subroutine cyclic_penta_solve(kind, lambda, ni, kk, ph, n, ld, il, jl, der, U1, U2, B1, B2, X, src, dst)
    !$omp declare target
    integer(C_INT), intent(in) :: kind, n, ld, il, jl
    real(C_DOUBLE), intent(in) :: lambda, ni, kk
    complex(C_DOUBLE_COMPLEX), intent(in) :: ph
    real(C_DOUBLE), intent(in) :: der(0:n - 1, 0:3, -2:2)
    real(C_DOUBLE), intent(inout) :: U1(ld, 0:n - 1), U2(ld, 0:n - 1), B1(ld, 0:n - 1), B2(ld, 0:n - 1)
    complex(C_DOUBLE_COMPLEX), intent(inout) :: X(ld, 0:n - 1)
    complex(C_DOUBLE_COMPLEX), intent(in) :: src(-2:n + 1, *)
    complex(C_DOUBLE_COMPLEX), intent(inout) :: dst(-2:n + 1, *)
    integer(C_INT) :: i, j, m
    real(C_DOUBLE) :: a(-2:2), l, rp, t1, t2, p, q            ! row i: entries and border columns, scaled by 1/pivot
    real(C_DOUBLE) :: u11, u12, p1, q1, u21, u22, p2, q2       ! rows i-1 and i-2, scaled
    real(C_DOUBLE) :: w, w1, w2, v, v1, v2                     ! rows 0 and 1 of the inverse unit upper factor at i, i-1, i-2
    real(C_DOUBLE) :: sp0, sq0, sp1, sq1                       ! their inner products with the border columns
    real(C_DOUBLE) :: c1m4, c1m3, c2m3, d11, d12, d21, d22
    complex(C_DOUBLE_COMPLEX) :: bx, x1, x2, sx0, sx1, cph     ! the right-hand side: row i, i-1, i-2, the inner products
    complex(C_DOUBLE_COMPLEX) :: xm1, pm1, qm1, xm2, pm2, qm2  ! back-substituted rows m-1, m-2 as xm - pm x(n-2) - qm x(n-1)
    complex(C_DOUBLE_COMPLEX) :: t0p, t0q, t1p, t1q            ! the same for rows 0, 1 (the sums)
    complex(C_DOUBLE_COMPLEX) :: c10, c20, c21, s1, s2, m11, m12, m21, m22, det, xb1, xb2, yb1, yb2, xk, xk1, xk2
    integer(C_INT), parameter :: nb = 4                        ! rows per iteration of the backward sweep
    real(C_DOUBLE) :: u1s(0:nb - 1), u2s(0:nb - 1), b1s(0:nb - 1), b2s(0:nb - 1)
    complex(C_DOUBLE_COMPLEX) :: xs(0:nb - 1)
    integer(C_INT) :: k

    m = n - 2
    cph = conjg(ph)
    ! Forward sweep over the interior rows: generate row i, take the wrapped
    ! entries of rows 0 and 1 (their phase conjg(ph) factored out) as the
    ! border columns, eliminate with rows i-2 and i-1, divide by the pivot,
    ! store the upper part, the border columns and the substituted
    ! right-hand side; accumulate the first two rows of the back substitution.
    u11 = 0.0d0; u12 = 0.0d0; x1 = 0.0d0; p1 = 0.0d0; q1 = 0.0d0
    u21 = 0.0d0; u22 = 0.0d0; x2 = 0.0d0; p2 = 0.0d0; q2 = 0.0d0
    w1 = 0.0d0; w2 = 0.0d0; v1 = 0.0d0; v2 = 0.0d0
    sx0 = 0.0d0; sp0 = 0.0d0; sq0 = 0.0d0; sx1 = 0.0d0; sp1 = 0.0d0; sq1 = 0.0d0
    do i = 0, m - 1
      do j = -2, 2
        a(j) = coef(kind, lambda, ni, kk, n, der, i, j)
      end do
      p = 0.0d0; q = 0.0d0
      if (i == 0) then
        p = a(-2); q = a(-1)
      end if
      if (i == 1) q = a(-2)
      bx = rhs_row(kind, n, der, src, i, jl)
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
      U1(il, i) = t1; U2(il, i) = t2; B1(il, i) = p; B2(il, i) = q; X(il, i) = bx
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
    ! Back-substituted values at the rows coupled to the border, as
    ! x = xm - pm x(n-2) - qm x(n-1): rows m-1 and m-2 from the registers
    ! (U2(m-2), U1(m-1) point at x(n-2) and U2(m-1) at x(n-1)), rows 0 and
    ! 1 from the sums (w(m-2), w(m-1) weight those same entries).
    xm1 = x1; pm1 = cph*p1 + u11; qm1 = cph*q1 + u12
    xm2 = x2 - u21*xm1; pm2 = cph*p2 + u22 - u21*pm1; qm2 = cph*q2 - u21*qm1
    t0p = cph*sp0 + w2*u22 + w1*u11; t0q = cph*sq0 + w1*u12
    t1p = cph*sp1 + v2*u22 + v1*u11; t1q = cph*sq1 + v1*u12
    ! Border rows: their interior couplings (C) and their 2x2 block (D).
    c1m4 = coef(kind, lambda, ni, kk, n, der, n - 2, -2)
    c1m3 = coef(kind, lambda, ni, kk, n, der, n - 2, -1)
    c10 = coef(kind, lambda, ni, kk, n, der, n - 2, 2)*ph
    c2m3 = coef(kind, lambda, ni, kk, n, der, n - 1, -2)
    c20 = coef(kind, lambda, ni, kk, n, der, n - 1, 1)*ph
    c21 = coef(kind, lambda, ni, kk, n, der, n - 1, 2)*ph
    d11 = coef(kind, lambda, ni, kk, n, der, n - 2, 0)
    d12 = coef(kind, lambda, ni, kk, n, der, n - 2, 1)
    d21 = coef(kind, lambda, ni, kk, n, der, n - 1, -1)
    d22 = coef(kind, lambda, ni, kk, n, der, n - 1, 0)
    ! 2x2 Schur complement for the border.
    s1 = rhs_row(kind, n, der, src, n - 2, jl) - (c1m4*xm2 + c1m3*xm1 + c10*sx0)
    s2 = rhs_row(kind, n, der, src, n - 1, jl) - (c2m3*xm1 + c20*sx0 + c21*sx1)
    m11 = d11 - (c1m4*pm2 + c1m3*pm1 + c10*t0p)
    m12 = d12 - (c1m4*qm2 + c1m3*qm1 + c10*t0q)
    m21 = d21 - (c2m3*pm1 + c20*t0p + c21*t1p)
    m22 = d22 - (c2m3*qm1 + c20*t0q + c21*t1q)
    det = m11*m22 - m12*m21
    xb1 = (m22*s1 - m12*s2)/det
    xb2 = (m11*s2 - m21*s1)/det
    dst(n - 2, jl) = xb1
    dst(n - 1, jl) = xb2
    ! Backward sweep of b minus the border columns times the border, into
    ! dst, seeded with the border (rows m-2, m-1 reach it through U1, U2).
    ! nb rows per iteration, their loads issued together before any of them
    ! is used: the compiler keeps the loads of a row behind the stores of
    ! the previous one, and one memory latency per row would bound the sweep.
    yb1 = cph*xb1; yb2 = cph*xb2
    xk1 = xb1; xk2 = xb2
    do i = m - 1, nb - 1, -nb
      do k = 0, nb - 1
        xs(k) = X(il, i - k); b1s(k) = B1(il, i - k); b2s(k) = B2(il, i - k); u1s(k) = U1(il, i - k); u2s(k) = U2(il, i - k)
      end do
      do k = 0, nb - 1
        xk = xs(k) - b1s(k)*yb1 - b2s(k)*yb2 - u1s(k)*xk1 - u2s(k)*xk2
        dst(i - k, jl) = xk
        xk2 = xk1; xk1 = xk
      end do
    end do
    do j = i, 0, -1                                            ! the rows left over
      xk = X(il, j) - B1(il, j)*yb1 - B2(il, j)*yb2 - U1(il, j)*xk1 - U2(il, j)*xk2
      dst(j, jl) = xk
      xk2 = xk1; xk1 = xk
    end do
  end subroutine cyclic_penta_solve

end module hst_linsolve
