! The decomposition (x-z pencils times y slabs), the transposes between
! the two pencil layouts, the ghost-row exchange between slabs and the
! allgather of the line solver's reduced systems.
!
! The nproc = npxz*npy ranks form npy slabs of nyB = ny/npy rows, each
! split into npxz x-z pencils: in spectral space (z-pencil) a rank owns the
! x modes nx0:nxN and every z mode of its rows, in physical space
! (x-pencil) the z lines nz0:nzN and every x point.  Moving between the
! two is one alltoall over the npxz ranks of the slab (comm_xz) per field;
! the coupling along y (five-point stencils, line solves) goes between
! neighbouring slabs through the four ghost rows of every field
! (exchange_ghost_rows) and, for the line solves, through the allgather
! of the slabs' reduced systems over the y column (allgather_y_start, comm_y);
! see hst_linsolve.  One slab per node keeps the alltoalls on NVLink
! (mpirun --map-by ppr:npxz:node).  The physics files know nothing of this
! (DESIGN.md 7 (i)); with npy = 1 this file is the x-z pencil code of the
! main branch, the ghost rows being the shear-periodic wrap of the slab.
!
! A transpose is done in two halves so that the alltoall of one field can
! overlap the transforms of the next (the channel's CHANNEL_OVERLAPPING):
! transpose_*_start packs the field and starts its alltoall,
! transpose_*_finish waits for it and unpacks.  Two send and receive
! buffers alternate between fields (double buffering), so at most two
! transposes are in flight and start(m+2) must follow finish(m).  The
! alltoall goes through MPI (CUDA-aware, the buffers stay on the device;
! MPI_Ialltoall, waited for in finish) or, in a build with NCCL=1 and one
! GPU per rank, through NCCL as grouped send/recv pairs on a second CUDA
! stream, ordered against the OpenMP target stream by two events, so that
! the host never waits (deck parameter transport).  The y exchanges use
! a second NCCL communicator over the y column on the target stream
! itself (MPI on the device buffers otherwise): across nodes MPI's device
! path is several times slower than NCCL's (FINDINGS.md).
!
! Taken from channel/src/mpi/mpi_transpose.f90 with the y-slab machinery
! and HIP removed.  The pack kernels are block copies (the alltoall
! permutes whole blocks, so the leading index stays leading); the change of
! leading index happens on the receive side, in one tiled transpose kernel
! that also converts the two layouts directly on a single rank.
module hst_mpi

  use, intrinsic :: iso_c_binding
  use mpi_f08
  use hst_params
#ifdef HAVE_CUDA
  use hst_fft, only: target_stream
  use cudafor
#endif
#ifdef HAVE_NCCL
  use omp_lib, only: omp_get_num_devices, omp_get_default_device
#endif

  implicit none
  private

  public :: setup_decomposition, free_mpi, exchange_ghost_rows, allgather_y_start, allgather_y_wait, NPY_MAX
  public :: transpose_zTOx_start, transpose_zTOx_finish, transpose_xTOz_start, transpose_xTOz_finish

  ! the two buffer pairs of the double buffering
  complex(C_DOUBLE_COMPLEX), allocatable, target, save :: sendbuf(:, :), recvbuf(:, :)
  integer(C_INT), save :: sendcount                ! elements per peer, one field
  !$omp declare target(sendcount)
  logical, save :: transpose_is_local
  logical, save :: use_nccl = .false., use_nccl_y = .false.   ! NCCL for the alltoall (comm_xz); for the y exchanges (comm_y)
  integer, save :: node_ranks                                   ! ranks per node (setup_decomposition)
  integer(C_INT), parameter :: NPY_MAX = 8                      ! slabs per line: the reduced system is at most 16 x 16 (hst_linsolve)
  type(MPI_Request), save :: req(2)
#ifdef HAVE_CUDA
  integer(kind=cuda_stream_kind), save :: compute_stream, comm_stream
  type(cudaEvent), save :: ev_packed(2), ev_done(2)  ! pack done (compute stream), alltoall done (comm stream)
  type(cudaEvent), save :: ev_rec(2), ev_gath(2)     ! records written (compute stream), gathered (comm stream), per workspace
#endif
  integer(C_INT), parameter :: TILE = 32, ROWS_PER_THREAD = 4   ! transpose_tiled: tile edge, rows per thread
  type(MPI_Comm), save :: comm_xz, comm_y                       ! the ranks of a slab; the ranks of a y column
  real(C_DOUBLE), save :: t_ghost = 0.0d0, t_gather = 0.0d0     ! time in the y exchanges (timing = .true.)
  ! the two rows sent to and received from each neighbouring slab (exchange_ghost_rows)
  complex(C_DOUBLE_COMPLEX), allocatable, target, save :: ghost_send(:, :, :, :), ghost_recv(:, :, :, :)

#ifdef HAVE_NCCL
  ! NCCL through its C prototypes (nccl.h).  The Fortran module of NVHPC
  ! wants CUDA Fortran device arrays, which OpenMP-mapped arrays are not.
  type, bind(c) :: nccl_unique_id
    integer(C_INT8_T) :: bytes(128)
  end type nccl_unique_id
  integer(C_INT), parameter :: NCCL_UINT8 = 1
  type(C_PTR), save :: nccl_comm = C_NULL_PTR, nccl_stream = C_NULL_PTR     ! the slab's communicator, on comm_stream
  type(C_PTR), save :: nccl_comm_y = C_NULL_PTR, nccl_ystream = C_NULL_PTR  ! the y column's, on the compute stream
  interface
    function ncclGetUniqueId(id) bind(c, name='ncclGetUniqueId') result(r)
      import :: nccl_unique_id, C_INT
      type(nccl_unique_id) :: id
      integer(C_INT) :: r
    end function ncclGetUniqueId
    function ncclCommInitRank(comm, nranks, id, rank) bind(c, name='ncclCommInitRank') result(r)
      import :: nccl_unique_id, C_INT, C_PTR
      type(C_PTR) :: comm
      integer(C_INT), value :: nranks, rank
      type(nccl_unique_id), value :: id
      integer(C_INT) :: r
    end function ncclCommInitRank
    function ncclCommDestroy(comm) bind(c, name='ncclCommDestroy') result(r)
      import :: C_INT, C_PTR
      type(C_PTR), value :: comm
      integer(C_INT) :: r
    end function ncclCommDestroy
    function ncclGroupStart() bind(c, name='ncclGroupStart') result(r)
      import :: C_INT
      integer(C_INT) :: r
    end function ncclGroupStart
    function ncclGroupEnd() bind(c, name='ncclGroupEnd') result(r)
      import :: C_INT
      integer(C_INT) :: r
    end function ncclGroupEnd
    function ncclSend(buf, count, dtype, peer, comm, stream) bind(c, name='ncclSend') result(r)
      import :: C_INT, C_PTR, C_SIZE_T
      type(C_PTR), value :: buf, comm, stream
      integer(C_SIZE_T), value :: count
      integer(C_INT), value :: dtype, peer
      integer(C_INT) :: r
    end function ncclSend
    function ncclRecv(buf, count, dtype, peer, comm, stream) bind(c, name='ncclRecv') result(r)
      import :: C_INT, C_PTR, C_SIZE_T
      type(C_PTR), value :: buf, comm, stream
      integer(C_SIZE_T), value :: count
      integer(C_INT), value :: dtype, peer
      integer(C_INT) :: r
    end function ncclRecv
    function ncclAllGather(sendbuf, recvbuf, count, dtype, comm, stream) bind(c, name='ncclAllGather') result(r)
      import :: C_INT, C_PTR, C_SIZE_T
      type(C_PTR), value :: sendbuf, recvbuf, comm, stream
      integer(C_SIZE_T), value :: count
      integer(C_INT), value :: dtype
      integer(C_INT) :: r
    end function ncclAllGather
  end interface
#endif
  integer :: ierr

contains

  ! npxz = nproc/npy ranks each own nxB = (nx+1)/npxz x modes in spectral
  ! space and nzB = nzd/npxz z lines in physical space (the alltoall needs
  ! both splits to be even), and npy slabs own nyB = ny/npy rows each (at
  ! least 8, so that the line solver's four rows next to the borders are
  ! distinct).  The npxz ranks of a slab are consecutive: ipy = iproc/npxz,
  ! ipxz = mod(iproc, npxz).  npy = 0 (the default) lets the code choose:
  ! one slab per node when that fits the grid, so that every alltoall stays
  ! inside a node, otherwise one slab.  With more than one slab the ranks
  ! of a node must be consecutive in MPI_COMM_WORLD, as mpirun --map-by
  ! ppr:N:node and Slurm's block distribution give.
  subroutine setup_decomposition()
    integer(C_SIZE_T) :: n
    type(MPI_Comm) :: node
    integer :: node_rank, nnodes, npxz_node
    logical :: consecutive

    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, node, ierr)
    call MPI_Comm_size(node, node_ranks, ierr)
    call MPI_Comm_rank(node, node_rank, ierr)
    call MPI_Comm_free(node, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, node_ranks, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)
    consecutive = (node_rank == mod(iproc, node_ranks))
    call MPI_Allreduce(MPI_IN_PLACE, consecutive, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD, ierr)
    consecutive = consecutive .and. mod(nproc, node_ranks) == 0      ! the same number of ranks on every node
    nnodes = nproc/node_ranks
    if (npy == 0) then
      npy = 1
      npxz_node = nproc/nnodes
      if (consecutive .and. nnodes > 1 .and. nnodes <= NPY_MAX .and. mod(ny, nnodes) == 0 .and. ny/nnodes >= 8 &
          .and. mod(nx + 1, npxz_node) == 0 .and. mod(nzd, npxz_node) == 0) npy = nnodes
      if (has_terminal .and. nnodes > 1 .and. npy == 1) write (*, '(A,I0,A)') &
        '   npy = 0: one slab per node (', nnodes, ' nodes) does not fit this grid or rank layout, taking one slab'
    else if (npy > 1 .and. .not. consecutive) then
      if (has_terminal) print *, 'ERROR: with npy > 1 every node must hold the same number of ranks, consecutive in', &
        ' MPI_COMM_WORLD (mpirun --map-by ppr:N:node, Slurm block distribution)'
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    if (npy < 1 .or. mod(nproc, npy) /= 0 .or. mod(ny, npy) /= 0 .or. ny/max(npy, 1) < 8) then
      if (has_terminal) then
        print *, 'ERROR: npy must divide nproc and ny, with at least 8 rows per slab.'
        print *, '       nproc =', nproc, ' ny =', ny, ' npy =', npy
      end if
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    npxz = nproc/npy
    ipy = iproc/npxz
    ipxz = mod(iproc, npxz)
    call MPI_Comm_split(MPI_COMM_WORLD, ipy, ipxz, comm_xz, ierr)
    call MPI_Comm_split(MPI_COMM_WORLD, ipxz, ipy, comm_y, ierr)
    if (mod(nx + 1, npxz) /= 0 .or. mod(nzd, npxz) /= 0) then
      if (has_terminal) then
        print *, 'ERROR: the number of x-z pencils must divide both nx+1 and nzd.'
        print *, '       nx+1 =', nx + 1, ' nzd =', nzd, ' npxz =', npxz
      end if
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    nx0 = ipxz*(nx + 1)/npxz
    nxN = (ipxz + 1)*(nx + 1)/npxz - 1
    nxB = nxN - nx0 + 1
    nz0 = ipxz*nzd/npxz
    nzN = (ipxz + 1)*nzd/npxz - 1
    nzB = nzN - nz0 + 1
    nyB = ny/npy
    ny0 = ipy*nyB
    nyN = ny0 + nyB - 1
    has_average = (nx0 == 0)
    !$omp target update to(nx0, nxN, nxB, nz0, nzN, nzB, ny0, nyN, ny, ni, S)
    if (has_terminal) write (*, '(A,I5,A,I3,A,I3,A,I5,A,I5,A,I5)') '   ranks =', nproc, ' (', npxz, ' x-z pencils x', npy, &
      ' y slabs)   nxB =', nxB, '   nzB =', nzB, '   nyB =', nyB
#ifdef HAVE_CUDA
    compute_stream = transfer(target_stream(), compute_stream)
#endif
    if (npy > 1) then
      allocate (ghost_send(2, -nz:nz, nx0:nxN, 2), ghost_recv(2, -nz:nz, nx0:nxN, 2))
      ghost_send = 0; ghost_recv = 0
      !$omp target enter data map(to: ghost_send, ghost_recv)
    end if

    transpose_is_local = (nzB == nzd)
    sendcount = nxB*nzB*(nyN - ny0 + 5)            ! one field, ghost rows included
    !$omp target update to(sendcount)
    n = 1
    if (.not. transpose_is_local) n = int(npxz, C_SIZE_T)*int(sendcount, C_SIZE_T)
    allocate (sendbuf(n, 2), recvbuf(n, 2))
    sendbuf = 0; recvbuf = 0
    !$omp target enter data map(alloc: sendbuf, recvbuf)
    call setup_transport()
  end subroutine setup_decomposition

  subroutine free_mpi()
#ifdef HAVE_NCCL
    if (use_nccl) ierr = ncclCommDestroy(nccl_comm)
    if (use_nccl_y) ierr = ncclCommDestroy(nccl_comm_y)
#endif
#ifdef HAVE_CUDA
    if (.not. transpose_is_local) then
      ierr = cudaStreamDestroy(comm_stream)
      ierr = cudaEventDestroy(ev_packed(1)); ierr = cudaEventDestroy(ev_packed(2))
      ierr = cudaEventDestroy(ev_done(1)); ierr = cudaEventDestroy(ev_done(2))
    end if
#endif
    !$omp target exit data map(delete: sendbuf, recvbuf)
    deallocate (sendbuf, recvbuf)
    if (npy > 1) then
      !$omp target exit data map(delete: ghost_send, ghost_recv)
      deallocate (ghost_send, ghost_recv)
      ! the transfers alone (after the device has finished the pack), a
      ! part of the phases of hst_timer's table
      if (timing .and. has_terminal) write (*, '(A,F9.5,A,F9.5,A)') '     of which y exchange (transfers only): ghost rows', &
        t_ghost/max(istep, 1_C_SIZE_T), ' s/step, reduced systems (exposed)', t_gather/max(istep, 1_C_SIZE_T), ' s/step'
    end if
    call MPI_Comm_free(comm_xz, ierr)
    call MPI_Comm_free(comm_y, ierr)
  end subroutine free_mpi

  !------------------------------------------------------------------------
  ! The four ghost rows of a field with the layout of a component of V
  !------------------------------------------------------------------------
  ! The two rows above the slab are the two lowest rows of the slab above,
  ! the two below the two highest rows of the slab below, and across the
  ! box edge the images carry the shear-periodic phase of the displacements
  ! shift_x, shift_z of the upper image (hst_derivatives):
  !   f(ny) = f(0) ph,  f(ny+1) = f(1) ph,  f(-1) = f(ny-1) conjg(ph),  f(-2) = f(ny-2) conjg(ph).
  ! With one slab both neighbours are the rank itself and the exchange is
  ! this wrap, done in place.  With more, the two rows for each neighbour
  ! are packed (a row is strided in memory), exchanged over comm_y with
  ! NCCL on the compute stream (two send/receive pairs between the pack
  ! and the unpack, the host does not wait) or with MPI on the device
  ! buffers, and unpacked with the phase on the slabs at the box edge.
  subroutine exchange_ghost_rows(field, shift_x, shift_z)
    complex(C_DOUBLE_COMPLEX), intent(inout) :: field(ny0 - 2:, -nz:, nx0:)
    real(C_DOUBLE), intent(in) :: shift_x, shift_z
    integer(C_INT) :: ix, iz, k, up, down
    complex(C_DOUBLE_COMPLEX) :: ph, f_below, f_above
    real(C_DOUBLE) :: t0
#ifdef HAVE_NCCL
    integer(C_SIZE_T) :: nbytes
    integer :: r
#endif

    if (npy == 1) then
      !$omp target teams distribute parallel do collapse(2) default(none) &
      !$omp shared(field, nx0, nxN, nz, ny0, nyN, alfa0, beta0, shift_x, shift_z) private(ix, iz, ph)
      do ix = nx0, nxN
        do iz = -nz, nz
          ph = exp(dcmplx(0.0d0, -(alfa0*ix*shift_x + beta0*iz*shift_z)))
          field(nyN + 1, iz, ix) = field(ny0, iz, ix)*ph
          field(nyN + 2, iz, ix) = field(ny0 + 1, iz, ix)*ph
          field(ny0 - 1, iz, ix) = field(nyN, iz, ix)*conjg(ph)
          field(ny0 - 2, iz, ix) = field(nyN - 1, iz, ix)*conjg(ph)
        end do
      end do
      return
    end if
    ! pack: (.., 1) the two lowest rows, for the slab below; (.., 2) the two highest, for the slab above
    !$omp target teams distribute parallel do collapse(3) default(none) &
    !$omp shared(field, ghost_send, nx0, nxN, nz, ny0, nyN) private(ix, iz, k)
    do ix = nx0, nxN
      do iz = -nz, nz
        do k = 1, 2
          ghost_send(k, iz, ix, 1) = field(ny0 + k - 1, iz, ix)
          ghost_send(k, iz, ix, 2) = field(nyN + k - 2, iz, ix)
        end do
      end do
    end do
    up = mod(ipy + 1, npy)
    down = mod(ipy - 1 + npy, npy)
#ifdef HAVE_CUDA
    !$omp target data use_device_addr(ghost_send, ghost_recv)
#endif
    if (use_nccl_y) then
#ifdef HAVE_NCCL
      ! the two rows for the slab above and those for the slab below as
      ! two groups (with two slabs both neighbours are the same rank)
      nbytes = 16_C_SIZE_T*int(size(ghost_send, 1)*size(ghost_send, 2)*size(ghost_send, 3), C_SIZE_T)
      if (timing) then
        r = cudaStreamSynchronize(compute_stream); t0 = MPI_Wtime()
      end if
      r = ncclGroupStart()
      r = ncclSend(c_loc(ghost_send(1, -nz, nx0, 2)), nbytes, NCCL_UINT8, up, nccl_comm_y, nccl_ystream)
      r = ncclRecv(c_loc(ghost_recv(1, -nz, nx0, 1)), nbytes, NCCL_UINT8, down, nccl_comm_y, nccl_ystream)
      r = ncclGroupEnd()
      if (r /= 0) error stop 'NCCL ghost row exchange failed'
      r = ncclGroupStart()
      r = ncclSend(c_loc(ghost_send(1, -nz, nx0, 1)), nbytes, NCCL_UINT8, down, nccl_comm_y, nccl_ystream)
      r = ncclRecv(c_loc(ghost_recv(1, -nz, nx0, 2)), nbytes, NCCL_UINT8, up, nccl_comm_y, nccl_ystream)
      r = ncclGroupEnd()
      if (r /= 0) error stop 'NCCL ghost row exchange failed'
      if (timing) then
        r = cudaStreamSynchronize(compute_stream); t_ghost = t_ghost + MPI_Wtime() - t0
      end if
#endif
    else
#ifdef HAVE_CUDA
      ierr = cudaStreamSynchronize(compute_stream)
#endif
      t0 = MPI_Wtime()
      call MPI_Sendrecv(ghost_send(:, :, :, 2), size(ghost_send, 1)*size(ghost_send, 2)*size(ghost_send, 3), &
                        MPI_DOUBLE_COMPLEX, up, 1, ghost_recv(:, :, :, 1), size(ghost_recv, 1)*size(ghost_recv, 2)*size(ghost_recv, 3), &
                        MPI_DOUBLE_COMPLEX, down, 1, comm_y, MPI_STATUS_IGNORE, ierr)
      call MPI_Sendrecv(ghost_send(:, :, :, 1), size(ghost_send, 1)*size(ghost_send, 2)*size(ghost_send, 3), &
                        MPI_DOUBLE_COMPLEX, down, 2, ghost_recv(:, :, :, 2), size(ghost_recv, 1)*size(ghost_recv, 2)*size(ghost_recv, 3), &
                        MPI_DOUBLE_COMPLEX, up, 2, comm_y, MPI_STATUS_IGNORE, ierr)
      if (timing) t_ghost = t_ghost + MPI_Wtime() - t0
    end if
#ifdef HAVE_CUDA
    !$omp end target data
#endif
    ! unpack: (.., 1) came from below (rows ny0-2, ny0-1), (.., 2) from above (rows nyN+1, nyN+2)
    !$omp target teams distribute parallel do collapse(3) default(none) &
    !$omp shared(field, ghost_recv, nx0, nxN, nz, ny0, nyN, ipy, npy, alfa0, beta0, shift_x, shift_z) &
    !$omp private(ix, iz, k, ph, f_below, f_above)
    do ix = nx0, nxN
      do iz = -nz, nz
        do k = 1, 2
          ph = exp(dcmplx(0.0d0, -(alfa0*ix*shift_x + beta0*iz*shift_z)))
          f_below = 1.0d0; if (ipy == 0) f_below = conjg(ph)
          f_above = 1.0d0; if (ipy == npy - 1) f_above = ph
          field(ny0 - 3 + k, iz, ix) = ghost_recv(k, iz, ix, 1)*f_below
          field(nyN + k, iz, ix) = ghost_recv(k, iz, ix, 2)*f_above
        end do
      end do
    end do
  end subroutine exchange_ghost_rows

  ! The allgather over the y column of the line solver's records
  ! (hst_linsolve): a(:, first:first+count-1, s+1) is slab s's block of one
  ! workspace w; block ipy holds this rank's records, the others are filled
  ! with the other slabs' (one send and one receive per other slab: the
  ! blocks of a workspace are not contiguous over the slabs, so NCCL's
  ! allgather does not apply).  NCCL: on the communication stream, which
  ! waits for the compute stream through ev_rec(w); allgather_y_wait makes
  ! the compute stream wait for the transfer through ev_gath(w), so that
  ! the forward sweep of the other workspace runs meanwhile and the host
  ! never blocks.  MPI: on the device buffer, the host waits.
  subroutine allgather_y_start(a, first, count, w)
    real(C_DOUBLE), intent(inout), contiguous, target :: a(:, :, :)
    integer(C_INT), intent(in) :: first, count, w
    integer :: s, peer
    real(C_DOUBLE) :: t0
#ifdef HAVE_NCCL
    integer(C_SIZE_T) :: nbytes
    integer :: r
#endif
    if (npy == 1) return
#ifdef HAVE_CUDA
    !$omp target data use_device_addr(a)
#endif
    if (use_nccl_y) then
#ifdef HAVE_NCCL
      nbytes = 8_C_SIZE_T*int(size(a, 1), C_SIZE_T)*int(count, C_SIZE_T)
      r = cudaEventRecord(ev_rec(w), compute_stream)
      r = cudaStreamWaitEvent(comm_stream, ev_rec(w), 0)
      r = ncclGroupStart()
      do s = 1, npy - 1
        peer = mod(ipy + s, npy)
        r = ncclSend(c_loc(a(1, first, ipy + 1)), nbytes, NCCL_UINT8, peer, nccl_comm_y, nccl_stream)
        r = ncclRecv(c_loc(a(1, first, peer + 1)), nbytes, NCCL_UINT8, peer, nccl_comm_y, nccl_stream)
      end do
      r = ncclGroupEnd()
      if (r /= 0) error stop 'NCCL allgather over the y column failed'
      r = cudaEventRecord(ev_gath(w), comm_stream)
#endif
    else
#ifdef HAVE_CUDA
      ierr = cudaStreamSynchronize(compute_stream)
#endif
      t0 = MPI_Wtime()
      do s = 1, npy - 1                       ! shift s: send to slab ipy + s, receive slab ipy - s's block
        peer = mod(ipy - s + npy, npy)
        call MPI_Sendrecv(a(:, first:first + count - 1, ipy + 1), size(a, 1)*count, MPI_DOUBLE_PRECISION, mod(ipy + s, npy), 3, &
                          a(:, first:first + count - 1, peer + 1), size(a, 1)*count, MPI_DOUBLE_PRECISION, peer, 3, &
                          comm_y, MPI_STATUS_IGNORE, ierr)
      end do
      if (timing) t_gather = t_gather + MPI_Wtime() - t0
    end if
#ifdef HAVE_CUDA
    !$omp end target data
#endif
  end subroutine allgather_y_start

  ! The compute stream waits for the gather of workspace w (with timing the
  ! host measures the part not hidden behind the compute stream's work).
  subroutine allgather_y_wait(w)
    integer(C_INT), intent(in) :: w
#ifdef HAVE_NCCL
    integer :: r
    real(C_DOUBLE) :: t0
    if (npy == 1 .or. .not. use_nccl_y) return
    if (timing) then
      r = cudaStreamSynchronize(compute_stream); t0 = MPI_Wtime()
      r = cudaEventSynchronize(ev_gath(w)); t_gather = t_gather + MPI_Wtime() - t0
    end if
    r = cudaStreamWaitEvent(compute_stream, ev_gath(w), 0)
#endif
  end subroutine allgather_y_wait

  ! transport = 'nccl' needs a build with NCCL=1 and one GPU per rank;
  ! 'auto' takes NCCL when both hold and MPI otherwise.  One rank needs no
  ! transport at all.  On the GPU the alltoall runs on its own stream
  ! (NCCL) or is started by the host once the compute stream has packed
  ! (MPI); the events order the two streams.  The y exchanges (ghost rows,
  ! reduced systems) go through a second NCCL communicator over the y
  ! column, the ghost rows on the compute stream itself, the records on
  ! the communication stream (allgather_y_start), or through MPI on the
  ! device buffers.
  subroutine setup_transport()
#ifdef HAVE_NCCL
    type(nccl_unique_id) :: id
    integer :: r
#endif
    character(len=4) :: xz_name, y_name
    use_nccl = .false.
    use_nccl_y = .false.
    if (transpose_is_local .and. npy == 1) return
#ifdef HAVE_CUDA
    ierr = cudaStreamCreateWithFlags(comm_stream, cudaStreamNonBlocking)
    ierr = cudaEventCreateWithFlags(ev_packed(1), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_packed(2), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_done(1), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_done(2), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_rec(1), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_rec(2), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_gath(1), cudaEventDisableTiming)
    ierr = cudaEventCreateWithFlags(ev_gath(2), cudaEventDisableTiming)
#endif
    if (transport /= 'mpi') then
#ifdef HAVE_NCCL
      if (node_ranks <= omp_get_num_devices()) then
        r = cudaSetDevice(omp_get_default_device())
        if (.not. transpose_is_local) then
          ! one NCCL communicator per slab, its id made by the slab's first rank
          if (ipxz == 0) r = ncclGetUniqueId(id)
          call MPI_Bcast(id%bytes, 128, MPI_BYTE, 0, comm_xz, ierr)
          r = ncclCommInitRank(nccl_comm, npxz, id, ipxz)
          if (r /= 0) then
            if (has_terminal) print *, 'ERROR: ncclCommInitRank failed with code', r
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
          end if
          nccl_stream = transfer(comm_stream, nccl_stream)
          use_nccl = .true.
        end if
        if (npy > 1) then
          ! and one per y column, its id made by the column's first slab
          if (ipy == 0) r = ncclGetUniqueId(id)
          call MPI_Bcast(id%bytes, 128, MPI_BYTE, 0, comm_y, ierr)
          r = ncclCommInitRank(nccl_comm_y, npy, id, ipy)
          if (r /= 0) then
            if (has_terminal) print *, 'ERROR: ncclCommInitRank (y column) failed with code', r
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
          end if
          nccl_ystream = transfer(compute_stream, nccl_ystream)
          nccl_stream = transfer(comm_stream, nccl_stream)
          use_nccl_y = .true.
        end if
      else if (transport == 'nccl') then
        if (has_terminal) print *, 'ERROR: transport = nccl needs one GPU per rank'
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
#else
      if (transport == 'nccl') then
        if (has_terminal) print *, 'ERROR: transport = nccl needs a build with NCCL=1'
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
#endif
    end if
    if (has_terminal) then
      xz_name = merge('nccl', 'mpi ', use_nccl)
      if (transpose_is_local) xz_name = 'none'
      y_name = merge('nccl', 'mpi ', use_nccl_y)
      if (npy > 1) then
        write (*, '(A,A,A,A)') '   alltoall transport: ', xz_name, ',  y exchange: ', y_name
      else
        write (*, '(A,A)') '   alltoall transport: ', xz_name
      end if
    end if
  end subroutine setup_transport

  !------------------------------------------------------------------------
  ! z-pencil Vz(iz, ix, iy)  <->  x-pencil Vx(ix, iz, iy), one field
  !------------------------------------------------------------------------
  ! The send buffer is a copy: block dest holds Vz(dest*nzB + iz, ix, iy)
  ! (or Vx(dest*nxB + ix, iz, iy)) with the leading index still leading,
  ! so both sides of the pack run contiguously and it moves at the memory
  ! bandwidth.  The receive buffer holds the same blocks from every source,
  ! and taking them apart into the other pencil layout is where the leading
  ! index changes: that is the tiled transpose below, which also serves the
  ! one-rank case, where the two layouts are converted in place of the
  ! alltoall.

  subroutine pack_zTOx(Vz, send)
    complex(C_DOUBLE_COMPLEX), intent(in) :: Vz(:, :, :)
    complex(C_DOUBLE_COMPLEX), intent(out) :: send(:)
    integer(C_SIZE_T) :: iy, ix, iz, dest, p
    integer(C_INT) :: ny_batch
    ny_batch = size(Vz, 3)
    !$omp target teams distribute parallel do collapse(4) default(none) &
    !$omp shared(Vz, send, ny_batch, nxB, nzB, npxz, sendcount) private(iy, ix, iz, dest, p)
    do dest = 0, npxz - 1
      do iy = 1, ny_batch
        do ix = 1, nxB
          do iz = 1, nzB
            p = dest*sendcount + iz + nzB*(ix - 1) + nzB*nxB*(iy - 1)
            send(p) = Vz(dest*nzB + iz, ix, iy)
          end do
        end do
      end do
    end do
  end subroutine pack_zTOx

  subroutine pack_xTOz(Vx, send)
    complex(C_DOUBLE_COMPLEX), intent(in) :: Vx(:, :, :)
    complex(C_DOUBLE_COMPLEX), intent(out) :: send(:)
    integer(C_SIZE_T) :: iy, ix, iz, dest, p
    integer(C_INT) :: ny_batch
    ny_batch = size(Vx, 3)
    !$omp target teams distribute parallel do collapse(4) default(none) &
    !$omp shared(Vx, send, ny_batch, nxB, nzB, npxz, sendcount) private(iy, ix, iz, dest, p)
    do dest = 0, npxz - 1
      do iy = 1, ny_batch
        do iz = 1, nzB
          do ix = 1, nxB
            p = dest*sendcount + ix + nxB*(iz - 1) + nxB*nzB*(iy - 1)
            send(p) = Vx(dest*nxB + ix, iz, iy)
          end do
        end do
      end do
    end do
  end subroutine pack_xTOz

  ! B(jb*(block - 1) + j, i, plane) = A(i, j, plane, block): the leading
  ! index of A (i, contiguous) becomes the second index of B, for every
  ! plane (one y row of one field) and, for the receive buffer, every block
  ! (one source rank, whose modes start at jb*(block - 1) in B).  On the GPU
  ! each thread block moves one TILE x TILE tile through shared memory,
  ! reading A along i and writing B along j, so both sides are coalesced; a
  ! plain loop leaves one side strided by a whole line and runs at 40% of
  ! the bandwidth.  This is the one CUDA Fortran kernel of the code: the
  ! OpenMP forms tried (teams distribute + parallel do, teams loop + loop)
  ! either leave the tile in global memory or generate slow inner loops,
  ! and both lose to the plain loop (FINDINGS.md).  The kernel runs on the
  ! OpenMP target stream, in order with the transforms and the alltoall.
  subroutine transpose_tiled(A, lda, n1, n2, nplanes, nblocks, B, ldb, jb)
    integer(C_INT), intent(in) :: lda, n1, n2, nplanes, nblocks, ldb, jb
    complex(C_DOUBLE_COMPLEX), intent(in), target :: A(lda, n2, nplanes, nblocks)
    complex(C_DOUBLE_COMPLEX), intent(inout), target :: B(ldb, n1, nplanes)
#ifdef HAVE_CUDA
    complex(C_DOUBLE_COMPLEX), device, pointer :: dA(:, :, :, :), dB(:, :, :)
    type(c_devptr) :: pA, pB
    integer(kind=cuda_stream_kind) :: stream
    !$omp target data use_device_addr(A, B)
    pA = transfer(c_loc(A), pA); pB = transfer(c_loc(B), pB)
    !$omp end target data
    call c_f_pointer(pA, dA, [lda, n2, nplanes, nblocks])
    call c_f_pointer(pB, dB, [ldb, n1, nplanes])
    stream = transfer(target_stream(), stream)
    call transpose_tile_kernel<<<dim3((n1 + TILE - 1)/TILE, (n2 + TILE - 1)/TILE, nplanes*nblocks), &
                                 dim3(TILE, TILE/ROWS_PER_THREAD, 1), 0, stream>>> (dA, dB, n1, n2, nplanes, jb)
#else
    integer(C_INT) :: block, plane, i, j
    do block = 1, nblocks
      do plane = 1, nplanes
        do j = 1, n2
          do i = 1, n1
            B(jb*(block - 1) + j, i, plane) = A(i, j, plane, block)
          end do
        end do
      end do
    end do
#endif
  end subroutine transpose_tiled

#ifdef HAVE_CUDA
  ! One thread block per tile: blockIdx (i tile, j tile, plane and block),
  ! TILE x TILE/ROWS_PER_THREAD threads, each doing ROWS_PER_THREAD rows.
  ! The tile is padded by one so that the transposed read has no bank
  ! conflicts.
  attributes(global) subroutine transpose_tile_kernel(A, B, n1, n2, nplanes, jb)
    integer(C_INT), value :: n1, n2, nplanes, jb
    complex(C_DOUBLE_COMPLEX), device, intent(in) :: A(:, :, :, :)
    complex(C_DOUBLE_COMPLEX), device, intent(inout) :: B(:, :, :)
    complex(C_DOUBLE_COMPLEX), shared :: t(TILE + 1, TILE)   ! (Fortran: not "tile", that is TILE)
    integer(C_INT) :: i0, j0, plane, block, tx, ty, k
    i0 = (blockIdx%x - 1)*TILE
    j0 = (blockIdx%y - 1)*TILE
    plane = mod(blockIdx%z - 1, nplanes) + 1
    block = (blockIdx%z - 1)/nplanes + 1
    tx = threadIdx%x
    ty = threadIdx%y
    do k = ty, TILE, TILE/ROWS_PER_THREAD
      if (i0 + tx <= n1 .and. j0 + k <= n2) t(tx, k) = A(i0 + tx, j0 + k, plane, block)
    end do
    call syncthreads()
    do k = ty, TILE, TILE/ROWS_PER_THREAD
      if (j0 + tx <= n2 .and. i0 + k <= n1) B(jb*(block - 1) + j0 + tx, i0 + k, plane) = t(k, tx)
    end do
  end subroutine transpose_tile_kernel
#endif

  ! The collective of the transposes, over the ranks of the slab, started
  ! on buffer pair b after the pack and waited for before the unpack.  On the GPU the buffers stay on
  ! the device: the use_device_addr block hands MPI (CUDA-aware) or NCCL
  ! their device addresses.  NCCL: one send and one receive per peer in a
  ! group (NCCL has no alltoall) on the communication stream, which waits
  ! for the pack through ev_packed, and the compute stream waits for the
  ! transfer through ev_done, so the host never blocks.  MPI: the host
  ! waits for the pack (and everything before it on the compute stream)
  ! and posts a non-blocking alltoall.
  subroutine alltoall_start(b)
    integer, intent(in) :: b
#ifdef HAVE_NCCL
    integer(C_SIZE_T) :: nbytes
    integer :: peer, r
#endif
#ifdef HAVE_CUDA
    !$omp target data use_device_addr(sendbuf, recvbuf)
#endif
    if (use_nccl) then
#ifdef HAVE_NCCL
      r = cudaEventRecord(ev_packed(b), compute_stream)
      r = cudaStreamWaitEvent(comm_stream, ev_packed(b), 0)
      nbytes = 16_C_SIZE_T*int(sendcount, C_SIZE_T)
      r = ncclGroupStart()
      do peer = 0, npxz - 1
        r = ncclSend(c_loc(sendbuf(peer*sendcount + 1, b)), nbytes, NCCL_UINT8, peer, nccl_comm, nccl_stream)
        if (r /= 0) error stop 'ncclSend failed'
        r = ncclRecv(c_loc(recvbuf(peer*sendcount + 1, b)), nbytes, NCCL_UINT8, peer, nccl_comm, nccl_stream)
        if (r /= 0) error stop 'ncclRecv failed'
      end do
      r = ncclGroupEnd()
      if (r /= 0) error stop 'ncclGroupEnd failed'
      r = cudaEventRecord(ev_done(b), comm_stream)
#endif
    else
#ifdef HAVE_CUDA
      ierr = cudaStreamSynchronize(compute_stream)
#endif
      call MPI_Ialltoall(sendbuf(:, b), int(sendcount), MPI_DOUBLE_COMPLEX, &
                         recvbuf(:, b), int(sendcount), MPI_DOUBLE_COMPLEX, comm_xz, req(b), ierr)
      if (ierr /= MPI_SUCCESS) error stop 'MPI_Ialltoall failed'
    end if
#ifdef HAVE_CUDA
    !$omp end target data
#endif
  end subroutine alltoall_start

  subroutine alltoall_wait(b)
    integer, intent(in) :: b
    if (use_nccl) then
#ifdef HAVE_CUDA
      ierr = cudaStreamWaitEvent(compute_stream, ev_done(b), 0)
#endif
    else
      call MPI_Wait(req(b), MPI_STATUS_IGNORE, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'MPI_Wait failed'
    end if
  end subroutine alltoall_wait

  ! Field m of a batch uses buffer pair 1, 2, 1, ...: start(m) may only
  ! follow finish(m-2).
  integer function pair(m)
    integer(C_INT), intent(in) :: m
    pair = mod(m - 1, 2) + 1
  end function pair

  ! The layouts of the transpose calls (block = source rank + 1):
  !   one rank    Vx(ix, iz, iy)             = Vz(iz, ix, iy)
  !   many ranks  Vx(src*nxB + ix, iz, iy)   = recv(iz, ix, iy, src)
  ! On one rank the two layouts are converted directly in finish.
  subroutine transpose_zTOx_start(Vz, m)
    complex(C_DOUBLE_COMPLEX), intent(in), contiguous :: Vz(:, :, :)
    integer(C_INT), intent(in) :: m
    if (transpose_is_local) return
    if (size(Vz, 3)*nxB*nzB /= sendcount) error stop 'transpose_zTOx: whole fields only'
    call pack_zTOx(Vz, sendbuf(:, pair(m)))
    call alltoall_start(pair(m))
  end subroutine transpose_zTOx_start

  subroutine transpose_zTOx_finish(Vz, Vx, m)
    complex(C_DOUBLE_COMPLEX), intent(in), contiguous :: Vz(:, :, :)
    complex(C_DOUBLE_COMPLEX), intent(out), contiguous :: Vx(:, :, :)
    integer(C_INT), intent(in) :: m
    if (transpose_is_local) then
      call transpose_tiled(Vz, nzd, nzd, nxB, size(Vz, 3), 1, Vx, size(Vx, 1), 0)
    else
      call alltoall_wait(pair(m))
      call transpose_tiled(recvbuf(:, pair(m)), nzB, nzB, nxB, size(Vz, 3), npxz, Vx, size(Vx, 1), nxB)
    end if
  end subroutine transpose_zTOx_finish

  !   one rank    Vz(iz, ix, iy)             = Vx(ix, iz, iy)
  !   many ranks  Vz(src*nzB + iz, ix, iy)   = recv(ix, iz, iy, src)
  subroutine transpose_xTOz_start(Vx, m)
    complex(C_DOUBLE_COMPLEX), intent(in), contiguous :: Vx(:, :, :)
    integer(C_INT), intent(in) :: m
    if (transpose_is_local) return
    if (size(Vx, 3)*nxB*nzB /= sendcount) error stop 'transpose_xTOz: whole fields only'
    call pack_xTOz(Vx, sendbuf(:, pair(m)))
    call alltoall_start(pair(m))
  end subroutine transpose_xTOz_start

  subroutine transpose_xTOz_finish(Vx, Vz, m)
    complex(C_DOUBLE_COMPLEX), intent(in), contiguous :: Vx(:, :, :)
    complex(C_DOUBLE_COMPLEX), intent(out), contiguous :: Vz(:, :, :)
    integer(C_INT), intent(in) :: m
    if (transpose_is_local) then
      call transpose_tiled(Vx, size(Vx, 1), nxB, nzd, size(Vx, 3), 1, Vz, nzd, 0)
    else
      call alltoall_wait(pair(m))
      call transpose_tiled(recvbuf(:, pair(m)), nxB, nxB, nzB, size(Vx, 3), npxz, Vz, nzd, nzB)
    end if
  end subroutine transpose_xTOz_finish

end module hst_mpi
