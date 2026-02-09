MPISchurComplements
===================

See also https://johnomotani.github.io/MPISchurComplements.jl/.

Solves the matrix system
```math
\left(\begin{array}{cc}
A & B\\
C & D
\end{array}\right)
\cdot\left(\begin{array}{c}
x\\
y
\end{array}\right)
=\left(\begin{array}{c}
u\\
v
\end{array}\right)
```
using a Schur complement factorization. This is useful when the block structure
of the matrix can be arranged so that $A$ is significantly simpler to solve
than the full matrix.

The package is parallelised using shared- and distributed-memory MPI, details
below. It does not provide a solver for the matrix $A$ - this must be supplied
at setup.

Schur complement factorization
------------------------------

Use the first row $A.x+B.y=u$ to eliminate $x$ from the second row $C.x+D.y=v$:
```math
(D - C.A^{-1}\cdot B).y = v - C.A^{-1}.u.
```
$S=(D - C.A^{-1}\cdot B)$ is the 'Schur complement' of $A$. It is a dense
matrix, which is computed when the factorization is constructed or updated.
Computing the Schur complement matrix, and computing the right-hand-side each
time the matrix system is solved both require the application of the action of
$A^{-1}$, but otherwise only matrix multiplication and addition/subtraction.

To complete the solve, we also need the action of $S^{-1}$. An LU factorization
of $S$ is computed, and used to apply $S^{-1}$. This allows $y$ to be computed
```math
y = S^{-1}.(v - C.A^{-1}.u),
```
which can then be substituted back into the first row to find $x$
```math
x = A^{-1}.u - A^{-1}.B.y.
```
$A^{-1}.B$ is a dense matrix. It must be computed as an intermediate step when
constructing $S$, and so can be stored and multiplied as a dense matrix (this
is the default). Optionally, when $B$ is sparse and the solver that applies
$A^{-1}$ is very fast, it might be more optimal to first calculate $B.y$, and
then use the $A^{-1}$ solver for a second time (the first was to calculate
$A^{-1}.u$).

Overall there are two 'matrix solves' (to calculate $A^{-1}.u$, and the
application of $S^{-1}$), with optionally a second application of $A^{-1}$ as
just described. The other operations are matrix multiplications and vector
addition/subtraction. Matrix multiplication and vector arithmetic operations
are fully parallelised. The $A^{-1}$ operation is external to this package, but
is assumed to be well parallelised. The $S^{-1}$ operation may be done in
serial using the LU solver from `LinearAlgebra`, which is likely to be optimal
for small $S$ (e.g. roughly smaller than 1000x1000). For larger $S$, a
parallelised LU solver is provided in the `DenseLUs` submodule, which might
give some speedup.

Usage
-----

The `MPISchurComplement` matrix factorization struct is created by calling
```julia
mpi_schur_complement(A_factorization, B, C, D, owned_top_vector_entries,
                     owned_bottom_vector_entries; distributed_comm=dcomm)
```
as a first example using just distributed-MPI parallelism (for shared-memory
see below). Other optional arguments are described in the docstring for
`mpi_schur_complement`. `dcomm` is an MPI communicator including all the ranks
that participate in this solve.

The matrix entries, solution vectors and right-hand-side vectors are assumed to
be domain-decomposed so that different MPI ranks own different parts of the
matrices/vectors. However, matrix/vector entries may be duplicated either on
different ranks (which may be useful for halo regions that join different
subdomains) or even on the same rank (which might be useful for enforcing
periodicity).

`owned_top_vector_entries` gives the 'global' indices of the locally-owned
entries of $u$, and `owned_bottom_vector_entries` gives the 'global' indices of
the locally-owned entries of $v$. `owned_top_vector_entries` and
`owned_bottom_vector_entries` can contain 'overlapping' indices, that are
repeated on the same or different MPI ranks. The complete set of indices in
`owned_top_vector_entries` or `owned_bottom_vector_entries` across all MPI
ranks does not have to be consecutive. For example, this might be useful if the
entries in $y$ are some subset of grid points that divides a full grid into
disconnected segments - in that case the global indices of the full grid can be
used directly for both $x$ and $y$. Internally, the indices are converted to
values that increase consecutively from 1, but have the same order as those
passed in `owned_top_vector_entries` and `owned_bottom_vector_entries`.

`A_factorization` is an object that applies the action of $A^{-1}$ when called
as
```julia
ldiv!(x, A_factorization, u)
```
where `x` and `u` are the values at the positions given by
`owned_top_vector_entries`.

By default `B`, `C`, and `D` are assumed to be given at the same points as
`owned_top_vector_entries` and `owned_bottom_vector_entries`: the rows of `B`
and the columns of `C` are given by `owned_top_vector_entries`; the columns of
`B`, the rows of `C` and both rows and columns of `D` are given by
`owned_bottom_vector_entries`. Optionally, different ranges can be given for
the rows of `C` or the columns of `B` and `D`.
[Developer note: it was technically more convenient to allow different rows for
`C`, but logically it would make more sense to allow different columns for `C`,
the same as `B` and `D`, as this would allow describing discretisation stencils
that use points from different subdomains. This could probably be supported if
it was needed?]

The requirements for the input matrices and the right-hand-side and solution
vectors at 'overlapping' points are different. The right-hand-side vector
should have identical entries for overlapping points on all the processes that
own that point; similarly the solution vector will have identical values of
everlapping points on all processes that own those points. In contrast, the
distributed input matrices should be constructed so that for each overlapping
point the sum of the values gathered from all processes that own that point
gives the value in the assembled, global matrix.

### Shared-memory MPI parallelism

Shared-memory MPI parallism is supported, in addition to distributed memory
parallelism. Shared-memory parallelism can increase efficiency as data does not
need to be copied between processes that belong to a block of processes that
share memory. It requires handling of `MPI.Win` objects that are associated to
the shared-memory arrays - see the ['Complete example'](#Complete-example)
section below.

When shared-memory MPI parallelism is used, the matrices `B`, `C` and `D`, and
the right-hand-side `b` and solution `x` vectors should all be shared memory
arrays.  In addition, functions `allocate_shared_float` and
`allocate_shared_int` must be passed to the `mpi_schur_complement()`
constructor. These will be used to allocate some external buffer arrays, and
must be provided because the caller must handle freeing the `MPI.Win` objects
associated with the shared-memory arrays, because `MPI.free(win::MPI.Win)` is a
collective operation, and so cannot be handled by the garbage collector.

### Complete example

An example using both distributed- and shared-memory parallelism. Note that
`FakeMPILU` is a place-holder for some efficiently parallelised solver for
$A^{-1}$.
```julia
using MPISchurComplements
using MPISchurComplements: FakeMPILU
using MPI

n_local_top = 9
n_local_bottom = 3

if !MPI.Initialized()
    MPI.Init()
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

if nproc > 1 && nproc % 2 != 0
    error("`nproc` not divisible by 2")
end

shared_nproc = min(2, nproc)

distributed_rank = rank ÷ shared_nproc
distributed_nproc = nproc ÷ shared_nproc
shared_comm = MPI.Comm_split(comm, distributed_rank, 0)
shared_comm_rank = MPI.Comm_rank(shared_comm)
distributed_comm = MPI.Comm_split(comm, shared_comm_rank == 0 ? 0 : nothing, 0)

local_win_store_float = MPI.Win[]
allocate_shared_float = (dims...)->begin
    if shared_comm_rank == 0
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end
    win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local,
                                              shared_comm)
    array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
    push!(local_win_store_float, win)
    if shared_comm_rank == 0
        array .= NaN
    end
    MPI.Barrier(shared_comm)
    return array
end

local_win_store_int = MPI.Win[]
allocate_shared_int = (dims...)->begin
    if shared_comm_rank == 0
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end
    win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local,
                                              shared_comm)
    array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
    push!(local_win_store_int, win)
    if shared_comm_rank == 0
        array .= typemin(Int64)
    end
    MPI.Barrier(shared_comm)
    return array
end

A_local = allocate_shared_float(n_local_top, n_local_top)
B_local = allocate_shared_float(n_local_top, n_local_bottom)
C_local = allocate_shared_float(n_local_bottom, n_local_top)
D_local = allocate_shared_float(n_local_bottom, n_local_bottom)
u_local = allocate_shared_float(n_local_top)
v_local = allocate_shared_float(n_local_bottom)
x_local = allocate_shared_float(n_local_top)
y_local = allocate_shared_float(n_local_bottom)

if shared_comm_rank == 0
    A_local .= rand(n_local_top, n_local_top)
    B_local .= rand(n_local_top, n_local_bottom)
    C_local .= rand(n_local_bottom, n_local_top)
    D_local .= rand(n_local_bottom, n_local_bottom)
    u_local .= rand(n_local_top)
    v_local .= rand(n_local_bottom)

    # Ensure overlapping points in u and v are identical on different processes.
    if distributed_rank > 0
        req_left1 = MPI.Isend(@view(u_local[1:1]), distributed_comm; dest=distributed_rank-1)
        req_left2 = MPI.Isend(@view(v_local[1:1]), distributed_comm; dest=distributed_rank-1)
    else
        req_left1 = MPI.REQUEST_NULL
        req_left2 = MPI.REQUEST_NULL
    end
    if distributed_rank < distributed_nproc - 1
        req_right1 = MPI.Irecv!(@view(u_local[end:end]), distributed_comm; source=distributed_rank+1)
        req_right2 = MPI.Irecv!(@view(v_local[end:end]), distributed_comm; source=distributed_rank+1)
    else
        req_right1 = MPI.REQUEST_NULL
        req_right2 = MPI.REQUEST_NULL
    end
    MPI.Waitall([req_left1, req_left2, req_right1, req_right2])
end
MPI.Barrier(shared_comm)

owned_top_vector_entries = distributed_rank*(n_local_top-1)+1:(distributed_rank+1)*(n_local_top-1)+1
owned_bottom_vector_entries = distributed_rank*(n_local_bottom-1)+1:(distributed_rank+1)*(n_local_bottom-1)+1

A_factorization = FakeMPILU(A_local, owned_top_vector_entries,
                            owned_top_vector_entries; comm=distributed_comm,
                            shared_comm)

Alu = mpi_schur_complement(A_factorization, B_local, C_local, D_local,
                           owned_top_vector_entries,
                           owned_bottom_vector_entries; comm, shared_comm,
                           distributed_comm, allocate_shared_float,
                           allocate_shared_int)

ldiv!(x_local, y_local, Alu, u_local, v_local)

# Clean up MPI.Win objects from shared arrays
if local_win_store_float !== nothing
    # Free the MPI.Win objects, because if they are free'd by the garbage collector
    # it may cause an MPI error or hang.
    for w ∈ local_win_store_float
        MPI.free(w)
    end
    resize!(local_win_store_float, 0)
end
if local_win_store_int !== nothing
    # Free the MPI.Win objects, because if they are free'd by the garbage collector
    # it may cause an MPI error or hang.
    for w ∈ local_win_store_int
        MPI.free(w)
    end
    resize!(local_win_store_int, 0)
end
```
