MPISchurComplements
===================

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
see below). `dcomm` is an MPI communicator including all the ranks that
participate in this solve.

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
vectors at 'overlappping' points are different.

### Shared-memory MPI parallelism

### Optional arguments
