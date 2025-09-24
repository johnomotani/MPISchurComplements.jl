"""
Use the Schur complement technique to solve a block-2x2 matrix system, parallelised with
MPI.

Solve the matrix system
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
\\cdot\\left(\\begin{array}{c}
x\\\\
y
\\end{array}\\right)
=\\left(\\begin{array}{c}
u\\\\
v
\\end{array}\\right)
```
"""
module MPISchurComplements

export MPISchurComplement, mpi_schur_complement, update_schur_complement!, ldiv!

using LinearAlgebra
import LinearAlgebra: ldiv!
using MPI

mutable struct MPISchurComplement{TA,TB,TC,TSC,TSCF,TAiu,Ttv,Tbv,Tgy,Tscomm,Tsync}
    A_factorization::TA
    Ainv_dot_B::TB
    B_global_column_range::UnitRange{Int64}
    C::TC
    C_global_row_range::UnitRange{Int64}
    C_local_row_range::UnitRange{Int64}
    D_global_column_range::UnitRange{Int64}
    D_local_column_range::UnitRange{Int64}
    schur_complement::TSC
    schur_complement_factorization::TSCF
    schur_complement_local_range::UnitRange{Int64}
    Ainv_dot_u::TAiu
    top_vec_buffer::Ttv
    top_vec_local_size::Int64
    bottom_vec_buffer::Tbv
    bottom_vec_local_size::Int64
    global_y::Tgy
    top_vec_global_size::Int64
    bottom_vec_global_size::Int64
    owned_top_vector_entries::UnitRange{Int64}
    global_top_vector_range::UnitRange{Int64}
    local_top_vector_range::UnitRange{Int64}
    owned_bottom_vector_entries::UnitRange{Int64}
    global_bottom_vector_range::UnitRange{Int64}
    local_bottom_vector_range::UnitRange{Int64}
    distributed_comm::MPI.Comm
    distributed_rank::Int64
    shared_comm::Tscomm
    shared_rank::Int64
    synchronize_shared::Tsync
end

"""
    mpi_schur_complement(A_factorization, B::AbstractMatrix,
                         B_global_column_range::UnitRange{Int64}, C::AbstractMatrix,
                         C_global_row_range::UnitRange{Int64}, D::AbstractMatrix,
                         D_global_column_range::UnitRange{Int64},
                         owned_top_vector_entries::UnitRange{Int64},
                         owned_bottom_vector_entries::UnitRange{Int64};
                         distributed_comm::MPI.Comm=MPI.COMM_SELF,
                         shared_comm::Union{MPI.Comm,Nothing}=nothing,
                         allocate_array::Union{Function,Nothing}=nothing)

Initialise an MPISchurComplement struct representing a 2x2 block-structured matrix
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
```

`A_factorization` should be a matrix factorization of `A`, or an object with similar
functionality, that can be passed to the `ldiv!()` function to solve a matrix system.

`B` and `D` are only used to initialize the MPISchurComplement, they are not stored. A
reference to `C` is stored. Only the locally owned parts of `B`, `C` and `D` should be
passed. `owned_top_vector_entries` gives the range of global indices that are owned by
this process in the top block of the state vector, `owned_bottom_vector_entries` the same
for the bottom block. `B` should contain only the rows corresponding to
`owned_top_vector_entries` and the columns corresponding to the global indices given by
`B_global_column_range` (these do not have to be all the global indices, when it is known
that the locally owned rows of `B` are zero outside of `B_global_column_range`). `C`
should contain only the columns corresponding to `owned_top_vector_entries` and the rows
corresponding to the global indices given by `C_global_row_range` (these do not have to be
all the global indices, when it is known that the locally owned columns of `C` are zero
outside of `C_global_row_range`). `D` should contain only the rows corresponding to
`owned_bottom_vector_entries` and the columns corresponding to the global indices given by
`D_global_column_range` (these do not have to be all the global indices, when it is known
that the locally owned rows of `D` are zero outside of `D_global_column_range`). When
shared-memory MPI parallelism is used, `B`, `C`, and `D` should all be shared memory
arrays which are identical on all processes in `shared_comm` (or non-shared identical
copies, but that would be an inefficient use of memory).

`distributed_comm` and `shared_comm` are the MPI communicators to use for
distributed-memory and shared-memory communications.

`allocate_array` is required when `shared_comm` is passed. It should be passed a function
that will be used to allocate various buffer arrays. It should return arrays with the same
element type as `A_factorization`, `B`, `C`, and `D`. This is necessary as the MPI
shared-memory 'windows' (`MPI.Win`) have to be stored and (eventually) freed. The
'windows' must be managed externally, as if they are freed by garbage collection, the
freeing is not synchronized between different processes, causing errors.

`synchronize_shared` can be passed a function that synchronizes all processes in
`shared_comm`. This may be useful if a custom synchronization is required for debugging,
etc. If it is not passed, `MPI.Barrier(shared_comm)` will be used.
"""
function mpi_schur_complement(A_factorization, B::AbstractMatrix,
                              B_global_column_range::UnitRange{Int64}, C::AbstractMatrix,
                              C_global_row_range::UnitRange{Int64}, D::AbstractMatrix,
                              D_global_column_range::UnitRange{Int64},
                              owned_top_vector_entries::UnitRange{Int64},
                              owned_bottom_vector_entries::UnitRange{Int64};
                              distributed_comm::MPI.Comm=MPI.COMM_SELF,
                              shared_comm::Union{MPI.Comm,Nothing}=nothing,
                              allocate_array::Union{Function,Nothing}=nothing,
                              synchronize_shared::Union{Function,Nothing}=nothing)

    if distributed_comm != MPI.COMM_NULL
        distributed_nproc = MPI.Comm_size(distributed_comm)
        distributed_rank = MPI.Comm_rank(distributed_comm)
    else
        distributed_nproc = -1
        distributed_rank = -1
    end
    if shared_comm === nothing
        shared_nproc = 1
        shared_rank = 0
    else
        shared_nproc = MPI.Comm_size(shared_comm)
        shared_rank = MPI.Comm_rank(shared_comm)
    end

    top_vec_local_size = length(owned_top_vector_entries)
    bottom_vec_local_size = length(owned_bottom_vector_entries)
    if shared_rank == 0
        top_vec_global_size = MPI.Allreduce(top_vec_local_size, +, distributed_comm)
        bottom_vec_global_size = MPI.Allreduce(bottom_vec_local_size, +, distributed_comm)
    end
    if shared_comm !== nothing
        if shared_rank == 0
            MPI.Bcast(top_vec_global_size, 0, shared_comm)
            MPI.Bcast(bottom_vec_global_size, 0, shared_comm)
        else
            top_vec_global_size = MPI.Bcast(-1, 0, shared_comm)
            bottom_vec_global_size = MPI.Bcast(-1, 0, shared_comm)
        end
    end

    @boundscheck size(A_factorization, 1) == top_vec_global_size || error(BoundsError, " Rows in A_factorization do not match size of 'top vector'.")
    @boundscheck size(A_factorization, 2) == top_vec_global_size || error(BoundsError, " Columns in A_factorization do not match size of 'top vector'.")
    @boundscheck size(B, 1) == top_vec_local_size || error(BoundsError, " Rows in B do not match locally-owned 'top vector' entries.")
    @boundscheck size(B, 2) == length(B_global_column_range) || error(BoundsError, " Columns in B do not match B_global_column_range.")
    @boundscheck size(C, 1) == length(C_global_row_range) || error(BoundsError, " Rows in C do not match C_global_row_range.")
    @boundscheck size(C, 2) == top_vec_local_size || error(BoundsError, " Columns in C do not match locally-owned 'top vector' entries.")
    @boundscheck size(D, 1) == bottom_vec_local_size || error(BoundsError, " Rows in D do not match locally-owned 'bottom vector' entries.")
    @boundscheck size(D, 2) == length(D_global_column_range) || error(BoundsError, " Columns in D do not match size of 'bottom vector'.")

    data_type = eltype(D)

    if shared_comm !== nothing && allocate_array === nothing
        error("when `shared_comm` is passed, `allocate_array` argument is required, "
              * "because it is necessary to manage the MPI.Win objects to ensure they "
              * "are not garbage collected, because garbage collection is not "
              * "necessarily synchronized between different processes.")
    elseif allocate_array === nothing
        allocate_array = (args...) -> zeros(data_type, args...)
    end

    if shared_comm === nothing
        synchronize_shared = ()->nothing
        C_local_row_range = 1:length(C_global_row_range)
        D_local_column_range = 1:length(D_global_column_range)
        schur_complement_local_range = 1:bottom_vec_global_size
        global_top_vector_range = owned_top_vector_entries
        local_top_vector_range = 1:top_vec_local_size
        global_bottom_vector_range = owned_bottom_vector_entries
        local_bottom_vector_range = 1:bottom_vec_local_size
    else
        if synchronize_shared === nothing
            synchronize_shared = ()->MPI.Barrier(shared_comm)
        end

        function get_shared_ranges(global_range)
            # Select a subset of C_global_row_range that will be handled locally.
            n = length(global_range)
            local_n_list = fill(n ÷ shared_nproc, shared_nproc)
            n_missing = n - sum(local_n_list)
            # Add extra points to processes so that local_n_list adds up to n.
            # n_missing will always be less than shared_nproc.
            for i ∈ 0:n_missing-1
                local_n_list[end-i] += 1
            end
            imin = sum(local_n_list[1:shared_rank]) + 1
            imax = imin + local_n_list[shared_rank+1] - 1

            local_range = imin:imax
            new_global_range = global_range[local_range]

            return new_global_range, local_range
        end

        C_global_row_range, C_local_row_range = get_shared_ranges(C_global_row_range)
        D_global_column_range, D_local_column_range = get_shared_ranges(D_global_column_range)
        _, schur_complement_local_range = get_shared_ranges(1:bottom_vec_global_size)
        global_top_vector_range, local_top_vector_range = get_shared_ranges(owned_top_vector_entries)
        global_bottom_vector_range, local_bottom_vector_range = get_shared_ranges(owned_bottom_vector_entries)
    end

    # Allocate buffer arrays
    Ainv_dot_B = allocate_array(top_vec_local_size, bottom_vec_global_size)
    Ainv_dot_u = allocate_array(top_vec_local_size)
    schur_complement = allocate_array(bottom_vec_global_size, bottom_vec_global_size)
    top_vec_buffer = allocate_array(top_vec_local_size)
    bottom_vec_buffer = allocate_array(bottom_vec_global_size)
    global_y = allocate_array(bottom_vec_global_size)

    for i ∈ 1:bottom_vec_global_size
        if i ∈ B_global_column_range
            @views ldiv!(Ainv_dot_B[:,i], A_factorization,
                         B[:,i-B_global_column_range.start+1])
        else
            # Pass `nothing` to indicate this part of B is empty. Depends on
            # implementation of A_factorization to support this!
            @views ldiv!(Ainv_dot_B[:,i], A_factorization, nothing)
        end
    end

    # Initialise to zero, because when C does not include all rows, the matrix
    # multiplication below would not initialise all elements.
    schur_complement[:,schur_complement_local_range] .= 0.0
    synchronize_shared()
    # We store locally all columns in Ainv_dot_B (only local rows) and all rows of C (only
    # local columns). Therefore we can take the matrix product Ainv_dot_B*C with the local
    # chunks, then do a sum-reduce to get the final result. The schur_complement buffer is
    # full size on every rank.
    @views mul!(schur_complement[C_global_row_range,:], C[C_local_row_range,:],
                Ainv_dot_B, -1.0, 0.0)
    synchronize_shared()
    # Only get the local rows for D, so just add these to the local rows of
    # schur_complement.
    @views @. schur_complement[owned_bottom_vector_entries,D_global_column_range] += D[:,D_local_column_range]
    synchronize_shared()
    if shared_rank == 0
        MPI.Reduce!(schur_complement, +, distributed_comm; root=0)
        if distributed_rank == 0
            schur_complement_factorization = lu!(schur_complement)
        else
            schur_complement_factorization = nothing
        end
    else
        schur_complement_factorization = nothing
    end

    sc_factorization = MPISchurComplement(A_factorization, Ainv_dot_B,
                                          B_global_column_range, C, C_global_row_range,
                                          C_local_row_range, D_global_column_range,
                                          D_local_column_range, schur_complement,
                                          schur_complement_factorization,
                                          schur_complement_local_range, Ainv_dot_u,
                                          top_vec_buffer, top_vec_local_size,
                                          bottom_vec_buffer, bottom_vec_local_size,
                                          global_y, top_vec_global_size,
                                          bottom_vec_global_size,
                                          owned_top_vector_entries,
                                          global_top_vector_range, local_top_vector_range,
                                          owned_bottom_vector_entries,
                                          global_bottom_vector_range,
                                          local_bottom_vector_range, distributed_comm,
                                          distributed_rank, shared_comm, shared_rank,
                                          synchronize_shared)

    return sc_factorization
end

"""
    update_schur_complement!(sc::MPISchurComplement, A, B::AbstractMatrix,
                             C::AbstractMatrix, D::AbstractMatrix)

Update the matrix which is being solved by `sc`.

`A` will be passed to `lu!(sc.A_factorization, A)`, so should be as required by the LU
implementation being used for `sc.A_factorization`.

`B`, `C`, and `D` should be the same shapes, and represent the same global index ranges,
as the inputs to `mpi_schur_complement()` used to construct `sc`.
"""
function update_schur_complement!(sc::MPISchurComplement, A, B::AbstractMatrix,
                                  C::AbstractMatrix, D::AbstractMatrix)
    @boundscheck length(sc.owned_top_vector_entries) == size(A, 1) || error(BoundsError, " Number of rows in A does not match number of locally owned top_vector_entries")
    @boundscheck size(sc.Ainv_dot_B, 1) == size(B, 1) || error(BoundsError, " Number of rows in B does not match number of rows in original Ainv_dot_B")
    @boundscheck length(sc.B_global_column_range) == size(B, 2) || error(BoundsError, " Number of columns in B does not match number of columns in original B")
    @boundscheck size(sc.C) == size(C) || error(BoundsError, " Size of C does not match size of original C")
    @boundscheck length(sc.owned_bottom_vector_entries) == size(D, 1) || error(BoundsError, " Number of rows in D does not match number of locally owned bottom_vector_entries")
    @boundscheck sc.D_local_column_range.stop ≤ size(D, 2) || error(BoundsError, " Number of columns in D is smaller than the largest index in D_local_column_range")

    distributed_comm = sc.distributed_comm
    distributed_rank = sc.distributed_rank
    A_factorization = sc.A_factorization
    Ainv_dot_B = sc.Ainv_dot_B
    schur_complement = sc.schur_complement
    synchronize_shared = sc.synchronize_shared

    lu!(A_factorization, A)

    B_global_column_range = sc.B_global_column_range
    for i ∈ 1:sc.bottom_vec_global_size
        if i ∈ B_global_column_range
            @views ldiv!(Ainv_dot_B[:,i], A_factorization,
                         B[:,i-B_global_column_range.start+1])
        else
            # Pass `nothing` to indicate this part of B is empty. Depends on
            # implementation of A_factorization to support this!
            @views ldiv!(Ainv_dot_B[:,i], A_factorization, nothing)
        end
    end

    sc.C = C

    # Initialise to zero, because when C does not include all rows, the matrix
    # multiplication below would not initialise all elements.
    schur_complement[:,sc.schur_complement_local_range] .= 0.0
    synchronize_shared()
    @views mul!(schur_complement[sc.C_global_row_range,:], C[sc.C_local_row_range,:],
                Ainv_dot_B, -1.0, 0.0)
    synchronize_shared()
    @views @. schur_complement[sc.owned_bottom_vector_entries,sc.D_global_column_range] += D[:,sc.D_local_column_range]
    synchronize_shared()
    if sc.shared_rank == 0
        MPI.Reduce!(schur_complement, +, distributed_comm; root=0)
        if distributed_rank == 0
            sc.schur_complement_factorization = lu!(schur_complement)
        end
    end

    return nothing
end

"""
    ldiv!(x::AbstractVector, y::AbstractVector, sc::MPISchurComplement,
          u::AbstractVector, v::AbstractVector)

Solve the 2x2 block-structured matrix system
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
\\cdot
\\left(\\begin{array}{c}
x\\\\
y
\\end{array}\\right)
=
\\left(\\begin{array}{c}
u\\\\
v
\\end{array}\\right)
```
for \$x\$ and \$v\$.

Only the locally owned parts of `u` and `v` should be passed, and the locally owned parts
of the solution will be written into `x` and `y`. This means that `x` and `u` should
correspond to the global indices in `sc.owned_top_vector_entries` and `y` and `v` should
correspond to the global indices in `sc.owned_bottom_vector_entries`. When shared memory
is used, `x` and `y` must be shared-memory arrays (identical on every process in
`sc.shared_comm`). `u` and `v` should also be shared memory arrays (although non-shared
identical copies would also work).
"""
function ldiv!(x::AbstractVector, y::AbstractVector, sc::MPISchurComplement,
               u::AbstractVector, v::AbstractVector)
    @boundscheck size(sc.top_vec_buffer) == size(u) || error(BoundsError, " Size of u does not match size of top_vec_buffer")
    @boundscheck size(sc.top_vec_buffer) == size(x) || error(BoundsError, " Size of x does not match size of top_vec_buffer")
    @boundscheck (length(sc.owned_bottom_vector_entries),) == size(v) || error(BoundsError, " Size of v does not match size of bottom_vector_buffer")
    @boundscheck (length(sc.owned_bottom_vector_entries),) == size(y) || error(BoundsError, " Size of y does not match size of bottom_vector_buffer")

    distributed_comm = sc.distributed_comm
    distributed_rank = sc.distributed_rank
    A_factorization = sc.A_factorization
    Ainv_dot_B = sc.Ainv_dot_B
    schur_complement_factorization = sc.schur_complement_factorization
    Ainv_dot_u = sc.Ainv_dot_u
    top_vec_buffer = sc.top_vec_buffer
    local_top_vector_range = sc.local_top_vector_range
    bottom_vec_buffer = sc.bottom_vec_buffer
    global_bottom_vector_range = sc.global_bottom_vector_range
    local_bottom_vector_range = sc.local_bottom_vector_range
    schur_complement_local_range = sc.schur_complement_local_range
    synchronize_shared = sc.synchronize_shared

    ldiv!(Ainv_dot_u, A_factorization, u)

    # Initialise to zero, because when C does not include all rows, the matrix
    # multiplication below would not initialise all elements.
    bottom_vec_buffer[schur_complement_local_range] .= 0.0
    synchronize_shared()
    # Need all rows of C, but only the local columns.
    @views mul!(bottom_vec_buffer[sc.C_global_row_range], sc.C[sc.C_local_row_range,:],
                Ainv_dot_u, -1.0, 0.0)
    synchronize_shared()

    # Only have the local entries of v, so add those to the local entries in
    # bottom_vec_buffer before recducing.
    @views @. bottom_vec_buffer[global_bottom_vector_range] += v[local_bottom_vector_range]
    synchronize_shared()

    global_y = sc.global_y
    if sc.shared_rank == 0
        MPI.Reduce!(bottom_vec_buffer, +, distributed_comm; root=0)

        if distributed_rank == 0
            ldiv!(global_y, schur_complement_factorization, bottom_vec_buffer)
        end
        MPI.Bcast!(global_y, distributed_comm; root=0)
    end
    synchronize_shared()

    # Need all columns of Ainv_dot_B, but only the local rows.
    @views mul!(top_vec_buffer[local_top_vector_range],
                Ainv_dot_B[local_top_vector_range,:], global_y)
    @. x[local_top_vector_range] = Ainv_dot_u[local_top_vector_range,:] - top_vec_buffer[local_top_vector_range]

    @views @. y[local_bottom_vector_range] = global_y[global_bottom_vector_range]
    synchronize_shared()

    return nothing
end

end # module MPISchurComplements
