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
using SparseArrays

mutable struct MPISchurComplement{Trange <: Union{UnitRange{Int64},Vector{Int64}},
                                  TA,TAiB,TAiBl,TB,TC,TSC,TSCF,TAiu,Ttv,Tbv,Tgy,TBob,
                                  Tscomm,Tsync}
    A_factorization::TA
    Ainv_dot_B::TAiB
    Ainv_dot_B_local::TAiBl
    B::TB
    B_global_column_range::Trange
    C::TC
    C_global_row_range::Trange
    C_local_row_range::UnitRange{Int64}
    D_global_column_range::Trange
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
    owned_top_vector_entries::Trange
    local_top_vector_range::UnitRange{Int64}
    local_top_vector_overlaps::Vector{Trange}
    B_overlap_buffers::TBob
    overlap_comms::Vector{MPI.Comm}
    owned_bottom_vector_entries::Trange
    global_bottom_vector_range::Trange
    global_bottom_vector_range_no_overlap::Trange
    local_bottom_vector_range::UnitRange{Int64}
    local_bottom_vector_range_no_overlap::UnitRange{Int64}
    distributed_comm::MPI.Comm
    distributed_rank::Int64
    shared_comm::Tscomm
    shared_rank::Int64
    synchronize_shared::Tsync
    use_sparse::Bool
    separate_Ainv_B::Bool
end

"""
Get communicators which include this rank and one neighbouring rank.
"""
function get_overlap_comm(distributed_comm::MPI.Comm, other_rank)
    this_rank = MPI.Comm_rank(distributed_comm)

    # Creating communicators is a collective operation. We will construct MPI.Group
    # objects to construct the communicators here - the communicator construction is then
    # collective on all processes in a group. Therefore we need to split the ranks up so
    # that upper boundaries of even ranks link with lower boundaries of odd ranks in one
    # step, and the reverse in a second step.
    ranks = Cint[this_rank, other_rank]
    sort!(ranks)
    group = MPI.Group_incl(MPI.Comm_group(distributed_comm), ranks)
    comm = MPI.Comm_create_group(distributed_comm, group, 0)

    return comm
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
                         allocate_array::Union{Function,Nothing}=nothing,
                         use_sparse=true, separate_Ainv_B=false)

Initialise an MPISchurComplement struct representing a 2x2 block-structured matrix
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
```

`A_factorization` should be a matrix factorization of `A`, or an object with similar
functionality, that can be passed to the `ldiv!()` function to solve a matrix system.

`B` and `D` are only used to initialize the MPISchurComplement, they are not stored. If
`sparse=false` is passed, a reference to `C` is stored. Only the locally owned parts of
`B`, `C` and `D` should be passed. `B`, `C` and `D` may be modified by this function.
`owned_top_vector_entries` gives the range of global indices that are owned by this
process in the top block of the state vector, `owned_bottom_vector_entries` the same for
the bottom block. `B` should contain only the rows corresponding to
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
copies, but that would be an inefficient use of memory). When `owned_top_vector_entries`
and/or `owned_bottom_vector_entries` are ranges that overlap between different
distributed-MPI ranks, the matrices `A`, `B`, `C`, and `D` should be set up so that adding
together the matrices passed on each distributed-MPI rank gives the full, global matrix
(i.e. there should be no double-counting of overlapping entries, but any fraction of the
overlap can be passed on any distributed rank, as long as the contributions from each
distributed rank add up to the full value).

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

By default, makes `C` a sparse matrix. Pass `use_sparse=false` to force `C` to be dense.

If `B` is very sparse, and `A_factorization` is efficiently parallelised, it might be more
efficient to compute `A \\ (B.y)` in two steps (`B.y` then `A \\ ()`) rather than
multiplying by the dense `Ainv_dot_B`. If `separate_Ainv_B=true` is passed, this will be
done (requires `use_sparse=true`). There is no saving in setup time because `Ainv_dot_B`
still has to be calculated, to be multiplied by `C` when calculating `schur_complement`.
There is also no memory saving as a dense B-sized buffer array is needed.
"""
function mpi_schur_complement(A_factorization, B::AbstractMatrix,
                              B_global_column_range::Trange, C::AbstractMatrix,
                              C_global_row_range::Trange, D::AbstractMatrix,
                              D_global_column_range::Trange,
                              owned_top_vector_entries::Trange,
                              owned_bottom_vector_entries::Trange;
                              distributed_comm::MPI.Comm=MPI.COMM_SELF,
                              shared_comm::Union{MPI.Comm,Nothing}=nothing,
                              allocate_array::Union{Function,Nothing}=nothing,
                              synchronize_shared::Union{Function,Nothing}=nothing,
                              use_sparse=true, separate_Ainv_B=false) where Trange <: Union{UnitRange{Int64},Vector{Int64}}

    data_type = eltype(D)

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
        top_vec_global_size = MPI.Allreduce(maximum(owned_top_vector_entries), max, distributed_comm)
        bottom_vec_global_size = MPI.Allreduce(maximum(owned_bottom_vector_entries), max, distributed_comm)

        # Collect a range of row/column indices from all distributed ranks, and return as
        # a Vector{Trange}.
        function get_distributed_ranges(local_range::UnitRange{Int64})
            local_range_vec = [local_range.start, local_range.stop]
            distributed_ranges_vec = MPI.Allgather(local_range_vec, distributed_comm)
            distributed_ranges = [distributed_ranges_vec[i]:distributed_ranges_vec[i+1]
                                  for i ∈ 1:2:2*distributed_nproc]
            return distributed_ranges
        end
        function get_distributed_ranges(local_range::Vector{Int64})
            vec_sizes = MPI.Allgather(Cint(length(local_range)), distributed_comm)
            distributed_ranges_vec = similar(local_range, sum(vec_sizes))
            MPI.Allgatherv!(local_range, MPI.VBuffer(distributed_ranges_vec, vec_sizes),
                            distributed_comm)
            distributed_ranges = Vector{Int64}[]
            imin = 1
            for idist ∈ 1:distributed_nproc
                imax = imin + vec_sizes[idist] - 1
                push!(distributed_ranges, distributed_ranges_vec[imin:imax])
                imin = imax + 1
            end
            return distributed_ranges
        end

        # Find all overlaps (i.e. intersections) between locally-owned range and
        # remotely-owned ranges on all other processes.
        function get_overlaps(local_range::T) where T
            distributed_ranges = get_distributed_ranges(local_range)
            overlaps = T[]
            for idist ∈ 1:distributed_nproc
                # Remember distributed_rank is 0-based index, but idist is 1-based.
                if idist - 1 == distributed_rank || distributed_comm == MPI.COMM_NULL
                    # Don't care about overlap of this rank with itself.
                    # Make start/stop negative so they are easy to filter out later.
                    push!(overlaps, -1:-2)
                else
                    push!(overlaps, intersect(local_range, distributed_ranges[idist]))
                end
            end
            return overlaps
        end

        # For any ranks that have a 'top vector overlap' that is not empty, need to create
        # a communicator so that the different ranks can sum-reduce their entries of the
        # `B` matrix.
        all_top_vector_overlaps = get_overlaps(owned_top_vector_entries)
        local_top_vector_overlaps = Trange[]
        B_overlap_buffers = Matrix{data_type}[]
        overlap_comms = MPI.Comm[]
        # Need to make sure all of overlap_comms are created in a consistent order on all
        # processes to avoid deadlock. To do this, sort the overlaps by the lowest index
        # value in the overlap. The overlap communicators are created for pairs of
        # processes. There might be more than two processes sharing a single overlap, e.g.
        # at a corner of a multidimensional domain decomposition - break these 'ties' by
        # sorting by distributed_rank (everything is already sorted by distributed_rank,
        # so we do not need to do any more work to achieve this).
        sorted_overlap_ranks = sortperm(collect(length(o) == 0 ? -1 : first(o) for o ∈ all_top_vector_overlaps))
        # Filter out empty overlaps.
        filter!((i) -> length(all_top_vector_overlaps[i]) > 0, sorted_overlap_ranks)
        top_vector_offset = first(owned_top_vector_entries) - 1
        function find_local_top_vector_inds(global_inds)
            # Assume global_inds is sorted, and owned_top_vector_entries is sorted.
            local_inds = similar(global_inds)
            lind = 0
            for (i, gind) ∈ enumerate(global_inds)
                # As `global_inds` is sorted, there is no need to search already-checked
                # entries (although this is probably an unnecessary optimisation!).
                lind = @views searchsortedfirst(owned_top_vector_entries[lind+1:end], gind) + lind
                local_inds[i] = lind
            end
            return local_inds
        end
        for idist ∈ sorted_overlap_ranks
            this_overlap = all_top_vector_overlaps[idist]
            if Trange === UnitRange{Int64}
                this_local_overlap = this_overlap.start-top_vector_offset:this_overlap.stop-top_vector_offset
            else
                this_local_overlap = find_local_top_vector_inds(this_overlap)
            end
            push!(local_top_vector_overlaps, this_local_overlap)
            push!(B_overlap_buffers, zeros(data_type, length(this_overlap),
                                           bottom_vec_global_size))
            push!(overlap_comms, get_overlap_comm(distributed_comm, idist-1))
        end

        # For 'bottom vector' need to work out a set of non-overlapping ranges so that
        # each distributed rank owns a unique set of entries.
        bottom_vector_overlaps = get_overlaps(owned_bottom_vector_entries)
        owned_bottom_vector_entries_no_overlap = copy(owned_bottom_vector_entries)
        # Say that this process 'owns' an entry if there is no process with a lower rank
        # that also owns that entry.
        # Note distributed_rank is a 0-based index, but idist is 1-based.
        if Trange === UnitRange{Int64}
            for idist ∈ 1:distributed_rank
                other_proc_overlap = bottom_vector_overlaps[idist]
                if other_proc_overlap.stop ≥ owned_bottom_vector_entries_no_overlap.start
                    owned_bottom_vector_entries_no_overlap = other_proc_overlap.stop+1:owned_bottom_vector_entries_no_overlap.stop
                end
            end
            bottom_vector_offset = owned_bottom_vector_entries.start - 1
            local_bottom_vector_range_no_overlap = owned_bottom_vector_entries_no_overlap.start-bottom_vector_offset:owned_bottom_vector_entries_no_overlap.stop-bottom_vector_offset
        else
            for idist ∈ 1:distributed_rank
                other_proc_overlap = bottom_vector_overlaps[idist]
                filter!((i) -> i ∉ other_proc_overlap, owned_bottom_vector_entries_no_overlap)
            end
            bottom_vector_offset = owned_bottom_vector_entries[1] - 1
            local_bottom_vector_range_no_overlap = owned_bottom_vector_entries_no_overlap .- bottom_vector_offset
        end
    else
        top_vec_global_size = nothing
        bottom_vec_global_size = nothing
        local_top_vector_overlaps = Trange[]
        B_overlap_buffers = nothing
        overlap_comms = MPI.Comm[]
        owned_bottom_vector_entries_no_overlap = nothing
    end
    if shared_comm !== nothing
        function shared_broadcast_int(i::Union{Int64,Nothing})
            if shared_rank == 0
                if i === nothing
                    error("`i` should not be `nothing` on root of shared-memory communicator")
                end
                MPI.Bcast(i, 0, shared_comm)
                return i
            else
                return MPI.Bcast(-1, 0, shared_comm)
            end
        end
        function shared_broadcast_range(r::Union{Trange,Nothing})
            if shared_rank == 0
                if r === nothing
                    error("`r` should not be `nothing` on root of shared-memory communicator")
                end
                MPI.bcast(r, 0, shared_comm)
                return r
            else
                if Trange === UnitRange{Int64}
                    range = MPI.bcast(1:0, 0, shared_comm)
                else
                    range = MPI.bcast(Int64[], 0, shared_comm)
                end
                return range
            end
        end

        top_vec_global_size = shared_broadcast_int(top_vec_global_size)
        bottom_vec_global_size = shared_broadcast_int(bottom_vec_global_size)
        owned_bottom_vector_entries_no_overlap = shared_broadcast_range(owned_bottom_vector_entries_no_overlap)
    end

    @boundscheck size(A_factorization, 1) == top_vec_global_size || error(BoundsError, " Rows in A_factorization do not match size of 'top vector'.")
    @boundscheck size(A_factorization, 2) == top_vec_global_size || error(BoundsError, " Columns in A_factorization do not match size of 'top vector'.")
    @boundscheck size(B, 1) == top_vec_local_size || error(BoundsError, " Rows in B do not match locally-owned 'top vector' entries.")
    @boundscheck size(B, 2) == length(B_global_column_range) || error(BoundsError, " Columns in B do not match B_global_column_range.")
    @boundscheck size(C, 1) == length(C_global_row_range) || error(BoundsError, " Rows in C do not match C_global_row_range.")
    @boundscheck size(C, 2) == top_vec_local_size || error(BoundsError, " Columns in C do not match locally-owned 'top vector' entries.")
    @boundscheck size(D, 1) == bottom_vec_local_size || error(BoundsError, " Rows in D do not match locally-owned 'bottom vector' entries.")
    @boundscheck size(D, 2) == length(D_global_column_range) || error(BoundsError, " Columns in D do not match size of 'bottom vector'.")

    if shared_comm !== nothing && allocate_array === nothing
        error("when `shared_comm` is passed, `allocate_array` argument is required, "
              * "because it is necessary to manage the MPI.Win objects to ensure they "
              * "are not garbage collected, because garbage collection is not "
              * "necessarily synchronized between different processes.")
    elseif allocate_array === nothing
        allocate_array = (args...) -> zeros(data_type, args...)
    end

    local_bottom_vector_offset = searchsortedfirst(owned_bottom_vector_entries, owned_bottom_vector_entries_no_overlap[1]) - 1
    if shared_comm === nothing
        synchronize_shared = ()->nothing
        C_local_row_range = 1:length(C_global_row_range)
        D_local_column_range = 1:length(D_global_column_range)
        schur_complement_local_range = 1:bottom_vec_global_size
        local_top_vector_range = 1:top_vec_local_size
        global_bottom_vector_range = owned_bottom_vector_entries
        local_bottom_vector_range = 1:bottom_vec_local_size
        global_bottom_vector_range_no_overlap = owned_bottom_vector_entries_no_overlap
        local_bottom_vector_range_no_overlap = 1+local_bottom_vector_offset:length(owned_bottom_vector_entries_no_overlap)+local_bottom_vector_offset
    else
        if synchronize_shared === nothing
            synchronize_shared = ()->MPI.Barrier(shared_comm)
        end

        function get_shared_ranges(global_range; local_offset=0)
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

            local_range = imin+local_offset:imax+local_offset
            new_global_range = global_range[imin:imax]

            return new_global_range, local_range
        end

        C_global_row_range, C_local_row_range = get_shared_ranges(C_global_row_range)
        D_global_column_range, D_local_column_range = get_shared_ranges(D_global_column_range)
        _, schur_complement_local_range = get_shared_ranges(1:bottom_vec_global_size)
        _, local_top_vector_range = get_shared_ranges(owned_top_vector_entries)
        global_bottom_vector_range, local_bottom_vector_range = get_shared_ranges(owned_bottom_vector_entries)
        global_bottom_vector_range_no_overlap, local_bottom_vector_range_no_overlap = get_shared_ranges(owned_bottom_vector_entries_no_overlap; local_offset=local_bottom_vector_offset)
    end

    # Allocate buffer arrays
    Ainv_dot_B = allocate_array(top_vec_local_size, bottom_vec_global_size)
    if separate_Ainv_B
        if !use_sparse
            error("It will always be more expensive to use `separate_Ainv_B` when "
                  * "`use_sparse=false`.")
        end
        Ainv_dot_B_local = nothing
        B_local = sparse(zeros(data_type, 1, 1))
    else
        # Store the chunk of Ainv_dot_B needed by this shared-memory process as a contiguous
        # array.
        Ainv_dot_B_local = Matrix{data_type}(undef, length(local_top_vector_range), bottom_vec_global_size)
        B_local = nothing
    end
    Ainv_dot_u = allocate_array(top_vec_local_size)
    schur_complement = allocate_array(bottom_vec_global_size, bottom_vec_global_size)
    top_vec_buffer = allocate_array(top_vec_local_size)
    bottom_vec_buffer = allocate_array(bottom_vec_global_size)
    global_y = allocate_array(bottom_vec_global_size)

    fake_C = zeros(data_type, 1, 1)
    if use_sparse
        fake_C = sparse(fake_C)
    end

    if shared_rank == 0 && distributed_rank == 0
        schur_complement_factorization = lu!(ones(data_type, 1, 1))
    else
        schur_complement_factorization = nothing
    end

    sc_factorization = MPISchurComplement(A_factorization, Ainv_dot_B, Ainv_dot_B_local,
                                          B_local, B_global_column_range, fake_C,
                                          C_global_row_range, C_local_row_range,
                                          D_global_column_range, D_local_column_range,
                                          schur_complement,
                                          schur_complement_factorization,
                                          schur_complement_local_range, Ainv_dot_u,
                                          top_vec_buffer, top_vec_local_size,
                                          bottom_vec_buffer, bottom_vec_local_size,
                                          global_y, top_vec_global_size,
                                          bottom_vec_global_size,
                                          owned_top_vector_entries,
                                          local_top_vector_range,
                                          local_top_vector_overlaps, B_overlap_buffers,
                                          overlap_comms, owned_bottom_vector_entries,
                                          global_bottom_vector_range,
                                          global_bottom_vector_range_no_overlap,
                                          local_bottom_vector_range,
                                          local_bottom_vector_range_no_overlap,
                                          distributed_comm, distributed_rank,
                                          shared_comm, shared_rank, synchronize_shared,
                                          use_sparse, separate_Ainv_B)

    update_schur_complement!(sc_factorization, missing, B, C, D)

    return sc_factorization
end

"""
    update_schur_complement!(sc::MPISchurComplement, A, B::AbstractMatrix,
                             C::AbstractMatrix, D::AbstractMatrix)

Update the matrix which is being solved by `sc`.

`A` will be passed to `lu!(sc.A_factorization, A)`, so should be as required by the LU
implementation being used for `sc.A_factorization`.

`B`, `C`, and `D` should be the same shapes, and represent the same global index ranges,
as the inputs to `mpi_schur_complement()` used to construct `sc`. `B`, `C`, and `D` may be
modified.
"""
function update_schur_complement!(sc::MPISchurComplement, A, B::AbstractMatrix,
                                  C::AbstractMatrix, D::AbstractMatrix)
    @boundscheck A === missing || length(sc.owned_top_vector_entries) == size(A, 1) || error(BoundsError, " Number of rows in A does not match number of locally owned top_vector_entries")
    @boundscheck size(sc.Ainv_dot_B, 1) == size(B, 1) || error(BoundsError, " Number of rows in B does not match number of rows in original Ainv_dot_B")
    @boundscheck length(sc.B_global_column_range) == size(B, 2) || error(BoundsError, " Number of columns in B does not match number of columns in original B")
    # Don't check size(C, 1) because we don't store the full row range. There will be an
    # out of bounds error from indexing by sc.C_local_row_range if C is too small.
    @boundscheck sc.top_vec_local_size == size(C, 2) || error(BoundsError, " Number of columns in C does not match original C")
    @boundscheck length(sc.owned_bottom_vector_entries) == size(D, 1) || error(BoundsError, " Number of rows in D does not match number of locally owned bottom_vector_entries")
    @boundscheck maximum(sc.D_local_column_range) ≤ size(D, 2) || error(BoundsError, " Number of columns in D is smaller than the largest index in D_local_column_range")

    A_factorization = sc.A_factorization
    Ainv_dot_B = sc.Ainv_dot_B
    Ainv_dot_B_local = sc.Ainv_dot_B_local
    B_global_column_range = sc.B_global_column_range
    C_global_row_range = sc.C_global_row_range
    D_global_column_range = sc.D_global_column_range
    D_local_column_range = sc.D_local_column_range
    schur_complement = sc.schur_complement
    schur_complement_local_range = sc.schur_complement_local_range
    owned_bottom_vector_entries = sc.owned_bottom_vector_entries
    local_top_vector_range = sc.local_top_vector_range
    local_top_vector_overlaps = sc.local_top_vector_overlaps
    B_overlap_buffers = sc.B_overlap_buffers
    overlap_comms = sc.overlap_comms
    distributed_comm = sc.distributed_comm
    distributed_rank = sc.distributed_rank
    shared_rank = sc.shared_rank
    synchronize_shared = sc.synchronize_shared
    use_sparse = sc.use_sparse
    separate_Ainv_B = sc.separate_Ainv_B

    # When `A===missing`, this was called from the `mpi_schur_complement()` constructor,
    # where we assume `A_factorization` was already initialized.
    if A !== missing
        lu!(A_factorization, A)
    end

    # Use `Ainv_dot_B` as a local-rows/global-columns sized buffer to collect `B` into.
    # This is slightly inefficient, as there will be chunks that are all-zero that we do
    # not need to collect, but this way seems the simplest to implement, as we need the
    # not-locally-owned columns of B in all locally owned rows, to pass to `ldiv!()`
    # below.
    Ainv_dot_B[local_top_vector_range,:] .= 0
    @views Ainv_dot_B[local_top_vector_range,B_global_column_range] .= B[local_top_vector_range,:]
    synchronize_shared()
    # Sum-reduce the overlapping rows of B (temporarily stored in `Ainv_dot_B`).
    if shared_rank == 0
        reqs = MPI.Request[]
        for (overlap_range, buffer, comm) ∈ zip(local_top_vector_overlaps,
                                                B_overlap_buffers, overlap_comms)
            @views buffer .= Ainv_dot_B[overlap_range, :]
            # Iallreduce seems not to be included in the nice Julia API, so have to use
            # lower level call here.
            this_req = MPI.Request()
            mpi_buffer = MPI.Buffer(buffer)
            mpi_op = MPI.Op(+, eltype(buffer))
            MPI.API.MPI_Iallreduce(MPI.IN_PLACE, mpi_buffer.data, mpi_buffer.count,
                                   mpi_buffer.datatype, mpi_op, comm, this_req)
            push!(reqs, this_req)
        end
        MPI.Waitall(reqs)
        for (overlap_range, buffer) ∈ zip(local_top_vector_overlaps, B_overlap_buffers)
            @views Ainv_dot_B[overlap_range, :] .= buffer
        end
    end
    synchronize_shared()

    if separate_Ainv_B
        # At this point `Ainv_dot_B` contains the dense array of `B`.
        sc.B = @views sparse(Ainv_dot_B[local_top_vector_range,:])
        synchronize_shared()
    end

    # When using shared memory, only store the slice of C that this process needs.
    C = @view C[sc.C_local_row_range, :]
    if use_sparse
        C = sparse(C)
    end
    sc.C = C

    ldiv!(A_factorization, Ainv_dot_B)

    # Initialise to zero, because when C does not include all rows, the matrix
    # multiplication below would not initialise all elements.
    schur_complement[:,schur_complement_local_range] .= 0.0
    synchronize_shared()
    if !separate_Ainv_B
        Ainv_dot_B_local .= Ainv_dot_B[local_top_vector_range,:]
    end
    # We store locally all columns in Ainv_dot_B (only local rows) and all rows of C (only
    # local columns). Therefore we can take the matrix product Ainv_dot_B*C with the local
    # chunks, then do a sum-reduce to get the final result. The schur_complement buffer is
    # full size on every rank.
    @views mul!(schur_complement[C_global_row_range,:], C, Ainv_dot_B, -1.0, 0.0)
    synchronize_shared()
    # Only get the local rows for D, so just add these to the local rows of
    # schur_complement.
    @views @. schur_complement[owned_bottom_vector_entries,D_global_column_range] += D[:,D_local_column_range]
    synchronize_shared()
    if shared_rank == 0
        MPI.Reduce!(schur_complement, +, distributed_comm; root=0)
    end

    if shared_rank == 0
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
    Ainv_dot_B_local = sc.Ainv_dot_B_local
    schur_complement_factorization = sc.schur_complement_factorization
    Ainv_dot_u = sc.Ainv_dot_u
    top_vec_buffer = sc.top_vec_buffer
    local_top_vector_range = sc.local_top_vector_range
    bottom_vec_buffer = sc.bottom_vec_buffer
    global_bottom_vector_range = sc.global_bottom_vector_range
    global_bottom_vector_range_no_overlap = sc.global_bottom_vector_range_no_overlap
    local_bottom_vector_range = sc.local_bottom_vector_range
    local_bottom_vector_range_no_overlap = sc.local_bottom_vector_range_no_overlap
    schur_complement_local_range = sc.schur_complement_local_range
    synchronize_shared = sc.synchronize_shared

    ldiv!(Ainv_dot_u, A_factorization, u)

    # Initialise to zero, because when C does not include all rows, the matrix
    # multiplication below would not initialise all elements.
    bottom_vec_buffer[schur_complement_local_range] .= 0.0
    synchronize_shared()
    # Need all rows of C, but only the local columns - this is all that is stored in sc.C.
    @views mul!(bottom_vec_buffer[sc.C_global_row_range], sc.C, Ainv_dot_u, -1.0, 0.0)
    synchronize_shared()

    # Only have the local entries of v, so add those to the local entries in
    # bottom_vec_buffer before recducing.
    # Need to avoid double counting of any overlapping entries in `v`.
    @views @. bottom_vec_buffer[global_bottom_vector_range_no_overlap] += v[local_bottom_vector_range_no_overlap]
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

    # Need all columns of Ainv_dot_B_local, but only the local rows.
    if sc.separate_Ainv_B
        # B_local is a sparse matrix, so this might sometimes be numerically cheaper than
        # multiplying by a dense, precomputed Ainv_dot_B_local`.
        @views mul!(top_vec_buffer[local_top_vector_range], sc.B, global_y)
        synchronize_shared()
        ldiv!(A_factorization, top_vec_buffer)
        synchronize_shared()
    else
        @views mul!(top_vec_buffer[local_top_vector_range], Ainv_dot_B_local, global_y)
    end
    @. x[local_top_vector_range] = Ainv_dot_u[local_top_vector_range,:] - top_vec_buffer[local_top_vector_range]

    @views @. y[local_bottom_vector_range] = global_y[global_bottom_vector_range]
    synchronize_shared()

    return nothing
end

end # module MPISchurComplements
