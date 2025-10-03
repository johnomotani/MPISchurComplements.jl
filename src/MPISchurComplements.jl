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
                                  Trangeno,Tscomm,Tsync}
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
    B_overlap_buffers_send::TBob
    B_overlap_buffers_recv::TBob
    overlap_ranks::Vector{Int64}
    owned_bottom_vector_entries::Trange
    global_bottom_vector_range::Trange
    global_bottom_vector_entries_no_overlap::Trange
    local_bottom_vector_range::UnitRange{Int64}
    local_bottom_vector_entries_no_overlap::Trangeno
    distributed_comm::MPI.Comm
    distributed_rank::Int64
    shared_comm::Tscomm
    shared_rank::Int64
    synchronize_shared::Tsync
    use_sparse::Bool
    separate_Ainv_B::Bool
end

"""
    remove_gaps_in_ranges!(distributed_ranges::Vector{UnitRange{Int64}})
    remove_gaps_in_ranges!(distributed_ranges::Vector{Vector{Int64}})

Where `distributed_ranges` does not include only a contiguous range (possibly overlapping)
of indices starting from 1, identify and remove any 'gaps' (indices 1 or greater and less
than the largest index in `distributed_ranges` which are not present in any element of
`distributed_ranges`) from the index ranges.
"""
function remove_gaps_in_ranges! end
function remove_gaps_in_ranges!(distributed_ranges::Vector{UnitRange{Int64}})
    sorted_ranges = sort(distributed_ranges; by=(x)->x.start)
    gaps = UnitRange{Int64}[]
    for i ∈ 1:length(sorted_ranges)-1
        this_stop = sorted_ranges[i].stop
        next_start = sorted_ranges[i+1].start
        if next_start > this_stop + 1
            push!(gaps, this_stop+1:next_start-1)
        end
    end

    if length(gaps) > 0
        for i ∈ 1:length(distributed_ranges)
            this_range = distributed_ranges[i]
            # Start here so that if the first index is >1 (i.e. there is an 'initial
            # gap') we also remove that.
            gaps_offset = sorted_ranges[1].start - 1
            for this_gap ∈ gaps
                if gaps.stop < this_range.start
                    gaps_offset += length(this_gap)
                else
                    # `gaps` is sorted, so no need to check more gaps.
                    break
                end
            end
            distributed_ranges[i] = this_range.start-gaps_offset:this_range.stop-gaps_offset
        end
    elseif sorted_ranges[1].start > 1
        offset = sorted_ranges[1].start - 1
        for i ∈ 1:length(distributed_ranges)
            this_range = distributed_ranges[i]
            distributed_ranges[i] = this_range.start-offset:this_range.stop-offset
        end
    end

    return nothing
end
function remove_gaps_in_ranges!(distributed_ranges::Vector{Vector{Int64}})
    # This may be a bit inefficient when the state vector size gets large. In that
    # case, it is suggested to use UnitRange{Int64}-based ranges.
    all_indices = vcat(distributed_ranges...)
    sort!(all_indices)
    gaps = UnitRange{Int64}[]
    prev_ind = 0
    for i ∈ all_indices
        if i > prev_ind + 1
            push!(gaps, prev_ind+1:i-1)
        end
        prev_ind = i
    end

    if length(gaps) > 0
        # Remove `gaps` from distributed_ranges
        for this_range ∈ distributed_ranges
            igap = 1
            n_gaps = length(gaps)
            this_gap = gaps[igap]
            # Start here so that if the first index is >1 (i.e. there is an
            # 'initial gap') we also remove that.
            gaps_offset = 0
            for i ∈ 1:length(this_range)
                this_ind = this_range[i]
                if this_ind > this_gap.start
                    while igap ≤ n_gaps && gaps[igap].stop < this_ind
                        gaps_offset += length(gaps[igap])
                        igap += 1
                    end
                    this_gap = gaps[min(igap, n_gaps)]
                end
                this_range[i] -= gaps_offset
            end
        end
    end

    return nothing
end

"""
    find_local_vector_inds(global_inds, owned_global_inds)

Find indices of `global_inds` within `owned_global_inds`. This gives the 'local indices'
corresponding to `global_inds`.
"""
function find_local_vector_inds(global_inds::AbstractArray, owned_global_inds)
    # Assume global_inds is sorted, and owned_global_inds is sorted.
    local_inds = similar(global_inds)
    lind = 0
    for (i, gind) ∈ enumerate(global_inds)
        # As `global_inds` is sorted, there is no need to search already-checked
        # entries (although this is probably an unnecessary optimisation!).
        lind = @views searchsortedfirst(owned_global_inds[lind+1:end], gind) + lind
        local_inds[i] = lind
    end
    return local_inds
end

"""
    mpi_schur_complement(A_factorization, B::AbstractMatrix, C::AbstractMatrix,
                         D::AbstractMatrix, owned_top_vector_entries::Trange,
                         owned_bottom_vector_entries::Trange;
                         B_global_column_range::Union{Trange,Nothing}=nothing,
                         C_global_row_range::Union{Trange,Nothing}=nothing,
                         D_global_column_range::Union{Trange,Nothing}=nothing,
                         distributed_comm::MPI.Comm=MPI.COMM_SELF,
                         shared_comm::Union{MPI.Comm,Nothing}=nothing,
                         allocate_array::Union{Function,Nothing}=nothing,
                         synchronize_shared::Union{Function,Nothing}=nothing,
                         use_sparse=true,
                         separate_Ainv_B=false) where Trange <: Union{UnitRange{Int64},Vector{Int64}}

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
the bottom block. The index ranges in `owned_top_vector_entries` and
`owned_bottom_vector_entries` may have gaps, not being a contiguous range starting from 1
(and similarly for `B_global_column_range`, `C_global_row_range` and
`D_global_column_range`, if they are passed, although they should all contain the same set
of indices, across all processes, as `owned_bottom_vector_entries`). This may be useful if
the 'bottom block' is actually, for example, a chunk from the middle of a grid, which
separates the 'top block' into block-diagonal pieces. The global indexing for the full
state vector can be used for both `owned_top_vector_entries` and
`owned_bottom_vector_entries`, and they will be converted (by shifting indices down to
remove the gaps) into a representation internal to the `MPISchurComplement` object that is
contiguous and starts at 1 for each block.

`B` should contain only the rows corresponding to `owned_top_vector_entries` and by
default the columns corresponding to `owned_bottom_vector_entries`. If a different range
of columns should be included, the corresponding global indices can be passed as
`B_global_column_range`.

`C` should contain only the columns corresponding to `owned_top_vector_entries` and by
default the rows corresponding to `owned_bottom_vector_entries`. If a different range of
rows should be included, the corresponding global indices can be passed as
`C_global_row_range`.

`D` should contain only the rows corresponding to `owned_bottom_vector_entries` and by
default the same columns. If a different range of columns should be included, the
corresponding global indices passed as `D_global_column_range`.

When shared-memory MPI parallelism is used, `B`, `C`, and `D` should all be shared memory
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
function mpi_schur_complement(A_factorization, B::AbstractMatrix, C::AbstractMatrix,
                              D::AbstractMatrix, owned_top_vector_entries::Trange,
                              owned_bottom_vector_entries::Trange;
                              B_global_column_range::Union{Trange,Nothing}=nothing,
                              C_global_row_range::Union{Trange,Nothing}=nothing,
                              D_global_column_range::Union{Trange,Nothing}=nothing,
                              distributed_comm::MPI.Comm=MPI.COMM_SELF,
                              shared_comm::Union{MPI.Comm,Nothing}=nothing,
                              allocate_array::Union{Function,Nothing}=nothing,
                              synchronize_shared::Union{Function,Nothing}=nothing,
                              use_sparse=true,
                              separate_Ainv_B=false) where Trange <: Union{UnitRange{Int64},Vector{Int64}}

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

        top_vector_distributed_ranges = get_distributed_ranges(owned_top_vector_entries)
        bottom_vector_distributed_ranges = get_distributed_ranges(owned_bottom_vector_entries)

        remove_gaps_in_ranges!(top_vector_distributed_ranges)
        owned_top_vector_entries = top_vector_distributed_ranges[distributed_rank+1]
        remove_gaps_in_ranges!(bottom_vector_distributed_ranges)
        owned_bottom_vector_entries = bottom_vector_distributed_ranges[distributed_rank+1]

        top_vec_global_size = maximum(maximum(r) for r ∈ top_vector_distributed_ranges)
        bottom_vec_global_size = maximum(maximum(r) for r ∈ bottom_vector_distributed_ranges)

        # Find all overlaps (i.e. intersections) between locally-owned range and
        # remotely-owned ranges on all other processes.
        function get_overlaps(local_range::T, distributed_ranges) where T
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

        # For 'bottom vector' need to work out a set of non-overlapping ranges so that
        # each distributed rank owns a unique set of entries.
        bottom_vector_overlaps = get_overlaps(owned_bottom_vector_entries,
                                              bottom_vector_distributed_ranges)
        owned_bottom_vector_entries_no_overlap = copy(owned_bottom_vector_entries)
        # Say that this process 'owns' an entry if there is no process with a lower rank
        # that also owns that entry.
        # Note distributed_rank is a 0-based index, but idist is 1-based.
        if Trange === UnitRange{Int64}
            for idist ∈ 1:distributed_rank
                other_proc_overlap = bottom_vector_overlaps[idist]
                if other_proc_overlap.stop ≥ owned_bottom_vector_entries_no_overlap.start
                    # This way of filtering to non-overlapping entries will not work for a
                    # multi-dimensional domain decomposition, but that case cannot be
                    # represented by `UnitRange{Int64}` ranges, so it is OK.
                    owned_bottom_vector_entries_no_overlap = other_proc_overlap.stop+1:owned_bottom_vector_entries_no_overlap.stop
                end
            end
        else
            for idist ∈ 1:distributed_rank
                other_proc_overlap = bottom_vector_overlaps[idist]
                filter!((i) -> i ∉ other_proc_overlap, owned_bottom_vector_entries_no_overlap)
            end
        end

        # For any ranks that have a 'top vector overlap' that is not empty, need to create
        # a communicator so that the different ranks can sum-reduce their entries of the
        # `B` matrix.
        all_top_vector_overlaps = get_overlaps(owned_top_vector_entries,
                                               top_vector_distributed_ranges)
        local_top_vector_overlaps = Trange[]
        B_overlap_buffers_send = Matrix{data_type}[]
        B_overlap_buffers_recv = Matrix{data_type}[]
        # Not sure overlap_ranks need to be sorted, but probably nicer if they are.
        sorted_overlap_ranks = sortperm(collect(length(o) == 0 ? -1 : first(o) for o ∈ all_top_vector_overlaps))
        # Filter out empty overlaps.
        filter!((i) -> length(all_top_vector_overlaps[i]) > 0, sorted_overlap_ranks)
        # MPI ranks are given by 0-based index, but sorted_overlap_ranks are 1-based
        overlap_ranks = sorted_overlap_ranks .- 1
        owned_top_vector_entries_no_overlap = copy(owned_top_vector_entries)
        top_vector_offset = first(owned_top_vector_entries) - 1
        for idist ∈ sorted_overlap_ranks
            this_overlap = all_top_vector_overlaps[idist]
            if Trange === UnitRange{Int64}
                this_local_overlap = this_overlap.start-top_vector_offset:this_overlap.stop-top_vector_offset
                if this_overlap.stop ≥ owned_top_vector_entries_no_overlap.start
                    # This way of filtering to non-overlapping entries will not work for a
                    # multi-dimensional domain decomposition, but that case cannot be
                    # represented by `UnitRange{Int64}` ranges, so it is OK.
                    owned_top_vector_entries_no_overlap = this_overlap.stop+1:owned_top_vector_entries_no_overlap.stop
                end
            else
                this_local_overlap = find_local_vector_inds(this_overlap, owned_top_vector_entries)
                filter!((i) -> i ∉ this_overlap, owned_top_vector_entries_no_overlap)
            end
            push!(local_top_vector_overlaps, this_local_overlap)
            push!(B_overlap_buffers_send, zeros(data_type, length(this_overlap),
                                                bottom_vec_global_size))
            push!(B_overlap_buffers_recv, zeros(data_type, length(this_overlap),
                                                bottom_vec_global_size))
        end
    else
        top_vec_global_size = nothing
        bottom_vec_global_size = nothing
        local_top_vector_overlaps = Trange[]
        overlap_ranks = Int64[]
        B_overlap_buffers_send = nothing
        B_overlap_buffers_recv = nothing
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
        owned_top_vector_entries = shared_broadcast_range(owned_top_vector_entries)
        owned_bottom_vector_entries = shared_broadcast_range(owned_bottom_vector_entries)
        owned_bottom_vector_entries_no_overlap = shared_broadcast_range(owned_bottom_vector_entries_no_overlap)
    end

    if B_global_column_range === nothing
        B_global_column_range = owned_bottom_vector_entries
    else
        if shared_rank == 0
            B_distributed_ranges = get_distributed_ranges(B_global_column_range)
            remove_gaps_in_ranges!(B_distributed_ranges)
            B_global_column_range = B_distributed_ranges[distributed_rank+1]
        else
            B_global_column_range = Trange === UnitRange{Int64} ? (-1:-2) : Int64[]
        end
        if shared_comm !== nothing
            B_global_column_range = shared_broadcast_range(B_global_column_range)
        end
    end
    if C_global_row_range === nothing
        C_global_row_range = owned_bottom_vector_entries
    else
        if shared_rank == 0
            C_distributed_ranges = get_distributed_ranges(C_global_row_range)
            remove_gaps_in_ranges!(C_distributed_ranges)
            C_global_column_range = C_distributed_ranges[distributed_rank+1]
        else
            C_global_column_range = Trange === UnitRange{Int64} ? (-1:-2) : Int64[]
        end
        if shared_comm !== nothing
            C_global_column_range = shared_broadcast_range(C_global_column_range)
        end
    end
    if D_global_column_range === nothing
        D_global_column_range = owned_bottom_vector_entries
    else
        if shared_rank == 0
            D_distributed_ranges = get_distributed_ranges(D_global_column_range)
            remove_gaps_in_ranges!(D_distributed_ranges)
            D_global_column_range = D_distributed_ranges[distributed_rank+1]
        else
            D_global_column_range = Trange === UnitRange{Int64} ? (-1:-2) : Int64[]
        end
        if shared_comm !== nothing
            D_global_column_range = shared_broadcast_range(D_global_column_range)
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
        global_bottom_vector_entries_no_overlap = owned_bottom_vector_entries_no_overlap
        if Trange === UnitRange{Int64}
            local_bottom_vector_entries_no_overlap = 1+local_bottom_vector_offset:length(owned_bottom_vector_entries_no_overlap)+local_bottom_vector_offset
        else
            local_bottom_vector_entries_no_overlap = find_local_vector_inds(owned_bottom_vector_entries_no_overlap, owned_bottom_vector_entries)
        end
    else
        if synchronize_shared === nothing
            synchronize_shared = ()->MPI.Barrier(shared_comm)
        end

        function get_shared_ranges(global_range)
            # Select a subset of global_range that will be handled locally.
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
            new_global_range = global_range[imin:imax]

            return new_global_range, local_range
        end
        function get_shared_partial_entries(global_entries, local_entries)
            # Select a subset of global_entries that will be handled locally.
            n = length(global_entries)
            local_n_list = fill(n ÷ shared_nproc, shared_nproc)
            n_missing = n - sum(local_n_list)
            # Add extra points to processes so that local_n_list adds up to n.
            # n_missing will always be less than shared_nproc.
            for i ∈ 0:n_missing-1
                local_n_list[end-i] += 1
            end
            imin = sum(local_n_list[1:shared_rank]) + 1
            imax = imin + local_n_list[shared_rank+1] - 1

            return global_entries[imin:imax], local_entries[imin:imax]
        end

        C_global_row_range, C_local_row_range = get_shared_ranges(C_global_row_range)
        D_global_column_range, D_local_column_range = get_shared_ranges(D_global_column_range)
        _, schur_complement_local_range = get_shared_ranges(1:bottom_vec_global_size)
        _, local_top_vector_range = get_shared_ranges(owned_top_vector_entries)
        global_bottom_vector_range, local_bottom_vector_range = get_shared_ranges(owned_bottom_vector_entries)
        global_bottom_vector_entries_no_overlap, local_bottom_vector_entries_no_overlap =
            get_shared_partial_entries(owned_bottom_vector_entries_no_overlap,
                                       find_local_vector_inds(owned_bottom_vector_entries_no_overlap,
                                                              owned_bottom_vector_entries))
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
                                          local_top_vector_overlaps,
                                          B_overlap_buffers_send, B_overlap_buffers_recv,
                                          overlap_ranks, owned_bottom_vector_entries,
                                          global_bottom_vector_range,
                                          global_bottom_vector_entries_no_overlap,
                                          local_bottom_vector_range,
                                          local_bottom_vector_entries_no_overlap,
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
    B_overlap_buffers_send = sc.B_overlap_buffers_send
    B_overlap_buffers_recv = sc.B_overlap_buffers_recv
    overlap_ranks = sc.overlap_ranks
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
        for (overlap_range, buffer_send, buffer_recv, overlap_rank) ∈
                zip(local_top_vector_overlaps, B_overlap_buffers_send,
                    B_overlap_buffers_recv, overlap_ranks)
            @views buffer_send .= Ainv_dot_B[overlap_range, :]
            # Iallreduce seems not to be included in the nice Julia API, so have to use
            # lower level call here.
            push!(reqs, MPI.Isend(buffer_send, distributed_comm; dest=overlap_rank))
            push!(reqs, MPI.Irecv!(buffer_recv, distributed_comm; source=overlap_rank))
        end
        MPI.Waitall(reqs)
        for (overlap_range, buffer) ∈ zip(local_top_vector_overlaps, B_overlap_buffers_recv)
            @views Ainv_dot_B[overlap_range, :] .+= buffer
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
    global_bottom_vector_entries_no_overlap = sc.global_bottom_vector_entries_no_overlap
    local_bottom_vector_range = sc.local_bottom_vector_range
    local_bottom_vector_entries_no_overlap = sc.local_bottom_vector_entries_no_overlap
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
    @views @. bottom_vec_buffer[global_bottom_vector_entries_no_overlap] += v[local_bottom_vector_entries_no_overlap]
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
