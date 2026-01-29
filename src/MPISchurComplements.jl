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

const Trange = Vector{Int64}

mutable struct MPISchurComplement{TA,TAiB,TAiBl,TB,TC,TSC,TSCF,TAiu,Ttv,Tbv,Tgy,TBob,
                                  Trangeno,Tscomm,Tsync}
    A_factorization::TA
    Ainv_dot_B::TAiB
    Ainv_dot_B_local::TAiBl
    B::TB
    B_global_column_range::Trange
    B_local_column_range::Trange
    B_local_column_repeats::Matrix{Int64}
    C::TC
    C_global_row_range_partial::Trange
    C_local_row_range_partial::Trange
    C_local_row_repeats::Matrix{Int64}
    D_global_column_range_partial::Trange
    D_local_column_range_partial::Trange
    D_local_column_repeats::Matrix{Int64}
    schur_complement::TSC
    schur_complement_factorization::TSCF
    schur_complement_local_range_partial::Trange
    Ainv_dot_u::TAiu
    top_vec_buffer::Ttv
    top_vec_local_size::Int64
    bottom_vec_buffer::Tbv
    bottom_vec_local_size::Int64
    global_y::Tgy
    top_vec_global_size::Int64
    bottom_vec_global_size::Int64
    owned_top_vector_entries::Trange
    local_top_vector_range_partial::UnitRange{Int64}
    local_top_vector_overlaps::Vector{Trange}
    local_top_vector_repeats::Matrix{Int64}
    B_overlap_buffers_send::TBob
    B_overlap_buffers_recv::TBob
    overlap_ranks::Vector{Int64}
    owned_bottom_vector_entries::Trange
    unique_bottom_vector_entries::Trange
    global_bottom_vector_range_partial::Trange
    global_bottom_vector_entries_no_overlap_partial::Trange
    local_bottom_vector_range_partial::UnitRange{Int64}
    local_bottom_vector_entries_no_overlap_partial::Trangeno
    local_bottom_vector_unique_entries::Vector{Int64}
    local_bottom_vector_repeats::Matrix{Int64}
    distributed_comm::MPI.Comm
    distributed_rank::Int64
    shared_comm::Tscomm
    shared_rank::Int64
    synchronize_shared::Tsync
    use_sparse::Bool
    separate_Ainv_B::Bool
end

"""
    remove_gaps_in_ranges!(distributed_ranges::Vector{Vector{Int64}})

Where `distributed_ranges` does not include only a contiguous range (possibly overlapping)
of indices starting from 1, identify and remove any 'gaps' (indices 1 or greater and less
than the largest index in `distributed_ranges` which are not present in any element of
`distributed_ranges`) from the index ranges.
"""
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
            # Allow for the possibility that this_range is not sorted, but assume that it
            # is mostly sorted (i.e. there are a small number of times when
            # this_ind < ind_previous) - otherwise the following would be inefficient.
            ind_previous = 0
            # Start here so that if the first index is >1 (i.e. there is an
            # 'initial gap') we also remove that.
            gaps_offset = 0
            for i ∈ 1:length(this_range)
                this_ind = this_range[i]
                if this_ind ≤ ind_previous
                    # Restart the gap search because the range is not sorted.
                    igap = 1
                    this_gap = gaps[igap]
                    gaps_offset = 0
                end
                ind_previous = this_ind
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
    separate_repeated_indices(inds)

Find any repeated indices in `inds`. The first occurence of and index is the 'real' entry,
and any subsequent occurences are 'repeats. Returns:
1. a Vector of 'real' entries
2. a Vector of the positions within the vector (i.e. the local indices) of the 'real' entries
3. a 2xn Matrix containing all the repeats, where the two entries in each column are the
   'real' index and a corresponding 'repeat'
"""
function separate_repeated_indices(inds)
    indices_with_repeats = eltype(inds)[]
    unique_inds = eltype(inds)[]
    for i ∈ inds
        if i ∈ unique_inds
            push!(indices_with_repeats, i)
        else
            push!(unique_inds, i)
        end
    end
    n_repeats = length(indices_with_repeats)
    # Get rid of any repeated repeats from indices_with_repeats.
    unique!(indices_with_repeats)

    repeats = zeros(Int64, 2, n_repeats)
    counter = 0
    for repeated ∈ indices_with_repeats
        all_repeats = findall(i -> i==repeated, inds)
        first_occurence = all_repeats[1]
        # Skip the first entry in all_repeats which is the position of the 'real' entry.
        for position ∈ all_repeats[2:end]
            counter += 1
            repeats[1,counter] = first_occurence
            repeats[2,counter] = position
        end
    end
    if counter != n_repeats
        error("Filled $counter columns of `repeats`, but expected $n_repeats repeats.")
    end

    local_unique_inds = find_local_vector_inds(unique_inds, inds)

    return unique_inds, local_unique_inds, repeats
end

"""
    find_local_vector_inds(global_inds, owned_global_inds)

Find indices of `global_inds` within `owned_global_inds`. This gives the 'local indices'
corresponding to `global_inds`.
"""
function find_local_vector_inds(global_inds::AbstractArray, owned_global_inds)
    local_inds = similar(global_inds)
    for (i, gind) ∈ enumerate(global_inds)
        # `owned_global_inds` is not necessarily sorted, so no obvious way to optimise
        # this.
        local_inds[i] = @views findfirst(i->i==gind, owned_global_inds)
    end
    return local_inds
end

"""
    mpi_schur_complement(A_factorization, B::AbstractMatrix, C::AbstractMatrix,
                         D::AbstractMatrix,
                         owned_top_vector_entries::Union{UnitRange{Int64},Vector{Int64}},
                         owned_bottom_vector_entries::Union{UnitRange{Int64},Vector{Int64}};
                         B_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                         C_global_row_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                         D_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                         distributed_comm::MPI.Comm=MPI.COMM_SELF,
                         shared_comm::Union{MPI.Comm,Nothing}=nothing,
                         allocate_array::Union{Function,Nothing}=nothing,
                         synchronize_shared::Union{Function,Nothing}=nothing,
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
contiguous and starts at 1 for each block. In addition, the indices passed do not have to
be unique (they can be repeated on different subdomains, and/or within a single subdomain)
or monotonically increasing - this allows for periodic dimensions where the final point in
the dimension should be identified with the first point. When there are repeated indices,
the vector entries passed for the right-hand-side (`u` and `v`) are assumed to be
identical at the locations given corresponding to repeated indices. Matrix entries in `B`,
`C`, and `D` corresponding to each repeated row or column index will be (in effect) summed
together into a single entry of an 'assembled' matrix (as an implementation detail, the
'assembled' matrix may not be constructed explicitly, but it serves to define the meaning
of repeated matrix entries).

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
                              D::AbstractMatrix,
                              owned_top_vector_entries::Union{UnitRange{Int64},Vector{Int64}},
                              owned_bottom_vector_entries::Union{UnitRange{Int64},Vector{Int64}};
                              B_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                              C_global_row_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                              D_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                              distributed_comm::MPI.Comm=MPI.COMM_SELF,
                              shared_comm::Union{MPI.Comm,Nothing}=nothing,
                              allocate_array::Union{Function,Nothing}=nothing,
                              synchronize_shared::Union{Function,Nothing}=nothing,
                              use_sparse=true, separate_Ainv_B=false)

    data_type = eltype(D)

    # Simpler to only support one type (`Vector{Int64}`) for ranges, so convert
    # UnitRange inputs to Vector.
    if isa(owned_top_vector_entries, UnitRange)
        owned_top_vector_entries = collect(owned_top_vector_entries)
    end
    if isa(owned_bottom_vector_entries, UnitRange)
        owned_bottom_vector_entries = collect(owned_bottom_vector_entries)
    end

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
        # Collect the row/column indices from all distributed ranks, and return as a
        # Vector{Trange}.
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

        # Include `init=0` in the `maximum()` calls in case any range is empty.
        top_vec_global_size = maximum(maximum(r; init=0)
                                      for r ∈ top_vector_distributed_ranges; init=0)
        bottom_vec_global_size = maximum(maximum(r; init=0)
                                         for r ∈ bottom_vector_distributed_ranges; init=0)

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
                elseif idist - 1 < distributed_rank
                    # Handle idist-1<distributed_rank and idist-1>distributed_rank
                    # separately because we want to ensure a consistent orderding of the
                    # intersection indices on all processes. `intersect` maintains the
                    # order of indices, but we also need to pass the sets in the same
                    # order. If we always pass the lower rank's indices first, we will get
                    # the same output on both processes involved in the overlap.
                    push!(overlaps, intersect(distributed_ranges[idist], local_range))
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
        for idist ∈ 1:distributed_rank
            other_proc_overlap = bottom_vector_overlaps[idist]
            filter!((i) -> i ∉ other_proc_overlap, owned_bottom_vector_entries_no_overlap)
        end
        # Filter any repeated entries out of owned_bottom_vector_entries_no_overlap.
        unique!(owned_bottom_vector_entries_no_overlap)
        unique_bottom_vector_entries, local_bottom_vector_unique_entries,
        local_bottom_vector_repeats =
            separate_repeated_indices(owned_bottom_vector_entries)

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
        top_vector_offset = first(owned_top_vector_entries) - 1
        for idist ∈ sorted_overlap_ranks
            this_overlap = all_top_vector_overlaps[idist]
            this_local_overlap = find_local_vector_inds(this_overlap, owned_top_vector_entries)
            push!(local_top_vector_overlaps, this_local_overlap)
            push!(B_overlap_buffers_send, zeros(data_type, length(this_overlap),
                                                bottom_vec_global_size))
            push!(B_overlap_buffers_recv, zeros(data_type, length(this_overlap),
                                                bottom_vec_global_size))
        end
        unique_top_vector_entries, local_top_vector_unique_entries,
        local_top_vector_repeats =
            separate_repeated_indices(owned_top_vector_entries)
    else
        top_vec_global_size = nothing
        bottom_vec_global_size = nothing
        local_top_vector_overlaps = Trange[]
        local_top_vector_repeats = nothing
        overlap_ranks = Int64[]
        B_overlap_buffers_send = nothing
        B_overlap_buffers_recv = nothing
        owned_bottom_vector_entries_no_overlap = nothing
        unique_bottom_vector_entries = nothing
        local_bottom_vector_unique_entries = nothing
        local_bottom_vector_repeats = nothing
    end
    if shared_comm !== nothing
        # Communicate index ranges, etc. to all processes on the shared-memory
        # communicator.
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
                range = MPI.bcast(Int64[], 0, shared_comm)
                return range
            end
        end
        function shared_broadcast_matrix(m::Union{Matrix{Int64},Nothing})
            if shared_rank == 0
                if m === nothing
                    error("`m` should not be `nothing` on root of shared-memory communicator")
                end
                MPI.bcast(m, 0, shared_comm)
                return m
            else
                mat = MPI.bcast(zeros(Int64, 0, 0), 0, shared_comm)
                return mat
            end
        end

        top_vec_global_size = shared_broadcast_int(top_vec_global_size)
        local_top_vector_repeats = shared_broadcast_matrix(local_top_vector_repeats)
        bottom_vec_global_size = shared_broadcast_int(bottom_vec_global_size)
        owned_top_vector_entries = shared_broadcast_range(owned_top_vector_entries)
        owned_bottom_vector_entries = shared_broadcast_range(owned_bottom_vector_entries)
        owned_bottom_vector_entries_no_overlap = shared_broadcast_range(owned_bottom_vector_entries_no_overlap)
        unique_bottom_vector_entries = shared_broadcast_range(unique_bottom_vector_entries)
        local_bottom_vector_unique_entries = shared_broadcast_range(local_bottom_vector_unique_entries)
        local_bottom_vector_repeats = shared_broadcast_matrix(local_bottom_vector_repeats)
    end

    # If Matrix ranges were not passed explicitly, set them from the top/bottom vector
    # ranges.
    if B_global_column_range === nothing
        B_global_column_range = unique_bottom_vector_entries
        B_local_column_range = local_bottom_vector_unique_entries
        B_local_column_repeats = local_bottom_vector_repeats
    else
        if isa(B_global_column_range, UnitRange)
            B_global_column_range = collect(B_global_column_range)
        end
        if shared_rank == 0
            B_distributed_ranges = get_distributed_ranges(B_global_column_range)
            remove_gaps_in_ranges!(B_distributed_ranges)
            B_global_column_range = B_distributed_ranges[distributed_rank+1]
        else
            B_global_column_range = Int64[]
        end
        if shared_comm !== nothing
            B_global_column_range = shared_broadcast_range(B_global_column_range)
        end
        B_global_column_range, B_local_column_range, B_local_column_repeats =
            separate_repeated_indices(B_global_column_range)
    end
    if C_global_row_range === nothing
        C_global_row_range = unique_bottom_vector_entries
        C_local_row_range = local_bottom_vector_unique_entries
        C_local_row_repeats = local_bottom_vector_repeats
    else
        if isa(C_global_row_range, UnitRange)
            C_global_row_range = collect(C_global_row_range)
        end
        if shared_rank == 0
            C_distributed_ranges = get_distributed_ranges(C_global_row_range)
            remove_gaps_in_ranges!(C_distributed_ranges)
            C_global_column_range = C_distributed_ranges[distributed_rank+1]
        else
            C_global_column_range = Int64[]
        end
        if shared_comm !== nothing
            C_global_column_range = shared_broadcast_range(C_global_column_range)
        end
        C_global_row_range, C_local_row_range, C_local_row_repeats =
            separate_repeated_indices(C_global_row_range)
    end
    if D_global_column_range === nothing
        D_global_column_range = unique_bottom_vector_entries
        D_local_column_range = local_bottom_vector_unique_entries
        D_local_column_repeats = local_bottom_vector_repeats
    else
        if isa(D_global_column_range, UnitRange)
            D_global_column_range = collect(D_global_column_range)
        end
        if shared_rank == 0
            D_distributed_ranges = get_distributed_ranges(D_global_column_range)
            remove_gaps_in_ranges!(D_distributed_ranges)
            D_global_column_range = D_distributed_ranges[distributed_rank+1]
        else
            D_global_column_range = Int64[]
        end
        if shared_comm !== nothing
            D_global_column_range = shared_broadcast_range(D_global_column_range)
        end
        D_global_column_range, D_local_column_range, D_local_column_repeats =
            separate_repeated_indices(D_global_column_range)
    end

    @boundscheck size(A_factorization, 1) == top_vec_global_size || error(BoundsError, " Rows in A_factorization do not match size of 'top vector'.")
    @boundscheck size(A_factorization, 2) == top_vec_global_size || error(BoundsError, " Columns in A_factorization do not match size of 'top vector'.")
    @boundscheck size(B, 1) == top_vec_local_size || error(BoundsError, " Rows in B do not match locally-owned 'top vector' entries.")
    @boundscheck size(B, 2) == length(B_local_column_range) + size(B_local_column_repeats, 2) || error(BoundsError, " Columns in B do not match index ranges.")
    @boundscheck size(C, 1) == length(C_local_row_range) + size(C_local_row_repeats, 2) || error(BoundsError, " Rows in C do not match index ranges.")
    @boundscheck size(C, 2) == top_vec_local_size || error(BoundsError, " Columns in C do not match locally-owned 'top vector' entries.")
    @boundscheck size(D, 1) == bottom_vec_local_size || error(BoundsError, " Rows in D do not match locally-owned 'bottom vector' entries.")
    @boundscheck size(D, 2) == length(D_local_column_range) + size(D_local_column_repeats, 2) || error(BoundsError, " Columns in D do not match index ranges.")

    if shared_comm !== nothing && allocate_array === nothing
        error("when `shared_comm` is passed, `allocate_array` argument is required, "
              * "because it is necessary to manage the MPI.Win objects to ensure they "
              * "are not garbage collected, because garbage collection is not "
              * "necessarily synchronized between different processes.")
    elseif allocate_array === nothing
        allocate_array = (args...) -> zeros(data_type, args...)
    end

    # Define indices that will be handled by this process in shared-memory-parallelised
    # operations.
    if shared_comm === nothing
        synchronize_shared = ()->nothing
        C_global_row_range_partial= C_global_row_range
        C_local_row_range_partial = C_local_row_range
        D_global_column_range_partial = D_global_column_range
        D_local_column_range_partial = D_local_column_range
        schur_complement_local_range_partial = collect(1:bottom_vec_global_size)
        local_top_vector_range_partial = 1:top_vec_local_size
        local_top_vector_unique_entries_partial = local_top_vector_unique_entries
        global_bottom_vector_range_partial = owned_bottom_vector_entries
        local_bottom_vector_range_partial = 1:bottom_vec_local_size
        global_bottom_vector_entries_no_overlap_partial = owned_bottom_vector_entries_no_overlap
        local_bottom_vector_entries_no_overlap_partial =
            find_local_vector_inds(owned_bottom_vector_entries_no_overlap,
                                   owned_bottom_vector_entries)
    else
        if synchronize_shared === nothing
            synchronize_shared = ()->MPI.Barrier(shared_comm)
        end

        function get_shared_partial_ranges(global_range, local_range)
            # Select a subset of global_range and local_range that will be handled
            # locally.
            if length(global_range) != length(local_range)
                error("expected global_range and local_range to have the same lengths. "
                      * "Got length(global_range)=$(length(global_range)), "
                      * "length(local_range)=$(length(local_range)).")
            end

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

            this_range = imin:imax
            new_global_range = global_range[imin:imax]
            new_local_range = local_range[imin:imax]

            return new_global_range, new_local_range
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

        C_global_row_range_partial, C_local_row_range_partial =
            get_shared_partial_ranges(C_global_row_range, C_local_row_range)
        D_global_column_range_partial, D_local_column_range_partial =
            get_shared_partial_ranges(D_global_column_range, D_local_column_range)
        _, schur_complement_local_range_partial =
            get_shared_partial_ranges(collect(1:bottom_vec_global_size),
                                      collect(1:bottom_vec_global_size))
        _, local_top_vector_range_partial =
            get_shared_partial_ranges(owned_top_vector_entries, 1:top_vec_local_size)
        global_bottom_vector_range_partial, local_bottom_vector_range_partial =
            get_shared_partial_ranges(owned_bottom_vector_entries,
                                      1:bottom_vec_local_size)
        global_bottom_vector_entries_no_overlap_partial,
        local_bottom_vector_entries_no_overlap_partial =
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
        Ainv_dot_B_local = Matrix{data_type}(undef, length(local_top_vector_range_partial), bottom_vec_global_size)
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
                                          B_local, B_global_column_range,
                                          B_local_column_range, B_local_column_repeats,
                                          fake_C, C_global_row_range_partial,
                                          C_local_row_range_partial, C_local_row_repeats,
                                          D_global_column_range_partial,
                                          D_local_column_range_partial,
                                          D_local_column_repeats, schur_complement,
                                          schur_complement_factorization,
                                          schur_complement_local_range_partial,
                                          Ainv_dot_u, top_vec_buffer, top_vec_local_size,
                                          bottom_vec_buffer, bottom_vec_local_size,
                                          global_y, top_vec_global_size,
                                          bottom_vec_global_size,
                                          owned_top_vector_entries,
                                          local_top_vector_range_partial,
                                          local_top_vector_overlaps,
                                          local_top_vector_repeats,
                                          B_overlap_buffers_send, B_overlap_buffers_recv,
                                          overlap_ranks, owned_bottom_vector_entries,
                                          unique_bottom_vector_entries,
                                          global_bottom_vector_range_partial,
                                          global_bottom_vector_entries_no_overlap_partial,
                                          local_bottom_vector_range_partial,
                                          local_bottom_vector_entries_no_overlap_partial,
                                          local_bottom_vector_unique_entries,
                                          local_bottom_vector_repeats, distributed_comm,
                                          distributed_rank, shared_comm, shared_rank,
                                          synchronize_shared, use_sparse, separate_Ainv_B)

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
    @boundscheck length(sc.B_local_column_range) + size(sc.B_local_column_repeats, 2) == size(B, 2) || error(BoundsError, " Number of columns in B does not match number of columns in original B")
    # Don't check size(C, 1) because we don't store the full row range. There will be an
    # out of bounds error from indexing by sc.C_local_row_range_partial if C is too small.
    @boundscheck sc.top_vec_local_size == size(C, 2) || error(BoundsError, " Number of columns in C does not match original C")
    @boundscheck length(sc.owned_bottom_vector_entries) == size(D, 1) || error(BoundsError, " Number of rows in D does not match number of locally owned bottom_vector_entries")
    @boundscheck length(sc.D_local_column_range_partial) == 0 || maximum(sc.D_local_column_range_partial) ≤ size(D, 2) || error(BoundsError, " Number of columns in D is smaller than the largest index in D_local_column_range")

    A_factorization = sc.A_factorization
    Ainv_dot_B = sc.Ainv_dot_B
    Ainv_dot_B_local = sc.Ainv_dot_B_local
    B_global_column_range = sc.B_global_column_range
    B_local_column_range = sc.B_local_column_range
    B_local_column_repeats = sc.B_local_column_repeats
    C_global_row_range_partial = sc.C_global_row_range_partial
    C_local_row_range_partial = sc.C_local_row_range_partial
    C_local_row_repeats = sc.C_local_row_repeats
    D_global_column_range_partial = sc.D_global_column_range_partial
    D_local_column_range_partial = sc.D_local_column_range_partial
    D_local_column_repeats = sc.D_local_column_repeats
    schur_complement = sc.schur_complement
    schur_complement_local_range_partial = sc.schur_complement_local_range_partial
    local_top_vector_repeats = sc.local_top_vector_repeats
    unique_bottom_vector_entries = sc.unique_bottom_vector_entries
    local_bottom_vector_unique_entries = sc.local_bottom_vector_unique_entries
    local_bottom_vector_repeats = sc.local_bottom_vector_repeats
    local_top_vector_range_partial = sc.local_top_vector_range_partial
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
    # When there are repeated entries in `B`, need to add them up into a single entry, and
    # then copy this entry into all the repeated positions. This converts the columns of
    # `B` into 'vectors' (with the same structure as `u`) that can be passed to
    # `A_factorization` to find `Ainv_dot_B`.
    if length(B_local_column_repeats) > 0 || length(local_top_vector_repeats) > 0
        # Add up entries that are repeated on this subdomain.
        if shared_rank == 0
            # Handle repeated columns and rows in serial for now. Look at this again if it
            # becomes a bottleneck.
            for (to, from) ∈ eachcol(B_local_column_repeats)
                @views B[:,to] .+= B[:,from]
            end
            for (to, from) ∈ eachcol(local_top_vector_repeats)
                @views B[to,B_local_column_range] .+= B[from,B_local_column_range]
            end
        end
        synchronize_shared()
    end
    Ainv_dot_B[local_top_vector_range_partial,:] .= 0
    @views Ainv_dot_B[local_top_vector_range_partial,B_global_column_range] .=
        B[local_top_vector_range_partial,B_local_column_range]
    synchronize_shared()
    # Add up the rows of B that overlap between different subdomains (temporarily stored
    # in `Ainv_dot_B`).  Note only non-repeated points in the overlaps are communicated,
    # to reduce the amount of communication.
    if shared_rank == 0
        reqs = MPI.Request[]
        for (overlap_range, buffer_send, buffer_recv, overlap_rank) ∈
                zip(local_top_vector_overlaps, B_overlap_buffers_send,
                    B_overlap_buffers_recv, overlap_ranks)
            @views buffer_send .= Ainv_dot_B[overlap_range,:]
            # Iallreduce seems not to be included in the nice Julia API, so have to use
            # lower level call here.
            push!(reqs, MPI.Isend(buffer_send, distributed_comm; dest=overlap_rank))
            push!(reqs, MPI.Irecv!(buffer_recv, distributed_comm; source=overlap_rank))
        end
        MPI.Waitall(reqs)
        for (overlap_range, buffer) ∈ zip(local_top_vector_overlaps, B_overlap_buffers_recv)
            @views Ainv_dot_B[overlap_range,:] .+= buffer
        end

        if length(B_local_column_repeats) > 0 || length(local_top_vector_repeats) > 0
            # Now that overlaps have been comunicated, all contributions have been added
            # the 'to' places, so we can now copy back these periodic entries to the
            # 'from' places.
            for (to, from) ∈ eachcol(local_top_vector_repeats)
                @views Ainv_dot_B[from,:] .= Ainv_dot_B[to,:]
            end
        end
    end
    synchronize_shared()

    # At this point `Ainv_dot_B` contains the dense array of `B`.
    if separate_Ainv_B
        sc.B = @views sparse(Ainv_dot_B[local_top_vector_range_partial,:])
        synchronize_shared()
    end
    ldiv!(A_factorization, Ainv_dot_B)

    # A representation of C is stored where no rows are repeated, so need to add up all
    # contributions from repeated row indices into a single row (that will then be
    # included in the stored `sc.C`, i.e. the 'to' rows are included in
    # `sc.C_local_row_range_partial` while the 'from' rows are not).
    if length(C_local_row_repeats) > 0
        if shared_rank == 0
            # Handle repeated columns in serial for now. Look at this again if it becomes a
            # bottleneck.
            for (to, from) ∈ eachcol(C_local_row_repeats)
                @views C[to,:] .+= C[from,:]
            end
        end
        synchronize_shared()
    end
    # When using shared memory, only store the slice of C that this process needs.
    if use_sparse
        C = @view C[sc.C_local_row_range_partial,:]
        C = sparse(C)
    else
        # Make a copy because C_local_row_range_partial might not be a contiguous range of
        # indices, but performance will be better if `C` is a contiguously-allocated
        # array.
        C = C[sc.C_local_row_range_partial,:]
    end
    sc.C = C

    # Initialise `schur_complement` to zero, because when `C` does not include all rows,
    # the matrix multiplication below would not initialise all elements.
    schur_complement[:,schur_complement_local_range_partial] .= 0.0
    synchronize_shared()

    # Read out the local entries of `Ainv_dot_B` here, rather than just after `Ainv_dot_B`
    # is calculated, in order to avoid adding another `synchronize_shared()` call.
    if !separate_Ainv_B
        Ainv_dot_B_local .= Ainv_dot_B[local_top_vector_range_partial,:]
    end

    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement` buffer is full size on every rank.
    @views mul!(schur_complement[C_global_row_range_partial,:], C, Ainv_dot_B, -1.0, 0.0)
    synchronize_shared()
    # Only get the local rows for D, so just add these to the local rows of
    # `schur_complement`.
    # As `schur_Complement` does not have any repeated entries, need to add up any locally
    # repeated entries of D (columns then rows) so that we can then select the 'assembled'
    # version of `D` to add to `schur_complement`. Any entries that are repeated on
    # different subdomains will be taken care of when the local contributions to
    # `schur_complement` are added together in the `MPI.Reduce!()` below, as there may be
    # non-zero contributions to some entries from multiple subdomains.
    if length(D_local_column_repeats) > 0
        if shared_rank == 0
            # Handle repeated columns in serial for now. Look at this again if it becomes a
            # bottleneck.
            for (to, from) ∈ eachcol(D_local_column_repeats)
                @views D[:,to] .+= D[:,from]
            end
        end
        synchronize_shared()
    end
    if length(local_bottom_vector_repeats) > 0
        if shared_rank == 0
            # Handle repeated columns in serial for now. Look at this again if it becomes a
            # bottleneck.
            for (to, from) ∈ eachcol(local_bottom_vector_repeats)
                @views D[to,:] .+= D[from,:]
            end
        end
        synchronize_shared()
    end
    @views @. schur_complement[unique_bottom_vector_entries,D_global_column_range_partial] +=
        D[local_bottom_vector_unique_entries,D_local_column_range_partial]
    synchronize_shared()
    if shared_rank == 0
        MPI.Reduce!(schur_complement, +, distributed_comm; root=0)
    end

    # `schur_complement` has been gathered/assembled onto the global rank-0 process, and
    # is now LU-factorized in serial.
    # Unless the original matrices were all block-diagonal in some consistent way (in
    # which case the solve could probably be done more efficiently by splitting the full
    # matrix into the disconnected pieces), `schur_complement` will generally be a dense
    # matrix, so not worth having an option for a sparse LU factorization here. Possibly
    # this LU factorization (and the corresponding `ldiv!()` using
    # `sc.schur_complement_factorization`) could be parallelised with shared memory and/or
    # distributed MPI, but we expect this step not to be a bottleneck, so it is done in
    # serial (at least for now).
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
    local_top_vector_range_partial = sc.local_top_vector_range_partial
    bottom_vec_buffer = sc.bottom_vec_buffer
    global_bottom_vector_range_partial = sc.global_bottom_vector_range_partial
    global_bottom_vector_entries_no_overlap_partial = sc.global_bottom_vector_entries_no_overlap_partial
    local_bottom_vector_range_partial = sc.local_bottom_vector_range_partial
    local_bottom_vector_entries_no_overlap_partial = sc.local_bottom_vector_entries_no_overlap_partial
    schur_complement_local_range_partial = sc.schur_complement_local_range_partial
    synchronize_shared = sc.synchronize_shared

    ldiv!(Ainv_dot_u, A_factorization, u)

    # Initialise to zero, because when C does not include all rows, the matrix
    # multiplication below would not initialise all elements.
    bottom_vec_buffer[schur_complement_local_range_partial] .= 0.0
    synchronize_shared()
    # Need all rows of C, but only the local columns - this is all that is stored in sc.C.
    @views mul!(bottom_vec_buffer[sc.C_global_row_range_partial], sc.C, Ainv_dot_u, -1.0, 0.0)
    synchronize_shared()

    # Only have the local entries of v, so add those to the local entries in
    # bottom_vec_buffer before recducing.
    # Need to avoid double counting of any overlapping entries in `v`.
    @views @. bottom_vec_buffer[global_bottom_vector_entries_no_overlap_partial] += v[local_bottom_vector_entries_no_overlap_partial]
    synchronize_shared()

    # `global_y` is solved in serial on the global rank-0 process, and then communicated
    # back to all other processes.
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
        @views mul!(top_vec_buffer[local_top_vector_range_partial], sc.B, global_y)
        synchronize_shared()
        ldiv!(A_factorization, top_vec_buffer)
        synchronize_shared()
    else
        @views mul!(top_vec_buffer[local_top_vector_range_partial], Ainv_dot_B_local, global_y)
    end
    @views @. x[local_top_vector_range_partial] = Ainv_dot_u[local_top_vector_range_partial] - top_vec_buffer[local_top_vector_range_partial]

    @views @. y[local_bottom_vector_range_partial] = global_y[global_bottom_vector_range_partial]
    synchronize_shared()

    return nothing
end

# Import FakeMPILUs to make it available for prototyping/testing in other packages.
# Not exported as part of public interface because it shouldn't be used in 'production'.
include("FakeMPILUs.jl")

end # module MPISchurComplements
