using ColumnPivotLUs
using Combinatorics
using Primes

# Implement efficient parallel LU factorization, based on ideas in:
# L. Grigori, J. Demmel, and H. Xiang, "CALU: a communication optimal LU factorization algorithm", SIAM Journal on Matrix Analysis and Applications, 32 (2011), pp. 1317-1350.
# and the ScaLAPACK design
# J. Choi, et al. "Design and implementation of the ScaLAPACK LU, QR, and Cholesky factorization routines", Scientific Programming 5.3 (1996), pp. 173-184.
#
# Also read
# E. Solomonik, and J. Demmel, "Communication-optimal parallel 2.5D matrix multiplication and LU factorization algorithms", European Conference on Parallel Processing. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011.
# but have not (yet?) implemented any '2.5D' algorithm.

# This is copied/modified from LinearAlgebra.ipiv2perm, but truncates the generation of the
# permutation vector after n entries.
function ipiv2perm_truncated(v::AbstractVector{T}, maxi::Integer,
                             truncatei::Integer) where T
    Base.require_one_based_indexing(v)
    p = T[1:maxi;]
    @inbounds for i in 1:truncatei
        p[i], p[v[i]] = p[v[i]], p[i]
    end
    return p[1:truncatei]
end

function setup_lu(m::Int64, n::Int64, tile_size::Int64, shared_comm::MPI.Comm,
                  shared_comm_rank::Int64, shared_comm_size::Int64,
                  distributed_comm_rank::Int64, distributed_comm_size::Int64,
                  datatype::Type, allocate_shared_float::Ff, allocate_shared_int::Fi,
                  synchronize_shared, timer) where {Ff,Fi}

    factors = allocate_shared_float(m, n)

    if shared_comm_rank == 0
        row_permutation = zeros(Int64, m)
    else
        row_permutation = zeros(Int64, 0)
    end

    # Each block owns a set of (tile_size,tile_size) tiles in the full matrix - the last
    # row and column of tiles may be shorter/narrower. The tiles are distributed in a
    # block-cyclic pattern. Each block owns sub-tiles in the k'th row in each group of K
    # columns, and in the l'th column of each group of L columns. We choose (abritrarily)
    # to make L≤K.
    distributed_comm_size_factors =
        [prod(x) for x in
         collect(unique(combinations(factor(Vector, distributed_comm_size))))]
    # Find the last factor ≤ sqrt(distributed_comm_size)
    factor_ind = findlast(x -> x≤sqrt(distributed_comm_size), distributed_comm_size_factors)
    group_L = distributed_comm_size_factors[factor_ind]
    group_K = distributed_comm_size ÷ group_L

    group_l, group_k = divrem(distributed_comm_rank, group_K) .+ 1

    group_row_height = group_K * tile_size
    group_n_rows = (n + group_row_height - 1) ÷ group_row_height

    group_col_height = group_L * tile_size
    group_n_cols = (m + group_col_height - 1) ÷ group_col_height

    function get_row_range(group_row)
        tile_row_start = (group_row - 1) * group_row_height + (group_k - 1) * tile_size + 1
        tile_row_end = min((group_row - 1) * group_row_height + group_k * tile_size, m)
        return tile_row_start:tile_row_end
    end

    function get_col_range(group_col)
        tile_col_start = (group_col - 1) * group_col_height + (group_l - 1) * tile_size + 1
        tile_col_end = min((group_col - 1) * group_col_height + group_l * tile_size, m)
        return tile_col_start:tile_col_end
    end

    factorization_matrix_parts_row_ranges = [get_row_range(group_row)
                                             for group_row ∈ 1:group_n_rows]
    factorization_matrix_parts_col_ranges = [get_col_range(group_col)
                                             for group_col ∈ 1:group_n_cols]
    factorization_locally_owned_rows = vcat((collect(r) for r ∈
                                             factorization_matrix_parts_row_ranges)...)

    # Store the locally-owned parts of the array in a joined-together 2D array
    # `factorization_matrix_storage`. This will be useful for some operations.
    # `factorization_matrix_parts` contains views into `factorization_matrix_storage`
    # corresponding to the locally-owned section of each tile.
    local_storage_m = sum(length(r) for r ∈ factorization_matrix_parts_row_ranges)
    local_storage_n = sum(length(c) for c ∈ factorization_matrix_parts_col_ranges)
    factorization_matrix_storage = allocate_shared_float(local_storage_m, local_storage_n)
    factorization_matrix_parts =
        [@view(factorization_matrix_storage[(group_row-1)*tile_size+1:min(group_row*tile_size,local_storage_m),
                                            (group_col-1)*tile_size+1:min(group_col*tile_size,local_storage_n)])
         for group_row ∈ 1:group_n_rows, group_col ∈ 1:group_n_cols]

    factorization_pivoting_buffer =
        allocate_shared_float(max(group_n_rows * tile_size * tile_size,
                                  group_K * tile_size * tile_size))
    if shared_comm_size > 1
        factorization_local_left_panel_buffer =
            Vector{datatype}(undef, (group_n_rows * tile_size + shared_comm_size - 1) ÷ shared_comm_size * tile_size)
    else
        # Do not need this buffer
        factorization_local_left_panel_buffer = zeros(datatype, 0)
    end

    # Generate a 'binary' tree of disributed-memory ranks that will participate in the
    # disributed-memory part of the parallel pivot generation.
    distributed_size_prime_factors = factor(Vector, group_K)
    factorization_pivot_generation_distributed_tree_sizes = [group_K]
    # `reverse()` so that we divide by the largest factors first. Should not affect
    # overall run time as the order of the factors just shuffles around work between
    # different levels, but this order minimises the total work done, which might
    # marginally help power consumption?
    for f in reverse(distributed_size_prime_factors)
        push!(factorization_pivot_generation_distributed_tree_sizes,
              factorization_pivot_generation_distributed_tree_sizes[end] ÷ f)
    end

    factorization_pivoting_reduction_buffer = allocate_shared_float(tile_size * group_K,
                                                                    tile_size)
    factorization_pivoting_reduction_indices =
        allocate_shared_int(max(tile_size * group_K * shared_comm_size, 2 * tile_size,
                                local_storage_m))
    factorization_pivoting_reduction_indices_local = zeros(Int64, tile_size)
    factorization_source_rows = zeros(Int64, 2 * tile_size)
    factorization_locally_owned_swap_rows = zeros(Int64, 2 * tile_size)
    factorization_source_swap_labels = zeros(Int64, 2 * tile_size)
    factorization_row_swap_buffers = allocate_shared_float(tile_size, local_storage_n)
    factorization_swap_flags = zeros(UInt8, 2 * tile_size)

    if tile_size ≤ ColumnPivotLUs.block_size
        # No point using RowPivotLUMPI, as it will just pass through to `LAPACK.getrf!()`
        # for this `tile_size`. Instead, store an `ipiv` vector to use when calling
        # `LAPACK.getrf!()` directly.
        factorization_shared_lu = allocate_shared_int(m)
    else
        ipiv = allocate_shared_int(m)
        factorization_shared_lu = get_row_pivot_lu(ipiv, shared_comm;
                                                   synchronize=synchronize_shared,
                                                   timer=timer)
    end

    comm_requests = [MPI.REQUEST_NULL for _ ∈
                     1:max((1 + tile_size) * group_K, group_K + group_L, 2 * tile_size)]

    return (; factors, row_permutation, group_K, group_L, group_k, group_l,
            factorization_matrix_storage, factorization_matrix_parts,
            factorization_matrix_parts_row_ranges, factorization_matrix_parts_col_ranges,
            factorization_locally_owned_rows,
            factorization_pivot_generation_distributed_tree_sizes,
            factorization_pivoting_buffer, factorization_local_left_panel_buffer,
            factorization_pivoting_reduction_buffer,
            factorization_pivoting_reduction_indices,
            factorization_pivoting_reduction_indices_local, factorization_source_rows,
            factorization_locally_owned_swap_rows, factorization_source_swap_labels,
            factorization_row_swap_buffers, factorization_swap_flags,
            factorization_shared_lu, comm_requests)
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    @sc_timeit A_lu.timer "lu!" begin
        n_tiles = A_lu.n_tiles
        shared_comm_rank = A_lu.shared_comm_rank
        synchronize_shared = A_lu.synchronize_shared

        redistribute_matrix!(A_lu, A)

        # Initialize row_permutation, which will be permuted as we generate the pivots.
        if shared_comm_rank == 0
            A_lu.row_permutation .= 1:A_lu.m
        end

        for panel ∈ 1:n_tiles
            generate_pivots!(A_lu, panel)
            apply_pivots_from_sub_column!(A_lu, panel)
            update_sub_panel_off_diagonals!(A_lu, panel)
            update_bottom_right_block!(A_lu, panel)
        end

        gather_factors!(A_lu)
        fill_ldiv_tiles!(A_lu)
    end

    return A_lu
end

# For parallelized LU factorization, each block of ranks owns a certain cyclic subset of
# tiles of the matrix, in the 'local buffers'.
function redistribute_matrix!(A_lu, A)
    @sc_timeit A_lu.timer "redistribute_matrix!" begin
        distributed_comm = A_lu.distributed_comm
        shared_comm_rank = A_lu.shared_comm_rank
        matrix_parts = A_lu.factorization_matrix_parts
        row_ranges = A_lu.factorization_matrix_parts_row_ranges
        col_ranges = A_lu.factorization_matrix_parts_col_ranges
        synchronize_shared = A_lu.synchronize_shared

        if size(A) != (A_lu.m, A_lu.n)
            error("Expected `A` to be a $((A_lu.m,A_lu.n)) matrix buffer on every rank")
        end

        if shared_comm_rank == 0
            # As this block is just copying a one matrix into another, it must be memory
            # bandwidth limited, so not sure that trying to use shared-memory processes would
            # speed it up. Anyway, it is unlikely to be the main bottleneck, so not worth
            # performance testing (unless it turns out to be a limiting factor).
            MPI.Bcast!(A, distributed_comm; root=0)

            for (tile_j,cr) ∈ enumerate(col_ranges), (tile_i, rr) ∈ enumerate(row_ranges)
                @views matrix_parts[tile_i,tile_j] .= A[rr,cr]
            end
        end

        synchronize_shared()
    end

    return nothing
end

# Gather factorized matrix from `matrix_parts` into `factors`. Could probably do this
# without the `MPI.Allreduce!()` by passing individual tiles in `fill_ldiv_tiles!()` (i.e.
# `fill_ldiv_tiles!()` would use data from the distributed `matrix_parts` directly, and we
# would never store `factors`).
function gather_factors!(A_lu)
    @sc_timeit A_lu.timer "gather_factors!" begin
        shared_comm_rank = A_lu.shared_comm_rank
        distributed_comm = A_lu.distributed_comm
        factors = A_lu.factors
        matrix_parts = A_lu.factorization_matrix_parts
        row_ranges = A_lu.factorization_matrix_parts_row_ranges
        col_ranges = A_lu.factorization_matrix_parts_col_ranges
        synchronize_shared = A_lu.synchronize_shared

        if shared_comm_rank == 0
            # As this block is just copying a value or another matrix into a matrix, it must
            # be memory bandwidth limited, so not sure that trying to use shared-memory
            # processes would speed it up. Anyway, it is unlikely to be the main bottleneck,
            # so not worth performance testing (unless it turns out to be a limiting factor).
            factors .= 0.0
            for (tile_j,cr) ∈ enumerate(col_ranges), (tile_i, rr) ∈ enumerate(row_ranges)
                @views factors[rr,cr] .= matrix_parts[tile_i,tile_j]
            end
            MPI.Allreduce!(factors, +, distributed_comm)
        end

        synchronize_shared()
    end

    return nothing
end

function get_n_rows_from_k(k, panel_k, panel_group_row, group_K, n_local_pivots, n_tiles, m, tile_size)
    # first_group_row is the global group-row index of the first group-row
    # owned by the rank with row-index k that contributes to this panel.
    if k < panel_k
        first_group_row = panel_group_row * group_K + k
    elseif k > panel_k
        first_group_row = (panel_group_row - 1) * group_K + k
    else
        return n_local_pivots
    end
    if first_group_row > n_tiles
        # Next tile owned by this rank would be off the bottom of the
        # matrix, so no rows.
        return 0
    elseif first_group_row == n_tiles
        # The next tile owned by this rank is the last one, which may not
        # be full height.
        return m - (n_tiles - 1) * tile_size
    else
        # This block cannot own only the last section, so it will provide
        # a full tile-worth of rows.
        return tile_size
    end
end

# Define this function independently of `generate_pivots!()` to ensure it is type-stable.
# We gather the pivot indices and columns onto the process with `group_k==panel_k`, so
# when calculating 'ranks' within this reduction process, we do so relative to that
# process.
function distributed_memory_tree_pivot_generation!(
             level; n_participating, tree_sizes, participating_proc_ks, panel_group_row,
             panel_k, group_k, group_l, group_K, m, tile_size, this_tile_size, n_pivots,
             shared_comm_rank, synchronize_shared, distributed_comm, reqs,
             reduction_buffer, pivoting_buffer, pivoting_reduction_buffer,
             pivoting_reduction_indices, pivoting_reduction_indices_local, shared_lu,
             check_lu)

    n_procs_this_level = tree_sizes[level]
    if n_procs_this_level == 1
        return nothing
    end

    # `participating_proc_ks` is passed a Vector{Int64} of ranks when not all processes
    # in the panel participate in the factorisation. In this case a single step of
    # reduction is done, from all participating processes to a single gathering
    # process.
    if participating_proc_ks === nothing
        rank_step = n_participating ÷ n_procs_this_level

        # Rank within the processes that are participating at this level.
        level_rank = (group_k - panel_k) ÷ rank_step

        n_procs_next_level = tree_sizes[level+1]
        # Note that by construction, `tree_sizes[level]` is always divisible by
        # `tree_sizes[level+1]`.
        subgroup_size = n_procs_this_level ÷ n_procs_next_level
        level_rank_offset = (level_rank + n_procs_this_level) % subgroup_size
        is_gathering_block = level_rank_offset == 0
        rank_offsets = 1:subgroup_size-1

        # When not all ranks in the group row are participating, the source rank never
        # needs to wrap back around to start again from 0, so the `% group_K` is not
        # needed, but does not hurt.  When all ranks are participating there are
        # `group_K` in total, and the rank may need to wrap around (when panel_k > 1).
        source_ks = [(group_k - 1 + r_offset * rank_step) % group_K + 1
                     for r_offset ∈ rank_offsets]
    else
        source_ks = @view participating_proc_ks[2:end]
        n_procs_next_level = 1
        subgroup_size = length(participating_proc_ks)
        is_gathering_block = (group_k == participating_proc_ks[1])
        rank_offsets = 1:length(participating_proc_ks)-1
    end

    if is_gathering_block
        # Note that the gathering process owns the diagonal tile, and we only use this
        # branch of the code when the diagonal tile is not the last one, so here the
        # diagonal tile is always full-sized.
        n_reduced_rows = this_tile_size
        for (r_offset, source_k) ∈ zip(rank_offsets, source_ks)
            # Usually we get this_tile_size rows from each source, but sometimes a
            # source that owns rows at the edge of the matrix might send fewer.
            if source_k < panel_k
                source_first_global_row = (panel_group_row * group_K + source_k - 1) * tile_size + 1
            else
                source_first_global_row = ((panel_group_row - 1) * group_K + source_k - 1) * tile_size + 1
            end
            n_rows_from_source = min(this_tile_size, m - source_first_global_row + 1)
            n_reduced_rows += n_rows_from_source
            if shared_comm_rank == 0
                source_offset = r_offset * this_tile_size

                source_r = (group_l - 1) * group_K + source_k - 1
                reqs[r_offset] =
                    MPI.Irecv!(@view(reduction_buffer[source_offset+1:(source_offset+n_rows_from_source),:]),
                               distributed_comm; source=source_r, tag=1)
                reqs[subgroup_size-1+r_offset] =
                    MPI.Irecv!(@view(pivoting_reduction_indices[source_offset+1:source_offset+n_rows_from_source]),
                               distributed_comm; source=source_r, tag=2)
            end
        end

        if shared_comm_rank == 0
            # Wait for all panel columns to arrive. Do not need indices yet, so wait for
            # those later.
            MPI.Waitall(reqs[1:subgroup_size-1])
        end

        # Need to keep the unfactorized columns, so copy the data into another buffer
        # for factorization.
        factorization_buffer =
            reshape(@view(pivoting_buffer[1:n_reduced_rows*this_tile_size]),
                    n_reduced_rows, this_tile_size)
        if shared_comm_rank == 0
            factorization_buffer .= @view reduction_buffer[1:n_reduced_rows,:]
        end

        synchronize_shared()

        if isa(shared_lu, AbstractVector{Int64})
            if shared_comm_rank == 0
                # `tile_size` is small, so doing a serial solve using LAPACK. In this
                # case `shared_lu` is a Vector{Int64} that can be used as the `ipiv`
                # argument.
                LAPACK.getrf!(factorization_buffer, shared_lu; check=check_lu)
            end
            ipiv = shared_lu
        else
            lu!(shared_lu, factorization_buffer)
            ipiv = shared_lu.ipiv
        end
        if shared_comm_rank == 0
            buffer_pivot_indices = ipiv2perm_truncated(ipiv, n_reduced_rows,
                                                       this_tile_size)

            # Wait for all indices to arrive.
            MPI.Waitall(reqs[subgroup_size:2*subgroup_size-2])

            # Use `pivoting_reduction_indices_local` as an intermediate to avoid having to
            # implicitly allocate a buffer to copy indices.
            pivoting_reduction_indices_local .= @view pivoting_reduction_indices[buffer_pivot_indices]
            @views pivoting_reduction_indices[1:this_tile_size] .= pivoting_reduction_indices_local

            if n_procs_next_level == 1
                # Copy the factorized diagonal_sub_tile into pivoting_reduction_buffer.
                @views reshape(pivoting_reduction_buffer[1:this_tile_size*this_tile_size],
                               this_tile_size, this_tile_size) .=
                    factorization_buffer[1:this_tile_size,1:this_tile_size]
            else
                # Apply row swaps to reduction buffer so that the pivot rows are moved
                # to the first `this_tile_size` rows.
                # Could parallelise this with shared-memory...
                ColumnPivotLUs.apply_row_swaps!(reduction_buffer, ipiv,
                                                this_tile_size, this_tile_size)
            end
        end

        return distributed_memory_tree_pivot_generation!(
                   level + 1; n_participating, tree_sizes, participating_proc_ks=nothing,
                   panel_group_row, panel_k, group_k, group_l, group_K, m, tile_size,
                   this_tile_size, n_pivots, shared_comm_rank, synchronize_shared,
                   distributed_comm, reqs, reduction_buffer, pivoting_buffer,
                   pivoting_reduction_buffer, pivoting_reduction_indices,
                   pivoting_reduction_indices_local, shared_lu, check_lu)
    else
        if shared_comm_rank == 0
            # Send pivot rows and indices to the gathering process.
            if participating_proc_ks === nothing
                gathering_rank_k = (group_k - 1 - level_rank_offset * rank_step + group_K) % group_K + 1
            else
                gathering_rank_k = participating_proc_ks[1]
            end
            gathering_rank = (group_l - 1) * group_K + gathering_rank_k - 1
            reqs[1] = MPI.Isend(@view(reduction_buffer[1:n_pivots,:]),
                                distributed_comm; dest=gathering_rank, tag=1)
            reqs[2] = MPI.Isend(@view(pivoting_reduction_indices[1:n_pivots]),
                                distributed_comm; dest=gathering_rank, tag=2)
            MPI.Waitall(reqs[1:2])
        end
        return nothing
    end

    return nothing
end

function generate_pivots!(A_lu, panel)
    @sc_timeit A_lu.timer "generate_pivots!" begin
        group_l = A_lu.group_l
        group_L = A_lu.group_L
        if (panel - 1) % group_L + 1 != group_l
            # This rank does not participate in pivot generation for this panel.
            return nothing
        end
        distributed_comm = A_lu.distributed_comm
        distributed_comm_rank = A_lu.distributed_comm_rank
        shared_comm_rank = A_lu.shared_comm_rank
        shared_comm_size = A_lu.shared_comm_size
        reqs = A_lu.comm_requests
        m = A_lu.m
        n_tiles = A_lu.n_tiles
        tile_size = A_lu.tile_size
        group_k = A_lu.group_k
        group_K = A_lu.group_K
        pivot_generation_distributed_tree_sizes = A_lu.factorization_pivot_generation_distributed_tree_sizes
        pivoting_buffer = A_lu.factorization_pivoting_buffer
        pivoting_reduction_buffer = A_lu.factorization_pivoting_reduction_buffer
        pivoting_reduction_indices = A_lu.factorization_pivoting_reduction_indices
        pivoting_reduction_indices_local = A_lu.factorization_pivoting_reduction_indices_local
        matrix_storage = A_lu.factorization_matrix_storage
        locally_owned_rows = A_lu.factorization_locally_owned_rows
        shared_lu = A_lu.factorization_shared_lu
        check_lu = A_lu.check_lu
        synchronize_shared = A_lu.synchronize_shared

        # Find the on-or-below diagonal part of the sub-column that is owned by this rank.
        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        if group_k < panel_k
            first_local_row = panel_group_row * tile_size + 1
        else
            first_local_row = (panel_group_row - 1) * tile_size + 1
        end
        last_local_row = size(matrix_storage, 1)
        first_local_col = (panel_group_col - 1) * tile_size + 1
        last_local_col = min(panel_group_col * tile_size, size(matrix_storage, 2))
        this_tile_size = last_local_col - first_local_col + 1
        total_local_rows = max(last_local_row - first_local_row + 1, 0)

        # Copy data into pivoting_buffer to be LU factorized.
        total_local_cols = max(last_local_col - first_local_col + 1, 0)
        copy_cols_per_rank = (total_local_cols + shared_comm_size - 1) ÷ shared_comm_size
        copy_col_range = shared_comm_rank*copy_cols_per_rank+first_local_col:min((shared_comm_rank+1)*copy_cols_per_rank,total_local_cols)+first_local_col-1
        copy_ncol = length(copy_col_range)
        if copy_ncol > 0
            copy_buffer_offset = shared_comm_rank * total_local_rows * copy_cols_per_rank
            copy_buffer_size = copy_ncol * total_local_rows
            copy_pivot_buffer = reshape(@view(pivoting_buffer[copy_buffer_offset+1:copy_buffer_offset+copy_buffer_size]),
                                        total_local_rows, copy_ncol)
            copy_pivot_buffer .= @view matrix_storage[first_local_row:last_local_row,copy_col_range]
        end
        synchronize_shared()

        # Within each shared-memory block, we do LU factorisation directly, with a solver that
        # is parallelised if `tile_size > 64` (although the parallelisation is not likely to
        # be very efficient unless the tile size is much larger, but this is OK because the
        # serial factorisation of an mx64 column is so efficient that it is hard to make it
        # faster by parallelising (LAPACK/BLAS probably gets less than a factor 2 speedup with
        # threads for this shape of matrix).
        # If there is more than one shared-memory block among the rows (`group_K > 1`), then
        # between shared-memory blocks (over distributed-MPI) do 'tournament pivoting'.
        # -> Root processes of the shared-memory blocks (the processes in `distributed_comm`)
        #    do a 'binary' tree reduction to reduce `distributed_comm_size * this_tile_size`
        #    pivots to `this_tile_size` pivots.
        # When there are not enough pivots to have `2*this_tile_size` columns per process,
        # some steps will be skipped until the number of processes participating is small
        # enough.

        # Factorise the locally-owned column.
        # The memory contained in `pivot_buffer` has already been filled with the entries from
        # `matrix_storage`.
        pivot_buffer = reshape(@view(pivoting_buffer[1:total_local_rows*this_tile_size]),
                               total_local_rows, this_tile_size)
        if isa(shared_lu, AbstractVector{Int64})
            if shared_comm_rank == 0
                # `tile_size` is small, so doing a serial solve using LAPACK. In this case
                # `shared_lu` is a Vector{Int64} that can be used as the `ipiv` argument.
                LAPACK.getrf!(pivot_buffer, shared_lu; check=check_lu)
            end
            ipiv = shared_lu
            synchronize_shared()
        else
            lu!(shared_lu, pivot_buffer)
            ipiv = shared_lu.ipiv
        end

        n_pivots = min(this_tile_size, total_local_rows)
        if shared_comm_rank == 0
            # Initial set of indices that will be permuted as pivoting progresses.
            local_pivot_indices = @view pivoting_reduction_indices_local[1:n_pivots]
            local_pivot_indices .=
                ipiv2perm_truncated(ipiv, total_local_rows, n_pivots) .+ (first_local_row - 1)
        end

        reduction_buffer = @view pivoting_reduction_buffer[1:this_tile_size * group_K,
                                                           1:this_tile_size]

        if group_K == 1
            # All rows are local to the block, so no need for further reduction,
            # factorisation, or local->global index conversion. Just need to copy the
            # factorized diagonal_sub_tile into pivoting_reduction_buffer.
            if group_k == panel_k && shared_comm_rank == 0
                @views reshape(pivoting_reduction_buffer[1:this_tile_size^2], this_tile_size,
                        this_tile_size) .=
                    pivot_buffer[1:this_tile_size,1:this_tile_size]
                @views pivoting_reduction_indices[1:n_pivots] .= locally_owned_rows[local_pivot_indices]
            end
        else
            if shared_comm_rank == 0
                # Define a reduction buffer. Note that we just reshape the full
                # `pivoting_reduction_buffer`, but only use some number of rows at the
                # beginning of the buffer - usually this buffer is 'too big'. Similarly
                # `pivoting_reduction_indices` is 'too big'.
                local_pivot_indices = @view pivoting_reduction_indices_local[1:n_pivots]
                # Convert the 'local' pivot indices to global ones.
                # Copy the local matrix rows into a reduction buffer.
                for i ∈ 1:n_pivots
                    local_pivot = local_pivot_indices[i]
                    pivoting_reduction_indices[i] = locally_owned_rows[local_pivot]
                    @views reduction_buffer[i,:] .=
                        matrix_storage[local_pivot,first_local_col:last_local_col]
                end
            end

            group_rows_in_panel = n_tiles - panel + 1
            if group_rows_in_panel < group_K
                # Not all distributed ranks in the group-column own entries in the left
                # panel. It would be silly to send zero-size messages. This should only
                # happen for very small panels, so efficiency is not too important. To
                # avoid needing to factorize `group_rows_in_panel`, set up a tree which
                # just reduces to one process in a single step.
                tree_sizes = [group_rows_in_panel, 1]
                n_participating = group_rows_in_panel
                is_participating = n_pivots > 0
                participating_proc_ks = [(k - 1) % group_K + 1
                                         for k ∈ panel_k:panel_k+n_participating-1]
            else
                tree_sizes = pivot_generation_distributed_tree_sizes
                n_participating = group_K
                is_participating = true
                participating_proc_ks = nothing
            end

            if is_participating
                distributed_memory_tree_pivot_generation!(
                    1; n_participating, tree_sizes, participating_proc_ks, panel_group_row,
                    panel_k, group_k, group_l, group_K, m, tile_size, this_tile_size,
                    n_pivots, shared_comm_rank, synchronize_shared, distributed_comm, reqs,
                    reduction_buffer, pivoting_buffer, pivoting_reduction_buffer,
                    pivoting_reduction_indices, pivoting_reduction_indices_local, shared_lu,
                    check_lu)

                if tree_sizes[1] == 1 && shared_comm_rank == 0
                    # `distributed_memory_tree_pivot_geneation!()` will return immediately, so
                    # need to copy diagonal tile into pivoting_reduction_buffer.
                    @views reshape(pivoting_reduction_buffer[1:this_tile_size^2], this_tile_size,
                                   this_tile_size) .=
                        pivot_buffer[1:this_tile_size,1:this_tile_size]
                end
            end
        end

        synchronize_shared()
    end

    return nothing
end

function apply_pivots_from_sub_column!(A_lu, panel)
    @sc_timeit A_lu.timer "apply_pivots_from_sub_column!" begin
        row_permutation = A_lu.row_permutation
        distributed_comm = A_lu.distributed_comm
        distributed_comm_rank = A_lu.distributed_comm_rank
        shared_comm_rank = A_lu.shared_comm_rank
        reqs = A_lu.comm_requests
        m = A_lu.m
        tile_size = A_lu.tile_size
        group_K = A_lu.group_K
        group_l = A_lu.group_l
        group_L = A_lu.group_L
        matrix_storage = A_lu.factorization_matrix_storage
        row_swap_buffers = A_lu.factorization_row_swap_buffers
        swap_flags = A_lu.factorization_swap_flags

        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        first_global_col = (panel - 1) * tile_size + 1
        last_global_col = min(panel * tile_size, m)
        this_tile_size = last_global_col - first_global_col + 1
        local_diagonal_tile_offset = (panel_group_row - 1) * tile_size

        # Broadcast the pivot indices for this sub column to all ranks.
        sub_column_pivot_indices = @view A_lu.factorization_pivoting_reduction_indices[1:this_tile_size]
        if shared_comm_rank == 0
            diagonal_distributed_rank = (panel_l - 1) * group_K + panel_k - 1
            MPI.Bcast!(sub_column_pivot_indices, distributed_comm;
                       root=diagonal_distributed_rank)
        end

        # As this function is just array copies and communication, we do not use any
        # shared-memory parallelism.
        if shared_comm_rank == 0
            diagonal_block_indices = (panel-1)*tile_size+1:min(panel*tile_size, m)
            first_diag = diagonal_block_indices[1]
            last_diag = diagonal_block_indices[end]

            # By default, just send the row from the diagonal block back to the original
            # position of the 'pivot row' that is being moved into the diagonal block.
            # However, if the row in the diagonal block is itself a 'pivot row', then we need
            # to send it to the correct position within the diagonal block. Here we set up
            # `diagonal_block_row_destination_indices` with the position where each row of the
            # diagonal block should be sent to.
            sorted_pivot_indices = sort(sub_column_pivot_indices)

            # All pivot indices are at or below the first row of the diagonal block, so we can
            # find the indices that are within the diagonal block from the first entries of
            # the sorted list of pivot indices.
            split_ind = searchsortedlast(sorted_pivot_indices, last_diag)
            pivot_indices_inside_diagonal_block = sorted_pivot_indices[1:split_ind]
            pivot_indices_outside_diagonal_block = sorted_pivot_indices[split_ind+1:end]
            outside_diagonal_block_counter = 1

            diagonal_block_row_destination_indices = copy(sub_column_pivot_indices)
            for i ∈ 1:length(sub_column_pivot_indices)
                diag_index = diagonal_block_indices[i]
                if diag_index ∈ pivot_indices_inside_diagonal_block
                    # Pivot destination of this row is within the diagonal block, so send it
                    # directly.
                    i2 = findfirst(isequal(diag_index), sub_column_pivot_indices)
                    diagonal_block_row_destination_indices[i] = diagonal_block_indices[i2]
                else
                    # The destination of this row must be set to something that is outside the
                    # diagonal block.
                    diagonal_block_row_destination_indices[i] =
                        pivot_indices_outside_diagonal_block[outside_diagonal_block_counter]
                    outside_diagonal_block_counter += 1
                end
            end

            # Permute the global pivot vector with the current panel's pivot indices.
            diag_tile_global_inds = (panel-1)*tile_size+1:min(panel*tile_size,m)
            diag_tile_unpermuted_inds = row_permutation[diag_tile_global_inds]
            pivot_rows_unpermuted_inds = row_permutation[sub_column_pivot_indices]
            # Need to use diagonal_block_row_destination_indices to avoid errors when some of
            # the pivot rows are within the diagonal block.
            row_permutation[diagonal_block_row_destination_indices] .=
                diag_tile_unpermuted_inds
            row_permutation[diag_tile_global_inds] .= pivot_rows_unpermuted_inds

            # We aim to minimise the number of copy operations needed to complete the
            # effect of all row swaps from this panel. As we want only one set of MPI
            # communications, we cannot directly use `ipiv` as generated by LAPACK, as
            # that would mean doing sequentially a set of row swaps. Instead we have
            # generated the complete set of sources and destinations of each diagonal row.
            # To make sure all operations are completed, we define a vector of flags for
            # each diagonal row involved in the swaps (any off-diagonal rows owned by the
            # diagonal-owning rank are always to be swapped to/from diagonal rows, and
            # off-diagonal rows on other ranks the row must be both sent and received over
            # MPI so in either case we do not need to track their states) where the flag
            # can indicate that the buffer is not yet updated, has been emptied (into a
            # row buffer or transferred to another row), or has been completed (final data
            # has been transferred in, or no change is needed).
            # We first set up the MPI communications that are needed:
            #   * For every row sent and/or received over MPI, copy the original data into
            #     a row buffer, then start an MPI.Isend of the data in the row buffer
            #     and/or an MPI.Irecv into the row.
            #   * If this diagonal row receieves over MPI, mark row as completed, as no
            #     other data needs to be transferred into the row, we only have to wait
            #     for all MPI transfers to complete, which we will do at the end of this
            #     function.
            #   * If the diagonal row sends but does not receive, mark the row as empty,
            #     as the data has been transferred out into the row buffer, but the row is
            #     not filled with the final data yet.
            #   * If the row has no MPI communications, but does not need any swap, mark
            #     as completed.
            # In second and third passes that only involves the diagonal-owning rank, we
            # transfer all the locally-owned data as needed. Second pass:
            #   * Loop through the diagonal-tile rows until the first empty row is found
            #     (restarting the loop on the following row after processing each empty
            #     row). Note that rows outside the diagonal tile cannot be 'empty'.
            #   * Starting from that first empty row, transfer the data from the 'source'
            #     row into the empty row, and mark the row as 'completed'. Since the
            #     source row is now empty, transfer data into that from its source (and
            #     mark as completed), continuing until data was transferred from a
            #     'complete' row (when the source row is 'complete' the data has to be
            #     transferred from the row buffer, not from `matrix_storage`).
            # After the second pass there are no empty rows, but it may be that there are
            # rows that are not 'complete' because they are part of a cycle of swaps among
            # rows owned by the diagonal-owning rank. Therefore we need a third pass:
            #   * Loop through the diagonal rows until the first incomplete row
            #     (restarting the loop from the row after this until there are no more
            #     incomplete rows). Note that rows outside the diagonal tile cannot be
            #     'incomplete'.
            #   * Copy the incomplete row into a row buffer (and mark it as complete), and
            #     save which row it is.  Copy the data from this rows source into the row.
            #     Move to the now-empty source and copy its source into it (and mark it as
            #     complete), repeating until the source is the original 'incomplete' row,
            #     whose data is copied from the row buffer.
            # After the third pass, all rows should be completed. Note that on each pass,
            # we loop through the rows only once, so the number of operations increases
            # only linearly with `tile_size` (the maximum number of rows involved in the
            # swaps is `2*tile_size`).
            # This method involves more loops and more checking flags than the
            # `ipiv`-based method, but reduces the amount of data copied. It is not
            # obvious which would be faster in serial, but the method used here is
            # preferred because it minimises the number of MPI data transfers, and it is
            # not clear how to mix single-transfer copies for MPI operations with
            # sequential row-swap operations for local transfers, without some similar
            # amount of flag-checking.

            # First pass - queue up MPI communications.
            ###########################################
            req_counter = 0
            # Flag 0x0 means 'complete', 0x1 means 'empty' and 0x2 means 'incomplete'.
            swap_flags .= 0x2
            owning_rank_k_diag = (panel - 1) % group_K
            owning_rank_diag = (group_l - 1) * group_K + owning_rank_k_diag

            source_rows = @view A_lu.factorization_source_rows[1:2*this_tile_size]
            source_rows .= -1
            # We also need to record which locally-owned rows are involved in the swaps.
            locally_owned_swap_rows = A_lu.factorization_locally_owned_swap_rows
            @views locally_owned_swap_rows[1:this_tile_size] .=
                local_diagonal_tile_offset+1:local_diagonal_tile_offset+this_tile_size
            @views locally_owned_swap_rows[this_tile_size+1:end] .= -1
            n_off_diagonal_dest_rows = 0

            for (iswap, (idiag, isource, idest)) ∈
                    enumerate(zip(diagonal_block_indices, sub_column_pivot_indices,
                                  diagonal_block_row_destination_indices))
                if idiag == isource == idest
                    # No swap to do.
                    swap_flags[iswap] = 0x0
                    continue
                end

                diag_tile_row_offset = (idiag - 1) % tile_size + 1

                source_tile_row, source_tile_row_offset = divrem(isource - 1, tile_size) .+ 1
                owning_rank_k_source = (source_tile_row - 1) % group_K
                owning_rank_source = (group_l - 1) * group_K + owning_rank_k_source

                dest_tile_row, dest_tile_row_offset = divrem(idest - 1, tile_size) .+ 1
                owning_rank_k_dest = (dest_tile_row - 1) % group_K
                owning_rank_dest = (group_l - 1) * group_K + owning_rank_k_dest

                if distributed_comm_rank == owning_rank_diag == owning_rank_source == owning_rank_dest
                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset
                    source_rows[iswap] = source_storage_row
                    if dest_tile_row > panel_group_row
                        # The 'destination row' is owned by the diagonal-tile-owning rank,
                        # but is not in the diagonal tile. Therefore the 'source row' of
                        # 'destination row' is within the diagonal tile, and we want to
                        # record which row it is.
                        n_off_diagonal_dest_rows += 1
                        diag_storage_row = ((panel - 1) ÷ group_K) * tile_size + diag_tile_row_offset
                        source_rows[this_tile_size+n_off_diagonal_dest_rows] = diag_storage_row
                        dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset
                        locally_owned_swap_rows[this_tile_size+n_off_diagonal_dest_rows] = dest_storage_row
                    end
                elseif distributed_comm_rank == owning_rank_diag == owning_rank_source
                    # Copy out data from the row and start MPI.Isend()

                    diag_storage_row = ((panel - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    row_swap_buffer = @view row_swap_buffers[iswap,:]
                    row_swap_buffer .= @view matrix_storage[diag_storage_row,:]

                    # diag -> dest
                    reqs[req_counter+=1] = MPI.Isend(row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_dest, tag=iswap)

                    # Mark as 'empty'.
                    swap_flags[iswap] = 0x1

                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset
                    source_rows[iswap] = source_storage_row
                elseif distributed_comm_rank == owning_rank_diag == owning_rank_dest
                    # Copy out data from the row that we own, to later copy to destination
                    # row.

                    diag_storage_row = ((panel - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    diag_row_data = @view matrix_storage[diag_storage_row,:]

                    row_swap_buffer = @view row_swap_buffers[iswap,:]
                    row_swap_buffer .= diag_row_data

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, distributed_comm;
                                                      source=owning_rank_source, tag=iswap)

                    # Mark as 'complete' because it will be complete once MPI
                    # communications have completed.
                    swap_flags[iswap] = 0x0

                    if dest_tile_row > panel_group_row
                        # The 'destination row' is owned by the diagonal-tile-owning rank,
                        # but is not in the diagonal tile. Therefore the 'source row' of
                        # 'destination row' is within the diagonal tile, and we want to
                        # record which row it is.
                        n_off_diagonal_dest_rows += 1
                        source_rows[this_tile_size+n_off_diagonal_dest_rows] = diag_storage_row
                        dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset
                        locally_owned_swap_rows[this_tile_size+n_off_diagonal_dest_rows] = dest_storage_row
                    end
                elseif distributed_comm_rank == owning_rank_diag
                    diag_storage_row = ((panel - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    diag_row_data = @view matrix_storage[diag_storage_row,:]

                    # Copy out data from the row that we own, to later copy to destination
                    # row.
                    row_swap_buffer = @view row_swap_buffers[iswap,:]
                    row_swap_buffer .= diag_row_data

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, distributed_comm;
                                                      source=owning_rank_source, tag=iswap)
                    # diag -> dest
                    reqs[req_counter+=1] = MPI.Isend(row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_dest, tag=iswap)

                    # Mark as 'complete' because it will be complete once MPI
                    # communications have completed.
                    swap_flags[iswap] = 0x0
                elseif distributed_comm_rank == owning_rank_source
                    # Copy out data from the row that we own, to send to owning_rank_diag.

                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset

                    row_swap_buffer = @view row_swap_buffers[iswap,:]
                    row_swap_buffer .= @view matrix_storage[source_storage_row,:]

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Isend(row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_diag, tag=iswap)
                end
            end
            if distributed_comm_rank != owning_rank_diag
                # Now that all rows on this rank have been copied out into buffers to
                # send, can call MPI.IRecv!() for all of them.
                for (iswap, (idiag, isource, idest)) ∈
                        enumerate(zip(diagonal_block_indices, sub_column_pivot_indices,
                                      diagonal_block_row_destination_indices))
                    if idiag == isource == idest
                        # No swap to do.
                        continue
                    end

                    panel = (idiag - 1) ÷ tile_size + 1
                    owning_rank_k_diag = (panel - 1) % group_K

                    dest_tile_row, dest_tile_row_offset = divrem(idest - 1, tile_size) .+ 1
                    owning_rank_k_dest = (dest_tile_row - 1) % group_K
                    owning_rank_dest = (group_l - 1) * group_K + owning_rank_k_dest

                    if distributed_comm_rank == owning_rank_dest
                        dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset

                        # source -> diag
                        reqs[req_counter+=1] =
                            MPI.Irecv!(@view(matrix_storage[dest_storage_row,:]),
                                       distributed_comm; source=owning_rank_diag, tag=iswap)
                    end
                end
            end

            if distributed_comm_rank == owning_rank_diag
                source_rows = @view source_rows[1:this_tile_size+n_off_diagonal_dest_rows]
                locally_owned_swap_rows = @view locally_owned_swap_rows[1:this_tile_size+n_off_diagonal_dest_rows]
                swap_flags = @view swap_flags[1:this_tile_size+n_off_diagonal_dest_rows]
                source_swap_labels = @view A_lu.factorization_source_swap_labels[1:this_tile_size+n_off_diagonal_dest_rows]

                # Figure out the index within locally_owned_swap_rows of each source row, and
                # store in source_swap_labels.
                for (i, source_row) ∈ enumerate(source_rows)
                    # Operation like `findfirst()`, but some entries will not be found.
                    # `findfirst()` returns `nothing` when the entry is not found, but
                    # that would introduce type instability, so search with a loop
                    # instead.
                    source_label = -1
                    for i ∈ 1:length(locally_owned_swap_rows)
                        if locally_owned_swap_rows[i] == source_row
                            source_label = i
                            break
                        end
                    end
                    source_swap_labels[i] = source_label
                end

                # Second pass - fill all chains that start with an empty row.
                #############################################################
                for next_potential_empty_row ∈ 1:this_tile_size
                    if swap_flags[next_potential_empty_row] == 0x1
                        label = next_potential_empty_row
                        source_label = source_swap_labels[label]
                        while true
                            if swap_flags[source_label] == 0x0
                                # 'Completed' row, which must be within the diagonal tile.
                                # Copy data from the row swap buffers.
                                irow = locally_owned_swap_rows[label]
                                @views matrix_storage[irow,:] .=
                                    row_swap_buffers[source_label,:]
                                swap_flags[label] = 0x0
                                break
                            elseif swap_flags[source_label] == 0x2
                                # 'Not-completed' row, copy data matrix storage.
                                irow = locally_owned_swap_rows[label]
                                isource = locally_owned_swap_rows[source_label]
                                @views matrix_storage[irow,:] .=
                                    matrix_storage[isource,:]
                                swap_flags[label] = 0x0
                                label = source_label
                                source_label = source_swap_labels[label]
                            else
                                error("Chain did not terminate in a completed row. This "
                                      * "should never happen.")
                            end
                        end
                    end
                end

                # Third pass - move data around chains that are cycles of incomplete rows.
                ##########################################################################
                for next_potential_incomplete_label ∈ 1:this_tile_size
                    if swap_flags[next_potential_incomplete_label] == 0x2
                        row_buffer = @view row_swap_buffers[next_potential_incomplete_label,:]
                        # Copy out data from this row so that we can put it back into the last
                        # row in the cycle.
                        irow = locally_owned_swap_rows[next_potential_incomplete_label]
                        row_buffer .= @view matrix_storage[irow,:]

                        label = next_potential_incomplete_label
                        next_label = source_swap_labels[label]
                        swap_flags[next_potential_incomplete_label] = 0x0
                        while next_label != next_potential_incomplete_label
                            this_row = locally_owned_swap_rows[label]
                            next_row = locally_owned_swap_rows[next_label]
                            @views matrix_storage[this_row,:] .= matrix_storage[next_row,:]
                            swap_flags[next_label] = 0x0
                            label = next_label
                            next_label = source_swap_labels[label]
                        end

                        # Copy in the data from start_row to this final row to complete the
                        # cycle.
                        this_row = locally_owned_swap_rows[label]
                        @views matrix_storage[this_row,:] .= row_buffer
                    end
                end
            end
            MPI.Waitall(reqs[1:req_counter])
        end
    end

    return nothing
end

function get_shared_local_row_range(A_lu, first_local_row)
    shared_comm_rank = A_lu.shared_comm_rank
    shared_comm_size = A_lu.shared_comm_size

    total_local_rows = size(A_lu.factorization_matrix_storage, 1)
    n_local_rows = total_local_rows - first_local_row + 1
    rows_per_rank = (n_local_rows + shared_comm_size - 1) ÷ shared_comm_size
    offset = shared_comm_rank * rows_per_rank
    return first_local_row+offset:min(offset+first_local_row+rows_per_rank-1,total_local_rows), offset
end

function get_shared_local_col_range(A_lu, first_local_col)
    shared_comm_rank = A_lu.shared_comm_rank
    shared_comm_size = A_lu.shared_comm_size

    total_local_cols = size(A_lu.factorization_matrix_storage, 2)
    n_local_cols = total_local_cols - first_local_col + 1
    cols_per_rank = (n_local_cols + shared_comm_size - 1) ÷ shared_comm_size
    offset = shared_comm_rank * cols_per_rank
    return first_local_col+offset:min(offset+first_local_col+cols_per_rank-1,total_local_cols), offset
end

function update_sub_panel_off_diagonals!(A_lu, panel)
    @sc_timeit A_lu.timer "update_sub_panel_off_diagonals!" begin
        n_tiles = A_lu.n_tiles
        distributed_comm_rank = A_lu.distributed_comm_rank
        shared_comm_rank = A_lu.shared_comm_rank
        pivoting_reduction_buffer = A_lu.factorization_pivoting_reduction_buffer
        matrix_parts = A_lu.factorization_matrix_parts
        group_K = A_lu.group_K
        group_L = A_lu.group_L
        shared_comm_size = A_lu.shared_comm_size
        synchronize_shared = A_lu.synchronize_shared

        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        diagonal_distributed_rank = (panel_l - 1) * group_K + panel_k - 1
        if panel == n_tiles
            # No remaining matix to update, so no need for communication or off-diagonal
            # update. Only need to copy LU-factorized block into matrix_storage
            if shared_comm_rank == 0
                if distributed_comm_rank == diagonal_distributed_rank
                    this_part = matrix_parts[panel_group_row,panel_group_col]
                    this_tile_size = size(this_part, 1)
                    diagonal_sub_tile =
                        reshape(@view(pivoting_reduction_buffer[1:this_tile_size*this_tile_size]),
                                this_tile_size, this_tile_size)
                    # Copy diagonal_sub_tile into the matrix storage.
                    this_part .= diagonal_sub_tile
                end
            end
            return nothing
        end
        tile_size = A_lu.tile_size
        distributed_comm = A_lu.distributed_comm
        matrix_storage = A_lu.factorization_matrix_storage
        reqs = A_lu.comm_requests
        group_k = A_lu.group_k
        group_l = A_lu.group_l
        row_buffers = A_lu.factorization_row_swap_buffers
        # Can reuse this buffer as a column buffer, as it is big enough.
        col_buffers = A_lu.factorization_pivoting_buffer
        local_col_buffer_storage = A_lu.factorization_local_left_panel_buffer

        first_panel_col = (panel_group_col - 1) * tile_size + 1
        last_panel_col = min(panel_group_col * tile_size, size(matrix_storage, 2))
        this_tile_size = last_panel_col - first_panel_col + 1
        # Note that the size of the diagonal tile that we communicate/process here is always
        # `(tile_size,tile_size)`. The tile can only be smaller than `tile_size` when it is
        # the last diagonal tile, but that one does not need communication/processing here. So
        # unlike some other functions, we do not need to calculate `this_tile_size` - we can
        # just use `tile_size`.

        req_counter = 0
        # While updating the sub-column, we also distribute the diagonal sub-tile to ranks
        # on the same sub-row as the diagonal sub-tile. This can be done in parallel with
        # updating the rest of the sub-column.
        #
        # It would be nicer to do these communications with collective MPI calls (e.g.
        # `MPI.Bcast!()`), but that would require many communicators (one for each pattern
        # of operation), which would be complicated to keep track of. There is also a
        # hard-coded limit on the maximum number of MPI communicators, so it is best not
        # to create them too freely.

        if distributed_comm_rank == diagonal_distributed_rank
            # The top tile_size*tile_size part of pivoting_reduction_buffer on
            # this rank contains the diagonal sub-tile that was calculated in
            # `generate_pivots!()`.
            diagonal_sub_tile =
                reshape(@view(pivoting_reduction_buffer[1:tile_size*tile_size]),
                        tile_size, tile_size)

            if shared_comm_rank == 0
                # Send the diagonal sub-tile to the other ranks in this sub-column.
                # Send to below-diagonal ranks in the group first, as these have slightly more
                # work to do.
                rank_offset = (group_l - 1) * group_K # Offset of ranks in sub-column.
                for k ∈ vcat(panel_k+1:group_K, 1:panel_k-1)
                    r = rank_offset + k - 1
                    # MPI.jl doesn't like ReshapedArray type, but we only need to communicate
                    # the underlying storage, so use `parent()` to extract a SubArray which
                    # MPI.jl can handle.
                    reqs[req_counter+=1] = MPI.Isend(parent(diagonal_sub_tile),
                                                     distributed_comm; dest=r)
                end

                # Send the diagonal sub-tile to the ranks in the same sub-row.
                for l ∈ vcat(panel_l+1:group_L, 1:panel_l-1)
                    r = (l - 1) * group_K + group_k - 1
                    # MPI.jl doesn't like ReshapedArray type, but we only need to communicate
                    # the underlying storage, so use `parent()` to extract a SubArray which
                    # MPI.jl can handle.
                    reqs[req_counter+=1] = MPI.Isend(parent(diagonal_sub_tile),
                                                     distributed_comm; dest=r)
                end
            end
        elseif group_k == panel_k
            # Receive the diagonal sub-block.
            diagonal_sub_tile =
                reshape(@view(pivoting_reduction_buffer[1:tile_size*tile_size]),
                        tile_size, tile_size)
            if shared_comm_rank == 0
                # MPI.jl doesn't like ReshapedArray type, but we only need to communicate the
                # underlying storage, so use `parent()` to extract a SubArray which MPI.jl can
                # handle.
                MPI.Recv!(parent(diagonal_sub_tile), distributed_comm;
                          source=diagonal_distributed_rank)
            end
        elseif group_l == panel_l
            diagonal_sub_tile =
                reshape(@view(pivoting_reduction_buffer[1:tile_size*tile_size]),
                        tile_size, tile_size)
            if shared_comm_rank == 0
                # MPI.jl doesn't like ReshapedArray type, but we only need to communicate the
                # underlying storage, so use `parent()` to extract a SubArray which MPI.jl can
                # handle.
                MPI.Recv!(parent(diagonal_sub_tile), distributed_comm;
                          source=diagonal_distributed_rank)
            end
        end

        if shared_comm_rank == 0 && distributed_comm_rank == diagonal_distributed_rank
            # Copy diagonal_sub_tile into the matrix storage.
            matrix_parts[panel_group_row,panel_group_col] .= diagonal_sub_tile
        end

        synchronize_shared()

        if group_l == panel_l
            # This branch now includes the rank that owns the diagonal sub-tile.

            # diagonal_sub_tile now contains both the L and U factors of the current
            # diagonal sub-tile. Now need to apply U^-1 from the right to the
            # locally-owned part of the sub-column that is below the diagonal sub-tile.
            if group_k ≤ panel_k
                first_storage_row = panel_group_row * tile_size + 1
            else
                first_storage_row = (panel_group_row - 1) * tile_size + 1
            end
            first_storage_col = (panel_group_col - 1) * tile_size + 1
            last_storage_col = min(panel_group_col * tile_size, size(matrix_storage, 2))

            shared_local_row_range, row_offset = get_shared_local_row_range(A_lu, first_storage_row)

            if length(shared_local_row_range) > 0
                n_local_rows = length(shared_local_row_range)
                n_rows = size(matrix_storage, 1) - first_storage_row + 1
                n_cols = last_storage_col - first_storage_col + 1
                buffer_size = n_local_rows * n_cols

                # Copy matrix entries into contiguous buffer to improve efficiency.
                col_buffer =
                    @view(reshape(@view(col_buffers[1:n_rows*n_cols]),
                                  n_rows, n_cols)[row_offset+1:row_offset+n_local_rows,:])
                if shared_comm_size > 1
                    # Need a local buffer which is contiguous in memory to copy into, because
                    # trsm!() cannot handle the non-contiguous col_buffer.
                    local_col_buffer = reshape(@view(local_col_buffer_storage[1:buffer_size]),
                                               n_local_rows, n_cols)
                else
                    # No shared memory parallelism, so we do not split up col_buffer here, so
                    # there is no need for local buffers.
                    local_col_buffer = col_buffer
                end

                local_below_diagonal_sub_column =
                    @view matrix_storage[shared_local_row_range,
                                         first_storage_col:last_storage_col]

                local_col_buffer .= local_below_diagonal_sub_column

                # Need to solve M*U=A for M, where A are the original matrix elements of the
                # sub-column, and U is the upper-triangular factor of the diagonal sub-tile.
                trsm!('R', 'U', 'N', 'N', 1.0, diagonal_sub_tile, local_col_buffer)

                # Copy buffer back into shared_storage and matrix storage.
                if shared_comm_size > 1
                    col_buffer .= local_col_buffer
                end
                local_below_diagonal_sub_column .= local_col_buffer
            end
        end

        if group_k == panel_k
            # This branch now includes the rank that owns the diagonal sub-tile.

            # The whole sub-row to the right of the diagonal sub-block needs to be updated
            # by solving L*M=A to find M, where A are the original matrix entries and L is
            # the lower-triangular factor from the diagonal sub-block.

            # Locally-owned rows/column to be copied into row_buffer
            first_storage_row = (panel_group_row - 1) * tile_size + 1
            last_storage_row = min(panel_group_row * tile_size, size(matrix_storage, 1))
            if group_l ≤ panel_l
                first_storage_col = panel_group_col * tile_size + 1
            else
                first_storage_col = (panel_group_col - 1) * tile_size + 1
            end

            shared_local_col_range, col_offset = get_shared_local_col_range(A_lu, first_storage_col)

            if length(shared_local_col_range) > 0
                n_rows = last_storage_row - first_storage_row + 1
                n_cols = length(shared_local_col_range)
                buffer_size = n_rows * n_cols
                buffer_offset = col_offset * n_rows

                # Copy matrix entries into contiguous buffer to improve efficiency.
                row_buffer =
                    reshape(@view(row_buffers[buffer_offset+1:buffer_offset+buffer_size]),
                            n_rows, n_cols)

                local_above_diagonal_sub_row =
                    @view matrix_storage[first_storage_row:last_storage_row,
                                         shared_local_col_range]

                # Copy rows into contiguous buffer, for efficiency and so that we can use the
                # LAPACK routine that requires this.
                row_buffer .= local_above_diagonal_sub_row

                # Update the 'row' part of the current sub-panel, using the L factor of the
                # diagonal sub-tile that is stored in `diagonal_sub_tile`.
                trsm!('L', 'L', 'N', 'U', 1.0, diagonal_sub_tile, row_buffer)

                # Copy buffer back into matrix storage.
                local_above_diagonal_sub_row .= row_buffer
            end
        end

        synchronize_shared()

        if shared_comm_rank == 0 && distributed_comm_rank == diagonal_distributed_rank
            # Diagonal rank can now ensure all communications have completed.
            MPI.Waitall(reqs[1:req_counter])
        end
    end

    return nothing
end

function update_bottom_right_block!(A_lu, panel)
    @sc_timeit A_lu.timer "update_bottom_right_block!" begin
        n_tiles = A_lu.n_tiles
        if panel == n_tiles
            # No remaining matix to update.
            return nothing
        end
        shared_comm_rank = A_lu.shared_comm_rank
        tile_size = A_lu.tile_size
        group_k = A_lu.group_k
        group_K = A_lu.group_K
        group_l = A_lu.group_l
        group_L = A_lu.group_L
        matrix_storage = A_lu.factorization_matrix_storage
        distributed_comm = A_lu.distributed_comm
        reqs = A_lu.comm_requests
        synchronize_shared = A_lu.synchronize_shared

        row_buffers = A_lu.factorization_row_swap_buffers
        # Can reuse this buffer as a column buffer, as it is big enough.
        col_buffers = A_lu.factorization_pivoting_buffer

        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1

        # The left panel and top panel are currently stored in contiguous buffers on the
        # panel_l sub-column and panel_k sub-row. We need to broadcast these buffers to all
        # ranks in the same row and column respectively.

        if group_k ≤ panel_k
            left_panel_first_storage_row = panel_group_row * tile_size + 1
        else
            left_panel_first_storage_row = (panel_group_row -1) * tile_size + 1
        end
        left_panel_last_storage_row = size(matrix_storage, 1)
        left_panel_first_storage_col = (panel_group_col - 1) * tile_size + 1
        left_panel_last_storage_col = panel_group_col * tile_size
        left_panel_n_rows = left_panel_last_storage_row - left_panel_first_storage_row + 1
        left_panel_n_cols = left_panel_last_storage_col - left_panel_first_storage_col + 1
        left_panel_buffer_size = left_panel_n_rows * left_panel_n_cols

        left_panel_buffer =
            reshape(@view(col_buffers[1:left_panel_buffer_size]), left_panel_n_rows,
                    left_panel_n_cols)

        # Locally-owned rows/column to be copied into row_buffer
        top_panel_first_storage_row = (panel_group_row - 1) * tile_size + 1
        top_panel_last_storage_row = panel_group_row * tile_size
        if group_l ≤ panel_l
            top_panel_first_storage_col = panel_group_col * tile_size + 1
        else
            top_panel_first_storage_col = (panel_group_col - 1) * tile_size + 1
        end
        top_panel_last_storage_col = size(matrix_storage, 2)
        top_panel_n_rows = top_panel_last_storage_row - top_panel_first_storage_row + 1
        top_panel_n_cols = top_panel_last_storage_col - top_panel_first_storage_col + 1
        top_panel_buffer_size = top_panel_n_rows * top_panel_n_cols
        top_panel_buffer = reshape(@view(row_buffers[1:top_panel_buffer_size]),
                                   top_panel_n_rows, top_panel_n_cols)

        if shared_comm_rank == 0
            request_counter = 0
            if group_l == panel_l
                # Send left panel to all ranks in the same row.
                for l ∈ 1:group_L
                    if l == panel_l
                        # Don't need to send from this rank to itself.
                        continue
                    end
                    r = (l - 1) * group_K + group_k - 1
                    reqs[request_counter+=1] = MPI.Isend(parent(parent(left_panel_buffer)),
                                                         distributed_comm; dest=r)
                end
            else
                left_panel_r = (panel_l - 1) * group_K + group_k - 1
                reqs[request_counter+=1] = MPI.Irecv!(parent(parent(left_panel_buffer)),
                                                      distributed_comm; source=left_panel_r)
            end

            if group_k == panel_k
                # Send top panel to all ranks in the same column.
                for k ∈ 1:group_K
                    if k == panel_k
                        # Don't need to send from this rank to itself.
                        continue
                    end
                    r = (group_l - 1) * group_K + k - 1
                    # MPI.jl doesn't like ReshapedArray type, but we only need to communicate
                    # the underlying storage, so use `parent()` to extract a SubArray which
                    # MPI.jl can handle.
                    reqs[request_counter+=1] = MPI.Isend(parent(top_panel_buffer),
                                                         distributed_comm; dest=r)
                end
            else
                top_panel_r = (group_l - 1) * group_K + panel_k - 1
                # MPI.jl doesn't like ReshapedArray type, but we only need to communicate the
                # underlying storage, so use `parent()` to extract a SubArray which MPI.jl can
                # handle.
                reqs[request_counter+=1] = MPI.Irecv!(parent(top_panel_buffer),
                                                      distributed_comm; source=top_panel_r)
            end

            MPI.Waitall(reqs[1:request_counter])
        end

        if !(group_l == panel_l && group_k == panel_k)
            synchronize_shared()
        end

        shared_local_col_range, col_offset =
            get_shared_local_col_range(A_lu, top_panel_first_storage_col)

        top_panel_shared_local_n_cols = length(shared_local_col_range)
        if top_panel_shared_local_n_cols > 0
            top_panel_shared_local_buffer_size = top_panel_n_rows * top_panel_shared_local_n_cols
            top_panel_shared_local_buffer_offset = col_offset * top_panel_n_rows
            top_panel_shared_local_buffer =
                reshape(@view(row_buffers[top_panel_shared_local_buffer_offset+1:top_panel_shared_local_buffer_offset+top_panel_shared_local_buffer_size]),
                        top_panel_n_rows, top_panel_shared_local_n_cols)


            # Perform the update of the bottom-right block.
            remaining_block = @view matrix_storage[left_panel_first_storage_row:end,
                                                   shared_local_col_range]
            mul!(remaining_block, left_panel_buffer, top_panel_shared_local_buffer, -1.0, 1.0)
        end

        synchronize_shared()
    end

    return nothing
end

function fill_ldiv_tiles!(A_lu)
    @sc_timeit A_lu.timer "fill_ldiv_tiles!" begin
        factors = A_lu.factors
        my_L_tiles = A_lu.my_L_tiles
        my_L_tile_row_ranges = A_lu.my_L_tile_row_ranges
        my_L_tile_col_ranges = A_lu.my_L_tile_col_ranges
        my_U_tiles = A_lu.my_U_tiles
        my_U_tile_row_ranges = A_lu.my_U_tile_row_ranges
        my_U_tile_col_ranges = A_lu.my_U_tile_col_ranges

        for (step, (row_range, col_range)) ∈ enumerate(zip(my_L_tile_row_ranges,
                                                           my_L_tile_col_ranges))
            if !isempty(row_range)
                @views my_L_tiles[1:length(row_range),1:length(col_range),step] .=
                    factors[row_range, col_range]
            end
        end

        for (step, (row_range, col_range)) ∈ enumerate(zip(my_U_tile_row_ranges,
                                                           my_U_tile_col_ranges))
            if !isempty(row_range)
                @views my_U_tiles[1:length(row_range),1:length(col_range),step] .=
                    factors[row_range, col_range]
            end
        end
    end
    return nothing
end
