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

function setup_lu(m::Int64, n::Int64, tile_size::Int64, shared_comm_rank::Int64,
                  shared_comm_size::Int64, distributed_comm_rank::Int64,
                  distributed_comm_size::Int64, datatype::Type, allocate_shared_float::Ff,
                  allocate_shared_int::Fi) where {Ff,Fi}

    factors = allocate_shared_float(m, n)

    if shared_comm_rank == 0
        col_permutation = zeros(Int64, m)
    else
        col_permutation = zeros(Int64, 0)
    end

    # Each block owns a set of (tile_size,tile_size) tiles in the full matrix - the last
    # row and column of tiles may be shorter/narrower. The tiles are distributed in a
    # block-cyclic pattern. Each block owns sub-tiles in the k'th row in each group of K
    # columns, and in the l'th column of each group of L columns. We choose (abritrarily)
    # to make K≤L.
    distributed_comm_size_factors =
        [prod(x) for x in
         collect(unique(combinations(factor(Vector, distributed_comm_size))))]
    # Find the last factor ≤ sqrt(distributed_comm_size)
    factor_ind = findlast(x -> x≤sqrt(distributed_comm_size), distributed_comm_size_factors)
    group_K = distributed_comm_size_factors[factor_ind]
    group_L = distributed_comm_size ÷ group_K

    group_k, group_l = divrem(distributed_comm_rank, group_L) .+ 1

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
    factorization_locally_owned_cols = vcat((collect(c) for c ∈
                                             factorization_matrix_parts_col_ranges)...)

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

    # Generate a 'binary' tree of disributed-memory ranks that will participate in the
    # disributed-memory part of the parallel pivot generation.
    distributed_size_prime_factors = factor(Vector, group_L)
    factorization_pivot_generation_distributed_tree_sizes = [group_L]
    # `reverse()` so that we divide by the largest factors first. Should not affect
    # overall run time as the order of the factors just shuffles around work between
    # different levels, but this order minimises the total work done, which might
    # marginally help power consumption?
    for f in reverse(distributed_size_prime_factors)
        push!(factorization_pivot_generation_distributed_tree_sizes,
              factorization_pivot_generation_distributed_tree_sizes[end] ÷ f)
    end

    # Generate a 'binary' tree of shared-memory ranks that will participate in the
    # shared-memory part of the parallel pivot generation.
    shared_size_prime_factors = factor(Vector, shared_comm_size)
    factorization_pivot_generation_shared_tree_sizes = [shared_comm_size]
    # `reverse()` so that we divide by the largest factors first. Should not affect
    # overall run time as the order of the factors just shuffles around work between
    # different levels, but this order minimises the total work done, which might
    # marginally help power consumption?
    for f in reverse(shared_size_prime_factors)
        push!(factorization_pivot_generation_shared_tree_sizes,
              factorization_pivot_generation_shared_tree_sizes[end] ÷ f)
    end

    factorization_pivoting_buffer = allocate_shared_float(group_n_cols * tile_size *
                                                          tile_size)
    factorization_jpiv = Vector{Int64}(undef, tile_size)
    factorization_pivoting_reduction_buffer = allocate_shared_float(tile_size * group_L
                                                                    * tile_size)
    factorization_pivoting_reduction_indices =
        allocate_shared_int(max(tile_size * group_L * shared_comm_size, 2 * tile_size))
    factorization_pivoting_reduction_indices_local = zeros(Int64, tile_size)
    factorization_source_cols = zeros(Int64, 2 * tile_size)
    factorization_panel_row_owned_swap_cols = zeros(Int64, 2 * tile_size)
    factorization_source_swap_labels = zeros(Int64, 2 * tile_size)
    factorization_col_swap_buffers = allocate_shared_float(local_storage_m, tile_size)
    factorization_swap_flags = zeros(UInt8, 2 * tile_size)
    comm_requests = [MPI.REQUEST_NULL for _ ∈
                     1:max((1 + tile_size) * group_L, group_K + group_L, 2 * tile_size)]

    # Indices and row/column sizes for when we want to divide the shared-memory processes
    # into a rectangular grid. Note shared_comm_i and shared_comm_j are 0-based indices.
    shared_comm_size_factors =
        [prod(x) for x in collect(unique(combinations(factor(Vector, shared_comm_size))))]
    shared_factor_ind = findlast(x -> x≤sqrt(shared_comm_size), shared_comm_size_factors)
    shared_comm_I = shared_comm_size_factors[shared_factor_ind]
    shared_comm_J = shared_comm_size ÷ shared_comm_I
    shared_comm_i, shared_comm_j = divrem(shared_comm_rank, shared_comm_J)

    return (; factors, col_permutation, group_K, group_L, group_k, group_l,
            factorization_matrix_storage, factorization_matrix_parts,
            factorization_matrix_parts_row_ranges, factorization_matrix_parts_col_ranges,
            factorization_locally_owned_cols,
            factorization_pivot_generation_distributed_tree_sizes,
            factorization_pivot_generation_shared_tree_sizes,
            factorization_pivoting_buffer, factorization_jpiv,
            factorization_pivoting_reduction_buffer,
            factorization_pivoting_reduction_indices,
            factorization_pivoting_reduction_indices_local, factorization_source_cols,
            factorization_panel_row_owned_swap_cols, factorization_source_swap_labels,
            factorization_col_swap_buffers, factorization_swap_flags, comm_requests,
            shared_comm_i, shared_comm_j, shared_comm_I, shared_comm_J)
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    @sc_timeit A_lu.timer "lu!" begin
        n_tiles = A_lu.n_tiles
        shared_comm_rank = A_lu.shared_comm_rank
        synchronize_shared = A_lu.synchronize_shared

        redistribute_matrix!(A_lu, A)

        # Initialize col_permutation, which will be permuted as we generate the pivots.
        if shared_comm_rank == 0
            A_lu.col_permutation .= 1:A_lu.n
        end

        for panel ∈ 1:n_tiles
            @sc_timeit A_lu.timer "panel=$panel" begin
                @sc_timeit A_lu.timer "generate_pivots!" begin
                    generate_pivots!(A_lu, panel)
                end
                apply_pivots_from_sub_row!(A_lu, panel)
                update_sub_panel_off_diagonals!(A_lu, panel)
                update_bottom_right_block!(A_lu, panel)
            end
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

function get_n_cols_from_l(l, panel_l, panel_group_col, group_L, n_local_pivots, n_tiles, n, tile_size)
    # first_group_col is the global group-column index of the first group-column owned by
    # the rank with column-index l that contributes to this panel.
    if l < panel_l
        first_group_col = panel_group_col * group_L + l
    elseif l > panel_l
        first_group_col = (panel_group_col - 1) * group_L + l
    else
        return n_local_pivots
    end
    if first_group_col > n_tiles
        # Next tile owned by this rank would be off the right of the
        # matrix, so no cols.
        return 0
    elseif first_group_col == n_tiles
        # The next tile owned by this rank is the last one, which may not
        # be full width.
        return n - (n_tiles - 1) * tile_size
    else
        # This block cannot own only the last section, so it will provide
        # a full tile-worth of columns.
        return tile_size
    end
end

function generate_pivots!(A_lu, panel)
    group_k = A_lu.group_k
    group_K = A_lu.group_K
    if (panel - 1) % group_K + 1 != group_k
        # This rank does not participate in pivot generation for this panel.
        return nothing
    end
    distributed_comm = A_lu.distributed_comm
    shared_comm_rank = A_lu.shared_comm_rank
    shared_comm_size = A_lu.shared_comm_size
    reqs = A_lu.comm_requests
    n = A_lu.n
    n_tiles = A_lu.n_tiles
    tile_size = A_lu.tile_size
    group_l = A_lu.group_l
    group_L = A_lu.group_L
    pivot_generation_distributed_tree_sizes = A_lu.factorization_pivot_generation_distributed_tree_sizes
    pivot_generation_shared_tree_sizes = A_lu.factorization_pivot_generation_shared_tree_sizes
    pivoting_buffer = A_lu.factorization_pivoting_buffer
    jpiv = A_lu.factorization_jpiv
    pivoting_reduction_buffer = A_lu.factorization_pivoting_reduction_buffer
    pivoting_reduction_indices = A_lu.factorization_pivoting_reduction_indices
    matrix_storage = A_lu.factorization_matrix_storage
    locally_owned_cols = A_lu.factorization_locally_owned_cols
    synchronize_shared = A_lu.synchronize_shared

    # Find the on-or-below diagonal part of the sub-column that is owned by this rank.
    panel_group_row = (panel - 1) ÷ group_K + 1
    panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
    if group_l < panel_l
        first_local_col = panel_group_col * tile_size + 1
    else
        first_local_col = (panel_group_col - 1) * tile_size + 1
    end
    first_local_row = (panel_group_row - 1) * tile_size + 1
    last_local_row = min(panel_group_row * tile_size, size(matrix_storage, 1))
    this_tile_size = last_local_row - first_local_row + 1
    pivoting_reduction_indices_local = @view A_lu.factorization_pivoting_reduction_indices_local[1:this_tile_size]

    # Copy data into pivoting_buffer to be LU factorized.
    total_local_cols = max(size(matrix_storage, 2) - first_local_col + 1, 0)
    copy_cols_per_rank = (total_local_cols + shared_comm_size - 1) ÷ shared_comm_size
    copy_col_range = shared_comm_rank*copy_cols_per_rank+first_local_col:min((shared_comm_rank+1)*copy_cols_per_rank,total_local_cols)+first_local_col-1
    copy_ncol = length(copy_col_range)
    if copy_ncol > 0
        copy_buffer_offset = (copy_col_range[1]-1) * this_tile_size
        copy_buffer_size = copy_ncol * this_tile_size
        copy_pivot_buffer = reshape(@view(pivoting_buffer[copy_buffer_offset+1:copy_buffer_offset+copy_buffer_size]),
                                    this_tile_size, copy_ncol)
        copy_pivot_buffer .= @view matrix_storage[first_local_row:last_local_row,copy_col_range]
    end
    if shared_comm_rank == 0
        # Initial set of indices that will be permuted as pivoting progresses.
        @views pivoting_reduction_indices[1:total_local_cols] .=
            first_local_col:size(matrix_storage,2)
    end
    synchronize_shared()

    # Do 'tournament pivoting' in two stages:
    #    Each shared-memory block does a 'binary' tree reduction to reduce
    #    `shared_n_cols` pivots to `this_tile_size` pivots.
    # -> Root processes of the shared-memory blocks (the processes in
    #    `distributed_comm`) do a 'binary' tree reduction to reduce
    #    `distributed_comm_size * this_tile_size` pivots to `this_tile_size` pivots.
    # When there are not enough pivots to have `2*this_tile_size` columns per process,
    # some steps will be skipped until the number of processes participating is small
    # enough.
    shared_n_cols = size(matrix_storage, 2) - first_local_col + 1
    shared_n_levels = length(pivot_generation_shared_tree_sizes)
    shared_pivot_buffer =
        reshape(@view(pivoting_buffer[1:this_tile_size*total_local_cols]),
                this_tile_size, total_local_cols)
    shared_n_levels = length(pivot_generation_shared_tree_sizes)
    function shared_memory_tree_pivot_generation!(level, n_cols)
        # Note that all processes in shared_comm have to go through all levels of
        # `shared_memory_tree_pivot_generation()` so that we can use
        # `synchronize_shared()` within the recursive calls, instead of having to
        # create separate communicators for every subset of processes that might be
        # participating.
        if level > shared_n_levels
            return nothing
        end
        if level < shared_n_levels && n_cols ≤ pivot_generation_shared_tree_sizes[level+1] * 2 * this_tile_size
            # Not enough work to do on this level, skip to the next level.
            return shared_memory_tree_pivot_generation(level + 1, n_cols)
        end
        level_nproc = pivot_generation_shared_tree_sizes[level]
        cols_per_shared_proc = max((n_cols + level_nproc - 1) ÷ level_nproc,
                                   2 * this_tile_size)

        if shared_comm_rank * cols_per_shared_proc < n_cols
            col_offset = shared_comm_rank * cols_per_shared_proc
            panel_buffer_offset = col_offset * this_tile_size
            panel_buffer_n_cols = length(this_shared_proc_cols)
            panel_buffer_size = panel_buffer_n_cols * this_tile_size
            this_lu_panel_buffer =
                reshape(@view(pivoting_buffer[panel_buffer_offset+1:panel_buffer_offset+panel_buffer_size]),
                        this_tile_size, panel_buffer_n_cols)
            # LU factorize this locally-owned part of the row to get the pivots, we
            # will then reduce the locally-found pivot columns with those found by all
            # other processes on the shared-memory communicator.
            # In the tournament pivoting algorithm implemented in `generate_pivots!()`
            # [Grigori et al. (2011)], the locally-owned block may be singular here.
            # Even if the block is singular, we can still generate pivots and
            # construct LU factors. These pivot columns can then be used for the
            # tournament pivoting.
            column_pivot_lu!(this_lu_panel_buffer, jpiv)
            shared_local_pivot_indices =
                ipiv2perm_truncated(jpiv, n_cols,
                                    min(this_tile_size, panel_buffer_n_cols))
            for (i,local_pivot) ∈ enumerate(shared_local_pivot_indices)
                pivoting_reduction_indices_local[i] = pivoting_reduction_indices[local_pivot+col_offset]
            end
        end
        synchronize_shared()

        # Update `pivoting_reduction_indices` and shared_pivot_buffer with the top
        # `this_tile_size` pivot indices and columns.
        if shared_comm_rank * cols_per_shared_proc < n_cols
            local_offset = shared_comm_rank * this_tile_size
            for i ∈ 1:length(shared_local_pivot_indices)
                local_pivot = pivoting_reduction_indices_local[i]
                pivoting_reduction_indices[local_offset+i] = local_pivot
                @views shared_pivot_buffer[:,i+local_offset] .=
                    matrix_storage[first_local_row:last_local_row,local_pivot]
            end
        end
        synchronize_shared()

        # Continue on the next level, with fewer processes.
        shared_memory_tree_pivot_generation(level + 1, level_nproc * this_tile_size)
    end

    shared_memory_tree_pivot_generation!(1, shared_n_cols)

    # Need to define this function outside the if-block to avoid precompilation
    # problems.
    # We gather the pivot indices and columns onto the process with
    # `group_l==panel_l`, so when calculating 'ranks' within this reduction process,
    # we do so relative to that process.
    function distributed_memory_tree_pivot_generation!(level, n_participating,
                                                       tree_sizes)
        n_procs_this_level = tree_sizes[level]
        if n_procs_this_level == 1
            return nothing
        end

        rank_step = n_participating ÷ n_procs_this_level

        # Rank within the processes that are participating at this level.
        level_rank = (group_l - panel_l) ÷ rank_step

        n_procs_next_level = tree_sizes[level+1]
        level_rank_offset = level_rank % n_procs_next_level
        is_gathering_proc = level_rank_offset == 0

        if is_gathering_proc
            # Note that the gathering process owns the diagonal tile, and we
            # only use this branch of the code when the diagonal tile is not
            # the last one, so here the diagonal tile is always full-sized.
            n_reduced_cols = this_tile_size
            for r_offset ∈ 1:rank_step-1
                # When not all ranks in the group row are participating, the
                # source rank never needs to wrap back around to start again
                # from 0, so the `% group_L` is not needed, but does not hurt.
                # When all ranks are participating there are `group_L` in
                # total, and the rank may need to wrap around (when panel_l >
                # 1).
                source_l = (group_l - 1 + r_offset * rank_step) % group_L + 1

                # Usually we get this_tile_size rows from each source, but
                # sometimes a source that owns columns at the edge of the
                # matrix might send fewer.
                if source_l < panel_l
                    source_first_global_col = panel_group_col * group_L * tile_size + 1
                else
                    source_first_global_col = (panel_group_col - 1) * group_L * tile_size + 1
                end
                n_cols_from_source = min(this_tile_size, n - source_first_global_col + 1)
                n_reduced_cols += n_cols_from_source
                source_offset = r_offset * this_tile_size

                source_r = (group_k - 1) * group_L + source_l - 1
                # MPI.jl doesn't like ReshapedArray type of reduction_buffer,
                # but we only need to communicate the underlying storage, so
                # communicate pivoting_reduction_buffer directly.
                reqs[r_offset] =
                    MPI.Irecv!(@view(pivoting_reduction_buffer[this_tile_size*source_offset+1:this_tile_size*(source_offset+n_cols_from_source)]),
                               distributed_comm; source=source_r, tag=1)
                reqs[rank_step-1+r_offset] =
                    MPI.Irecv!(@view(pivoting_reduction_indices[source_offset+1:source_offset+n_cols_from_source]),
                               distributed_comm; source=source_r, tag=2)
            end

            # Wait for all panel columns to arrive. Do not need indices yet,
            # so wait for those later.
            MPI.Waitall(@view(reqs[1:rank_step-1]))

            # Need to keep the unfactorized columns, so copy the data into
            # another buffer for factorization.
            factorization_buffer =
                reshape(@view(pivoting_buffer[1:this_tile_size*n_reduced_cols]),
                        this_tile_size, n_reduced_cols)
            factorization_buffer .= @view pivoting_reduction_buffer[:,1:n_reduced_cols]

            column_pivot_lu!(factorization_buffer, jpiv)
            buffer_pivot_indices = ipiv2perm_truncated(jpiv, n_reduced_cols,
                                                       this_tile_size)

            # Wait for all indices to arrive.
            MPI.Waitall(@view(reqs[rank_step:2*rank_step-2]))

            # Use `pivoting_reduction_indices_local` as an intermediate to
            # avoid having to implicitly allocate a buffer to copy indices.
            pivoting_reduction_indices_local .= @view pivoting_reduction_indices[buffer_pivot_indices]
            @views pivoting_reduction_indices[1:this_tile_size] .= pivoting_reduction_indices_local

            return distributed_memory_tree_pivot_generation!(level + 1,
                                                             n_participating,
                                                             tree_sizes)
        else
            # Send pivot columns and indices to the gathering
            # process.
            gathering_rank_l = (group_l - 1 - level_rank_offset * rank_step) % group_L + 1
            gathering_rank = (group_k - 1) * group_L + gathering_rank_l - 1
            # MPI.jl doesn't like ReshapedArray type of reduction_buffer, but
            # we only need to communicate the underlying storage, so
            # communicate pivoting_reduction_buffer directly.
            reqs[1] = MPI.Isend(@view(pivoting_reduction_buffer[1:this_tile_size*n_local_cols]),
                                distributed_comm; dest=gathering_rank, tag=1)
            reqs[2] = MPI.Isend(@view(pivoting_reduction_indices[1:n_local_cols]),
                                distributed_comm; dest=gathering_rank, tag=2)
            MPI.Waitall(@view(reqs[1:2]))
            return nothing
        end

        return nothing
    end

    if shared_comm_rank == 0
        if group_L == 1
            # All columns are local to the block, so no need for further reduction,
            # factorisation, or local->global index conversion.
        else
            n_local_cols = min(this_tile_size, total_local_cols)
            # Define a reduction buffer. Note that we just reshape the full
            # `pivoting_reduction_buffer`, but only use some number of columns at the
            # beginning of the buffer - usually this buffer is 'too big'. Similarly
            # `pivoting_reduction_indices` is 'too big'.
            reduction_buffer =
                reshape(pivoting_reduction_buffer, this_tile_size, group_L *
                        tile_size)
            # Convert the 'local' pivot indices to global ones.
            local_pivot_indices = @view pivoting_reduction_indices_local[1:n_local_cols]
            local_pivot_indices .= @view pivoting_reduction_indices[1:n_local_cols]
            # Copy the local matrix columns into a reduction buffer.
            for i ∈ 1:n_local_cols
                local_pivot = local_pivot_indices[i]
                pivoting_reduction_indices[i] = locally_owned_cols[local_pivot]
                @views reduction_buffer[:,i] .=
                    matrix_storage[first_local_row:last_local_row,local_pivot]
            end

            group_rows_in_panel = n_tiles - panel + 1
            if group_rows_in_panel < group_L
                # Not all distributed ranks in the group-row own entries in the top
                # panel. It would be silly to send zero-size messages. This should
                # only happen for very small panels, so efficiency is not too
                # important. To avoid needing to factorize `group_rows_in_panel`, set
                # up a tree which just reduces to one process in a single step.
                tree_sizes = [group_rows_in_panel, 1]
                n_participating = group_rows_in_panel
                is_participating = n_local_cols > 0
            else
                tree_sizes = pivot_generation_distributed_tree_sizes
                n_participating = group_L
                is_participating = true
            end

            if is_participating
                distributed_memory_tree_pivot_generation!(1, n_participating, tree_sizes)
            end
        end
    end

    synchronize_shared()

    return nothing
end

function apply_pivots_from_sub_row!(A_lu, panel)
    @sc_timeit A_lu.timer "apply_pivots_from_sub_row!" begin
        col_permutation = A_lu.col_permutation
        distributed_comm = A_lu.distributed_comm
        distributed_comm_rank = A_lu.distributed_comm_rank
        shared_comm_rank = A_lu.shared_comm_rank
        reqs = A_lu.comm_requests
        n = A_lu.n
        tile_size = A_lu.tile_size
        group_k = A_lu.group_k
        group_K = A_lu.group_K
        group_L = A_lu.group_L
        matrix_storage = A_lu.factorization_matrix_storage
        col_swap_buffers = A_lu.factorization_col_swap_buffers
        swap_flags = A_lu.factorization_swap_flags

        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        first_global_row = (panel - 1) * tile_size + 1
        last_global_row = min(panel * tile_size, n)
        this_tile_size = last_global_row - first_global_row + 1
        local_panel_column_offset = (panel_group_col - 1) * tile_size
        owning_rank_panel_column = (group_k - 1) * group_L + panel_l - 1

        # Broadcast the pivot indices for this sub column to all ranks.
        sub_row_pivot_indices = @view A_lu.factorization_pivoting_reduction_indices[1:this_tile_size]
        if shared_comm_rank == 0
            owning_rank_diag = (panel_k - 1) * group_L + panel_l - 1
            MPI.Bcast!(sub_row_pivot_indices, distributed_comm; root=owning_rank_diag)
        end

        # As this function is just array copies and communication, we do not use any
        # shared-memory parallelism.
        if shared_comm_rank == 0
            panel_column_indices = (panel-1)*tile_size+1:min(panel*tile_size, n)
            last_panel_column = panel_column_indices[end]

            # By default, just send the column from the panel column back to the original
            # position of the 'pivot column' that is being moved into the panel column.
            # However, if the column in the panel column is itself a 'pivot column', then
            # we need to send it to the correct position within the panel column. Here we
            # set up `panel_col_destination_indices` with the position where each column
            # of the panel column should be sent to.
            sorted_pivot_indices = sort(sub_row_pivot_indices)

            # All pivot indices are in or right of the first column of the panel column,
            # so we can find the indices that are within the panel column from the first
            # entries of the sorted list of pivot indices.
            split_ind = searchsortedlast(sorted_pivot_indices, last_panel_column)
            pivot_indices_inside_panel_column = sorted_pivot_indices[1:split_ind]
            pivot_indices_outside_panel_column = sorted_pivot_indices[split_ind+1:end]
            outside_panel_column_counter = 1

            panel_col_destination_indices = copy(sub_row_pivot_indices)
            for i ∈ 1:length(sub_row_pivot_indices)
                panel_index = panel_column_indices[i]
                if panel_index ∈ pivot_indices_inside_panel_column
                    # Pivot destination of this column is within the panel_column, so send
                    # it directly.
                    i2 = findfirst(isequal(panel_index), sub_row_pivot_indices)
                    panel_col_destination_indices[i] = panel_column_indices[i2]
                else
                    # The destination of this column must be set to something that is
                    # outside the panel column block.
                    panel_col_destination_indices[i] =
                        pivot_indices_outside_panel_column[outside_panel_column_counter]
                    outside_panel_column_counter += 1
                end
            end

            # Permute the global pivot vector with the current panel's pivot indices.
            panel_col_global_inds = (panel-1)*tile_size+1:min(panel*tile_size,n)
            panel_col_unpermuted_inds = col_permutation[panel_col_global_inds]
            pivot_cols_unpermuted_inds = col_permutation[sub_row_pivot_indices]
            # Need to use panel_col_destination_indices to avoid errors when some of
            # the pivot columns are within the panel column.
            col_permutation[panel_col_destination_indices] .=
                panel_col_unpermuted_inds
            col_permutation[panel_col_global_inds] .= pivot_cols_unpermuted_inds

            # We aim to minimise the number of copy operations needed to complete the
            # effect of all column swaps from this panel. As we want only one set of MPI
            # communications, we cannot directly use `jpiv` as generated by
            # column_pivot_lu!(), as that would mean doing sequentially a set of column
            # swaps. Instead we have generated the complete set of sources and
            # destinations of each panel column.  To make sure all operations are
            # completed, we define a vector of flags for each panel column involved in the
            # swaps (any off-panel columns owned by the panel-owning rank are always to be
            # swapped to/from panel columns, and off-panel columns on other ranks the
            # column must be both sent and received over MPI so in either case we do not
            # need to track their states) where the flag can indicate that the buffer is
            # not yet updated, has been emptied (into a column buffer or transferred to
            # another column), or has been completed (final data has been transferred in,
            # or no change is needed).
            # We first set up the MPI communications that are needed:
            #   * For every column sent and/or received over MPI, copy the original data
            #     into a column buffer, then start an MPI.Isend of the data in the column
            #     buffer and/or an MPI.Irecv into the column.
            #   * If this panel column receieves over MPI, mark column as completed, as no
            #     other data needs to be transferred into the column, we only have to wait
            #     for all MPI transfers to complete, which we will do at the end of this
            #     function.
            #   * If the panel column sends but does not receive, mark the column as
            #     empty, as the data has been transferred out into the column buffer, but
            #     the column is not filled with the final data yet.
            #   * If the column has no MPI communications, but does not need any swap,
            #     mark as completed.
            # In second and third passes that only involves the panel-column-owning rank,
            # we transfer all the locally-owned data as needed. Second pass:
            #   * Loop through the panel columns until the first empty column is found
            #     (restarting the loop on the following column after processing each empty
            #     column). Note that columns outside the panel column cannot be 'empty'.
            #   * Starting from that first empty column, transfer the data from the
            #     'source' column into the empty column, and mark the column as
            #     'completed'. Since the source column is now empty, transfer data into
            #     that from its source (and mark as completed), continuing until data was
            #     transferred from a 'complete' column (when the source column is
            #     'complete' the data has to be transferred from the column buffer, not
            #     from `matrix_storage`).
            # After the second pass there are no empty columns, but it may be that there
            # are columns that are not 'complete' because they are part of a cycle of
            # swaps among columns owned by the panel column owning rank. Therefore we need
            # a third pass:
            #   * Loop through the panel columns until the first incomplete column
            #     (restarting the loop from the column after this until there are no more
            #     incomplete columns). Note that columns outside the panel column cannot
            #     be 'incomplete'.
            #   * Copy the incomplete column into a column buffer (and mark it as
            #     complete), and save which column it is.  Copy the data from this columns
            #     source into the column.  Move to the now-empty source and copy its
            #     source into it (and mark it as complete), repeating until the source is
            #     the original 'incomplete' column, whose data is copied from the column
            #     buffer.
            # After the third pass, all columns should be completed. Note that on each
            # pass, we loop through the columns only once, so the number of operations
            # increases only linearly with `tile_size` (the maximum number of columns
            # involved in the swaps is `2*tile_size`).
            # This method involves more loops and more checking flags than the
            # `jpiv`-based method, but reduces the amount of data copied. It is not
            # obvious which would be faster in serial, but the method used here is
            # preferred because it minimises the number of MPI data transfers, and it is
            # not clear how to mix single-transfer copies for MPI operations with
            # sequential column-swap operations for local transfers, without some similar
            # amount of flag-checking.

            # First pass - queue up MPI communications.
            ###########################################
            req_counter = 0
            # Flag 0x0 means 'complete', 0x1 means 'empty' and 0x2 means 'incomplete'.
            swap_flags .= 0x2

            source_cols = @view A_lu.factorization_source_cols[1:2*this_tile_size]
            source_cols .= -1
            # We also need to record which locally-owned columns are involved in the
            # swaps.
            panel_row_owned_swap_cols = A_lu.factorization_panel_row_owned_swap_cols
            @views panel_row_owned_swap_cols[1:this_tile_size] .=
                local_panel_column_offset+1:local_panel_column_offset+this_tile_size
            @views panel_row_owned_swap_cols[this_tile_size+1:end] .= -1
            n_off_panel_dest_cols = 0

            for (iswap, (ipanel, isource, idest)) ∈
                    enumerate(zip(panel_column_indices, sub_row_pivot_indices,
                                  panel_col_destination_indices))
                if ipanel == isource == idest
                    # No swap to do.
                    swap_flags[iswap] = 0x0
                    continue
                end

                panel_col_offset = (ipanel - 1) % tile_size + 1

                source_tile_col, source_tile_col_offset = divrem(isource - 1, tile_size) .+ 1
                owning_rank_l_source = (source_tile_col - 1) % group_L
                owning_rank_source = (group_k - 1) * group_L + owning_rank_l_source

                dest_tile_col, dest_tile_col_offset = divrem(idest - 1, tile_size) .+ 1
                owning_rank_l_dest = (dest_tile_col - 1) % group_L
                owning_rank_dest = (group_k - 1) * group_L + owning_rank_l_dest

                if distributed_comm_rank == owning_rank_panel_column == owning_rank_source == owning_rank_dest
                    source_storage_col = ((source_tile_col - 1) ÷ group_L) * tile_size + source_tile_col_offset
                    source_cols[iswap] = source_storage_col
                    if dest_tile_col > panel_group_col
                        # The 'destination column' is owned by the panel-column-owning
                        # rank, but is not in the panel column. Therefore the 'source
                        # column' of 'destination column' is within the panel column, and
                        # we want to record which column it is.
                        n_off_panel_dest_cols += 1
                        panel_storage_col = ((panel - 1) ÷ group_L) * tile_size + panel_col_offset
                        source_cols[this_tile_size+n_off_panel_dest_cols] = panel_storage_col
                        dest_storage_col = ((dest_tile_col - 1) ÷ group_L) * tile_size + dest_tile_col_offset
                        panel_row_owned_swap_cols[this_tile_size+n_off_panel_dest_cols] = dest_storage_col
                    end
                elseif distributed_comm_rank == owning_rank_panel_column == owning_rank_source
                    # Copy out data from the colu and start MPI.Isend()

                    panel_storage_col = ((panel - 1) ÷ group_L) * tile_size + panel_col_offset

                    col_swap_buffer = @view col_swap_buffers[:,iswap]
                    col_swap_buffer .= @view matrix_storage[:,panel_storage_col]

                    # panel -> dest
                    reqs[req_counter+=1] = MPI.Isend(col_swap_buffer, distributed_comm;
                                                     dest=owning_rank_dest, tag=iswap)

                    # Mark as 'empty'.
                    swap_flags[iswap] = 0x1

                    source_storage_col = ((source_tile_col - 1) ÷ group_L) * tile_size + source_tile_col_offset
                    source_cols[iswap] = source_storage_col
                elseif distributed_comm_rank == owning_rank_panel_column == owning_rank_dest
                    # Copy out data from the column that we own, to later copy to
                    # destination column.

                    panel_storage_col = ((panel - 1) ÷ group_L) * tile_size + panel_col_offset

                    panel_col_data = @view matrix_storage[:,panel_storage_col]

                    col_swap_buffer = @view col_swap_buffers[:,iswap]
                    col_swap_buffer .= panel_col_data

                    # source -> panel
                    reqs[req_counter+=1] = MPI.Irecv!(panel_col_data, distributed_comm;
                                                      source=owning_rank_source, tag=iswap)

                    # Mark as 'complete' because it will be complete once MPI
                    # communications have completed.
                    swap_flags[iswap] = 0x0

                    if dest_tile_col > panel_group_col
                        # The 'destination column' is owned by the panel-column-owning
                        # rank, but is not in the panel column. Therefore the 'source
                        # column' of 'destination column' is within the panel column, and
                        # we want to record which column it is.
                        n_off_panel_dest_cols += 1
                        source_cols[this_tile_size+n_off_panel_dest_cols] = panel_storage_col
                        dest_storage_col = ((dest_tile_col - 1) ÷ group_L) * tile_size + dest_tile_col_offset
                        panel_row_owned_swap_cols[this_tile_size+n_off_panel_dest_cols] = dest_storage_col
                    end
                elseif distributed_comm_rank == owning_rank_panel_column
                    panel_storage_col = ((panel - 1) ÷ group_L) * tile_size + panel_col_offset

                    panel_col_data = @view matrix_storage[:,panel_storage_col]

                    # Copy out data from the column that we own, to later copy to
                    # destination column.
                    col_swap_buffer = @view col_swap_buffers[:,iswap]
                    col_swap_buffer .= panel_col_data

                    # source -> panel
                    reqs[req_counter+=1] = MPI.Irecv!(panel_col_data, distributed_comm;
                                                      source=owning_rank_source, tag=iswap)
                    # panel -> dest
                    reqs[req_counter+=1] = MPI.Isend(col_swap_buffer, distributed_comm;
                                                     dest=owning_rank_dest, tag=iswap)

                    # Mark as 'complete' because it will be complete once MPI
                    # communications have completed.
                    swap_flags[iswap] = 0x0
                elseif distributed_comm_rank == owning_rank_source
                    # Copy out data from the column that we own, to send to
                    # owning_rank_panel_column.

                    source_storage_col = ((source_tile_col - 1) ÷ group_L) * tile_size + source_tile_col_offset

                    col_swap_buffer = @view col_swap_buffers[:,iswap]
                    col_swap_buffer .= @view matrix_storage[:,source_storage_col]

                    # source -> panel
                    reqs[req_counter+=1] = MPI.Isend(col_swap_buffer, distributed_comm;
                                                     dest=owning_rank_panel_column, tag=iswap)
                end
            end
            if distributed_comm_rank != owning_rank_panel_column
                # Now that all columns on this rank have been copied out into buffers to
                # send, can call MPI.IRecv!() for all of them.
                for (iswap, (ipanel, isource, idest)) ∈
                        enumerate(zip(panel_column_indices, sub_row_pivot_indices,
                                      panel_col_destination_indices))
                    if ipanel == isource == idest
                        # No swap to do.
                        continue
                    end

                    dest_tile_col, dest_tile_col_offset = divrem(idest - 1, tile_size) .+ 1
                    owning_rank_l_dest = (dest_tile_col - 1) % group_L
                    owning_rank_dest = (group_k - 1) * group_L + owning_rank_l_dest

                    if distributed_comm_rank == owning_rank_dest
                        dest_storage_col = ((dest_tile_col - 1) ÷ group_L) * tile_size + dest_tile_col_offset

                        # source -> panel
                        reqs[req_counter+=1] =
                            MPI.Irecv!(@view(matrix_storage[:,dest_storage_col]),
                                       distributed_comm; source=owning_rank_panel_column, tag=iswap)
                    end
                end
            end

            if distributed_comm_rank == owning_rank_panel_column
                source_cols = @view source_cols[1:this_tile_size+n_off_panel_dest_cols]
                panel_row_owned_swap_cols = @view panel_row_owned_swap_cols[1:this_tile_size+n_off_panel_dest_cols]
                swap_flags = @view swap_flags[1:this_tile_size+n_off_panel_dest_cols]
                source_swap_labels = @view A_lu.factorization_source_swap_labels[1:this_tile_size+n_off_panel_dest_cols]

                # Figure out the index within panel_row_owned_swap_cols of each source
                # column, and store in source_swap_labels.
                for (i, source_col) ∈ enumerate(source_cols)
                    # Operation like `findfirst()`, but some entries will not be found.
                    # `findfirst()` returns `nothing` when the entry is not found, but
                    # that would introduce type instability, so search with a loop
                    # instead.
                    source_label = -1
                    for i ∈ 1:length(panel_row_owned_swap_cols)
                        if panel_row_owned_swap_cols[i] == source_col
                            source_label = i
                            break
                        end
                    end
                    source_swap_labels[i] = source_label
                end

                # Second pass - fill all chains that start with an empty column.
                ################################################################
                for next_potential_empty_col ∈ 1:this_tile_size
                    if swap_flags[next_potential_empty_col] == 0x1
                        label = next_potential_empty_col
                        source_label = source_swap_labels[label]
                        while true
                            if swap_flags[source_label] == 0x0
                                # 'Completed' column, which must be within the panel
                                # column.
                                # Copy data from the column swap buffers.
                                icol = panel_row_owned_swap_cols[label]
                                @views matrix_storage[:,icol] .=
                                    col_swap_buffers[:,source_label]
                                swap_flags[label] = 0x0
                                break
                            elseif swap_flags[source_label] == 0x2
                                # 'Not-completed' column, copy data matrix storage.
                                icol = panel_row_owned_swap_cols[label]
                                isource = panel_row_owned_swap_cols[source_label]
                                @views matrix_storage[:,icol] .=
                                    matrix_storage[:,isource]
                                swap_flags[label] = 0x0
                                label = source_label
                                source_label = source_swap_labels[label]
                            else
                                error("Chain did not terminate in a completed column. "
                                      * "This should never happen.")
                            end
                        end
                    end
                end

                # Third pass - move data around chains that are cycles of incomplete
                # columns.
                ####################################################################
                for next_potential_incomplete_label ∈ 1:this_tile_size
                    if swap_flags[next_potential_incomplete_label] == 0x2
                        col_buffer = @view col_swap_buffers[:,next_potential_incomplete_label]
                        # Copy out data from this column so that we can put it back into
                        # the last column in the cycle.
                        icol = panel_row_owned_swap_cols[next_potential_incomplete_label]
                        col_buffer .= @view matrix_storage[:,icol]

                        label = next_potential_incomplete_label
                        next_label = source_swap_labels[label]
                        swap_flags[next_potential_incomplete_label] = 0x0
                        while next_label != next_potential_incomplete_label
                            this_col = panel_row_owned_swap_cols[label]
                            next_col = panel_row_owned_swap_cols[next_label]
                            @views matrix_storage[:,this_col] .= matrix_storage[:,next_col]
                            swap_flags[next_label] = 0x0
                            label = next_label
                            next_label = source_swap_labels[label]
                        end

                        # Copy in the data from start_col to this final column to complete
                        # the cycle.
                        this_col = panel_row_owned_swap_cols[label]
                        @views matrix_storage[:,this_col] .= col_buffer
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

function get_shared_local_range(n, first_index, proc, nproc)
    n_local_entries = n - first_index + 1
    entries_per_proc = (n_local_entries + nproc - 1) ÷ nproc
    offset = proc * entries_per_proc
    return first_index+offset:min(offset+first_index+entries_per_proc-1,n), offset
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
        diagonal_distributed_rank = (panel_k - 1) * group_L + panel_l - 1
        if panel == n_tiles
            # No remaining matix to update, so no need for communication or off-diagonal
            # update. Only need to copy LU-factorized block into matrx_storage
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
        # Can reuse this buffer as a row buffer, as it is big enough.
        row_buffers = A_lu.factorization_pivoting_buffer
        col_buffers = A_lu.factorization_col_swap_buffers

        first_panel_row = (panel_group_row - 1) * tile_size + 1
        last_panel_row = min(panel_group_row * tile_size, size(matrix_storage, 1))
        this_tile_size = last_panel_row - first_panel_row + 1
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
                for k ∈ vcat(panel_k+1:group_K, 1:panel_k-1)
                    r = (k - 1) * group_L + group_l - 1
                    # MPI.jl doesn't like ReshapedArray type, but we only need to communicate
                    # the underlying storage, so use `parent()` to extract a SubArray which
                    # MPI.jl can handle.
                    reqs[req_counter+=1] = MPI.Isend(parent(diagonal_sub_tile),
                                                     distributed_comm; dest=r)
                end

                # Send the diagonal sub-tile to the ranks in the same sub-row.
                rank_offset = (group_k - 1) * group_L # Offset of ranks in sub-row.
                for l ∈ vcat(panel_l+1:group_L, 1:panel_l-1)
                    r = rank_offset + l - 1
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

                col_buffer =
                    @view(reshape(@view(col_buffers[1:n_rows*n_cols]),
                                  n_rows, n_cols)[row_offset+1:row_offset+n_local_rows,:])

                local_below_diagonal_sub_column =
                    @view matrix_storage[shared_local_row_range,
                                         first_storage_col:last_storage_col]

                col_buffer .= local_below_diagonal_sub_column

                # Need to solve M*U=A for M, where A are the original matrix elements of the
                # sub-column, and U is the upper-triangular factor of the diagonal sub-tile.
                trsm!('R', 'U', 'N', 'N', 1.0, diagonal_sub_tile, col_buffer)

                local_below_diagonal_sub_column .= col_buffer
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

            shared_local_col_range, col_offset =
                get_shared_local_range(size(matrix_storage, 2), first_storage_col,
                                       shared_comm_rank, shared_comm_size)

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
        shared_comm_size = A_lu.shared_comm_size
        shared_comm_i = A_lu.shared_comm_i
        shared_comm_j = A_lu.shared_comm_j
        shared_comm_I = A_lu.shared_comm_I
        shared_comm_J = A_lu.shared_comm_J
        tile_size = A_lu.tile_size
        group_k = A_lu.group_k
        group_K = A_lu.group_K
        group_l = A_lu.group_l
        group_L = A_lu.group_L
        matrix_storage = A_lu.factorization_matrix_storage
        distributed_comm = A_lu.distributed_comm
        reqs = A_lu.comm_requests
        synchronize_shared = A_lu.synchronize_shared

        # Can reuse this buffer as a row buffer, as it is big enough.
        row_buffers = A_lu.factorization_pivoting_buffer
        col_buffers = A_lu.factorization_col_swap_buffers

        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1

        # The left panel and top panel are currently stored in contiguous buffers on the
        # panel_l sub-column and panel_k sub-row. We need to broadcast these buffers to all
        # ranks in the same row and column respectively.

        if group_k ≤ panel_k
            left_panel_first_storage_row = panel_group_row * tile_size + 1
        else
            left_panel_first_storage_row = (panel_group_row - 1) * tile_size + 1
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
                    r = (group_k - 1) * group_L + l - 1
                    reqs[request_counter+=1] = MPI.Isend(parent(parent(left_panel_buffer)),
                                                         distributed_comm; dest=r)
                end
            else
                left_panel_r = (group_k - 1) * group_L + panel_l - 1
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
                    r = (k - 1) * group_L + group_l - 1
                    # MPI.jl doesn't like ReshapedArray type, but we only need to communicate
                    # the underlying storage, so use `parent()` to extract a SubArray which
                    # MPI.jl can handle.
                    reqs[request_counter+=1] = MPI.Isend(parent(top_panel_buffer),
                                                         distributed_comm; dest=r)
                end
            else
                top_panel_r = (panel_k - 1) * group_L + group_l - 1
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
            get_shared_local_range(size(matrix_storage, 2), top_panel_first_storage_col,
                                   shared_comm_j, shared_comm_J)

        shared_local_row_range, row_offset =
            get_shared_local_range(size(matrix_storage, 1), left_panel_first_storage_row,
                                   shared_comm_i, shared_comm_I)

        top_panel_shared_local_n_cols = length(shared_local_col_range)
        left_panel_shared_local_n_rows = length(shared_local_row_range)
        if top_panel_shared_local_n_cols > 0 && left_panel_shared_local_n_rows > 0
            top_panel_shared_local_buffer =
                @view top_panel_buffer[:,col_offset+1:col_offset+top_panel_shared_local_n_cols]

            left_panel_shared_local_buffer =
                @view left_panel_buffer[row_offset+1:row_offset+left_panel_shared_local_n_rows,:]

            # Perform the update of the bottom-right block.
            remaining_block = @view matrix_storage[shared_local_row_range,
                                                   shared_local_col_range]
            mul!(remaining_block, left_panel_shared_local_buffer,
                 top_panel_shared_local_buffer, -1.0, 1.0)
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
