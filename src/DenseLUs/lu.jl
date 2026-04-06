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

    factorization_pivoting_buffer = allocate_shared_float(group_n_rows * tile_size *
                                                          tile_size)
    factorization_pivoting_reduction_buffer = allocate_shared_float(tile_size * group_K
                                                                    * tile_size)
    factorization_pivoting_reduction_indices = allocate_shared_int(tile_size * group_K * shared_comm_size)
    factorization_row_swap_buffers = allocate_shared_float(2 * tile_size, local_storage_n)
    comm_requests = [MPI.REQUEST_NULL for _ ∈
                     1:max((1 + tile_size) * group_K, group_K + group_L, 2 * tile_size)]

    return (; factors, row_permutation, group_K, group_L, group_k, group_l,
            factorization_matrix_storage, factorization_matrix_parts,
            factorization_matrix_parts_row_ranges, factorization_matrix_parts_col_ranges,
            factorization_locally_owned_rows, factorization_pivoting_buffer,
            factorization_pivoting_reduction_buffer,
            factorization_pivoting_reduction_indices, factorization_row_swap_buffers,
            comm_requests)
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
        pivoting_buffer = A_lu.factorization_pivoting_buffer
        pivoting_reduction_buffer = A_lu.factorization_pivoting_reduction_buffer
        pivoting_reduction_indices = A_lu.factorization_pivoting_reduction_indices
        matrix_storage = A_lu.factorization_matrix_storage
        locally_owned_rows = A_lu.factorization_locally_owned_rows
        synchronize_shared = A_lu.synchronize_shared

        # Find the on-or-below diagonal part of the sub-column that is owned by this rank.
        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        if group_k < panel_k
            first_local_row = panel_group_row * tile_size + 1
        else
            first_local_row = (panel_group_row - 1) * tile_size + 1
        end
        first_local_col = (panel_group_col - 1) * tile_size + 1
        last_local_col = min(panel_group_col * tile_size, size(matrix_storage, 2))
        this_tile_size = last_local_col - first_local_col + 1

        # Do 'tournament pivoting' in three stages:
        #    All processes in each `shared_comm`.
        # -> Root process of each shared-memory block, which is every process in
        #    `distributed_comm`.
        # -> Global root process, which is the root procees of `distributed_comm`.
        n_local_rows = size(matrix_storage, 1) - first_local_row + 1
        rows_per_shared_proc = max((n_local_rows + shared_comm_size - 1) ÷ shared_comm_size,
                                   this_tile_size)
        this_shared_proc_rows =
            (shared_comm_rank*rows_per_shared_proc+first_local_row:min((shared_comm_rank+1)*rows_per_shared_proc+first_local_row-1,size(matrix_storage,1)))

        # We keep at least one tile per shared-mem process, so sometimes some processes may
        # have no work to do.
        n_active_shared_procs = (n_local_rows + rows_per_shared_proc - 1) ÷ rows_per_shared_proc
        last_proc_n_rows = n_local_rows - (n_active_shared_procs - 1) * rows_per_shared_proc
        last_proc_missing_rows = max(0, this_tile_size - last_proc_n_rows)
        if shared_comm_rank < n_active_shared_procs
            panel_buffer_offset = shared_comm_rank * rows_per_shared_proc * this_tile_size
            panel_buffer_n_rows = length(this_shared_proc_rows)
            panel_buffer_size = panel_buffer_n_rows * this_tile_size
            this_lu_panel_buffer = reshape(@view(pivoting_buffer[panel_buffer_offset+1:panel_buffer_offset+panel_buffer_size]),
                                           panel_buffer_n_rows, this_tile_size)
            this_lu_panel_buffer .= @view matrix_storage[this_shared_proc_rows,
                                                         first_local_col:last_local_col]
            # LU factorize this locally-owned part of the column to get the pivots, we will
            # then reduce the locally-found pivot rows with those found by all other processes
            # on the shared-memory communicator.
            # In the tournament pivoting algorithm implemented in `generate_pivots!()`
            # [Grigori et al. (2011)], the locally-owned block may be singular here. Even if
            # the block is singular, we can still generate pivots and construct LU factors.
            # These pivot rows can then be used for the tournament pivoting.
            shared_local_lu = lu!(this_lu_panel_buffer; allowsingular=true)
            shared_local_pivot_indices =
                ipiv2perm_truncated(shared_local_lu.ipiv, panel_buffer_n_rows,
                                    min(this_tile_size, size(this_lu_panel_buffer, 1)))
            indices_offset = shared_comm_rank * this_tile_size
            n_indices = min(this_tile_size,length(shared_local_pivot_indices))
            @views @. pivoting_reduction_indices[indices_offset+1:indices_offset+n_indices] =
                shared_local_pivot_indices[1:n_indices] + this_shared_proc_rows[1] - 1
        end
        synchronize_shared()

        if shared_comm_rank == 0
            candidate_pivot_indices = @view pivoting_reduction_indices[1:n_active_shared_procs*this_tile_size-last_proc_missing_rows]
            if n_active_shared_procs == 0
                # Only this process did any work in the previous step, so no need to re-do LU
                # factorization here.
                local_pivot_indices = candidate_pivot_indices
                global_pivot_indices = locally_owned_rows[local_pivot_indices]
            else
                n_buffer_rows = length(candidate_pivot_indices)
                # Construct a reshaped view so that lu_panel_buffer is a contiguously-allocated
                # array. Need this complication because we need a different number of rows for
                # each `panel`, but we use column-major storage for arrays, so slicing the rows of a
                # 2D buffer would give non-contiguous storage.
                lu_panel_buffer = reshape(@view(pivoting_buffer[1:n_buffer_rows*this_tile_size]),
                                          n_buffer_rows, this_tile_size)
                lu_panel_buffer .= @view matrix_storage[candidate_pivot_indices,first_local_col:last_local_col]

                # LU factorize this locally-owned part of the column to get the pivots, we will
                # then reduce the locally-found pivot rows with those found by all other blocks
                # that share this column.
                # In the tournament pivoting algorithm implemented in `generate_pivots!()`
                # [Grigori et al. (2011)], the locally-owned block may be singular here. Even if
                # the block is singular, we can still generate pivots and construct LU factors.
                # These pivot rows can then be used for the tournament pivoting.
                local_lu = lu!(lu_panel_buffer; allowsingular=true)
                # Get the rows for just the first `this_tile_size` pivots (or all the rows, if
                # less than `this_tile_size`), which is the number need to find in the end, after
                # reducing over all blocks.
                # Using the our custom ipiv2perm_truncated is slightly more efficient than
                # constructing the full permutation vector and then selecting only the first
                # `section_width` entries from it.
                local_panel_pivot_indices =
                    ipiv2perm_truncated(local_lu.ipiv, n_buffer_rows,
                                        min(this_tile_size, size(lu_panel_buffer, 1)))
                local_pivot_indices = candidate_pivot_indices[local_panel_pivot_indices]
                global_pivot_indices = locally_owned_rows[local_pivot_indices]
            end

            # Collect all the local pivot rows and indices onto the
            # `panel_k` block.
            if group_k == panel_k
                n_rows_from_k = [get_n_rows_from_k(k, panel_k, panel_group_row, group_K,
                                                   length(local_panel_pivot_indices),
                                                   n_tiles, m, tile_size)
                                 for k ∈ 1:group_K]
                k_rows_end = cumsum(n_rows_from_k)
                n_reduced_rows = k_rows_end[end]
                n_cols = last_local_col - first_local_col + 1
                reduced_buffer =
                    reshape(@view(pivoting_reduction_buffer[1:n_reduced_rows*n_cols]),
                            n_reduced_rows, n_cols)
                reduced_row_indices = @view pivoting_reduction_indices[1:n_reduced_rows]

                # Post receives for the rows and row indices from other blocks.
                req_counter = 0
                for k ∈ 1:group_K
                    if k == group_k
                        # This rank does not need to communicate with itself!
                        continue
                    end
                    if n_rows_from_k[k] == 0
                        # No rows to colect
                        continue
                    end
                    # Each rank in this column is offset from the next/previous in the
                    # distributed communicator `distributed_comm` by 1.
                    rank_k = distributed_comm_rank + (k - group_k)
                    if k == 1
                        first_row = 1
                    else
                        first_row = k_rows_end[k-1] + 1
                    end
                    k_counter = 0
                    for row ∈ first_row:k_rows_end[k]
                        k_counter += 1
                        reqs[req_counter+=1] =
                            MPI.Irecv!(@view(reduced_buffer[row,:]), distributed_comm;
                                       source=rank_k, tag=k_counter)
                    end
                    k_counter += 1
                    reqs[req_counter+=1] =
                        MPI.Irecv!(@view(reduced_row_indices[first_row:k_rows_end[k]]),
                                   distributed_comm; source=rank_k, tag=k_counter)
                end

                # Copy in the local contributions
                k = group_k
                if group_k == 1
                    first_row = 1
                else
                    first_row = k_rows_end[k-1]+1
                end
                @views reduced_buffer[first_row:k_rows_end[k],:] .=
                    matrix_storage[local_pivot_indices,first_local_col:last_local_col]
                @views reduced_row_indices[first_row:k_rows_end[k]] .= global_pivot_indices

                MPI.Waitall(reqs[1:req_counter])

                # Do an LU factorization on the reduced rows. This gives the final pivot
                # indices, and also the (section_width,section_width) top-left block of the LU
                # factors.
                local_lu = lu!(reduced_buffer)
                buffer_pivot_indices = ipiv2perm_truncated(local_lu.ipiv, n_reduced_rows,
                                                           n_cols)
                # Is OK to re-use the pivoting_reduction_indices buffer, as we are already
                # finished with `reduced_row_indices`.
                # The pivot indices stored in this buffer will be broadcast to all ranks in
                # `apply_pivots_from_sub_column!()`.
                sub_column_pivot_indices = @view pivoting_reduction_indices[1:this_tile_size]
                sub_column_pivot_indices .= reduced_row_indices[buffer_pivot_indices]

                # For use later, only need the factorized diagonal tile. This is contained in
                # reduced_buffer, but not contiguously, as reduced_buffer can contain more
                # than just the diagonal tile. Therefore copy the first n_cols rows from each
                # column into contiguous storage at the beginning of reduced_buffer.
                # Note that we copy the entries in a loop rather than copying all the entries
                # from each column in a broadcast operation because it can happen that the
                # number of rows is less than 2*n_cols, in which case the source and
                # destination of the copy would overlap, which might (?) cause problems. This
                # operation should not be a significant computational cost, so no need to
                # super-optimize.
                if size(reduced_buffer, 1) > n_cols
                    linear_index = n_cols
                    for col in 2:n_cols
                        for row in 1:n_cols
                            reduced_buffer[linear_index+=1] = reduced_buffer[row,col]
                        end
                    end
                end
            else
                # Get the local pivot rows ready for collection.
                if length(local_pivot_indices) > 0
                    collecting_rank = (panel_l - 1) * group_K + panel_k - 1
                    req_counter = 0
                    for row ∈ local_pivot_indices
                        req_counter += 1
                        reqs[req_counter] =
                            MPI.Isend(@view(matrix_storage[row,first_local_col:last_local_col]),
                                      distributed_comm; dest=collecting_rank, tag=req_counter)
                    end
                    req_counter += 1
                    reqs[req_counter] =
                        MPI.Isend(global_pivot_indices, distributed_comm;
                                  dest=collecting_rank, tag=req_counter)
                    MPI.Waitall(reqs[1:req_counter])
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

        panel_k = (panel - 1) % group_K + 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        first_global_col = (panel - 1) * tile_size + 1
        last_global_col = min(panel * tile_size, m)
        this_tile_size = last_global_col - first_global_col + 1

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
                    i2 = findfirst(x -> x==diag_index, sub_column_pivot_indices)
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

            # Need to store parameters for receives that must be delayed until after all sends
            # have started.
            isolated_receive_info = NTuple{3,Int64}[]

            # First copy all the rows to be set into row_swap_buffers, and post sends. We then
            # do the receives and copies into the matrix rows in a second step, to ensure that
            # no row is overwritten before being copied out.
            req_counter = 0
            for (iswap, (idiag, isource, idest)) ∈
                    enumerate(zip(diagonal_block_indices, sub_column_pivot_indices,
                                  diagonal_block_row_destination_indices))
                if idiag == isource == idest
                    # No swap to do.
                    continue
                end

                diag_row_swap_buffer = @view row_swap_buffers[2*(iswap-1)+1,:]
                source_row_swap_buffer = @view row_swap_buffers[2*(iswap-1)+2,:]

                diag_tile_row, diag_tile_row_offset = divrem(idiag - 1, tile_size) .+ 1
                owning_rank_k_diag = (diag_tile_row - 1) % group_K
                owning_rank_diag = (group_l - 1) * group_K + owning_rank_k_diag

                source_tile_row, source_tile_row_offset = divrem(isource - 1, tile_size) .+ 1
                owning_rank_k_source = (source_tile_row - 1) % group_K
                owning_rank_source = (group_l - 1) * group_K + owning_rank_k_source

                dest_tile_row, dest_tile_row_offset = divrem(idest - 1, tile_size) .+ 1
                owning_rank_k_dest = (dest_tile_row - 1) % group_K
                owning_rank_dest = (group_l - 1) * group_K + owning_rank_k_dest

                if distributed_comm_rank == owning_rank_diag == owning_rank_source == owning_rank_dest
                    # Swaps are all among rows owned by this distributed rank.

                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset
                    diag_storage_row = ((diag_tile_row - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    # Copy the diag row into a buffer to send down to owning_rank_dest
                    # afterward.
                    diag_row_swap_buffer .= @view matrix_storage[diag_storage_row,:]

                    # source -> diag
                    @views matrix_storage[diag_storage_row,:] .=
                        matrix_storage[source_storage_row,:]
                elseif distributed_comm_rank == owning_rank_diag == owning_rank_source
                    # Copy out data from the row that we own, to send to owning_rank_dest.

                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset
                    diag_storage_row = ((diag_tile_row - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    diag_row_swap_buffer .= @view matrix_storage[diag_storage_row,:]

                    # diag -> dest
                    reqs[req_counter+=1] = MPI.Isend(diag_row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_dest, tag=iswap)

                    # source -> diag
                    @views matrix_storage[diag_storage_row,:] .=
                        matrix_storage[source_storage_row,:]
                elseif distributed_comm_rank == owning_rank_diag == owning_rank_dest
                    # Copy out data from the row that we own, to later copy to destination
                    # row.

                    diag_storage_row = ((diag_tile_row - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    diag_row_data = @view matrix_storage[diag_storage_row,:]

                    diag_row_swap_buffer .= diag_row_data

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, distributed_comm;
                                                      source=owning_rank_source, tag=iswap)
                elseif distributed_comm_rank == owning_rank_diag
                    diag_storage_row = ((diag_tile_row - 1) ÷ group_K) * tile_size + diag_tile_row_offset

                    diag_row_data = @view matrix_storage[diag_storage_row,:]

                    # Copy out data from the row that we own, to later copy to destination
                    # row.
                    diag_row_swap_buffer .= diag_row_data

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, distributed_comm;
                                                      source=owning_rank_source, tag=iswap)
                    # diag -> dest
                    reqs[req_counter+=1] = MPI.Isend(diag_row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_dest, tag=iswap)
                elseif distributed_comm_rank == owning_rank_source == owning_rank_dest
                    # Copy out data from the row that we own, to send to owning_rank_diag.

                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset

                    source_row_swap_buffer .= @view matrix_storage[source_storage_row,:]

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Isend(source_row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_diag, tag=iswap)
                elseif distributed_comm_rank == owning_rank_source
                    # Copy out data from the row that we own, to send to owning_rank_k1.

                    source_storage_row = ((source_tile_row - 1) ÷ group_K) * tile_size + source_tile_row_offset

                    source_row_swap_buffer .= @view matrix_storage[source_storage_row,:]

                    # source -> diag
                    reqs[req_counter+=1] = MPI.Isend(source_row_swap_buffer, distributed_comm;
                                                     dest=owning_rank_diag, tag=iswap)
                end
            end
            for (iswap, (idiag, isource, idest)) ∈
                    enumerate(zip(diagonal_block_indices, sub_column_pivot_indices,
                                  diagonal_block_row_destination_indices))
                if idiag == isource == idest
                    # No swap to do.
                    continue
                end

                diag_row_swap_buffer = @view row_swap_buffers[2*(iswap-1)+1,:]

                diag_tile_row, diag_tile_row_offset = divrem(idiag - 1, tile_size) .+ 1
                owning_rank_k_diag = (diag_tile_row - 1) % group_K
                owning_rank_diag = (group_l - 1) * group_K + owning_rank_k_diag

                source_tile_row, source_tile_row_offset = divrem(isource - 1, tile_size) .+ 1
                owning_rank_k_source = (source_tile_row - 1) % group_K
                owning_rank_source = (group_l - 1) * group_K + owning_rank_k_source

                dest_tile_row, dest_tile_row_offset = divrem(idest - 1, tile_size) .+ 1
                owning_rank_k_dest = (dest_tile_row - 1) % group_K
                owning_rank_dest = (group_l - 1) * group_K + owning_rank_k_dest

                if distributed_comm_rank == owning_rank_diag == owning_rank_source == owning_rank_dest
                    # Swaps are all among rows owned by this distributed rank.

                    dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset

                    # diag -> dest
                    @views matrix_storage[dest_storage_row,:] .= diag_row_swap_buffer
                elseif distributed_comm_rank == owning_rank_diag == owning_rank_dest
                    diag_storage_row = ((diag_tile_row - 1) ÷ group_K) * tile_size + diag_tile_row_offset
                    dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset

                    # diag -> dest
                    @views matrix_storage[dest_storage_row,:] .= diag_row_swap_buffer
                elseif distributed_comm_rank == owning_rank_source == owning_rank_dest
                    dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset

                    # source -> diag
                    reqs[req_counter+=1] =
                        MPI.Irecv!(@view(matrix_storage[dest_storage_row,:]),
                                   distributed_comm; source=owning_rank_diag, tag=iswap)
                elseif distributed_comm_rank == owning_rank_dest
                    dest_storage_row = ((dest_tile_row - 1) ÷ group_K) * tile_size + dest_tile_row_offset

                    # source -> diag
                    reqs[req_counter+=1] =
                        MPI.Irecv!(@view(matrix_storage[dest_storage_row,:]),
                                   distributed_comm; source=owning_rank_diag, tag=iswap)
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
        synchronize_shared = A_lu.synchronize_shared

        panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
        panel_group_col, panel_l = divrem(panel - 1, group_L) .+ 1
        diagonal_distributed_rank = (panel_l - 1) * group_K + panel_k - 1
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
        row_buffers = A_lu.factorization_row_swap_buffers
        # Can reuse this buffer as a column buffer, as it is big enough.
        col_buffers = A_lu.factorization_pivoting_buffer

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
                n_rows = length(shared_local_row_range)
                n_cols = last_storage_col - first_storage_col + 1
                buffer_size = n_rows * n_cols
                buffer_offset = row_offset * n_cols

                # Copy matrix entries into contiguous buffer to improve efficiency.
                # Use a transposed buffer here so that the storage is row-major, and the parts
                # (each of which is contiguous in memory) filled by different processes in
                # `shared_comm` concatenate together correctly to give the complete 'below
                # diagonal sub-column'.
                col_buffer =
                    transpose(reshape(@view(col_buffers[buffer_offset+1:buffer_offset+buffer_size]),
                                      n_cols, n_rows))

                local_below_diagonal_sub_column =
                    @view matrix_storage[shared_local_row_range,
                                         first_storage_col:last_storage_col]

                col_buffer .= local_below_diagonal_sub_column

                # Need to solve M*U=A for M, where A are the original matrix elements of the
                # sub-column, and U is the upper-triangular factor of the diagonal sub-tile.
                # However, as col_buffer is transposed, and `trsm!()` does not support 'M'
                # being transposed, we need to transpose the whole equation to U'*M'=A'. We
                # then tell trsm! that U' is transposed (the 'T' third argument) while M' and
                # A' are the 'transposed' version of col_buffer, but as col_buffer itself is a
                # 'transpose', they are contiguous, column-major, arrays.
                trsm!('L', 'U', 'T', 'N', 1.0, diagonal_sub_tile, transpose(col_buffer))

                # Copy buffer back into matrix storage.
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

        # Use a transposed buffer to match the column buffer used in
        # `update_sub_panel_off_diagonals!()`.
        left_panel_buffer =
            transpose(reshape(@view(col_buffers[1:left_panel_buffer_size]),
                              left_panel_n_cols, left_panel_n_rows))

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
