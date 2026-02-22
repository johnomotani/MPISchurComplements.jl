using LinearAlgebra: ipiv2perm # Don't think ipiv2perm is part of the public interface,
                               # but it is convenient for us and seems unlikely to change
                               # often.

function setup_lu(m::Int64, n::Int64, tile_size::Int64, shared_comm_rank::Int64,
                  shared_comm_size::Int64, distributed_comm_rank::Int64,
                  distributed_comm_size::Int64, allocate_shared_float::Ff) where Ff

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
    factor_ind = findlast(x -> x≤sqrt(distributed_comm_size))
    group_L = distributed_comm_size_factors[factor_ind]
    group_K = distributed_comm_size ÷ section_L

    group_l, group_k = divrem(distributed_comm_rank, group_K)
    # Previous line would create 0-based indices, switch to 1-based.
    group_k += 1
    group_l += 1

    group_row_height = group_K * tile_size
    group_n_rows = (n + group_row_height - 1) ÷ group_row_height

    group_col_width = group_L * tile_size
    group_n_cols = (m + group_col_height - 1) ÷ group_col_width

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
        [@view(factorization_matrix_storage[(group_row-1)*tile_size+1:min(group_row*tile_size,local_storage_n),
                                            (group_col-1)*tile_size+1:min(group_col*tile_size,local_storage_m)])
         for group_row ∈ 1:group_n_rows, group_col ∈ 1:group_n_cols]

    factorization_pivoting_buffer = allocate_shared_float(group_n_rows * tile_size *
                                                          tile_size)
    factorization_pivoting_reduction_buffer = allocate_shared_float(tile_size * group_K
                                                                    * tile_size)
    factorization_pivoting_reduction_indices = allocate_shared_int(tile_size * group_K)
    factorization_row_swap_buffers = zeros(datatype, 2 * tile_size, local_storage_n)
    pivot_requests = [MPI.REQUEST_NULL for _ ∈ 1:max(2*group_K, group_K + group_L)]

    return (; factors, row_permutation, group_K, group_L, group_k, group_l,
            factorization_matrix_storage, factorization_matrix_parts,
            factorization_matrix_parts_row_ranges, factorization_matrix_parts_col_ranges,
            factorization_locally_owned_rows, factorization_pivoting_buffer,
            factorization_pivoting_reduction_buffer,
            factorization_pivoting_reduction_indices, factorization_row_swap_buffer,
            pivot_requests)
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    factors = A_lu.factors
    row_permutation = A_lu.row_permutation
    n_tiles = A_lu.n_tiles

#    distributed_comm = A_lu.distributed_comm
#    synchronize_shared = A_lu.synchronize_shared
#    check_lu = A_lu.check_lu
#    if A_lu.is_root
#        # Factorize in serial for now. Could look at implementing a parallel version of
#        # this later. Could maybe borrow algorithms from
#        # https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl/ ?
#        factorization = lu!(A; check=check_lu)
#
#        factors .= factorization.factors
#        row_permutation .= factorization.p
#    end
#    if A_lu.shared_comm_rank == 0
#        req1 = temp_Ibcast!(factors, distributed_comm; root=0)
#        req2 = temp_Ibcast!(row_permutation, distributed_comm; root=0)
#        MPI.Waitall([req1, req2])
#    end
#    synchronize_shared()

    redistribute_matrix!(A_lu, A)

    for panel ∈ 1:n_tiles
        pivot_panel_factorization!(A_lu, panel)
        pivot_remaining_columns!(A_lu, panel)
        update_top_panel!(A_lu, panel)
        update_remaining_matrix!(A_lu, panel)
    end

    fill_ldiv_tiles!(A_lu)

    return A_lu
end

# For parallelized LU factorization, each block of ranks owns a certain cyclic subset of
# tiles of the matrix, in the 'local buffers'.
function redistribute_matrix!(A_lu, A)
    shared_comm_rank = A_lu.shared_comm_rank
    matrix_parts = A_lu.factorization_matrix_parts
    row_ranges = A_lu.factorization_matrix_parts_row_ranges
    col_ranges = A_lu.factorization_matrix_parts_col_ranges
    nt = A_lu.factorization_n_tiles

    if shared_comm_rank == 0
        for j ∈ 1:nt, i ∈ 1:nt
            @views matrix_parts[i,j] .= A[row_ranges[i],col_ranges[j]]
        end
    end

    return nothing
end

function pivot_panel_factorization!(A_lu, panel)
    generate_pivots!(A_lu, panel)
    apply_pivots_from_sub_column!(A_lu, panel)
    update_sub_panel_off_diagonals!(A_lu, panel)
    update_bottom_right_block!(A_lu, panel)

    return nothing
end

function generate_pivots!(A_lu, panel)
    group_L = A_lu.group_L
    group_l = A_lu.group_l
    if (panel - 1) % group_L + 1 != group_l
        # This rank does not participate in pivot generation for this panel.
        return nothing
    end
    comm = A_lu.comm
    rank = A_lu.comm_rank
    shared_comm_rank = A_lu.shared_comm_rank
    reqs = A_lu.pivot_requests
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
    this_matrix_column = @view matrix_storage[first_local_row:end,
                                              first_local_col:last_local_col]
    this_tile_size = last_local_col - first_local_col + 1
    if shared_comm_rank == 0
        n_panel_rows = max(size(matrix_storage, 1) - first_local_row + 1, 0)
        # Construct a reshaped view so that lu_panel_buffer is a contiguously-allocated
        # array. Need this complication because we need a different number of rows for
        # each `panel`, but we use column-major storage for arrays, so slicing the rows of a
        # 2D buffer would give non-contiguous storage.
        lu_panel_buffer = reshape(@view(pivoting_buffer[1:n_panel_rows*this_tile_size]),
                                  n_panel_rows, this_tile_size)
        lu_panel_buffer .= this_matrix_column

        # LU factorize this locally-owned part of the column to get the pivots, we will
        # then reduce the locally-found pivot rows with those found by all other blocks
        # that share this column.
        local_lu = lu!(lu_panel_buffer)
        # Get the rows for just the first `this_tile_size` pivots (or all the rows, if
        # less than `this_tile_size`), which is the number need to find in the end, after
        # reducing over all blocks.
        # Using the internal LinearAlgebra function ipiv2perm is slightly more efficient
        # than constructing the full permutation vector and then selecting only the first
        # `section_width` entries from it.
        local_pivot_indices = (ipiv2perm(local_lu.ipiv,
                                         min(this_tile_size, size(lu_panel_buffer, 1)))
                               .+ first_local_row .- 1)
        global_pivot_indices = locally_owned_rows[local_pivot_indices]

        # Collect all the local pivot rows and indices onto the
        # `panel_k` block.
        if group_k == panel_k
            function get_n_rows_from_k(k)
                if k < panel_k
                    first_group_row = panel_group_row * group_K + group_k
                elseif k > panel_k
                    first_group_row = (panel_group_row - 1) * group_K + group_k
                else
                    return length(local_pivot_indices)
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
            n_rows_from_k = [get_n_rows_from_k(k) for k ∈ 1:group_K]
            k_rows_end = cumsum(n_rows_from_k)
            n_reduced_rows = k_rows_end[end]
            reduced_buffer =
                reshape(@view pivoting_reduction_buffer[1:n_reduced_rows*tile_size],
                        n_reduced_rows, tile_size)
            reduced_row_indices = @view pivoting_reduction_indices[1:n_reduced_rows]

            # Post receives for the rows and row indices from other blocks.
            req_counter = 0
            for k ∈ 1:group_K
                if k == section_k
                    # This rank does not need to communicate with itself!
                    continue
                end
                if n_rows_from_k[k] == 0
                    # No rows to colect
                    continue
                end
                # Each rank in this column is offset from the next/previous in the global
                # communicator `comm` by `group_L`.
                rank_k = rank + (k - group_k) * group_L
                if k == 1
                    first_row = 1
                else
                    first_row = k_rows_end[k-1] + 1
                end
                reqs[req_counter+=1] =
                    MPI.Irecv(@view(reduced_buffer[first_row:k_rows_end[k],:]), comm;
                              source=rank_k, tag=1)
                reqs[req_counter+=1] =
                    MPI.Irecv(@view(reduced_row_indices[first_row:k_rows_end[k]], comm;
                                    source=rank_k)), tag=2
            end

            # Copy in the local contributions
            k = section_k
            if section_k == 1
                first_row = 1
            else
                first_row = k_rows_end[k-1]+1
            end
            @views reduced_buffer[first_row:k_rows_end[k],:] .=
                this_matrix_column[local_pivot_indices,:]
            @views reduced_row_indices[first_row:k_rows_end[k]] .= global_pivot_indices

            MPI.Waitall(@view(reqs[1:req_counter]))

            # Do an LU factorization on the reduced rows. This gives the final pivot
            # indices, and also the (section_width,section_width) top-left block of the LU
            # factors.
            local_lu = lu!(reduced_buffer)
            buffer_pivot_indices = ipiv2perm(local_lu.ipiv, tile_size)
            # Is OK to re-use the pivoting_reduction_indices buffer, as we are already
            # finished with `reduced_row_indices`.
            # The pivot indices stored in this buffer will be broadcast to all ranks in
            # `apply_pivots_from_sub_column!()`.
            sub_column_pivot_indices = @view pivoting_reduction_indices[1:tile_size]
            sub_column_pivot_indices .= reduced_row_indices[buffer_pivot_indices]
        else
            # Get the local pivot rows ready for collection.
            if length(local_pivot_indices) > 0
                local_pivots_buffer .= @view matrix_storage[local_pivot_indices,:]
                reqs[1] = MPI.Isend(local_pivots_buffer, comm; dest=collecting_rank, tag=1)
                reqs[2] = MPI.Isend(global_pivot_indices, comm; dest=collecting_rank, tag=2)
                MPI.Waitall(@view reqs[1:2])
            end
        end
    end

    return nothing
end

function apply_pivots_from_sub_column!(A_lu, panel)
    comm = A_lu.comm
    rank = A_lu.comm_rank
    shared_comm_rank = A_lu.shared_comm_rank
    reqs = A_lu.pivot_requests
    m = A_lu.m
    n_tiles = A_lu.factorization_n_tiles
    tile_size = A_lu.tile_size
    section_l = A_lu.section_l
    group_K = A_lu.group_K
    section_width = A_lu.section_width
    this_first_pivot_section_k = A_lu.first_pivot_section_k[sub_column]
    this_rank_section_height = length(A_lu.factorization_matrix_parts_col_ranges[1])
    matrix_parts = A_lu.factorization_matrix_parts
    row_swap_buffers = A_lu.factorization_row_swap_buffers

    panel_group_row, panel_k = divrem(panel - 1, group_K) .+ 1
    first_local_col = (panel_group_col - 1) * tile_size + 1
    last_local_col = min(panel_group_col * tile_size, size(matrix_storage, 2))
    this_tile_size = last_local_col - first_local_col + 1

    sub_column_pivot_indices = @view A_lu.factorization_pivoting_reduction_indices[1:this_tile_size]

    if shared_comm_rank == 0
        diagonal_rank = (panel_l - 1) * group_K + panel_k
        # Broadcast the pivot indices for this sub column to all ranks.
        MPI.Bcast!(sub_column_pivot_indices, comm; root=diagonal_rank)

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
        pivot_indices_in_diagonal_block = sorted_pivot_indices[sorted_pivot_indices .≤ last_diag]

        diagonal_block_row_destination_indices = copy(sub_column_pivot_indices)
        for i ∈ 1:length(sub_column_pivot_indices)
            diag_index = diagonal_block_indices[i]
            if diag_index ∈ pivot_indices_in_diagonal_block
                # Need to send this row to directly to its pivoted position.
                original_destination = diagonal_block_row_destination_indices[i]
                i2 = findfirst(x -> x==diag_index, sub_column_pivot_indices)
                diagonal_block_row_destination_indices[i] = diagonal_block_indices[i2]
                # Re-insert the original_destination in place of the destination of the
                # row that is replaced by `diag_index`.
                diagonal_block_row_destination_indices[i2] = original_destination
            end
        end

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
            owning_rank_k_diag = (diag_tile_row - 1) % group_K + 1
            owning_rank_diag = (group_l - 1) * group_K + owning_rank_k_diag

            source_tile_row, source_tile_row_offset = divrem(isource - 1, tile_size) .+ 1
            owning_rank_k_source = (source_tile_row - 1) % group_K + 1
            owning_rank_source = (group_l - 1) * group_K + owning_rank_k_source

            dest_tile_row, dest_tile_row_offset = divrem(idest - 1, tile_size) .+ 1
            owning_rank_k_dest = (dest_tile_row - 1) % group_K + 1
            owning_rank_dest = (group_l - 1) * group_K + owning_rank_k_dest

            if rank == owning_rank_diag == owning_rank_source == owning_rank_dest
                # Swaps are all among rows owned by this rank.

                # Copy the diag row into a buffer to send down to owning_rank_dest
                # afterward.
                diag_row_swap_buffer .= @view matrix_storage[diag_section_row,:]

                # source -> diag
                @views matrix_storage[diag_section_row,:] .=
                    matrix_storage[source_section_row,:]
            elseif rank == owning_rank_diag == owning_rank_source
                # Copy out data from the row that we own, to send to owning_rank_dest.
                diag_row_swap_buffer .= @view matrix_storage[diag_section_row,:]

                # diag -> dest
                reqs[req_counter+=1] = MPI.Isend(diag_row_swap_buffer, comm;
                                                 dest=owning_rank_dest, tag=iswap)

                # source -> diag
                @views matrix_storage[diag_section_row,:] .=
                    matrix_storage[source_section_row,:]
            elseif rank == owning_rank_diag == owning_rank_dest
                # Copy out data from the row that we own, to later copy to destination
                # row.
                diag_row_swap_buffer .= @view matrix_storage[diag_section_row,:]

                # source -> diag
                reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, comm;
                                                  source=owning_rank_source, tag=iswap)
            elseif rank == owning_rank_diag
                diag_row_data = @view matrix_storage[diag_section_row,:]

                # Copy out data from the row that we own, to later copy to destination
                # row.
                diag_row_swap_buffer .= diag_row_data

                # source -> diag
                reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, comm;
                                                  source=owning_rank_source, tag=iswap)
                # diag -> dest
                reqs[req_counter+=1] = MPI.Isend!(diag_row_swap_buffer, comm;
                                                  dest=owning_rank_dest, tag=iswap)
            elseif rank == owning_rank_source == owning_rank_dest
                # Copy out data from the row that we own, to send to owning_rank_diag.
                source_row_swap_buffer .= @view matrix_storage[source_section_row,:]

                # source -> diag
                reqs[req_counter+=1] = MPI.Isend(source_row_swap_buffer, comm;
                                                 dest=owning_rank_diag, tag=iswap)
            elseif rank == owning_rank_source
                # Copy out data from the row that we own, to send to owning_rank_k1.
                source_row_swap_buffer .= @view matrix_storage[source_section_row,:]

                # source -> diag
                reqs[req_counter+=1] = MPI.Isend(source_row_swap_buffer, comm;
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
            owning_rank_k_diag = (diag_tile_row - 1) % group_K + 1
            owning_rank_diag = (group_l - 1) * group_K + owning_rank_k_diag

            source_tile_row, source_tile_row_offset = divrem(isource - 1, tile_size) .+ 1
            owning_rank_k_source = (source_tile_row - 1) % group_K + 1
            owning_rank_source = (group_l - 1) * group_K + owning_rank_k_source

            dest_tile_row, dest_tile_row_offset = divrem(idest - 1, tile_size) .+ 1
            owning_rank_k_dest = (dest_tile_row - 1) % group_K + 1
            owning_rank_dest = (group_l - 1) * group_K + owning_rank_k_dest

            if rank == owning_rank_diag == owning_rank_source == owning_rank_dest
                # Swaps are all among rows owned by this rank.

                # diag -> dest
                @views matrix_storage[dest_section_row,:] .= diag_row_swap_buffer
            elseif rank == owning_rank_diag == owning_rank_dest
                # diag -> dest
                @views matrix_storage[dest_section_row,:] .=
                    matrix_storage[diag_section_row,:]
            elseif rank == owning_rank_source == owning_rank_dest
                # source -> diag
                reqs[req_counter+=1] =
                    MPI.Irecv!(@view matrix_storage[dest_section_row,:], comm;
                               source=owning_rank_diag, tag=iswap)
            elseif rank == owning_rank_dest
                # source -> diag
                reqs[req_counter+=1] =
                    MPI.Irecv!(@view matrix_storage[dest_section_row,:], comm;
                               source=owning_rank_diag, tag=iswap)
            end
        end
        MPI.Waitall(@view(reqs[1:req_counter]))
    end

    return nothing
end

function update_sub_panel_off_diagonals!(A_lu, panel, sub_column)
    comm = A_lu.comm
    rank = A_lu.comm_rank
    shared_comm_rank = A_lu.shared_comm_rank
    row_ranges = factorization_matrix_parts_row_ranges
    col_ranges = factorization_matrix_parts_col_ranges
    matrix_parts = A_lu.factorization_matrix_parts
    n_tiles = A_lu.factorization_n_tiles
    tile_width = A_lu.factorization_tile_width
    reqs = A_lu.pivot_requests
    section_k = A_lu.section_k
    section_K = A_lu.section_K
    section_L = A_lu.section_L
    section_width = A_lu.section_width
    this_first_pivot_section_k = A_lu.first_pivot_section_k[sub_column]
    pivoting_reduction_buffer = A_lu.factorization_pivoting_reduction_buffer

    diagonal_block_size = min(section_l * section_width, tile_width) -
                          (section_l - 1) * section_width
    this_l_width = length(col_ranges[1])
    # Row buffer will contain entries for the sub-row to the right of the diagonal
    # sub-tile.
    row_buffer = @view A_lu.factorization_row_swap_buffers[1:diagonal_block_size,
                                                           panel*this_l_width+1:end]

    req_counter = 0
    if shared_comm_rank == 0
        # While updating the sub-column, we also distribute the diagonal sub-tile to ranks
        # on the same section-row as the rank that owns the diagonal sub-tile, and collect
        # onto those same ranks the full data for the rows in the diagonal sub-tile. This
        # collection is needed because these sub-tiles will become the 'U' entries in the
        # row to the right of the current diagonal sub-tile, and the sub-tile for each
        # column will need to be communicated to all ranks in that column. This can be
        # done in parallel with updating the rest of the sub-column.
        #
        # It would be nicer to do these communications with collective MPI calls (e.g.
        # `MPI.Bcast!()`), but that would require many communicators (one for each pattern
        # of operation), which would be complicated to keep track of. There is also a
        # hard-coded limit on the maximum number of MPI communicators, so it is best not
        # to create them too freely.

        diagonal_rank = (sub_column - 1) * section_K + this_first_pivot_section_k 

        if rank == diagonal_rank
            # The top diagonal_block_size*diagonal_block_size part of pivoting_reduction_buffer on
            # this rank contains the diagonal sub-tile that was calculated in
            # `generate_pivots!()`.
            diagonal_sub_tile = 
                reshape(@view pivoting_reduction_buffer[1:diagonal_block_size*diagonal_block_size],
                        diagonal_block_size, diagonal_block_size)

            # Send the diagonal sub-tile to the other ranks in this sub-column.
            rank_offset = (section_l - 1) * section_K # Offset of ranks in sub-column.
            for k ∈ 1:section_K
                if k == section_k
                    continue
                end
                r = rank_offset + k - 1
                reqs[req_counter+=1] = MPI.Isend(diagonal_sub_tile, comm; dest=r)
            end

            # Send the diagonal sub-tile to the ranks in the same sub-row.
            for l ∈ 1:section_L
                if l == section_l
                    continue
                end
                r = (l - 1) * section_K + section_k - 1
                reqs[req_counter+=1] = MPI.Isend(diagonal_sub_tile, comm; dest=r)
            end
        elseif section_k == this_first_pivot_section_k
            # Receive the diagonal sub-block and gather the data for all the corresponding
            # rows in this section_l.
            diagonal_sub_tile = 
                reshape(@view pivoting_reduction_buffer[1:diagonal_block_size*diagonal_block_size],
                        diagonal_block_size, diagonal_block_size)
            reqs[req_counter+=1] = MPI.Irecv!(diagonal_sub_tile, comm; source=diagonal_rank)
        elseif section_k == this_first_pivot_section_k + 1
            # Might need to send part of locally-owned data to rank owning previous row.
            first_diagonal_row = (sub_column - 1) * diagonal_block_size + 1
            last_diagonal_row = first_diagonal_row + diagonal_block_size - 1
            first_row_this_rank = (section_k - 1) * section_height + 1
            last_row_this_rank = min(section_k * section_height, tile_size)
            if last_diagonal_row ≥ first_row_this_rank
                this_k_height = length(row_ranges[1])
                first_storage_row = (section_k - 1) * this_k_height + 1
                last_storage_row = first_storage_row + last_diagonal_row - first_row_this_rank
                # Send to rank on the previous section-row
                dest = (section_l - 1) * section_K + section_k - 2
                MPI.Send(@view(matrix_storage[first_storage_row:last_storage_row,
                                              panel*this_l_width+1:end]),
                         comm; dest=dest)
            end
        elseif section_l == sub_column
            diagonal_sub_tile = 
                reshape(@view pivoting_reduction_buffer[1:diagonal_block_size*diagonal_block_size],
                        diagonal_block_size, diagonal_block_size)

            MPI.Recv(diagonal_sub_tile, comm; source=diagonal_rank)
        end

        if section_l == sub_column
            # This branch now includes the rank that owns the diagonal sub-tile.

            # diagonal_sub_tile now contains both the L and U factors of the current
            # diagonal sub-tile. Now need to apply U^-1 from the right to the
            # locally-owned part of the sub-column that is below the diagonal sub-tile.
            sub_column_width = diagonal_block_size # because these ranks own the same
                                                   # sub-column as the diagonal sub-tile
            last_diagonal_row = first_diagonal_row + diagonal_block_size - 1
            first_row_this_rank = (section_k - 1) * section_height + 1
            offset = max(last_diagonal_row + 1, first_row_this_rank) - first_row_this_rank

            local_section_height = length(col_ranges[1])
            local_below_diagonal_sub_column =
                @view matrix_storage[(panel-1)*local_section_height+offset+1:end,
                                     (panel-1)*sub_column_width+1:panel*sub_column_width]

            # Need to solve M*U=A for M, where A are the original matrix elements of the
            # sub-column, and U is the upper-triangular factor of the diagonal sub-tile.
            # There does not seem to be a LAPACK routine for this when M and A are
            # column-major matrices (trsv! works with a transposed U, but if we transposed
            # M and A they would be row-major, which trsv! does not seem to support).
            # Therefore just use LinearAlgebra's `rdiv!()`, which does not seem to be as
            # optimized, but this step is probably not a bottleneck anyway (?).
            rdiv!(local_below_diagonal_sub_column, UpperTriangular(diagonal_sub_tile))
        end

        if section_k == this_first_pivot_section_k
            # This branch now includes the rank that owns the diagonal sub-tile.

            # The whole sub-row to the right of the diagonal sub-block needs to be updated
            # by solving L*M=A to find M, where A are the original matrix entries and L is
            # the lower-triangular factor from the diagonal sub-block.

            first_diagonal_row = (sub_column - 1) * diagonal_block_size + 1
            last_diagonal_row = first_diagonal_row + diagonal_block_size - 1
            first_row_this_rank = (section_k - 1) * section_height + 1
            last_row_this_rank = min(section_k * section_height, tile_size)
            local_n_rows = min(last_diagonal_row, last_row_this_rank) - first_diagonal_row + 1

            # Locally-owned rows to be copied into row_buffer
            first_storage_row = (section_k - 1) * this_k_height + first_diagonal_row -
                                first_row_this_rank + 1
            last_storage_row = first_storage_row + local_n_rows - 1

            if last_diagonal_row > last_row_this_rank
                # Need to collect some rows from the next rank.
                source = (section_l - 1) * section_K + section_k
                reqs[req_counter+=1] =
                    MPI.Recv!(@view(row_buffer[local_n_rows+1:end,:]), comm; source=source)
            end

            # Copy locally-owned rows into buffer.
            @views row_buffer[1:local_n_rows,:] .=
                matrix_storage[first_storage_row,last_storage_row,panel*this_l_width+1:end]

            if rank == diagonal_rank
                # Only need to wait for last receive to complete.
                MPI.Wait(reqs[req_counter])
                req_counter -= 1
            else
                # Need to complete both receives before this rank can continue to the next
                # operations.
                MPI.Waitall(@view(reqs[1:req_counter]))
            end

            # Update the 'row' part of the current sub-panel, using the L factor of the
            # diagonal sub-tile that is stored in `diagonal_sub_tile`.
            trsv!('L', 'N', 'U', diagonal_sub_tile, row_buffer)
        end

        if rank == diagonal_rank
            # Diagonal rank can now wait for all communications to complete.
            MPI.Waitall(@view(reqs[1:req_counter]))
        end
    end

    return nothing
end

function fill_ldiv_tiles!(A_lu)
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
    return nothing
end
