using LinearAlgebra: ipiv2perm # Don't think ipiv2perm is part of the public interface,
                               # but it is convenient for us and seems unlikely to change
                               # often.

function setup_lu(m::Int64, n::Int64, shared_comm_rank::Int64, shared_comm_size::Int64,
                  distributed_comm_rank::Int64, distributed_comm_size::Int64,
                  allocate_shared_float::Ff) where Ff

    factors = allocate_shared_float(m, n)

    if shared_comm_rank == 0
        row_permutation = zeros(Int64, m)
    else
        row_permutation = zeros(Int64, 0)
    end

    # Each block owns a rectangular section of each tile. Try to make sections as square
    # as possible, and when they cannot be exactly square (because distributed_comm_size
    # is not a square number) make them taller than they are wide, as this simplifies the
    # communication-avoiding (CA) pivoting implementation, and possibly reduces overall
    # communication.
    #
    # Tiles are indexed by (i,j) - i for row, j for column.
    # Sections within each tile are indexed by (k,l) - k for row, l for column.

    factorization_tile_size = 2048 # Should be settable, or related to tile_size?
    # Final bottom/right tiles may not be full size.
    factorization_n_tiles = (m + factorization_tile_size - 1) ÷ factorization_tile_size
    total_tiles = factorization_n_tiles^2

    distributed_comm_size_factors =
        [prod(x) for x in
         collect(unique(combinations(factor(Vector, distributed_comm_size))))]
    # Find the last factor ≤ sqrt(distributed_comm_size)
    factor_ind = findlast(x -> x≤sqrt(distributed_comm_size))
    section_L = distributed_comm_size_factors[factor_ind]
    section_K = distributed_comm_size ÷ section_L

    section_height = factorization_tile_size ÷ section_K
    section_width = factorization_tile_size ÷ section_L

    section_l, section_k = divrem(distributed_comm_rank, section_K)
    # Previous line would create 0-based indices, switch to 1-based.
    section_k += 1
    section_l += 1

    function get_row_range(tile_i)
        tile_row_start = (tile_i - 1) * factorization_tile_size + 1
        tile_row_end = min(tile_i * factorization_tile_size, m)
        section_row_start = tile_row_start + (section_k - 1) * section_height
        section_row_end = min(tile_row_start - 1 + section_k * section_height,
                              tile_row_end)
        return section_row_start:section_row_end
    end

    function get_col_range(tile_j)
        tile_col_start = (tile_j - 1) * factorization_tile_size
        tile_col_end = min(tile_j * factorization_tile_size, n)
        section_col_start = tile_row_start + (section_l - 1) * section_width
        section_col_end = min(tile_row_start - 1 + section_l * section_width,
                              tile_col_end)
        return section_col_start:section_col_end
    end

    factorization_matrix_parts_row_ranges = [get_row_range(tile_i)
                                             for tile_i ∈ 1:factorization_n_tiles]
    factorization_matrix_parts_col_ranges = [get_col_range(tile_j)
                                             for tile_j ∈ 1:factorization_n_tiles]
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
        [@view(factorization_matrix_storage[(tile_i-1)*section_height:min(tile_i*section_height,local_storage_n),
                                            (tile_j-1)*section_width:min(tile_j*section_width,local_storage_m)])
         for tile_i ∈ 1:n_tiles, tile_j ∈ 1:n_tiles]

    # Find the section-row-index of the rank containing the first diagonal row for each
    # sub-column.
    first_pivot_section_k = [((l - 1) * section_width) ÷ section_height + 1
                             for l ∈ 1:section_L]

    if shared_comm_rank == 0
        # The most blocks that will ever be owned by this process when calculating the
        # pivots of a column. We distribute the blocks cyclically among the processors
        # participating in the pivoting, so only need at most ~1/section_K of the total
        # number of tiles (this would be the amount needed for the first column
        # factorization), allowing for possible remainders.
        max_local_pivoting_blocks = (factorization_n_tiles + section_K - 1) ÷ section_K

        first_pivoting_row_index_buffers = Matrix{Int64}(undef, 2*section_height,
                                                         max_local_pivoting_blocks)
        first_section_pair_lu_buffer = Matrix{datatype}(undef, 2*section_height,
                                                        section_width)
        pivoting_row_index_buffers = Matrix{Int64}(undef, 2*section_width,
                                                   max_local_pivoting_blocks)
    else
        first_pivoting_row_index_buffers = zeros(Int64, 0, 0)
        first_section_pair_lu_buffer = zeros(datatype, 0, 0)
    end
    factorization_pivoting_buffer = allocate_shared_float(n_tiles * section_height *
                                                          section_width)
    factorization_pivoting_reduction_buffer =
        allocate_shared_float(section_width * section_K * section_width)
    factorization_pivoting_reduction_indices =
        allocate_shared_int(section_width * section_K)
    factorization_row_swap_buffers = zeros(datatype, section_width, local_storage_n)
    pivot_requests = [MPI.REQUEST_NULL for _ ∈ 1:max(2*section_K, section_K + section_L)]

    return (; factors, row_permutation, section_K, section_L, section_k, section_l,
            section_height, section_width, factorization_matrix_storage,
            factorization_matrix_parts, factorization_matrix_parts_row_ranges,
            factorization_matrix_parts_col_ranges, factorization_locally_owned_rows,
            factorization_tile_size, factorization_n_tiles, first_pivot_section_k,
            factorization_pivoting_buffer, factorization_pivoting_reduction_buffer,
            factorization_pivoting_reduction_indices, factorization_row_swap_buffers,
            pivot_requests)
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    factors = A_lu.factors
    row_permutation = A_lu.row_permutation
    factorization_n_tiles = A_lu.factorization_n_tiles
    distributed_comm = A_lu.distributed_comm
    synchronize_shared = A_lu.synchronize_shared
    check_lu = A_lu.check_lu

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

    for p ∈ 1:factorization_n_tiles
        pivot_panel_factorization!(A_lu, p)
        pivot_remaining_columns!(A_lu, p)
        update_top_panel!(A_lu, p)
        update_remaining_matrix!(A_lu, p)
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

function pivot_panel_factorization!(A_lu, p)
    section_L = A_lu.section_L
    section_l = A_lu.section_l
    n_tiles = A_lu.factorization_n_tiles

    for sub_column ∈ 1:section_L
        if p == n_tiles && sub_column == section_L
            # This is the last, bottom-right section, so we just LU in serial.
            if section_l == section_L && A_lu.section_k = A_lu.section_K
                section_width = length(A_lu.factorization_matrix_parts_col_ranges[end])
                last_block = @view A_lu.factorization_matrix_parts[end,end]
                # Note the in-place version `lu!` actually saves the L and U factors
                # into last_block.
                last_section_lu = lu!(last_block)
            end
            # LU is finished, and as we have completed the final section in serial, we do
            # not need to call the remaining functions.
            break
        end
        generate_pivots!(A_lu, p, sub_column)
        apply_pivots_from_sub_column!(A_lu, p, sub_column)
        update_sub_panel_off_diagonals!(A_lu, p, sub_column)
        update_bottom_right_block!(A_lu, p, sub_column)
    end

    return nothing
end

function generate_pivots!(A_lu, p, sub_column)
    if sub_column != A_lu.section_l
        # This rank does not participate in pivot generation for this sub-column.
        return nothing
    end
    comm = A_lu.comm
    rank = A_lu.comm_rank
    shared_comm_rank = A_lu.shared_comm_rank
    reqs = A_lu.pivot_requests
    m = A_lu.m
    n_tiles = A_lu.factorization_n_tiles
    tile_size = A_lu.factorization_tile_size
    section_k = A_lu.section_k
    section_K = A_lu.section_K
    section_L = A_lu.section_L
    section_height = A_lu.section_height
    section_width = A_lu.section_width
    this_first_pivot_section_k = A_lu.first_pivot_section_k[sub_column]
    pivoting_buffer = A_lu.factorization_pivoting_buffer
    pivoting_reduction_buffer = A_lu.factorization_pivoting_reduction_buffer
    pivoting_reduction_indices = A_lu.factorization_pivoting_reduction_indices
    matrix_storage = A_lu.factorization_matrix_storage
    col_ranges = A_lu.factorization_matrix_parts_col_ranges
    locally_owned_rows = A_lu.factorization_locally_owned_rows

    this_matrix_column =
        @view matrix_storage[:,(p-1)*section_width+1:min(p*section_width,size(matrix_storage,2))]
    if shared_comm_rank == 0
        # Find the first row of local storage that is on or below the matrix diagonal.
        if section_k < this_first_pivot_section_k
            first_local_row = (p + 1) * section_height
        elseif section_k == this_first_pivot_section_k
            # This block contains the first diagonal row in this column.
            global_diagonal_row = col_ranges[1]
            first_local_row = (p * section_height
                               + global_diagonal_row
                               - (p - 1) * tile_size
                               - section_k * section_height)
        else
            first_local_row = p * section_height
        end
        n_panel_rows = size(matrix_storage, 1) - first_local_row + 1
        # Construct a reshaped view so that lu_panel_buffer is a contiguously-allocated
        # array. Need this complication because we need a different number of rows for
        # each `p`, but we use column-major storage for arrays, so slicing the rows of a
        # 2D buffer would give non-contiguous storage.
        lu_panel_buffer = reshape(@view(pivoting_buffer[1:n_panel_rows*section_width]),
                                  n_panel_rows, section_width)
        lu_panel_buffer .= @view this_matrix_column[first_local_row:end,:]

        # LU factorize this locally-owned part of the column to get the pivots, we will
        # then reduce the locally-found pivot rows with those found by all other blocks
        # that share this column.
        local_lu = lu!(lu_panel_buffer)
        # Get the rows for just the first `section_width` pivots, which is the number need
        # to find in the end, after reducing over all blocks.
        # Using the internal LinearAlgebra function ipiv2perm is slightly more efficient
        # than constructing the full permutation vector and then selecting only the first
        # `section_width` entries from it.
        local_pivot_indices = (ipiv2perm(local_lu.ipiv,
                                        min(section_width, size(lu_panel_buffer, 1)))
                               .+ first_local_row .- 1)
        global_pivot_indices = locally_owned_rows[local_pivot_indices]

        # Collect all the local pivot rows and indices onto the
        # `this_first_pivot_section_k` block.
        if section_k == this_first_pivot_section_k
            function get_n_rows_from_k(k)
                if k < this_first_pivot_section_k
                    if p == n_tiles
                        # Last tile, but this block would only contribute starting at the
                        # next tile, so no rows.
                        return 0
                    elseif p == n_tiles - 1
                        # The next tile is the last one, which may not be full height.
                        # first_row is limited to maximum of `m + 1` so that if the
                        # section is entirely beyond the bottom of the matrix, we will
                        # return 0.
                        first_row = min((n_tiles - 1) * tile_size
                                        + (k - 1) * section_height + 1,
                                        m + 1
                                       )
                        last_row = min((n_tiles - 1) * tile_size + k * section_height, m)
                        return last_row - first_row + 1
                    else
                        # This block cannot own the last section, so its sections must be
                        # full-height (except possibly on the last tile, but the next tile
                        # is not the last tile). A full-height section has at least as
                        # many rows as section_width (we made this choice during setup).
                        # So this block will pass the full number of pivot rows
                        # (`section_width`).
                        return section_width
                    end
                elseif k > this_first_pivot_section_k
                    this_section_height =
                        (k < section_K ? section_height :
                         tile_size - (section_K - 1) * section_height)
                    # Contributions from all tile-rows before the last one
                    n_rows = this_section_height * (n_tiles - p)

                    # The last tile might not be full height, so get its height specially,
                    # similar to the `p == n_tiles - 1` case above.
                    last_tile_first_row = min((n_tiles - 1) * tile_size
                                    + (k - 1) * section_height + 1,
                                    m + 1
                                   )
                    last_tile_last_row = min((n_tiles - 1) * tile_size + k * section_height, m)
                    n_rows += last_tile_last_row - last_tile_first_row + 1
                    return min(n_rows, section_width)
                else
                    return length(local_pivot_indices)
                end
            end
            n_rows_from_k = [get_n_rows_from_k(k) for k ∈ 1:section_K]
            k_rows_end = cumsum(n_rows_from_k)
            n_reduced_rows = k_rows_end[end]
            reduced_buffer =
                reshape(@view pivoting_reduction_buffer[1:n_reduced_rows*section_width],
                        n_reduced_rows, section_width)
            reduced_row_indices = @view pivoting_reduction_indices[1:n_reduced_rows]

            # Post receives for the rows and row indices from other blocks.
            req_counter = 0
            for k ∈ 1:section_K
                if k == section_k
                    # This rank does not need to communicate with itself!
                    continue
                end
                if n_rows_from_k[k] == 0
                    # No rows to colect
                    continue
                end
                # Each rank in this column is offset from the next/previous in the global
                # communicator `comm` by `section_L`.
                rank_k = rank + (k - section_k) * section_L
                if k == 1
                    reqs[req_counter+=1] = MPI.Irecv(@view(reduced_buffer[1:k_rows_end[k],:]),
                                                      comm; source=rank_k, tag=1)
                    reqs[req_counter+=1] =
                        MPI.Irecv(@view(reduced_row_indices[1:k_rows_end[k]], comm;
                                        source=rank_k)), tag=2
                else
                    reqs[req_counter+=1] =
                        MPI.Irecv(@view(reduced_buffer[k_rows_end[k-1]+1:k_rows_end[k],:]),
                                        comm; source=rank_k, tag=1)
                    reqs[req_counter+=1] =
                        MPI.Irecv(@view(reduced_row_indices[k_rows_end[k-1]+1:k_rows_end[k]],
                                        comm; source=rank_k), tag=2)
                end
            end

            # Copy in the local contributions
            k = section_k
            if section_k == 1
                @views reduced_buffer[1:k_rows_end[k],:] .=
                    this_matrix_column[local_pivot_indices,:]
                @views reduced_row_indices[1:k_rows_end[k]] .= global_pivot_indices
            else
                @views reduced_buffer[1:k_rows_end[k],:] .=
                    this_matrix_column[local_pivot_indices,:]
                @views reduced_row_indices[k_rows_end[k-1]+1:k_rows_end[k]] .=
                    global_pivot_indices
            end

            MPI.Waitall(@view(reqs[1:req_counter]))

            # Do an LU factorization on the reduced rows. This gives the final pivot
            # indices, and also the (section_width,section_width) top-left block of the LU
            # factors.
            local_lu = lu!(reduced_buffer)
            buffer_pivot_indices = ipiv2perm(local_lu.ipiv, section_width)
            # Is OK to re-use the pivoting_reduction_indices buffer, as
            # we are already finished with `reduced_row_indices`.
            # The pivot indices stored in this buffer will be broadcast to all ranks in
            # `apply_pivots_from_sub_column!()`.
            sub_column_pivot_indices = @view pivoting_reduction_indices[1:section_width]
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

function apply_pivots_from_sub_column!(A_lu, p, sub_column)
    comm = A_lu.comm
    rank = A_lu.comm_rank
    shared_comm_rank = A_lu.shared_comm_rank
    reqs = A_lu.pivot_requests
    n_tiles = A_lu.factorization_n_tiles
    tile_size = A_lu.factorization_tile_size
    section_l = A_lu.section_l
    section_K = A_lu.section_K
    section_width = A_lu.section_width
    this_first_pivot_section_k = A_lu.first_pivot_section_k[sub_column]
    this_rank_section_height = length(A_lu.factorization_matrix_parts_col_ranges[1])
    matrix_parts = A_lu.factorization_matrix_parts
    row_swap_buffers = A_lu.factorization_row_swap_buffers
    sub_column_pivot_indices = @view pivoting_reduction_indices[1:section_width]

    if shared_comm_rank == 0
        broadcasting_rank = (sub_column - 1) * section_K + this_first_pivot_section_k
        # Broadcast the pivot indices for this sub column to all ranks.
        MPI.Bcast!(sub_column_pivot_indices, comm; root=broadcasting_rank)

        diagonal_block_indices = (sub_column-1)*section_width+1:min(sub_column*section_width, tile_size)
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

        req_counter = 0
        for (iswap, (idiag, isource, idest)) ∈
                enumerate(zip(diagonal_block_indices, sub_column_pivot_indices,
                              diagonal_block_row_destination_indices))
            if i1 == i2 == i3
                # No swap to do.
                continue
            end

            row_swap_buffer = @view row_swap_buffers[:,iswap]

            owning_rank_k_diag, diag_section_row_offset = divrem((idiag - 1) % tile_size, section_height) .+ 1
            diag_section_row = (p - 1) * this_rank_section_height + diag_section_row_offset
            owning_rank_diag = (section_l - 1) * section_K + owning_rank_k_diag

            source_tile, source_tile_offset = divrem(isource - 1, tile-size)
            owning_rank_k_source, source_section_row_offset = divrem(source_tile_offset, section_height) .+ 1
            source_section_row = (source_tile - 1) * this_rank_section_height + source_section_row_offset
            owning_rank_source = (section_l - 1) * section_K + owning_rank_k_source

            dest_tile, dest_tile_offset = divrem(idest - 1, tile-size)
            owning_rank_k_dest, dest_section_row_offset = divrem(dest_tile_offset, section_height) .+ 1
            dest_section_row = (dest_tile - 1) * this_rank_section_height + dest_section_row_offset
            owning_rank_dest = (section_l - 1) * section_K + owning_rank_k_dest

            if rank == owning_rank_k1 == owning_rank_k2
                # Swaps are all among rows owned by this rank.

                # Copy the diag row into a buffer to send down to i2 afterward.
                row_swap_buffer .= @view matrix_storage[diag_section_row,:]

                # source -> diag
                @views matrix_storage[diag_section_row,:] .=
                    matrix_storage[source_section_row,:]

                # diag -> dest
                matrix_storage[dest_section_row,:] .= row_swap_buffer
            elseif rank == owning_rank_k_diag
                diag_row_data = @view matrix_storage[diag_section_row,:]
                # Copy out data from the row that we own, to send to owning_rank_k2.
                row_swap_buffer .= diag_row_data

                # diag -> dest
                reqs[req_counter+=1] = MPI.Isend(row_swap_buffer, comm;
                                                 dest=owning_rank_dest, tag=iswap)
                # source -> diag
                reqs[req_counter+=1] = MPI.Irecv!(diag_row_data, comm;
                                                  source=owning_rank_source, tag=iswap)
            elseif rank == owning_rank_source && rank == owning_rank_dest
                source_row_data = @view matrix_storage[source_section_row,:]
                dest_row_data = @view matrix_storage[dest_section_row,:]
                # Copy out data from the row that we own, to send to owning_rank_k1.
                row_swap_buffer .= source_row_data

                # source -> diag
                reqs[req_counter+=1] = MPI.Isend(row_swap_buffer, comm;
                                                 dest=owning_rank_diag, tag=iswap)
                # diag -> dest
                reqs[req_counter+=1] = MPI.Irecv!(dest_row_data, comm;
                                                  source=owning_rank_diag, tag=iswap)
            elseif rank == owning_rank_source
                source_row_data = @view matrix_storage[source_section_row,:]
                # Copy out data from the row that we own, to send to owning_rank_k1.
                row_swap_buffer .= source_row_data

                # source -> diag
                reqs[req_counter+=1] = MPI.Isend(row_swap_buffer, comm;
                                                 dest=owning_rank_diag, tag=iswap)
            elseif rank == owning_rank_dest
                # Must make the MPI.Irecv!() call after the data has been copied out to be
                # sent (which must be happening at a different `iswap` to enter this
                # branch). Therefore store the row and the source rank so we can make the
                # MPI.Irecv!() calls after all the MPI.Isend!() have been made.
                push!(isolated_receive_info, (dest_section_row, owning_rank_diag, iswap))
            end
        end
        for (dest_section_row, owning_rank_diag, iswap) ∈ isolated_receive_info
            dest_row_data = @view matrix_storage[dest_section_row,:]

            # diag -> dest
            reqs[req_counter+=1] = MPI.Irecv!(dest_row_data, comm;
                                              source=owning_rank_diag, tag=iswap)
        end
        MPI.Waitall(@view(reqs[1:req_counter]))
    end

    return nothing
end

function update_sub_panel_off_diagonals!(A_lu, p, sub_column)
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
                                                           p*this_l_width+1:end]

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
                                              p*this_l_width+1:end]),
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
                @view matrix_storage[(p-1)*local_section_height+offset+1:end,
                                     (p-1)*sub_column_width+1:p*sub_column_width]

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
                matrix_storage[first_storage_row,last_storage_row,p*this_l_width+1:end]

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
