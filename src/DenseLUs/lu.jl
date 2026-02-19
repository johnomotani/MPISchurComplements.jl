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
    factorization_row_swap_buffers = zeros(datatype, section_width, section_width)
    pivot_requests = [MPI.REQUEST_NULL for _ ∈ 1:2*section_K]

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
        update_below_diagonal_sub_column!(A_lu, p, sub_column)
        update_right_sub_columns!(A_lu, p, sub_column)
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
                    reqs[2*(k-1)+1] = MPI.Irecv(@view(reduced_buffer[1:k_rows_end[k],:]),
                                                      comm; source=rank_k, tag=1)
                    reqs[2*(k-1)+2] =
                        MPI.Irecv(@view(reduced_row_indices[1:k_rows_end[k]], comm;
                                        source=rank_k)), tag=2
                else
                    reqs[2*(k-1)+1] =
                        MPI.Irecv(@view(reduced_buffer[k_rows_end[k-1]+1:k_rows_end[k],:]),
                                        comm; source=rank_k, tag=1)
                    reqs[2*(k-1)+2] =
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

            MPI.Waitall(reqs)

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
    matrix_parts = A_lu.factorization_matrix_parts
    row_swap_buffers = A_lu.factorization_row_swap_buffers
    sub_column_pivot_indices = @view pivoting_reduction_indices[1:section_width]

    if shared_comm_rank == 0
        broadcasting_rank = (sub_column - 1) * section_K + this_first_pivot_section_k
        # Broadcast the pivot indices for this sub column to all ranks.
        MPI.Bcast!(sub_column_pivot_indices, comm; root=broadcasting_rank)

        diagonal_block_indices = (sub_column-1)*section_width+1:min(sub_column*section_width, tile_size)

        for (iswap, (i1, i2)) ∈ enumerate(zip(diagonal_block_indices, sub_column_pivot_indices))
            if i1 == i2
                # No swap to do.
                continue
            end

            row_swap_buffer = @view row_swap_buffers[:,iswap]

            owning_rank_k1, i1_section_row = divrem((i1 - 1) % tile_size, section_height) .+ 1
            owning_rank1 = (section_l - 1) * section_K + owning_rank_k1

            owning_rank_k2, i2_section_row = divrem((i2 - 1) % tile_size, section_height) .+ 1
            owning_rank2 = (section_l - 1) * section_K + owning_rank_k2

            # Need to do the same row-swaps in all tile-columns.
            for tile_j ∈ 1:n_tiles
                # When this rank is in the sub_column currently being processed
                # (sub_column==sub_column_l), it only needs to send the diagonal row down to
                # the pivot row, as the diagonal block was already collected during the pivot
                # generation.
                if rank == owning_rank_k1 == owning_rank_k2
                    # Swaps are all among rows owned by this rank.

                    if sub_column == section_l && tile_j == p
                        # i1 -> i2
                        @views matrix_parts[i2_tile,tile_j][i2_section_row,:] .=
                            matrix_parts[i1_tile,tile_j][i1_section_row,:]
                    else
                        # Copy the i1 row into a buffer to send down to i2 afterward.
                        row_swap_buffer .= @view matrix_parts[i1_tile,tile_j][i1_section_row,:]

                        # i2 -> i1
                        @views matrix_parts[i1_tile,tile_j][i1_section_row,:] .=
                            matrix_parts[i2_tile,tile_j][i2_section_row,:]

                        # i1 -> i2
                        matrix_parts[i2_tile,tile_j][i2_section_row,:] .= row_swap_buffer
                    end
                elseif rank == owning_rank_k1
                    i1_row_data = @view matrix_parts[i1_tile,tile_j][i1_section_row,:]
                    if sub_column == section_l && tile_j == p
                        # i1 -> i2
                        reqs[2*(iswap-1)+1] = MPI.Isend(i1_row_data, comm;
                                                        dest=owning_rank_2, tag=iswap)
                    else
                        # Copy out data from the row that we own, to send to owning_rank_k2.
                        row_swap_buffer .= i1_row_data

                        # i1 -> i2
                        reqs[2*(iswap-1)+1] = MPI.Isend(row_swap_buffer, comm;
                                                        dest=owning_rank_2, tag=iswap)
                        # i2 -> i1
                        reqs[2*(iswap-1)+2] = MPI.Irecv!(i1_row_data, comm;
                                                         source=owning_rank_2, tag=iswap)
                    end
                elseif rank == owning_rank_k2
                    i2_row_data = @view matrix_parts[i2_tile,tile_j][i2_section_row,:]
                    if sub_column == section_l && tile_j == p
                        # i1 -> i2
                        reqs[2*(iswap-1)+2] = MPI.Irecv!(i2_row_data, comm;
                                                         source=owning_rank_1, tag=iswap)
                    else
                        # Copy out data from the row that we own, to send to owning_rank_k1.
                        row_swap_buffer .= i2_row_data

                        # i2 -> i1
                        reqs[2*(iswap-1)+1] = MPI.Isend(row_swap_buffer, comm;
                                                        dest=owning_rank_1, tag=iswap)
                        # i1 -> i2
                        reqs[2*(iswap-1)+2] = MPI.Irecv!(i2_row_data, comm;
                                                         source=owning_rank_2, tag=iswap)
                    end
                end
            end
        end
        MPI.Waitall(reqs)
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
