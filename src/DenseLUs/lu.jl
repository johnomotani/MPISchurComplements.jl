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
    factorization_matrix_parts = [allocate_shared_float(length(row_range),
                                                        length(col_range))
                                  for row_range ∈ factorization_matrix_parts_row_ranges,
                                      col_range ∈ factorization_matrix_parts_col_ranges]

    # This rank is the top of the column of diagonal and below-diagonal blocks that need
    # to be pivoted.
    diagonal_row = factorization_matrix_parts_col_ranges[1]
    first_pivot_section_k = (diagonal_row - 1) ÷ section_height + 1
    if first_pivot_section_k == section_k
        first_row_with_diagonal = (factorization_matrix_parts_col_ranges[1] -
                                   factorization_matrix_parts_row_ranges[1] + 1)
    else
        first_row_with_diagonal = -1
    end

    if shared_comm_rank == 0
        # The most blocks that will ever be owned by this process when calculating the
        # pivots of a column. We distribute the blocks cyclically among the processors
        # participating in the pivoting, so only need at most ~1/section_K of the total
        # number of tiles (this would be the amount needed for the first column
        # factorization), allowing for possible remainders.
        max_local_pivoting_blocks = (factorization_n_tiles + section_K - 1) ÷ section_K

        first_pivoting_buffers = Array{datatype,3}(undef, 2*section_height, section_width,
                                                   max_local_pivoting_blocks)
        first_pivoting_row_index_buffers = Matrix{Int64}(undef, 2*section_height,
                                                         max_local_pivoting_blocks)
        first_section_pair_lu_buffer = Matrix{datatype}(undef, 2*section_height,
                                                        section_width)
        pivoting_buffers = Array{datatype,3}(undef, 2*section_width, section_width,
                                             max_local_pivoting_blocks)
        pivoting_row_index_buffers = Matrix{Int64}(undef, 2*section_width,
                                                   max_local_pivoting_blocks)
        section_pair_lu_buffer = Matrix{datatype}(undef, 2*section_width, section_width)
    else
        first_pivoting_buffers = zeros(datatype, 0, 0, 0)
        first_pivoting_row_index_buffers = zeros(Int64, 0, 0)
        pivoting_buffers = zeros(datatype, 0, 0, 0)
        pivoting_row_index_buffers = zeros(Int64, 0, 0)
        first_section_pair_lu_buffer = zeros(datatype, 0, 0)
        section_pair_lu_buffer = zeros(datatype, 0, 0)
    end
    pivot_send_requests = [MPI.REQUEST_NULL for _ ∈ 1:factorization_n_tiles+2]
    pivot_recv_requests = [MPI.REQUEST_NULL for _ ∈ 1:factorization_n_tiles+2]

    return (; factors, row_permutation, section_K, section_L, section_k, section_l,
            section_height, section_width, factorization_matrix_parts,
            factorization_matrix_parts_row_ranges, factorization_matrix_parts_col_ranges,
            factorization_tile_size, factorization_n_tiles, first_pivot_section_k,
            first_row_with_diagonal, first_pivoting_buffers,
            first_pivoting_row_index_buffers, first_section_pair_lu_buffer,
            pivoting_buffers, pivoting_row_index_buffers, section_pair_lu_buffer,
            pivot_send_requests, pivot_recv_requests)
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
        if sub_column == section_l
            if p == n_tiles
                # This is the last, bottom-right section, so we just LU in serial.
                if A_lu.section_k = A_lu.section_K
                    section_width = length(A_lu.factorization_matrix_parts_col_ranges[end])
                    last_block = @view A_lu.factorization_matrix_parts[end-section_width+1:end,:,end]
                    # Note the in-place version `lu!` actually saves the L and U factors
                    # into last_block.
                    last_section_lu = lu!(last_block)
                end
                # LU is finished, there are no more sub columns to handle.
                break
            end

            # This rank owns sub_column, so participates in generating pivots.
            generate_pivots!(A_lu, p)
        end
        pivot_sub_column!(A_lu, p, sub_column)
        LU_sub_column!(A_lu, p, sub_column)
    end
end

function generate_pivots!(A_lu, p)
    if A_lu.section_K == 1
        error("does this case need special handling? No pieces to swap?")
    end
    if A_lu.section_K == 2
        error("does this case need special handling? second section might not be full size?")
    end
    comm = A_lu.comm
    rank = A_lu.comm_rank
    comm_size = A_lu.comm_rank
    recv_reqs = A_lu.pivot_recv_requests
    send_reqs = A_lu.pivot_send_requests
    n_tiles = A_lu.factorization_n_tiles
    tile_size = A_lu.factorization_tile_size
    section_k = A_lu.section_k
    section_K = A_lu.section_K
    section_height = A_lu.section_height
    section_width = A_lu.section_width
    first_pivot_section_k = A_lu.first_pivot_section_k
    first_pivoting_buffers = A_lu.first_pivoting_buffers
    first_pivoting_row_index_buffers = A_lu.first_pivoting_row_index_buffers
    first_section_pair_lu_buffer = A_lu.first_section_pair_lu_buffer
    pivoting_buffers = A_lu.first_pivoting_buffers
    pivoting_row_index_buffers = A_lu.first_pivoting_row_index_buffers
    matrix_parts = A_lu.factorization_matrix_parts
    row_ranges = A_lu.factorization_matrix_parts_row_ranges
    col_ranges = A_lu.factorization_matrix_parts_col_ranges

    # The first step of the tree-reduction of the pivot candidates is special, because the
    # input matrices are potentially non-square (they may be taller than they are wide
    # away from tile edges, and matrices from locations at the bottom of a tile may be
    # truncated). On all following steps, the input matrices are square, with a size
    # (section_width,section_width) given by the number of pivot rows we are searching
    # for. We therefore have a first step here which requires quite a bit of special
    # handling, followed by a call to a more generic function that recurses until the
    # binary tree has only one element left, which contains the final pivots.

    function receive_next_section!(counter, tile)
        next_part_rank = rank + section_L
        if section_k == section_K - 1
            # This is the second-last section on the tile, the next section is the last on
            # the tile, which might not be full height. `buffer_last_row` should be equal
            # to the height of the last two sections on the tile.
            buffer_last_row = tile_size - (section_K - 2) * section_height
        else
            # If this section is neither the last nor second-last on this tile, then the
            # next section is certainly full height.
            # If this section is the last row on this tile, then the next section is the
            # first one on the next tile, which is always full height.
            buffer_last_row = 2 * section_height
        end
        recv_reqs[2*counter+1] =
            MPI.Irecv!(@view first_pivoting_buffers[section_height:buffer_last_row,:,counter+1],
                       comm; source=next_part_rank, tag=tile)
        recv_reqs[2*counter+2] =
            MPI.Irecv!(@view
                       first_pivoting_row_index_buffers[section_height:buffer_last_row,counter+1],
                       comm; source=next_part_rank, tag=tile)
        return nothing
    end

    function receive_previous_section!(counter, tile)
        previous_part_rank = rank - section_L
        if section_k == 1
            # This is the first section on the tile, the previous section is the last on
            # the previous tile, which might not be full height. `buffer_first_row` should
            # be equal to the difference in height between a standard section and the last
            # section on the tile.
            buffer_first_row = tile_size - (section_K - 1) * section_height + 1
            tag = tile - 1
        else
            # For all other sections then the previous section is certainly full height.
            buffer_first_row = 1
            tag = tile
        end
        recv_reqs[2*counter+1] =
            MPI.Irecv!(@view first_pivoting_buffers[buffer_first_row:section_height,:,counter+1],
                       comm; source=next_part_rank, tag=tag)
        recv_reqs[2*counter+2] =
            MPI.Irecv!(@view
                       first_pivoting_row_index_buffers[buffer_first_row:section_height,counter+1],
                       comm; source=next_part_rank, tag=tag)
        return nothing
    end

    function send_to_next_section!(counter, tile)
        next_part_rank = rank + section_L
        send_reqs[2*counter+1] =
            MPI.Isend!(@view matrix_parts[:,:,tile], comm; dest=next_part_rank, tag=tile)
        send_reqs[2*counter+2] =
            MPI.Isend!(@view
                       first_pivoting_row_index_buffers[section_height:buffer_last_row,counter+1],
                       comm; source=next_part_rank, tag=tile)
        return nothing
    end

    function send_to_previous_section!(counter, tile)
        previous_part_rank = rank - section_L
        if section_k == 1
            # This is the first section on the tile, the previous section is the last on
            # the previous tile, which might not be full height. `buffer_first_row` should
            # be equal to the difference in height between a standard section and the last
            # section on the tile.
            buffer_first_row = tile_size - (section_K - 1) * section_height + 1
            tag = tile - 1
        else
            # For all other sections then the previous section is certainly full height.
            buffer_first_row = 1
            tag = tile
        end
        recv_reqs[2*counter+1] =
            MPI.Irecv!(@view first_pivoting_buffers[buffer_first_row:section_height,:,counter+1],
                       comm; source=next_part_rank, tag=tag)
        recv_reqs[2*counter+2] =
            MPI.Irecv!(@view first_pivoting_row_index_buffers[buffer_first_row:section_height,counter+1],
                       comm; source=next_part_rank, tag=tag)
        return nothing
    end

    # Only include parts on or below the matrix diagonal. The section that contains the
    # diagonal requires some special handling.
    # Notionally we loop over all the diagonal-and-below sections in each step below, but
    # this rank only handles its own sections, and so 'skips through' the loop.

    # First handle all the matrix-section communications
    n_received_sections = 0
    n_sent_sections = 0
    if section_k < first_pivot_section_k
        first_remaining_tile = p + 1
    else
        first_remaining_tile = p
    end
    section_K_is_odd = section_K % 2 == 1
    for (tile_counter, tile) ∈ enumerate(first_remaining_tile:n_tiles)
        # Find which section in the part of the column from diagonal down to the bottom of
        # the matrix we are currently handling.
        section_counter = section_k - first_pivot_section_k + 1 + (tile - p) * section_K
        section_pair_counter, section_pair_which = divrem(section_counter - 1, 2) .+ 1

        # We are setting up a binary tree. When section_counter is odd, this rank receives
        # a section from the next rank. When section_counter is even, this rank sends a
        # section to the next rank.
        # There is a set of blocks of two sections, and we want to evenly distribute them
        # among the ranks participating in this pivot calculation (of which there are
        # `section_K`). Therefore as we iterate through the pairs, the rank out of the two
        # holding the pair that owns the lowest existing number of pairs receives the
        # pair, breaking ties by choosing the lower section_k of the pair. To achieve
        # this: when section_K is odd, we just choose the first rank of the two owning the
        # pair every time; when section_K is even, we choose the first rank on odd loops
        # through the ranks, then the second on the even loops through the ranks.
        if section_K_is_odd
            if section_pair_which == 1
                # Receive the other member of the pair.
                receive_next_section!(n_received_sections, tile)
                n_received_sections += 1
            else
                send_to_previous_rank!(n_sent_sections, tile)
                n_sent_sections += 1
            end
        else
            if tile_counter % 2 == 1
                if section_pair_which == 1
                    # Receive the other member of the pair.
                    receive_next_section!(n_received_sections, tile)
                    n_received_sections += 1
                else
                    # Send to the rank holding the other member of the pair.
                    send_to_previous_rank!(n_sent_sections, tile)
                    n_sent_sections += 1
                end
            else
                if section_pair_which == 1
                    # Send to the rank holding the other member of the pair.
                    send_to_next_rank!(n_sent_sections, tile)
                    n_sent_sections += 1
                else
                    # Receive the other member of the pair.
                    receive_previous_section!(n_received_sections, tile)
                    n_received_sections += 1
                end
            end
        end
    end

    # For receiving ranks, fill in the locally-owned part, wait for the part being
    # communicated, and do the LU factorization to get the potential pivot rows.
    if first_pivot_section_k == section_k
        # This rank contains the first diagonal row. However, that row might not be at the
        # beginning of matrix_parts[p,p], so need to handle the possibility of only
        # including part of the locally-owned section. As the sections are taller than
        # they are wide (if they are non-square), the first two sections are certain to
        # contain all the diagonal rows, which are therefore included in `buffer1` below.
        # `generate_pivots!()` is not called for the very last square piece of the matrix,
        # so there are always at least two sections.
        first_row = A_lu.first_row_with_diagonal
        buffer1 = @view first_pivoting_buffers[first_row:2*section_height,:,1]
        lu_buffer1 = @view first_section_pair_lu_buffer[first_row:2*section_height,:]
        rowind_buffer1 = @view first_pivoting_row_index_buffers[first_row:2*section_height,1]
        @views first_pivoting_buffers[first_row:section_height,:,1] .=
            matrix_parts[first_row:end,:,p]
        section_start = (p - 1) * tile_size + (section_k - 1) * section_height
        @views rowind_buffer1[first_row:section_height,:,1] .=
            (section_start+first_row:section_start+section_height)
        MPI.Wait(recv_reqs[1])
        lu_buffer1 .= buffer1_lu
        buffer1_lu = lu!(lu_buffer1)
        MPI.Wait(recv_reqs[2])
        # Note that section_width is always ≤section_height, so this slice will always be
        # in range.
        pivot_candidates1 = rowind_buffer1[buffer1_lu.p[1:section_width]]
    end

    MPI.Waitall(send_reqs)
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
