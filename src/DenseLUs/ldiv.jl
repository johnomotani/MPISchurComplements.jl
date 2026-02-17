function setup_ldiv(m::Int64, datatype::Type, tile_size::Int64, shared_comm::MPI.Comm,
                    shared_comm_size::Int64, shared_comm_rank::Int64,
                    distributed_comm::MPI.Comm, distributed_comm_size::Int64,
                    distributed_comm_rank::Int64, is_root::Bool,
                    allocate_shared_float::Ff, allocate_shared_int::Fi) where {Ff,Fi}
    vec_buffer1 = allocate_shared_float(m)
    vec_buffer2 = allocate_shared_float(m)
    L_rhs_update_buffer = allocate_shared_float(m)
    U_rhs_update_buffer = allocate_shared_float(m)

    # Number of tiles. Note that the final tile may be smaller than the others.
    n_tiles = (m + tile_size - 1) ÷ tile_size

    # Generate a 'task list' to be executed by the processes in shared_comm when executing
    # L_solve!() and U_solve!(). The list will be executed in steps. At each step each
    # process will have either one or zero units of work to do, and there will be a
    # synchronize_shared() call after each step.
    n_steps_max = (n_tiles * (n_tiles + 1)) ÷ 2 # This is the number of steps if the tiles
                                                # were handled in serial.
    if shared_comm_rank == 0 && distributed_comm_rank > 0
        first_step_for_column = fill(0, n_tiles)
    else
        first_step_for_column = Int64[]
    end
    if is_root
        diagonal_indices = fill(0, n_steps_max)
        first_unhandled_column_in_row = ones(Int64, n_tiles)
        # The pairs of entries in `tiles_for_rank` are the (row,column) of a tile. We
        # build up a list of these Tuples for each rank, indicating the tile that that
        # rank should work on on each step.
        tiles_for_rank = fill(-1, 2, n_steps_max, shared_comm_size * distributed_comm_size)
        next_diagonal_tile = 1
        step = 1
        rows_with_tasks = zeros(Int64, shared_comm_size, distributed_comm_size)
        diagonal_distances_row_maxima = fill(typemin(Int64), n_tiles)
        first_incomplete_column = 1
        while next_diagonal_tile ≤ n_tiles
            this_diagonal_tile = next_diagonal_tile
            if first_unhandled_column_in_row[this_diagonal_tile] == this_diagonal_tile
                # All the off-diagonal tiles in this row have been handled already, so we can
                # do the solve using the triangular element from the block-diagonal.
                tiles_for_rank[1,step,1] = this_diagonal_tile
                tiles_for_rank[2,step,1] = this_diagonal_tile
                # As the root of shared_comm is solving a diagonal tile, do not give it an
                # off-diagonal tile on this step.
                next_diagonal_tile += 1
                first_unhandled_column_in_row[this_diagonal_tile] += 1
                diagonal_indices[step] = this_diagonal_tile
            else
                # Cannot operate on a diagonal element on this step.
                # Instead, the root of shared_comm works on tiles from the
                # this_diagonal_tile row.
                tiles_for_rank[1,step,1] = this_diagonal_tile
                this_column = first_unhandled_column_in_row[this_diagonal_tile]
                first_unhandled_column_in_row[this_diagonal_tile] += 1
                tiles_for_rank[2,step,1] = this_column
            end

            # Assign the other ranks tiles each from a different row, so that they will write
            # to distinct elements of the RHS vector. Pick the tile that is furthest from the
            # diagonal crossing `this_diagonal_tile`. Hopefully this gives a balance between
            # filling rows and filling columns, to minimize the number of tasks that have no
            # work to do.
            # Note using one-based indexing of ranks for convenience here, so `rank`
            # corresponds to `shared_comm_rank+1`.
            rows_with_tasks .= 0
            # Exclude all columns ≥this_diagonal_tile, as they cannot be processed yet.
            # The first remaining element in each row has the largest 'diagonal distince'
            # of all entries in the row, so only need to check that one.
            function set_row_maximum!(irow, first_unhandled_column_in_row,
                                      first_incomplete_column, this_diagonal_tile,
                                      diagonal_distances_row_maxima, n_tiles)
                icolumn = first_unhandled_column_in_row[irow]
                break_loop = false
                if icolumn ≥ this_diagonal_tile
                    diagonal_distances_row_maxima[irow] = typemin(Int64)
                elseif icolumn == first_incomplete_column
                    # Remaining rows all have the minimum in this column.
                    for jrow ∈ irow:n_tiles
                        diagonal_distances_row_maxima[jrow] = 2 * this_diagonal_tile - jrow - icolumn
                    end
                    break_loop = true
                else
                    diagonal_distances_row_maxima[irow] = 2 * this_diagonal_tile - irow - icolumn
                end
                return break_loop
            end
            for irow ∈ this_diagonal_tile+1:n_tiles
                break_loop = set_row_maximum!(irow, first_unhandled_column_in_row,
                                              first_incomplete_column, this_diagonal_tile,
                                              diagonal_distances_row_maxima, n_tiles)
                if break_loop
                    break
                end
            end
            this_diagonal_distances_row_maxima = @view diagonal_distances_row_maxima[this_diagonal_tile+1:n_tiles]
            for block ∈ 1:distributed_comm_size, sr ∈ 1:shared_comm_size
                rank = (block - 1) * shared_comm_size + sr
                if rank == 1 || length(this_diagonal_distances_row_maxima) == 0
                    continue
                end
                max_distance = maximum(this_diagonal_distances_row_maxima)
                if max_distance == typemin(Int64)
                    # No work available.
                else
                    for irow ∈ this_diagonal_tile+1:n_tiles
                        rowmax = diagonal_distances_row_maxima[irow]
                        if rowmax == max_distance && !(irow ∈ @view rows_with_tasks[:,block])
                            tiles_for_rank[1,step,rank] = irow
                            icolumn = first_unhandled_column_in_row[irow]
                            tiles_for_rank[2,step,rank] = icolumn
                            first_unhandled_column_in_row[irow] += 1
                            rows_with_tasks[sr,block] = irow
                            if icolumn == n_tiles
                                first_incomplete_column = icolumn
                            end

                            # Re-calculate diagonal_distances_row_maxima[irow] because we
                            # just removed the maximum-distance entry from this row.
                            set_row_maximum!(irow, first_unhandled_column_in_row,
                                             first_incomplete_column, this_diagonal_tile,
                                             diagonal_distances_row_maxima, n_tiles)

                            break
                        end
                    end
                end
            end

            step += 1
        end

        n_steps = Ref(step - 1)
        tiles_for_rank = tiles_for_rank[:,1:n_steps[],:]

        # Sort the tiles so that the shared_comm_rank=0 process on each block handles the
        # 'lowest'/'highest' row for the L/U solve. This avoids the need for a
        # synchronization call.
        for block ∈ 0:distributed_comm_size-1
            ranks = block*shared_comm_size+1:(block+1)*shared_comm_size
            for step ∈ 1:n_steps[]
                # A value of -1 indicates no tile, so sort this after any positive
                # integers.
                sorted_inds = sortperm(@view(tiles_for_rank[1,step,ranks]);
                                       lt=(x,y)->x==-1 ? false : y==-1 ? true : x < y)
                # Note do *not* use views to do this, as we want to create a permuted
                # intermediate copy, then copy that into the original array. In-place
                # would be trickier and any attempt at optimization would be overkill here
                # (see https://discourse.julialang.org/t/fastest-way-to-permute-array-given-some-permutation/49687).
                @. tiles_for_rank[:,step,ranks] .= tiles_for_rank[:,step,ranks[sorted_inds]]
            end
        end

        # Need to check which steps need synchronization after sorting.
        # While checking, attempt to re-arrange the tiles among the shared-memory ranks to
        # reduce the need to synchronize. Only do this by rearranging each step in turn.
        # Possibly there would be more optimal ways of doing this with some global
        # rearrangement of tiles/ranks?
        step_needs_synchronize = fill(false, n_steps[], distributed_comm_size)
        rank_worked_on_row_since_synchronize = zeros(Int64, n_tiles, distributed_comm_size)
        rank_worked_on_row_current_step = zeros(Int64, n_tiles, distributed_comm_size)
        this_diagonal_tile = 1
        for step ∈ 1:n_steps[]
            # Except when handling a diagonal tile, the 'next row' to the one being
            # handled by the global root is this_diagonal_tile+2, ...
            diagonal_offset = 2
            if diagonal_indices[step] > 0
                this_diagonal_tile = diagonal_indices[step]
                # ...but on the step where the diagonal tile is handled, the 'next row' is
                # this_diagonal_tile+1.
                diagonal_offset = 1
            end
            for block ∈ 1:distributed_comm_size #, sr ∈ 1:shared_comm_size
                ranks = [(block - 1) * shared_comm_size + sr for sr ∈ 1:shared_comm_size]
                this_step_rows = tiles_for_rank[1,step,ranks]

                # Modified array that skips entries that have no tile to work on. These
                # should be gathered at the end of this_step_rows.
                first_missing_tile = findfirst(x -> x==-1, this_step_rows)
                if first_missing_tile !== nothing
                    this_step_rows_no_missing = this_step_rows[1:first_missing_tile-1]
                    if any(this_step_rows[first_missing_tile:end] .≥ 0)
                        error("expected all tiles after first missing tile to be '-1'. "
                              * "Got $this_step_rows.")
                    end
                else
                    this_step_rows_no_missing = this_step_rows
                end

                sorted_this_step_rows = sort(this_step_rows)
                existing_processes_for_rows = rank_worked_on_row_since_synchronize[this_step_rows_no_missing,block]
                # Pad existing_processes_for_rows with zeros so that it has length
                # shared_comm_size.
                existing_processes_for_rows =
                    vcat(existing_processes_for_rows,
                         zeros(Int64, shared_comm_size -
                               length(existing_processes_for_rows)))

                counts = countmap(existing_processes_for_rows)
                # Any entries that are 0 are rows that have not been accessed since the
                # last synchronization, so any rank can take them.
                pop!(counts, 0, 0)
                if any(x > 1 for x in values(counts))
                    # Need to syncronize at the end of the previous step because there are
                    # multiple rows that have been accessed by the same rank, so we cannot
                    # distribute these rows among the ranks without synchronizing.
                    step_needs_synchronize[step-1, block] = true
                    rank_worked_on_row_since_synchronize[:,block] .= 0
                end

                free_rows = [r for r ∈ this_step_rows_no_missing
                             if rank_worked_on_row_since_synchronize[r,block] == 0]

                function swap_local_ranks!(a, b)
                    if a == b
                        return nothing
                    end
                    a_rank = (block-1)*shared_comm_size + a
                    b_rank = (block-1)*shared_comm_size + b
                    tiles_for_rank[:,step,[a_rank,b_rank]] .= tiles_for_rank[:,step,[b_rank,a_rank]]
                    this_step_rows[[a,b]] .= this_step_rows[[b,a]]
                    existing_processes_for_rows[[a,b]] .= existing_processes_for_rows[[b,a]]
                    return nothing
                end

                function delete_from_free_rows!(row)
                    irow = findfirst(x -> x==row, free_rows)
                    if irow !== nothing
                        deleteat!(free_rows, irow)
                    end
                    return nothing
                end

                root_rank = (block - 1) * shared_comm_size + 1

                if block == 1
                    # The root process always handle the first incomplete row. This is
                    # already set. Just need to delete this row from free_rows to prevent
                    # a later rank from trying to take it.
                    delete_from_free_rows!(tiles_for_rank[1,step,1])
                elseif sorted_this_step_rows[1] == this_diagonal_tile + diagonal_offset
                    # This row should be handled by the root of the shared-memory block as
                    # this avoids some synchronization. Note that this_step_rows is
                    # sorted, so only the first entry can be the second incomplete row, as
                    # the first incomplete row is always handled by the global root.
                    rank_with_row = findfirst(x -> x == this_diagonal_tile + diagonal_offset,
                                              this_step_rows)
                    swap_local_ranks!(1, rank_with_row)
                    if root_rank ∈ existing_processes_for_rows[2:end] ||
                            existing_processes_for_rows[2:end] ∉ (0, root_rank)
                        # As we have required this root_rank to take the row
                        # `this_diagonal_tile + diagonal_offset`, we need to
                        # re-synchronize if it has already accessed any other row since
                        # the last synchronization, or if another process has already
                        # accessed `this_diagonal_tile + diagonal_offset`.
                        step_needs_synchronize[step-1, block] = true
                        rank_worked_on_row_since_synchronize[:,block] .= 0
                        existing_processes_for_rows .= 0
                        # Need to re-create free_rows since
                        # rank_worked_on_row_since_synchronize is being reset.
                        # Don't need to worry about the row handled by the global-root,
                        # because this branch only handles `block > 1`.
                        free_rows = copy(this_step_rows_no_missing)
                    end
                    # Delete from free rows to stop another rank picking up this row.
                    delete_from_free_rows!(this_diagonal_tile + diagonal_offset)
                elseif root_rank ∈ existing_processes_for_rows
                    rank_with_row = findfirst(x -> x == root_rank, existing_processes_for_rows)
                    swap_local_ranks!(1, rank_with_row)
                else
                    # If there are no free rows, there is no work for this rank to do. Its
                    # tile should be assigned [-1,-1] already, or will be by a swap with
                    # another rank that needs to claim the tile that this rank currently
                    # owns.
                    if length(free_rows) > 0
                        # Take the lowest row that does not have an associated rank already.
                        irow = argmin(free_rows)
                        row = free_rows[irow]
                        deleteat!(free_rows, irow)
                        swap_local_ranks!(1, findfirst(x -> x == row, this_step_rows))
                    end
                end

                # Now handle the non-root ranks of the shared-memory block.
                for sr ∈ 2:shared_comm_size
                    rank = (block - 1) * shared_comm_size + sr
                    if rank ∈ existing_processes_for_rows
                        rank_with_row = findfirst(x -> x == rank, existing_processes_for_rows)
                        swap_local_ranks!(sr, rank_with_row)
                    else
                        # If there are no free rows, there is no work for this rank to do. Its
                        # tile should be assigned [-1,-1] already, or will be by a swap with
                        # another rank that needs to claim the tile that this rank currently
                        # owns.
                        if length(free_rows) > 0
                            # Take the lowest row that does not have an associated rank already.
                            irow = argmin(free_rows)
                            row = free_rows[irow]
                            deleteat!(free_rows, irow)
                            swap_local_ranks!(sr, findfirst(x -> x == row, this_step_rows))
                        end
                    end
                end

                # this_step_rows was updated as rows were swapped, so now contains the
                # final rows handled by each rank in the shared-memory block.
                for (sr, row) ∈ enumerate(this_step_rows)
                    if row < 0
                        continue
                    end
                    rank = (block - 1) * shared_comm_size + sr
                    if rank_worked_on_row_since_synchronize[row,block] ∉ (0, rank)
                        error("Sorting has failed! "
                              * "rank_worked_on_row_since_synchronize[$row,$block] "
                              * "should be either 0 or $rank, but is "
                              * "$(rank_worked_on_row_since_synchronize[row,block]).")
                    end
                    rank_worked_on_row_since_synchronize[row,block] = rank
                end
            end
            if diagonal_indices[step] > 0
                # Always need to synchronize after diagonal steps, so that the new
                # solution vector entries can be used by non-root proceses.
                step_needs_synchronize[step,:] .= true
                rank_worked_on_row_since_synchronize .= 0
            end
            rank_worked_on_row_current_step .= 0
        end
        # Always synchronize on the final step.
        step_needs_synchronize[end,:] .= true
        # Convert because Int64 is more convenient to communicate over MPI than Bool.
        step_needs_synchronize = Int64.(step_needs_synchronize)

        diagonal_indices = diagonal_indices[1:n_steps[]]

        MPI.Bcast!(n_steps, distributed_comm; root=0)
        MPI.Bcast!(n_steps, shared_comm; root=0)
        MPI.Bcast!(diagonal_indices, distributed_comm; root=0)

        step_needs_synchronize_this_block = allocate_shared_int(n_steps[])

        reqs = MPI.Request[]
        # First handle the processes in the same shared_comm as root.
        # Note that `sr`, `block`, and `rank` are 0-based indices.
        for sr ∈ 1:shared_comm_size-1
            push!(reqs, MPI.Isend(@view(tiles_for_rank[:,:,sr+1]), shared_comm;
                                  dest=sr))
        end
        step_needs_synchronize_this_block .= @view step_needs_synchronize[:,1]
        for block ∈ 1:distributed_comm_size-1
            # Send sr=0 last, because it is the one that does not need to be passed on by
            # the sr=0 process.
            for sr ∈ [1:shared_comm_size-1...,0]
                rank = block * shared_comm_size + sr
                push!(reqs, MPI.Isend(@view(tiles_for_rank[:,:,rank+1]), distributed_comm;
                                      dest=block, tag=sr))
            end
            push!(reqs, MPI.Isend(@view(step_needs_synchronize[:,block+1]),
                                  distributed_comm; dest=block, tag=shared_comm_size))
        end
        MPI.Waitall(reqs)
        my_tiles_for_rank = tiles_for_rank[:,:,1]
    elseif shared_comm_rank == 0
        n_steps = Ref(-1)
        MPI.Bcast!(n_steps, distributed_comm; root=0)
        MPI.Bcast!(n_steps, shared_comm; root=0)
        diagonal_indices = fill(0, n_steps[])
        MPI.Bcast!(diagonal_indices, distributed_comm; root=0)
        step_needs_synchronize_this_block = allocate_shared_int(n_steps[])
        # Receive tiles_for_rank for each process in this block, and then pass them on.
        # Use my_tiles_for_rank as a buffer to pass on these arrays.
        my_tiles_for_rank = zeros(Int64, 2, n_steps[])
        for rank ∈ 1:shared_comm_size-1
            MPI.Recv!(my_tiles_for_rank, distributed_comm; source=0, tag=rank)

            # Check for the last appearances of rows and first appearances of columns
            # while we are passing through my_tiles_for_rank to each process in the block.
            for step ∈ 1:n_steps[]
                column = my_tiles_for_rank[2,step]
                if column > 0 && (step < first_step_for_column[column] ||
                                  first_step_for_column[column] == 0)
                    first_step_for_column[column] = step
                end
            end

            MPI.Send(my_tiles_for_rank, shared_comm; dest=rank)
        end
        MPI.Recv!(my_tiles_for_rank, distributed_comm; source=0, tag=0)
        MPI.Recv!(step_needs_synchronize_this_block, distributed_comm; source=0,
                  tag=shared_comm_size)
        # Finish checking for the last appearances of rows and first appearances of
        # columns while we are passing through my_tiles_for_rank to each process in the
        # block.
        for step ∈ 1:n_steps[]
            column = my_tiles_for_rank[2,step]
            if column > 0 && (step < first_step_for_column[column] ||
                              first_step_for_column[column] == 0)
                first_step_for_column[column] = step
            end
        end
    else
        n_steps = Ref(-1)
        diagonal_indices = Int64[]
        MPI.Bcast!(n_steps, shared_comm; root=0)
        step_needs_synchronize_this_block = allocate_shared_int(n_steps[])
        my_tiles_for_rank = zeros(Int64, 2, n_steps[])
        MPI.Recv!(my_tiles_for_rank, shared_comm; source=0)
    end

    # Need to find the steps where the following step is the first one that uses the
    # solution on a certain tile.
    if shared_comm_rank == 0 && distributed_comm_rank > 0
        new_column_triggers = zeros(Int64, shared_comm_size, n_steps[])
        for (tile, step) ∈ enumerate(first_step_for_column)
            for i ∈ 1:shared_comm_size
                if step > 0 && new_column_triggers[i,step-1] == 0
                    new_column_triggers[i,step-1] = tile
                    break
                end
            end
        end
    else
        new_column_triggers = zeros(Int64, 0, 0)
    end

    # Also need to trigger synchronization after every step where we wait for 'new column'
    # data. The MPI.Wait() is only called on the root process of the shared-memory block,
    # so without a synchronize_shared() call, there would be a risk of other processes
    # using the data before the MPI.Wait() completed.
    if shared_comm_rank == 0 && distributed_comm_rank > 0
        for step ∈ 1:n_steps[]
            if any(@view(new_column_triggers[:,step]) .> 0)
                # We call MPI.Wait() one or more times on this step.
                step_needs_synchronize_this_block[step] = 1
            end
        end
    end

    # Store the tiles that will be handled by this process in contiguous arrays.
    function get_L_tile_index_range(itile)
        if itile == -1
            return 1:0
        else
            return (itile-1)*tile_size+1:min(itile*tile_size,m)
        end
    end
    my_L_tiles = fill(datatype(NaN), tile_size, tile_size, n_steps[])
    my_L_tile_row_ranges = Vector{UnitRange{Int64}}(undef, n_steps[])
    my_L_tile_col_ranges = Vector{UnitRange{Int64}}(undef, n_steps[])
    for i ∈ 1:n_steps[]
        tile = @view my_tiles_for_rank[:,i]
        my_L_tile_row_ranges[i] = get_L_tile_index_range(tile[1])
        my_L_tile_col_ranges[i] = get_L_tile_index_range(tile[2])
    end
    if shared_comm_rank == 0
        L_receive_requests = fill(MPI.REQUEST_NULL, n_tiles)
        L_send_requests = fill(MPI.REQUEST_NULL, n_tiles)
    else
        L_receive_requests = MPI.Request[]
        L_send_requests = MPI.Request[]
    end

    # When dealing with the upper-triangular 'U' matrix, we count the tiles from the
    # bottom-right corner, so bottom-to-top for rows, and right-to-left for columns.
    function get_U_tile_index_range(itile)
        if itile == -1
            return 1:0
        else
            return max(m-itile*tile_size+1,1):m-(itile-1)*tile_size
        end
    end
    my_U_tiles = fill(datatype(NaN), tile_size, tile_size, n_steps[])
    my_U_tile_row_ranges = Vector{UnitRange{Int64}}(undef, n_steps[])
    my_U_tile_col_ranges = Vector{UnitRange{Int64}}(undef, n_steps[])
    for i ∈ 1:n_steps[]
        tile = @view my_tiles_for_rank[:,i]
        my_U_tile_row_ranges[i] = get_U_tile_index_range(tile[1])
        my_U_tile_col_ranges[i] = get_U_tile_index_range(tile[2])
    end
    if shared_comm_rank == 0
        U_receive_requests = fill(MPI.REQUEST_NULL, n_tiles)
        U_send_requests = fill(MPI.REQUEST_NULL, n_tiles)
    else
        U_receive_requests = MPI.Request[]
        U_send_requests = MPI.Request[]
    end

    return (; my_L_tiles, my_L_tile_row_ranges, my_L_tile_col_ranges, L_receive_requests,
            L_send_requests, my_U_tiles, my_U_tile_row_ranges, my_U_tile_col_ranges,
            U_receive_requests, U_send_requests, diagonal_indices, new_column_triggers,
            step_needs_synchronize_this_block, vec_buffer1, vec_buffer2,
            L_rhs_update_buffer, U_rhs_update_buffer, n_tiles)
end

function ldiv!(A_lu::DenseLU{T}, b::AbstractVector{T}) where T
    return ldiv!(b, A_lu, b)
end

function ldiv!(x::AbstractVector{T}, A_lu::DenseLU{T}, b::AbstractVector{T}) where T
    is_root = A_lu.is_root
    row_permutation = A_lu.row_permutation
    b_permuted = A_lu.vec_buffer1
    y = A_lu.vec_buffer2
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    # Permute the RHS, storing in buffer2. This accounts for 'row permutations' that were
    # generated/used for 'pivoting' when the L and U factors were computed.
    if is_root
        # Could parallelise this?
        @views b_permuted .= b[row_permutation]
    end

    L_solve!(y, A_lu, b_permuted)
    U_solve!(x, A_lu, y)

    # Clean up MPI requests. These should all have been completed already, so this should
    # not take any time.
    if is_root
        MPI.Waitall(A_lu.L_receive_requests)
        MPI.Waitall(A_lu.U_receive_requests)
    elseif shared_comm_rank == 0
        MPI.Waitall(A_lu.L_send_requests)
        MPI.Waitall(A_lu.U_send_requests)
        MPI.Waitall(A_lu.L_receive_requests)
        MPI.Waitall(A_lu.U_receive_requests)
    end

    return x
end

function L_solve!(y, A_lu::DenseLU{T}, b) where T
    m = A_lu.m
    n_tiles = A_lu.n_tiles
    tile_size = A_lu.tile_size
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_row_ranges = A_lu.my_L_tile_row_ranges
    my_L_tile_col_ranges = A_lu.my_L_tile_col_ranges
    diagonal_indices = A_lu.diagonal_indices
    synchronize_shared = A_lu.synchronize_shared
    L_receive_requests = A_lu.L_receive_requests
    L_send_requests = A_lu.L_send_requests
    new_column_triggers = A_lu.new_column_triggers
    step_needs_synchronize_this_block = A_lu.step_needs_synchronize_this_block
    L_rhs_update_buffer = A_lu.L_rhs_update_buffer
    shared_comm_rank = A_lu.shared_comm_rank
    distributed_comm = A_lu.distributed_comm

    if shared_comm_rank == 0
        L_rhs_update_buffer .= 0.0
    end

    if A_lu.is_root
        for step ∈ 1:length(my_L_tile_row_ranges)
            diagonal_tile = diagonal_indices[step]
            row_range = my_L_tile_row_ranges[step]
            col_range = my_L_tile_col_ranges[step]
            if diagonal_tile > 0
                # Wait to ensure that contributions from all other blocks have been added
                # to `b`.
                MPI.Wait(L_receive_requests[diagonal_tile])
                # Root process always wrote to b[tile_range] on the previous step, so no
                # need to synchronize before this calculation.
                # Still need to add this block's contributions to `b`.
                @views @. y[col_range] = b[col_range] + L_rhs_update_buffer[col_range]
                # Need the [1:length(row_range),1:length(col_range)] selection, even
                # though for most tiles this is just the full range, because the last row
                # and column may have a different size
                @views trsv!('L', 'N', 'U',
                             my_L_tiles[1:length(row_range),1:length(col_range),step],
                             y[col_range])
                if diagonal_tile < n_tiles
                    L_send_requests[diagonal_tile] = temp_Ibcast!(@view(y[col_range]),
                                                                  distributed_comm; root=0)
                    # Start MPI.Ireduce!() ready for the next diagonal tile. MPI
                    # non-blocking collective operations have to be called in the same
                    # order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm),
                    # so we cannot start this operation earlier.
                    t = diagonal_tile+1
                    L_receive_requests[t] =
                        temp_Ireduce!(@view(b[(t-1)*tile_size+1:min(t*tile_size,m)]), +,
                                      distributed_comm; root=0)
                end
            else
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_L_tiles[1:length(row_range),:,step],
                             y[col_range], one(T), L_rhs_update_buffer[row_range])
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    else
        for step ∈ 1:length(my_L_tile_row_ranges)
            row_range = my_L_tile_row_ranges[step]
            col_range = my_L_tile_col_ranges[step]
            if !isempty(row_range)
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_L_tiles[1:length(row_range),:,step],
                             y[col_range], one(T), L_rhs_update_buffer[row_range])
            end
            if shared_comm_rank == 0
                # `diagonal_indices[step]` is non-zero if the root process is handling a
                # diagonal tile on this step.
                maybe_diagonal_tile = diagonal_indices[step]
                if maybe_diagonal_tile > 0
                    # Data from the maybe_diagonal_tile is available, so start the
                    # MPI.Ibcast!(). Also the maybe_diagonal_tile+1 row is guaranteed to be
                    # completed, as only the root process will handle any tiles from that row
                    # from this step on, so start the MPI.Ireduce!(). MPI non-blocking
                    # collective operations have to be called in the same order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm), so we
                    # have to match the order that these operations are started on the root
                    # process.
                    if maybe_diagonal_tile < n_tiles
                        L_receive_requests[maybe_diagonal_tile] =
                            temp_Ibcast!(@view(y[(maybe_diagonal_tile-1)*tile_size+1:min(maybe_diagonal_tile*tile_size, m)]),
                                         distributed_comm; root=0)
                        # We have sorted the tiles so that the shared_comm_rank=0 process
                        # always handles the lowest row in the block, so if
                        # `t` was handled on this step on this block, it was definitely
                        # handled on this rank, so we do not need to synchronize.
                        t = maybe_diagonal_tile + 1
                        L_send_requests[t] =
                            temp_Ireduce!(@view(L_rhs_update_buffer[(t-1)*tile_size+1:min(t*tile_size,m)]),
                                          +, distributed_comm; root=0)
                    end
                end
                # Ensure data required for the next tiles processed on the block has arrived.
                for tile ∈ @view new_column_triggers[:,step]
                    if tile == 0
                        # No more to do
                        break
                    end
                    MPI.Wait(L_receive_requests[tile])
                end
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    end

    return nothing
end

function U_solve!(x, A_lu::DenseLU{T}, y) where T
    m = A_lu.m
    n_tiles = A_lu.n_tiles
    tile_size = A_lu.tile_size
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_row_ranges = A_lu.my_U_tile_row_ranges
    my_U_tile_col_ranges = A_lu.my_U_tile_col_ranges
    diagonal_indices = A_lu.diagonal_indices
    synchronize_shared = A_lu.synchronize_shared
    U_receive_requests = A_lu.U_receive_requests
    U_send_requests = A_lu.U_send_requests
    new_column_triggers = A_lu.new_column_triggers
    step_needs_synchronize_this_block = A_lu.step_needs_synchronize_this_block
    U_rhs_update_buffer = A_lu.U_rhs_update_buffer
    shared_comm_rank = A_lu.shared_comm_rank
    distributed_comm = A_lu.distributed_comm

    if shared_comm_rank == 0
        U_rhs_update_buffer .= 0.0
    end

    if A_lu.is_root
        for step ∈ 1:length(my_U_tile_row_ranges)
            diagonal_tile = diagonal_indices[step]
            row_range = my_U_tile_row_ranges[step]
            col_range = my_U_tile_col_ranges[step]
            if diagonal_tile > 0
                # Wait to ensure that contributions from all other blocks have been added
                # to `y`.
                MPI.Wait(U_receive_requests[diagonal_tile])
                # Root process always wrote to b[tile_range] on the previous step, so no
                # need to synchronize before this calculation.
                # Still need to add this block's contributions to `y`.
                @views @. x[col_range] = y[col_range] + U_rhs_update_buffer[col_range]
                # Need the [1:length(row_range),1:length(col_range)] selection, even
                # though for most tiles this is just the full range, because the last row
                # and column may have a different size
                @views trsv!('U', 'N', 'N',
                             my_U_tiles[1:length(row_range),1:length(col_range),step],
                             x[col_range])
                U_send_requests[diagonal_tile] = temp_Ibcast!(@view(x[col_range]),
                                                              distributed_comm; root=0)
                if diagonal_tile < n_tiles
                    # Start MPI.Ireduce!() ready for the next diagonal tile. MPI
                    # non-blocking collective operations have to be called in the same
                    # order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm),
                    # so we cannot start this operation earlier.
                    t = diagonal_tile+1
                    U_receive_requests[t] =
                        temp_Ireduce!(@view(y[max(m-t*tile_size+1,1):m-(t-1)*tile_size]),
                                      +, distributed_comm; root=0)
                end
            else
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_U_tiles[1:length(row_range),:,step],
                             x[col_range], one(T), U_rhs_update_buffer[row_range])
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    else
        for step ∈ 1:length(my_U_tile_row_ranges)
            row_range = my_U_tile_row_ranges[step]
            col_range = my_U_tile_col_ranges[step]
            if !isempty(row_range)
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_U_tiles[1:length(row_range),:,step],
                             x[col_range], one(T), U_rhs_update_buffer[row_range])
            end
            # Get data required for the next tiles processed on the block.
            if shared_comm_rank == 0
                # `diagonal_indices[step]` is non-zero if the root process is handling a
                # diagonal tile on this step.
                maybe_diagonal_tile = diagonal_indices[step]
                if maybe_diagonal_tile > 0
                    # Data from the maybe_diagonal_tile is available, so start the
                    # MPI.Ibcast!(). Also the maybe_diagonal_tile+1 row is guaranteed to be
                    # completed, as only the root process will handle any tiles from that row
                    # from this step on, so start the MPI.Ireduce!(). MPI non-blocking
                    # collective operations have to be called in the same order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm), so we
                    # have to match the order that these operations are started on the root
                    # process.
                    U_receive_requests[maybe_diagonal_tile] =
                        temp_Ibcast!(@view(x[max(m-maybe_diagonal_tile*tile_size+1,1):m-(maybe_diagonal_tile-1)*tile_size]),
                                     distributed_comm; root=0)
                    if maybe_diagonal_tile < n_tiles
                        # We have sorted the tiles so that the shared_comm_rank=0 process
                        # always handles the lowest row in the block, so if
                        # `t` was handled on this step on this block, it was definitely
                        # handled on this rank, so we do not need to synchronize.
                        t = maybe_diagonal_tile + 1
                        U_send_requests[t] =
                            temp_Ireduce!(@view(U_rhs_update_buffer[max(m-t*tile_size+1,1):m-(t-1)*tile_size]),
                                          +, distributed_comm; root=0)
                    end
                end
                for tile ∈ @view new_column_triggers[:,step]
                    if tile == 0
                        # No more to do
                        break
                    end
                    MPI.Wait(U_receive_requests[tile])
                end
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    end

    return nothing
end

# Temporarily copy functions from https://github.com/JuliaParallel/MPI.jl/pull/827, until
# that PR is merged to provide MPI.Ireduce!()
temp_Ireduce!(sendrecvbuf, op, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Ireduce!(sendrecvbuf, op, root, comm, req)
temp_Ireduce!(sendbuf, recvbuf, op, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Ireduce!(sendbuf, recvbuf, op, root, comm, req)
function temp_Ireduce!(rbuf::MPI.RBuffer, op::Union{MPI.Op,MPI.MPI_Op}, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    # int MPI_Ireduce(const void* sendbuf, void* recvbuf, int count,
    #                 MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
    #                 MPI_Request* req)
    MPI.API.MPI_Ireduce(rbuf.senddata, rbuf.recvdata, rbuf.count, rbuf.datatype, op, root, comm, req)
    MPI.setbuffer!(req, rbuf)
    return req
end
temp_Ireduce!(rbuf::MPI.RBuffer, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Ireduce!(rbuf, MPI.Op(op, eltype(rbuf)), root, comm, req)
temp_Ireduce!(sendbuf, recvbuf, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Ireduce!(MPI.RBuffer(sendbuf, recvbuf), op, root, comm, req)
# inplace
function temp_Ireduce!(buf, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    if MPI.Comm_rank(comm) == root
        temp_Ireduce!(MPI.IN_PLACE, buf, op, root, comm, req)
    else
        temp_Ireduce!(buf, nothing, op, root, comm, req)
    end
end

# Temporarily copy functions from https://github.com/JuliaParallel/MPI.jl/pull/882, until
# that PR is merged to provide MPI.Ibcast!()
temp_Ibcast!(buf, comm::MPI.Comm; root::Integer=Cint(0)) =
    temp_Ibcast!(buf, root, comm)
function temp_Ibcast!(buf::MPI.Buffer, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest = MPI.Request())
    # int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
    #   MPI_Comm comm, MPI_Request *request)
    MPI.API.MPI_Ibcast(buf.data, buf.count, buf.datatype, root, comm, req)
    return req
end
function temp_Ibcast!(data, root::Integer, comm::MPI.Comm)
    temp_Ibcast!(MPI.Buffer(data), root, comm)
end
