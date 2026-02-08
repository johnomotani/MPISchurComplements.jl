module DenseLUs

export DenseLU, dense_lu

using LinearAlgebra
using LinearAlgebra.BLAS: trsv!, gemm!
using MPI

import LinearAlgebra: lu!, ldiv!

struct DenseLU{T,Tmat,Tvec,Tsync}
    m::Int64
    n::Int64
    factors::Tmat
    row_permutation::Vector{Int64}
    diagonal_L_tiles::Vector{Matrix{T}}
    my_L_tiles::Vector{Matrix{T}}
    my_L_tile_row_ranges::Vector{UnitRange{Int64}}
    my_L_tile_col_ranges::Vector{UnitRange{Int64}}
    diagonal_U_tiles::Vector{Matrix{T}}
    my_U_tiles::Vector{Matrix{T}}
    my_U_tile_row_ranges::Vector{UnitRange{Int64}}
    my_U_tile_col_ranges::Vector{UnitRange{Int64}}
    vec_buffer1::Tvec
    vec_buffer2::Tvec
    tile_size::Int64
    n_tiles::Int64
    shared_comm::MPI.Comm
    shared_comm_rank::Int64
    shared_comm_size::Int64
    synchronize_shared::Tsync
end

function dense_lu(A::AbstractMatrix, tile_size::Int64,
                  shared_comm::Union{MPI.Comm,Nothing}, allocate_shared_float::Function,
                  allocate_shared_int::Function; synchronize_shared=nothing,
                  skip_factorization=false)
    datatype = eltype(A)

    if shared_comm === nothing
        shared_comm = MPI.COMM_SELF
    end

    if synchronize_shared === nothing
        synchronize_shared = ()->MPI.Barrier(shared_comm)
    end

    m, n = size(A)
    if m != n
        error("Non-square matrices not supported in DenseLU. Got ($m,$n).")
    end
    factors = allocate_shared_float(m, n)
    vec_buffer1 = allocate_shared_float(m)
    vec_buffer2 = allocate_shared_float(m)

    shared_comm_rank = MPI.Comm_rank(shared_comm)
    shared_comm_size = MPI.Comm_size(shared_comm)

    if shared_comm_rank == 0
        row_permutation = zeros(Int64, m)
    else
        row_permutation = zeros(Int64, 0)
    end

    # Number of tiles. Note that the final tile may be smaller than the others.
    n_tiles = (m + tile_size - 1) ÷ tile_size

    # Generate a 'task list' to be executed by the processes in shared_comm when executing
    # L_solve!() and U_solve!(). The list will be executed in steps. At each step each
    # process will have either one or zero units of work to do, and there will be a
    # synchronize_shared() call after each step.
    n_steps_max = (n_tiles * (n_tiles + 1)) ÷ 2 # This is the number of steps if the tiles
                                                # were handled in serial.
    if shared_comm_rank == 0
        diagonal_tiles = fill(-1, n_steps_max)
        first_unhandled_column_in_row = ones(Int64, n_tiles)
        # The pairs of entries in `tiles_for_rank` are the (row,column) of a tile. We
        # build up a list of these Tuples for each rank, indicating the tile that that
        # rank should work on on each step.
        tiles_for_rank = fill(-1, 2, n_steps_max, shared_comm_size)
        next_diagonal_tile = 1
        step = 1
        rows_with_tasks = zeros(Int64, shared_comm_size)
        diagonal_distances_row_maxima = fill(typemin(Int64), n_tiles)
        first_incomplete_column = 1
        while next_diagonal_tile ≤ n_tiles
            this_diagonal_tile = next_diagonal_tile
            if first_unhandled_column_in_row[this_diagonal_tile] == this_diagonal_tile
                # All the off-diagonal tiles in this row have been handled already, so we can
                # do the solve using the triangular element from the block-diagonal.
                diagonal_tiles[step] = this_diagonal_tile
                # As the root of shared_comm is solving a diagonal tile, do not give it an
                # off-diagonal tile on this step.
                next_diagonal_tile += 1
                first_unhandled_column_in_row[this_diagonal_tile] += 1
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
            for rank ∈ 2:min(shared_comm_size, n_tiles - this_diagonal_tile)
                max_distance = maximum(this_diagonal_distances_row_maxima)
                if max_distance == typemin(Int64)
                    # No work available.
                else
                    for irow ∈ this_diagonal_tile+1:n_tiles
                        rowmax = diagonal_distances_row_maxima[irow]
                        if rowmax == max_distance && !(irow ∈ rows_with_tasks)
                            tiles_for_rank[1,step,rank] = irow
                            icolumn = first_unhandled_column_in_row[irow]
                            tiles_for_rank[2,step,rank] = icolumn
                            first_unhandled_column_in_row[irow] += 1
                            rows_with_tasks[rank] = irow
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
        diagonal_tiles = diagonal_tiles[1:n_steps[]]
        tiles_for_rank = tiles_for_rank[:,1:n_steps[],:]

        MPI.Bcast!(n_steps, shared_comm; root=0)

        reqs = MPI.Request[]
        for rank ∈ 1:shared_comm_size-1
            push!(reqs, MPI.Isend(@view(tiles_for_rank[:,:,rank+1]), shared_comm; dest=rank))
        end
        MPI.Waitall(reqs)

        my_tiles_for_rank = tiles_for_rank[:,:,1]
    else
        n_steps = Ref(-1)
        MPI.Bcast!(n_steps, shared_comm; root=0)
        my_tiles_for_rank = zeros(Int64, 2, n_steps[])
        MPI.Recv!(my_tiles_for_rank, shared_comm; source=0)
    end

    # Store the tiles that will be handled by this process in contiguous arrays.
    function get_L_tile_index_range(itile)
        return (itile-1)*tile_size+1:min(itile*tile_size,m)
    end
    diagonal_L_tiles = Matrix{datatype}[]
    if shared_comm_rank == 0
        sizehint!(diagonal_L_tiles, n_steps[])
    end
    my_L_tiles = Matrix{datatype}[]
    sizehint!(my_L_tiles, n_steps[])
    my_L_tile_row_ranges = UnitRange{Int64}[]
    sizehint!(my_L_tile_row_ranges, n_steps[])
    my_L_tile_col_ranges = UnitRange{Int64}[]
    sizehint!(my_L_tile_col_ranges, n_steps[])
    if shared_comm_rank == 0
        for i ∈ 1:n_steps[]
            diagonal_tile = diagonal_tiles[i]
            off_diagonal_tile = @view my_tiles_for_rank[:,i]
            if diagonal_tile == -1
                push!(diagonal_L_tiles, zeros(datatype, 0, 0))
                if off_diagonal_tile == (-1, -1)
                    error("Expected off diagonal tile on root when there is no diagonal "
                          * "tile.")
                end
                row_range = get_L_tile_index_range(off_diagonal_tile[1])
                col_range = get_L_tile_index_range(off_diagonal_tile[2])
                push!(my_L_tiles, zeros(datatype, length(row_range), length(col_range)))
                push!(my_L_tile_row_ranges, row_range)
                push!(my_L_tile_col_ranges, col_range)
            else
                push!(my_L_tiles, zeros(datatype, 0, 0))
                if off_diagonal_tile != [-1, -1]
                    error("Expected no off diagonal tile on root when there is a "
                          * "diagonal tile.")
                end
                row_range = get_L_tile_index_range(diagonal_tile)
                col_range = get_L_tile_index_range(diagonal_tile)
                push!(diagonal_L_tiles, zeros(datatype, length(row_range), length(col_range)))
                push!(my_L_tile_row_ranges, row_range)
                push!(my_L_tile_col_ranges, col_range)
            end
        end
    else
        for i ∈ 1:n_steps[]
            off_diagonal_tile = @view my_tiles_for_rank[:,i]
            if off_diagonal_tile == [-1, -1]
                push!(my_L_tiles, zeros(datatype, 0, 0))
                push!(my_L_tile_row_ranges, 1:0)
                push!(my_L_tile_col_ranges, 1:0)
            else
                row_range = get_L_tile_index_range(off_diagonal_tile[1])
                col_range = get_L_tile_index_range(off_diagonal_tile[2])
                push!(my_L_tiles, zeros(datatype, length(row_range), length(col_range)))
                push!(my_L_tile_row_ranges, row_range)
                push!(my_L_tile_col_ranges, col_range)
            end
        end
    end

    # When dealing with the upper-triangular 'U' matrix, we count the tiles from the
    # bottom-right corner, so bottom-to-top for rows, and right-to-left for columns.
    function get_U_tile_index_range(itile)
        return max(m-itile*tile_size+1,1):m-(itile-1)*tile_size
    end
    diagonal_U_tiles = Matrix{datatype}[]
    if shared_comm_rank == 0
        sizehint!(diagonal_U_tiles, n_steps[])
    end
    my_U_tiles = Matrix{datatype}[]
    sizehint!(my_U_tiles, n_steps[])
    my_U_tile_row_ranges = UnitRange{Int64}[]
    sizehint!(my_U_tile_row_ranges, n_steps[])
    my_U_tile_col_ranges = UnitRange{Int64}[]
    sizehint!(my_U_tile_col_ranges, n_steps[])
    if shared_comm_rank == 0
        for i ∈ 1:n_steps[]
            diagonal_tile = diagonal_tiles[i]
            off_diagonal_tile = @view my_tiles_for_rank[:,i]
            if diagonal_tile == -1
                push!(diagonal_U_tiles, zeros(datatype, 0, 0))
                if off_diagonal_tile == [-1, -1]
                    error("Expected off diagonal tile on root when there is no diagonal "
                          * "tile.")
                end
                row_range = get_U_tile_index_range(off_diagonal_tile[1])
                col_range = get_U_tile_index_range(off_diagonal_tile[2])
                push!(my_U_tiles, zeros(datatype, length(row_range), length(col_range)))
                push!(my_U_tile_row_ranges, row_range)
                push!(my_U_tile_col_ranges, col_range)
            else
                push!(my_U_tiles, zeros(datatype, 0, 0))
                if off_diagonal_tile != [-1, -1]
                    error("Expected no off diagonal tile on root when there is a "
                          * "diagonal tile.")
                end
                row_range = get_U_tile_index_range(diagonal_tile)
                col_range = get_U_tile_index_range(diagonal_tile)
                push!(diagonal_U_tiles, zeros(datatype, length(row_range), length(col_range)))
                push!(my_U_tile_row_ranges, row_range)
                push!(my_U_tile_col_ranges, col_range)
            end
        end
    else
        for i ∈ 1:n_steps[]
            off_diagonal_tile = @view my_tiles_for_rank[:,i]
            if off_diagonal_tile == [-1, -1]
                push!(my_U_tiles, zeros(datatype, 0, 0))
                push!(my_U_tile_row_ranges, 1:0)
                push!(my_U_tile_col_ranges, 1:0)
            else
                row_range = get_U_tile_index_range(off_diagonal_tile[1])
                col_range = get_U_tile_index_range(off_diagonal_tile[2])
                push!(my_U_tiles, zeros(datatype, length(row_range), length(col_range)))
                push!(my_U_tile_row_ranges, row_range)
                push!(my_U_tile_col_ranges, col_range)
            end
        end
    end

    A_lu =  DenseLU(m, n, factors, row_permutation, diagonal_L_tiles, my_L_tiles,
                    my_L_tile_row_ranges, my_L_tile_col_ranges, diagonal_U_tiles,
                    my_U_tiles, my_U_tile_row_ranges, my_U_tile_col_ranges, vec_buffer1,
                    vec_buffer2, tile_size, n_tiles, shared_comm, shared_comm_rank,
                    shared_comm_size, synchronize_shared)

    if !skip_factorization
        lu!(A_lu, A)
    end

    synchronize_shared()

    return A_lu
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    factors = A_lu.factors
    diagonal_L_tiles = A_lu.diagonal_L_tiles
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_row_ranges = A_lu.my_L_tile_row_ranges
    my_L_tile_col_ranges = A_lu.my_L_tile_col_ranges
    diagonal_U_tiles = A_lu.diagonal_U_tiles
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_row_ranges = A_lu.my_U_tile_row_ranges
    my_U_tile_col_ranges = A_lu.my_U_tile_col_ranges
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    if A_lu.shared_comm_rank == 0
        # Factorize in serial for now. Could look at implementing a parallel version of
        # this later. Could maybe borrow algorithms from
        # https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl/ ?
        factorization = lu!(A)

        A_lu.factors .= factorization.factors
        A_lu.row_permutation .= factorization.p
    end
    synchronize_shared()

    if shared_comm_rank == 0
        for (diagonal_tile, off_diagonal_tile, row_range, col_range) ∈
                zip(diagonal_L_tiles, my_L_tiles, my_L_tile_row_ranges,
                    my_L_tile_col_ranges)
            if isempty(diagonal_tile)
                @views off_diagonal_tile .= factors[row_range, col_range]
            else
                @views diagonal_tile .= factors[row_range, col_range]
            end
        end
    else
        for (off_diagonal_tile, row_range, col_range) ∈
                zip(my_L_tiles, my_L_tile_row_ranges, my_L_tile_col_ranges)
            @views off_diagonal_tile .= factors[row_range, col_range]
        end
    end

    if shared_comm_rank == 0
        for (diagonal_tile, off_diagonal_tile, row_range, col_range) ∈
                zip(diagonal_U_tiles, my_U_tiles, my_U_tile_row_ranges,
                    my_U_tile_col_ranges)
            if isempty(diagonal_tile)
                @views off_diagonal_tile .= factors[row_range, col_range]
            else
                @views diagonal_tile .= factors[row_range, col_range]
            end
        end
    else
        for (off_diagonal_tile, row_range, col_range) ∈
                zip(my_U_tiles, my_U_tile_row_ranges, my_U_tile_col_ranges)
            @views off_diagonal_tile .= factors[row_range, col_range]
        end
    end

    return A_lu
end

function ldiv!(A_lu::DenseLU{T}, b::AbstractVector{T}) where T
    return ldiv!(b, A_lu, b)
end

function ldiv!(x::AbstractVector{T}, A_lu::DenseLU{T}, b::AbstractVector{T}) where T
    row_permutation = A_lu.row_permutation
    b_permuted = A_lu.vec_buffer1
    y = A_lu.vec_buffer2
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    # Permute the RHS, storing in buffer2. This accounts for 'row permutations' that were
    # generated/used for 'pivoting' when the L and U factors were computed.
    if shared_comm_rank == 0
        # Could parallelise this?
        @views b_permuted .= b[row_permutation]
    end

    L_solve!(y, A_lu, b_permuted)
    U_solve!(x, A_lu, y)

    synchronize_shared()

    return x
end

function L_solve!(y, A_lu::DenseLU{T}, b) where T
    diagonal_L_tiles = A_lu.diagonal_L_tiles
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_row_ranges = A_lu.my_L_tile_row_ranges
    my_L_tile_col_ranges = A_lu.my_L_tile_col_ranges
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    for itask ∈ 1:length(my_L_tiles)
        row_range = my_L_tile_row_ranges[itask]
        col_range = my_L_tile_col_ranges[itask]
        if shared_comm_rank == 0
            diagonal_tile = diagonal_L_tiles[itask]
            if isempty(diagonal_tile)
                @views gemm!('N', 'N', -one(T), my_L_tiles[itask], y[col_range], one(T), b[row_range])
            else
                # shared_comm_rank=0 always wrote to b[tile_range] on the previous step, so no
                # need to synchronize before this calculation.
                @views y[col_range] .= b[col_range]
                @views trsv!('L', 'N', 'U', diagonal_tile, y[col_range])
            end
        else
            tile = my_L_tiles[itask]
            if !isempty(tile)
                @views gemm!('N', 'N', -one(T), tile, y[col_range], one(T), b[row_range])
            end
        end

        # Synchronize to avoid race conditions.
        synchronize_shared()
    end

    return nothing
end

function U_solve!(x, A_lu::DenseLU{T}, y) where T
    diagonal_U_tiles = A_lu.diagonal_U_tiles
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_row_ranges = A_lu.my_U_tile_row_ranges
    my_U_tile_col_ranges = A_lu.my_U_tile_col_ranges
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    for itask ∈ 1:length(my_U_tiles)
        row_range = my_U_tile_row_ranges[itask]
        col_range = my_U_tile_col_ranges[itask]
        if shared_comm_rank == 0
            diagonal_tile = diagonal_U_tiles[itask]
            if isempty(diagonal_tile)
                @views gemm!('N', 'N', -one(T), my_U_tiles[itask], x[col_range], one(T), y[row_range])
            else
                # shared_comm_rank=0 always wrote to b[tile_range] on the previous step, so no
                # need to synchronize before this calculation.
                @views x[col_range] .= y[col_range]
                @views trsv!('U', 'N', 'N', diagonal_tile, x[col_range])
            end
        else
            tile = my_U_tiles[itask]
            if !isempty(tile)
                @views gemm!('N', 'N', -one(T), tile, x[col_range], one(T), y[row_range])
            end
        end

        # Synchronize to avoid race conditions.
        synchronize_shared()
    end

    return nothing
end

end # module DenseLUs
