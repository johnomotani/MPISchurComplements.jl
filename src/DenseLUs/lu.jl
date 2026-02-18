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
    # is not a square number) make them wider than they are tall, as this increases the
    # parallelism (although also increases the amount of communication in some stages).
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
    section_K = distributed_comm_size_factors[factor_ind]
    section_L = distributed_comm_size ÷ section_K

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
        section_row_end = min(tile_row_start - 1 + section_k * section_height, tile_row_end)
        return section_row_start:section_row_end
    end

    function get_col_range(tile_j)
        tile_col_start = (tile_j - 1) * factorization_tile_size
        tile_col_end = min(tile_j * factorization_tile_size, n)
        section_col_start = tile_row_start + (section_l - 1) * section_width
        section_col_end = min(tile_row_start - 1 + section_l * section_width, tile_col_end)
        return section_col_start:section_col_end
    end

    factorization_matrix_parts_row_ranges = [get_row_range(tile_i) for tile_i ∈ 1:factorization_n_tiles]
    factorization_matrix_parts_col_ranges = [get_col_range(tile_j) for tile_j ∈ 1:factorization_n_tiles]
    factorization_matrix_parts = [transpose(allocate_shared_float(length(col_range), length(row_range)))
                                  for row_range ∈ factorization_matrix_parts_row_ranges,
                                      col_range ∈ factorization_matrix_parts_col_ranges]

    return (; factors, row_permutation, factorization_matrix_parts,
            factorization_matrix_parts_row_ranges, factorization_matrix_parts_col_ranges,
            factorization_tile_size, factorization_n_tiles)
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
# tiles of the matrix, in the 'local buffers'. We store the 'local buffers' in
# 'transposed' arrays, so that our storage is effectively row-major, which will make
# row-based operations (e.g. swapping, or splitting by row for matrix-vector
# multiplication) more efficient.
# A quick test suggests that when copying between a transposed and a non-transposed
# matrix, it is most efficient to index in the natural way for the matrix that we are
# copying *into*.
function redistribute_matrix!(A_lu, A)
    shared_comm_rank = A_lu.shared_comm_rank
    matrix_parts = A_lu.factorization_matrix_parts
    matrix_parts_row_ranges = A_lu.factorization_matrix_parts_row_ranges
    matrix_parts_col_ranges = A_lu.factorization_matrix_parts_col_ranges
    nt = A_lu.factorization_n_tiles

    if shared_comm_rank == 0
        for j ∈ 1:nt, i ∈ 1:nt
            # Direct looping seems to be faster than broadcasting when one of the arrays
            # is a transpose.
            for (local_k, global_k) ∈ enumerate(matrix_row_ranges[i]),
                    (local_l, global_l) ∈ enumerate(matrix_col_ranges[i])
                matrix_parts[local_k,local_l] = A[global_k,global_l]
            end
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
            @views my_L_tiles[1:length(row_range),1:length(col_range),step] .= factors[row_range, col_range]
        end
    end

    for (step, (row_range, col_range)) ∈ enumerate(zip(my_U_tile_row_ranges,
                                                       my_U_tile_col_ranges))
        if !isempty(row_range)
            @views my_U_tiles[1:length(row_range),1:length(col_range),step] .= factors[row_range, col_range]
        end
    end
    return nothing
end
