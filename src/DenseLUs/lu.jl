function setup_lu(m::Int64, n::Int64, shared_comm_rank::Int64,
                  allocate_shared_float::Ff) where Ff
    factors = allocate_shared_float(m, n)

    if shared_comm_rank == 0
        row_permutation = zeros(Int64, m)
    else
        row_permutation = zeros(Int64, 0)
    end

    return (; factors, row_permutation)
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    factors = A_lu.factors
    row_permutation = A_lu.row_permutation
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_row_ranges = A_lu.my_L_tile_row_ranges
    my_L_tile_col_ranges = A_lu.my_L_tile_col_ranges
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_row_ranges = A_lu.my_U_tile_row_ranges
    my_U_tile_col_ranges = A_lu.my_U_tile_col_ranges
    distributed_comm = A_lu.distributed_comm
    synchronize_shared = A_lu.synchronize_shared
    check_lu = A_lu.check_lu

    if A_lu.is_root
        # Factorize in serial for now. Could look at implementing a parallel version of
        # this later. Could maybe borrow algorithms from
        # https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl/ ?
        factorization = lu!(A; check=check_lu)

        factors .= factorization.factors
        row_permutation .= factorization.p
    end
    if A_lu.shared_comm_rank == 0
        req1 = temp_Ibcast!(factors, distributed_comm; root=0)
        req2 = temp_Ibcast!(row_permutation, distributed_comm; root=0)
        MPI.Waitall([req1, req2])
    end
    synchronize_shared()

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

    return A_lu
end
