module DenseLUs

export dense_lu

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
    my_L_tile_ranges::Vector{UnitRange{Int64}}
    diagonal_U_tiles::Vector{Matrix{T}}
    my_U_tiles::Vector{Matrix{T}}
    my_U_tile_ranges::Vector{UnitRange{Int64}}
    vec_buffer1::Tvec
    vec_buffer2::Tvec
    end_buffer::Vector{T}
    tile_size::Int64
    n_tiles::Int64
    last_tile_size::Int64
    shared_comm::MPI.Comm
    shared_comm_rank::Int64
    shared_comm_size::Int64
    synchronize_shared::Tsync
end

function dense_lu(A::AbstractMatrix, tile_size::Int64, shared_comm::MPI.Comm,
                  allocate_shared_float::Function, allocate_shared_int::Function;
                  synchronize_shared=nothing)
    datatype = eltype(A)

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

    # Number of full-size tiles
    n_tiles = m ÷ tile_size

    # The last tile is an arbitrary size (less than tile_size) to handle the case when
    # `m` is not a multiple of `tile_size`.
    last_tile_size = m - n_tiles * tile_size

    # Store the tiles that will be handled by this process in contiguous arrays.
    diagonal_L_tiles = Matrix{datatype}[]
    my_L_tiles = Matrix{datatype}[]
    my_L_tile_ranges = UnitRange{Int64}[]
    for tile ∈ 1:n_tiles
        if shared_comm_rank == 0
            push!(diagonal_L_tiles, zeros(datatype, tile_size, tile_size))
        end

        n_remaining_tiles = n_tiles - tile
        tiles_per_proc = max((n_remaining_tiles + shared_comm_size - 1) ÷ shared_comm_size, 1)

        # Note this my_first_tile may be greater than my_last tile for some procs when
        # there are not many tiles left to solve, in which case this process does no work.
        my_first_tile = tile + shared_comm_rank * tiles_per_proc + 1
        my_last_tile = min(tile + (shared_comm_rank + 1) * tiles_per_proc, n_tiles)

        if my_last_tile ≥ my_first_tile
            this_range = (my_first_tile-1)*tile_size+1:my_last_tile*tile_size
            push!(my_L_tile_ranges, this_range)
            push!(my_L_tiles, zeros(datatype, length(this_range), tile_size))
        else
            push!(my_L_tile_ranges, 1:0)
            push!(my_L_tiles, zeros(datatype, 0, 0))
        end
    end
    if last_tile_size > 0
        if shared_comm_rank == 0
            push!(diagonal_L_tiles, zeros(datatype, last_tile_size, last_tile_size))
        end

        # Handle tiles in reverse order here, so that the shared_comm_rank=0 process
        # handles the next-to-last tile (this allows us to skip synchronization in
        # L_solve!()).
        tiles_per_proc = max((n_tiles + shared_comm_size - 1) ÷ shared_comm_size, 1)
        my_first_tile = max(n_tiles - (shared_comm_rank + 1) * tiles_per_proc + 1, 1)
        my_last_tile = n_tiles - shared_comm_rank * tiles_per_proc
        if my_last_tile ≥ my_first_tile
            this_range = (my_first_tile-1)*tile_size+1:my_last_tile*tile_size
            push!(my_L_tile_ranges, this_range)
            push!(my_L_tiles, zeros(datatype, last_tile_size, length(this_range)))
        else
            push!(my_L_tile_ranges, 1:0)
            push!(my_L_tiles, zeros(datatype, 0, 0))
        end
    end

    diagonal_U_tiles = Matrix{datatype}[]
    my_U_tiles = Matrix{datatype}[]
    my_U_tile_ranges = UnitRange{Int64}[]
    for tile ∈ 1:n_tiles
        if shared_comm_rank == 0
            push!(diagonal_U_tiles, zeros(datatype, tile_size, tile_size))
        end

        n_remaining_tiles = n_tiles - tile
        tiles_per_proc = max((n_remaining_tiles + shared_comm_size - 1) ÷ shared_comm_size, 1)

        # Note this my_first_tile may be greater than my_last tile for some procs when
        # there are not many tiles left to solve, in which case this process does no work.
        my_first_tile = max(n_tiles - tile - (shared_comm_rank + 1) * tiles_per_proc + 1, 1)
        my_last_tile = n_tiles - tile - shared_comm_rank * tiles_per_proc

        if my_last_tile ≥ my_first_tile
            this_range = last_tile_size+(my_first_tile-1)*tile_size+1:last_tile_size+my_last_tile*tile_size
            push!(my_U_tile_ranges, this_range)
            push!(my_U_tiles, zeros(datatype, length(this_range), tile_size))
        else
            push!(my_U_tile_ranges, 1:0)
            push!(my_U_tiles, zeros(datatype, 0, 0))
        end
    end
    if last_tile_size > 0
        if shared_comm_rank == 0
            push!(diagonal_U_tiles, zeros(datatype, last_tile_size, last_tile_size))
        end

        # Handle tiles in reverse order here, so that the shared_comm_rank=0 process
        # handles the next-to-last tile (this allows us to skip synchronization in
        # U_solve!()).
        tiles_per_proc = max((n_tiles + shared_comm_size - 1) ÷ shared_comm_size, 1)
        my_first_tile = shared_comm_rank * tiles_per_proc + 1
        my_last_tile = min((shared_comm_rank + 1) * tiles_per_proc, n_tiles)
        if my_last_tile ≥ my_first_tile
            this_range = last_tile_size+(my_first_tile-1)*tile_size+1:last_tile_size+my_last_tile*tile_size
            push!(my_U_tile_ranges, this_range)
            push!(my_U_tiles, zeros(datatype, last_tile_size, length(this_range)))
        else
            push!(my_U_tile_ranges, 1:0)
            push!(my_U_tiles, zeros(datatype, 0, 0))
        end
    end

    end_buffer = zeros(datatype, last_tile_size)

    A_lu =  DenseLU(m, n, factors, row_permutation, diagonal_L_tiles, my_L_tiles,
                    my_L_tile_ranges, diagonal_U_tiles, my_U_tiles, my_U_tile_ranges,
                    vec_buffer1, vec_buffer2, end_buffer, tile_size, n_tiles,
                    last_tile_size, shared_comm, shared_comm_rank, shared_comm_size,
                    synchronize_shared)

    lu!(A_lu, A)

    synchronize_shared()

    return A_lu
end

function lu!(A_lu::DenseLU{T}, A::AbstractMatrix{T}) where T
    m = A_lu.m
    factors = A_lu.factors
    diagonal_L_tiles = A_lu.diagonal_L_tiles
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_ranges = A_lu.my_L_tile_ranges
    diagonal_U_tiles = A_lu.diagonal_U_tiles
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_ranges = A_lu.my_U_tile_ranges
    tile_size = A_lu.tile_size
    n_tiles = A_lu.n_tiles
    last_tile_size = A_lu.last_tile_size
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    if A_lu.shared_comm_rank == 0
        # Factorize in serial for now. Could look at implementing a parallel version of
        # this later.
        factorization = lu!(A)

        # The following is not the most efficient, as factorization.L and factorization.U
        # both allocate new intermediate matrices. Only actually need to copy the
        # lower-triangular elements of factorization.factors into L, and diagonal and
        # upper-triangular elements into U.  Optimise later!
        A_lu.factors .= factorization.factors
        A_lu.row_permutation .= factorization.p
    end
    synchronize_shared()

    for tile ∈ 1:n_tiles
        tile_range = (tile-1)*tile_size+1:tile*tile_size

        if shared_comm_rank == 0
            @views diagonal_L_tiles[tile] .= factors[tile_range,tile_range]
        end

        my_tiles_range = my_L_tile_ranges[tile]

        if !isempty(my_tiles_range)
            @views my_L_tiles[tile] .= factors[my_tiles_range,tile_range]
        end
    end
    if last_tile_size > 0
        # Handle remaining rows.
        tile_range = n_tiles*tile_size+1:m

        if shared_comm_rank == 0
            @views diagonal_L_tiles[end] .= factors[tile_range,tile_range]
        end

        my_tiles_range = my_L_tile_ranges[end]

        if !isempty(my_tiles_range)
            @views my_L_tiles[end] .= factors[tile_range,my_tiles_range]
        end
    end

    for tile ∈ 1:n_tiles
        tile_range = last_tile_size+(n_tiles-tile)*tile_size+1:last_tile_size+(n_tiles-tile+1)*tile_size

        if shared_comm_rank == 0
            @views diagonal_U_tiles[tile] .= factors[tile_range,tile_range]
        end

        my_tiles_range = my_U_tile_ranges[tile]

        if !isempty(my_tiles_range)
            @views my_U_tiles[tile] .= factors[my_tiles_range,tile_range]
        end
    end
    if last_tile_size > 0
        # Handle remaining rows.
        tile_range = 1:last_tile_size

        if shared_comm_rank == 0
            @views diagonal_U_tiles[end] .= factors[tile_range,tile_range]
        end

        my_tiles_range = my_U_tile_ranges[end]

        if !isempty(my_tiles_range)
            @views my_U_tiles[end] .= factors[tile_range,my_tiles_range]
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
    m = A_lu.m
    diagonal_L_tiles = A_lu.diagonal_L_tiles
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_ranges = A_lu.my_L_tile_ranges
    tile_size = A_lu.tile_size
    n_tiles = A_lu.n_tiles
    last_tile_size = A_lu.last_tile_size
    end_buffer = A_lu.end_buffer
    shared_comm_rank = A_lu.shared_comm_rank
    shared_comm_size = A_lu.shared_comm_size
    shared_comm = A_lu.shared_comm
    synchronize_shared = A_lu.synchronize_shared

    for tile ∈ 1:n_tiles
        tile_range = (tile-1)*tile_size+1:tile*tile_size
        if shared_comm_rank == 0
            # shared_comm_rank=0 always wrote to b[tile_range] on the previous step, so no
            # need to synchronize before this calculation.
            @views y[tile_range] .= b[tile_range]
            @views trsv!('L', 'N', 'U', diagonal_L_tiles[tile], y[tile_range])
        end

        # About to read the just-calculated values of `y` on all processes, so synchronize
        # first.
        synchronize_shared()

        my_tiles_range = my_L_tile_ranges[tile]

        if !isempty(my_tiles_range)
            @views gemm!('N', 'N', -one(T), my_L_tiles[tile], y[tile_range], one(T), b[my_tiles_range])
        end
    end

    if last_tile_size > 0
        # Handle remaining rows.
        tile_range = n_tiles*tile_size+1:m

        my_tiles_range = my_L_tile_ranges[end]

        if shared_comm_rank == 0
            # shared_comm_rank=0 updates `b` directly.
            # y[my_tiles_range] on this rank includes the final values of `y` that were
            # updated above, so no need to synchronize before this.
            @views gemm!('N', 'N', -one(T), my_L_tiles[end], y[my_tiles_range], one(T), b[tile_range])
            @views MPI.Reduce!(b[tile_range], +, shared_comm; root=0)
        elseif !isempty(my_tiles_range)
            end_buffer .= 0.0
            @views gemm!('N', 'N', -one(T), my_L_tiles[end], y[my_tiles_range], zero(T), end_buffer)
            @views MPI.Reduce!(end_buffer, +, shared_comm; root=0)
        else
            # end_buffer is already zero'ed.
            # This is slightly inefficient, but simpler than creating a special
            # communicator excluding the processes that end up in this branch.
            @views MPI.Reduce!(end_buffer, +, shared_comm; root=0)
        end

        if shared_comm_rank == 0
            @views y[tile_range] .= b[tile_range]
            @views trsv!('L', 'N', 'U', diagonal_L_tiles[end], y[tile_range])
        end
    end

    return nothing
end

function U_solve!(x, A_lu::DenseLU{T}, y) where T
    m = A_lu.m
    diagonal_U_tiles = A_lu.diagonal_U_tiles
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_ranges = A_lu.my_U_tile_ranges
    tile_size = A_lu.tile_size
    n_tiles = A_lu.n_tiles
    last_tile_size = A_lu.last_tile_size
    end_buffer = A_lu.end_buffer
    shared_comm_rank = A_lu.shared_comm_rank
    shared_comm_size = A_lu.shared_comm_size
    shared_comm = A_lu.shared_comm
    synchronize_shared = A_lu.synchronize_shared

    for tile ∈ 1:n_tiles
        tile_range = last_tile_size+(n_tiles-tile)*tile_size+1:last_tile_size+(n_tiles-tile+1)*tile_size
        if shared_comm_rank == 0
            # shared_comm_rank=0 always wrote to y[tile_range] on the previous step, so no
            # need to synchronize before this calculation.
            @views x[tile_range] .= y[tile_range]
            @views trsv!('U', 'N', 'N', diagonal_U_tiles[tile], x[tile_range])
        end

        # About to read the just-calculated values of `x` on all processes, so synchronize
        # first.
        synchronize_shared()

        my_tiles_range = my_U_tile_ranges[tile]

        if !isempty(my_tiles_range)
            @views gemm!('N', 'N', -one(T), my_U_tiles[tile], x[tile_range], one(T), y[my_tiles_range])
        end
    end

    if last_tile_size > 0
        # Handle remaining rows.
        tile_range = 1:last_tile_size

        my_tiles_range = my_U_tile_ranges[end]

        if shared_comm_rank == 0
            # shared_comm_rank=0 updates `y` directly.
            # x[my_tiles_range] on this rank includes the final values of `x` that were
            # updated above, so no need to synchronize before this.
            @views gemm!('N', 'N', -one(T), my_U_tiles[end], x[my_tiles_range], one(T), y[tile_range])
            @views MPI.Reduce!(y[tile_range], +, shared_comm; root=0)
        elseif !isempty(my_tiles_range)
            end_buffer .= 0.0
            @views gemm!('N', 'N', -one(T), my_U_tiles[end], x[my_tiles_range], zero(T), end_buffer)
            @views MPI.Reduce!(end_buffer, +, shared_comm; root=0)
        else
            # end_buffer is already zero'ed.
            # This is slightly inefficient, but simpler than creating a special
            # communicator excluding the processes that end up in this branch.
            @views MPI.Reduce!(end_buffer, +, shared_comm; root=0)
        end

        if shared_comm_rank == 0
            @views x[tile_range] .= y[tile_range]
            @views trsv!('U', 'N', 'N', diagonal_U_tiles[end], x[tile_range])
        end
    end

    return nothing
end

end # module DenseLUs
