module DenseLUs

export DenseLU, dense_lu

using LinearAlgebra
using LinearAlgebra.BLAS: trsv!, gemm!
using MPI
using StatsBase: countmap

import LinearAlgebra: lu!, ldiv!

@kwdef struct DenseLU{T,Tmat,Tvec,Tintvec,Tsync}
    m::Int64
    n::Int64
    factors::Tmat
    row_permutation::Vector{Int64}
    section_K::Int64
    section_L::Int64
    section_k::Int64
    section_l::Int64
    section_height::Int64
    section_width::Int64
    factorization_matrix_parts::Matrix{Transpose{T,Matrix{T}}}
    factorization_matrix_parts_row_ranges::Vector{UnitRange{Int64}}
    factorization_matrix_parts_col_ranges::Vector{UnitRange{Int64}}
    factorization_tile_size::Int64
    factorization_n_tiles::Int64
    first_pivot_section_k::Int64
    first_row_with_diagonal::Int64
    first_pivoting_buffers::Array{T,3}
    first_pivoting_row_index_buffers::Matrix{Int64}
    pivoting_buffers::Array{T,3}
    pivoting_row_index_buffers::Matrix{Int64}
    pivot_send_requests::Vector{MPI.Request}
    pivot_recv_requests::Vector{MPI.Request}
    my_L_tiles::Array{T,3}
    my_L_tile_row_ranges::Vector{UnitRange{Int64}}
    my_L_tile_col_ranges::Vector{UnitRange{Int64}}
    L_receive_requests::Vector{MPI.Request}
    L_send_requests::Vector{MPI.Request}
    my_U_tiles::Array{T,3}
    my_U_tile_row_ranges::Vector{UnitRange{Int64}}
    my_U_tile_col_ranges::Vector{UnitRange{Int64}}
    U_receive_requests::Vector{MPI.Request}
    U_send_requests::Vector{MPI.Request}
    diagonal_indices::Vector{Int64}
    new_column_triggers::Matrix{Int64}
    step_needs_synchronize_this_block::Tintvec
    vec_buffer1::Tvec
    vec_buffer2::Tvec
    L_rhs_update_buffer::Tvec
    U_rhs_update_buffer::Tvec
    tile_size::Int64
    n_tiles::Int64
    comm::MPI.Comm
    comm_rank::Int64
    comm_size::Int64
    shared_comm::MPI.Comm
    shared_comm_rank::Int64
    shared_comm_size::Int64
    distributed_comm::MPI.Comm
    distributed_comm_rank::Int64
    distributed_comm_size::Int64
    is_root::Bool
    synchronize_shared::Tsync
    check_lu::Bool
end

function dense_lu(A::AbstractMatrix, tile_size::Int64, comm::MPI.Comm,
                  shared_comm::MPI.Comm, distributed_comm::MPI.Comm,
                  allocate_shared_float::Function, allocate_shared_int::Function;
                  synchronize_shared=nothing, skip_factorization=false, check_lu=true)
    datatype = eltype(A)

    if synchronize_shared === nothing
        synchronize_shared = ()->MPI.Barrier(shared_comm)
    end

    m, n = size(A)
    if m != n
        error("Non-square matrices not supported in DenseLU. Got ($m,$n).")
    end

    comm_rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    shared_comm_rank = MPI.Comm_rank(shared_comm)
    shared_comm_size = MPI.Comm_size(shared_comm)

    # distributed comm is only needed on the root process of each shared-memory block.
    if shared_comm_rank == 0
        distributed_comm_rank = MPI.Comm_rank(distributed_comm)
        distributed_comm_size = MPI.Comm_size(distributed_comm)
    else
        # These values should not be used/required except on shared_comm_rank=0.
        distributed_comm_rank = -1
        distributed_comm_size = -1
    end
    is_root = (shared_comm_rank == 0 && distributed_comm_rank == 0)

    # setup_lu and setup_ldiv both return NamedTuples. All the entries in both those
    # NamedTuples are fields of the DenseLU struct, which we splat into the DenseLU
    # constructor to avoid having to type out long lists of variable names repeatedly.
    lu_variables =
        setup_lu(m, n, shared_comm_rank, shared_comm_size, distributed_comm_rank,
                 distributed_comm_size, allocate_shared_float)

    ldiv_variables =
        setup_ldiv(m, datatype, tile_size, shared_comm, shared_comm_size,
                   shared_comm_rank, distributed_comm, distributed_comm_size,
                   distributed_comm_rank, is_root, allocate_shared_float,
                   allocate_shared_int)

    A_lu =  DenseLU(; m, n, tile_size, comm, comm_rank, comm_size, shared_comm,
                    shared_comm_rank, shared_comm_size, distributed_comm,
                    distributed_comm_rank, distributed_comm_size, is_root,
                    synchronize_shared, check_lu, lu_variables..., ldiv_variables...)

    if !skip_factorization
        lu!(A_lu, A)
    end

    synchronize_shared()

    return A_lu
end

include("lu.jl")

include("ldiv.jl")

end # module DenseLUs
