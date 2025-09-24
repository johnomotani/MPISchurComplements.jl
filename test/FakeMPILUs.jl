"""
Inefficient distributed-MPI LU factorization, for testing.

Requires local rows of matrix to be passed on each process. Just gathers the matrix and
RHS vector to the root process, solves there, and scatters back.
"""
module FakeMPILUs

export FakeMPILU

import Base: size
using LinearAlgebra
import LinearAlgebra: ldiv!, lu!
using MPI
using SparseArrays

function gather_A(A_local, row_counts, A_global_column_range, comm)
    nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    if rank == 0
        total_size = sum(row_counts)
        A = zeros(eltype(A_local), total_size, total_size)

        row_minind = 1
        row_maxind = row_counts[1]
        A[row_minind:row_maxind,A_global_column_range] .= A_local

        row_minind = row_maxind + 1
        for iproc ∈ 2:nproc
            row_maxind = row_minind - 1 + row_counts[iproc]
            col_minind = MPI.Recv(Int64, comm; source=iproc-1)
            col_maxind = MPI.Recv(Int64, comm; source=iproc-1)
            @views MPI.Recv!(A[row_minind:row_maxind, col_minind:col_maxind], comm;
                             source=iproc-1)
            row_minind = row_maxind + 1
        end
    else
        MPI.Send(A_global_column_range.start, comm; dest=0)
        MPI.Send(A_global_column_range.stop, comm; dest=0)
        MPI.Send(A_local, comm; dest=0)
        A = nothing
    end

    return A
end

mutable struct FakeMPILU{T,TLU}
    Alu::TLU
    n::Int64
    m::Int64
    rhs_buffer::Vector{T}
    row_counts::Vector{Int64}
    A_global_column_range::UnitRange{Int64}
    comm::MPI.Comm
    rank::Cint
    sparse::Bool

    function FakeMPILU(A_local::AbstractMatrix,
                       A_global_column_range::Union{UnitRange{Int64},Nothing}=nothing;
                       comm::MPI.Comm=MPI.COMM_WORLD, sparse=false)

        data_type = eltype(A_local)
        nproc = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        if A_global_column_range === nothing
            # All columns were present in A_local, not just a subset.
            A_global_column_range = 1:size(A_local, 2)
        end
        local_vector_size = size(A_local, 1)

        if rank == 0
            sendbuf = Ref(local_vector_size)
            row_counts = zeros(Int64, nproc)
            MPI.Gather!(sendbuf, row_counts, comm; root=0)
            # total_size is given by the maximum column index in any chunk of A.
            total_size = MPI.Reduce(A_global_column_range.stop, max, comm; root=0)
            if sum(row_counts) != total_size
                error("Values of local_vector_size ($row_counts) that were passed do not add "
                      * "up to the matrix size ($total_size).")
            end

            A = gather_A(A_local, row_counts, A_global_column_range, comm)

            if sparse
                Alu = lu(sparse(A))
            else
                Alu = lu(A)
            end

            rhs_buffer = zeros(data_type, total_size)

            n, m = size(A)
            MPI.bcast((n,m), 0, comm)
        else
            sendbuf = Ref(local_vector_size)
            MPI.Gather!(sendbuf, nothing, comm; root=0)
            MPI.Reduce(A_global_column_range.stop, max, comm; root=0)

            gather_A(A_local, nothing, A_global_column_range, comm)

            Alu = nothing
            rhs_buffer = zeros(data_type, 0)
            row_counts = Int64[]
            (n, m) = MPI.bcast(nothing, 0, comm)
        end

        return new{data_type,typeof(Alu)}(Alu, n, m, rhs_buffer, row_counts,
                                          A_global_column_range, comm, rank, sparse)
    end
end

size(Alu::FakeMPILU) = (Alu.n, Alu.m)
function size(Alu::FakeMPILU, d::Integer)
    if d == 1
        return Alu.n
    elseif d == 2
        return Alu.m
    elseif d > 2
        return 1
    else
        error("arraysize: dimension out of range")
    end
end

function lu!(Alu::FakeMPILU, A_local::AbstractMatrix)
    A = gather_A(A_local, Alu.row_counts, Alu.A_global_column_range, Alu.comm)
    if Alu.rank == 0
        if Alu.sparse
            try
                lu!(Alu.Alu, sparse(A))
            catch e
                if !isa(e, ArgumentError)
                    rethrow(e)
                end
                println("FakeMPILU: Sparsity pattern of matrix changed, rebuilding "
                        * " LU from scratch")
                Alu.Alu = lu(sparse(A))
            end
        else
            Alu.Alu = lu(A)
        end
    end
    return Alu
end

function ldiv!(x::AbstractVector, Alu::FakeMPILU, b::AbstractVector)
    if size(x) != size(b)
        error("x and b should have the same size")
    end
    comm = Alu.comm
    rank = Alu.rank
    row_counts = Alu.row_counts
    if rank == 0
        rhs_buffer = VBuffer(Alu.rhs_buffer, row_counts)
    else
        rhs_buffer = nothing
    end

    MPI.Gatherv!(b, rhs_buffer, comm; root=0)

    if rank == 0
        ldiv!(Alu.Alu, Alu.rhs_buffer)
    end

    MPI.Scatterv!(rhs_buffer, x, comm; root=0)

    return x
end

function ldiv!(x::AbstractMatrix, Alu::FakeMPILU, b::AbstractMatrix)
    if size(x) != size(b)
        error("x and b should have the same size")
    end
    # Just do one at a time, because if we made rhs_buffer a Matrix, we wouldn't be able
    # to resize it to adapt to the size of b anyway.
    for i ∈ 1:size(b, 2)
        @views ldiv!(x[:,i], Alu, b[:,i])
    end
    return x
end

# Special version to fill in if b===nothing is passed, indicating that b is all-zeros. A
# non-fake implementation might make use of this information to increase efficiency.
function ldiv!(x, Alu::FakeMPILU, b::Nothing)
    b = zeros(eltype(x), size(x))
    return ldiv!(x, Alu, b)
end

function ldiv!(Alu::FakeMPILU, b)
    # Is fine to pass `b` in both arguments, because the input is copied out by the
    # MPI.Gatherv!, and output copied back by the MPI.Scatterv!.
    return ldiv!(b, Alu, b)
end

end # module FakeMPILUs

using .FakeMPILUs
