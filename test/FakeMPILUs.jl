"""
Inefficient distributed-MPI LU factorization, for testing.

Requires matrix to be passed on the root process. Just gathers the RHS vector to the root
process, solves there, and scatters back.
"""
module FakeMPILUs

export FakeMPILU

import Base: size
using LinearAlgebra
import LinearAlgebra: ldiv!, lu!
using MPI

mutable struct FakeMPILU{T,TLU}
    Alu::TLU
    n::Int64
    m::Int64
    rhs_buffer::Vector{T}
    counts::Vector{Int64}
    comm::MPI.Comm
    rank::Cint

    function FakeMPILU(A::Union{AbstractMatrix,Nothing}, local_vector_size;
                       comm::MPI.Comm=MPI.COMM_WORLD, data_type::DataType=Float64)

        nproc = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        if A === nothing && rank == 0
            error("Must pass `A` on root process, got `nothing`")
        elseif A !== nothing && rank != 0
            println("A was passed on non-root process, ignoring.")
        end

        if rank == 0
            if size(A, 1) != size(A, 2)
                error("A must be square, got $(size(A)).")
            end
            sendbuf = Ref(local_vector_size)
            counts = zeros(Int64, nproc)
            MPI.Gather!(sendbuf, counts, comm; root=0)
            total_size = sum(counts)
            if total_size != size(A, 1)
                error("Values of local_vector_size ($counts) that were passed do not add "
                      * "up to the matrix size ($(size(A, 1))).")
            end

            Alu = lu(A)
            rhs_buffer = zeros(data_type, total_size)

            n, m = size(A)
            MPI.bcast((n,m), 0, comm)
        else
            sendbuf = Ref(local_vector_size)
            MPI.Gather!(sendbuf, nothing, comm; root=0)

            Alu = nothing
            rhs_buffer = zeros(data_type, 0)
            counts = Int64[]
            (n, m) = MPI.bcast(nothing, 0, comm)
        end

        return new{data_type,typeof(Alu)}(Alu, n, m, rhs_buffer, counts, comm, rank)
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

function lu!(Alu::FakeMPILU, A::Union{AbstractMatrix,Nothing})
    rank = Alu.rank
    if A === nothing && rank == 0
        error("Must pass `A` on root process, got `nothing`")
    elseif rank == 0
        Alu.Alu = lu(A)
    elseif A !== nothing
        println("A was passed on non-root process, ignoring.")
    end
    return Alu
end

function ldiv!(x::AbstractVector, Alu::FakeMPILU, b::AbstractVector)
    if size(x) != size(b)
        error("x and b should have the same size")
    end
    comm = Alu.comm
    rank = Alu.rank
    counts = Alu.counts
    if rank == 0
        rhs_buffer = VBuffer(Alu.rhs_buffer, counts)
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
