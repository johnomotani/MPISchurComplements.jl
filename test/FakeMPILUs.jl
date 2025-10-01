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

function gather_A(A_local, total_size, global_row_range, global_column_range, comm)
    nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    if rank == 0
        A = zeros(eltype(A_local), total_size, total_size)

        A[global_row_range,global_column_range] .+= A_local

        for iproc ∈ 2:nproc
            row_minind = MPI.Recv(Int64, comm; source=iproc-1)
            row_maxind = MPI.Recv(Int64, comm; source=iproc-1)
            col_minind = MPI.Recv(Int64, comm; source=iproc-1)
            col_maxind = MPI.Recv(Int64, comm; source=iproc-1)
            A_chunk = zeros(eltype(A_local), row_maxind - row_minind + 1,
                            col_maxind - col_minind + 1)
            MPI.Recv!(A_chunk, comm; source=iproc-1)
            @views A[row_minind:row_maxind, col_minind:col_maxind] .+= A_chunk
        end
    else
        MPI.Send(global_row_range.start, comm; dest=0)
        MPI.Send(global_row_range.stop, comm; dest=0)
        MPI.Send(global_column_range.start, comm; dest=0)
        MPI.Send(global_column_range.stop, comm; dest=0)
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
    global_row_range::UnitRange{Int64}
    global_column_range::UnitRange{Int64}
    global_vector_ranges::Vector{UnitRange{Int64}}
    comm::MPI.Comm
    rank::Cint
    nproc::Cint
    shared_rank::Cint
    use_sparse::Bool

    function FakeMPILU(A_local::AbstractMatrix,
                       global_row_range::Union{UnitRange{Int64},Nothing}=nothing,
                       global_column_range::Union{UnitRange{Int64},Nothing}=nothing;
                       comm::MPI.Comm=MPI.COMM_WORLD,
                       shared_comm::Union{MPI.Comm,Nothing}=nothing, use_sparse=false)

        if shared_comm === nothing
            shared_comm = MPI.COMM_NULL
        end

        data_type = eltype(A_local)
        nproc = (comm == MPI.COMM_NULL ? Cint(-1) : MPI.Comm_size(comm))
        rank = (comm == MPI.COMM_NULL ? Cint(-1) : MPI.Comm_rank(comm))
        shared_rank = shared_comm == MPI.COMM_NULL ? Cint(0) : MPI.Comm_rank(shared_comm)
        if global_row_range === nothing
            # All rows were present in A_local, not just a subset.
            global_row_range = 1:size(A_local, 1)
        end
        if global_column_range === nothing
            # All columns were present in A_local, not just a subset.
            global_column_range = 1:size(A_local, 2)
        end

        if rank == 0 && shared_rank == 0
            # total_size is given by the maximum column index in any chunk of A.
            total_size = MPI.Allreduce(global_row_range.stop, max, comm)
            total_column_size = MPI.Allreduce(global_column_range.stop, max, comm)
            if total_column_size != total_size
                error("Values of global_row_range ($global_row_range) and "
                      * "global_column_range($global_column_range) that were passed do "
                      * "not give the same total size ($total_size vs. "
                      * "$total_column_size).")
            end

            A = gather_A(A_local, total_size, global_row_range, global_column_range, comm)

            if use_sparse
                Alu = lu(sparse(A))
            else
                Alu = lu(A)
            end

            rhs_buffer = zeros(data_type, total_size)

            n = total_size
            m = total_column_size

            MPI.bcast((n,m), 0, comm)
            shared_comm != MPI.COMM_NULL && MPI.bcast((n,m), 0, shared_comm)

            global_vector_ranges = [global_row_range]
            for iproc ∈ 1:nproc-1
                iproc_imin = MPI.Recv(Int64, comm; source=iproc)
                iproc_imax = MPI.Recv(Int64, comm; source=iproc)
                push!(global_vector_ranges, iproc_imin:iproc_imax)
            end
        elseif shared_rank == 0
            total_size = MPI.Allreduce(global_row_range.stop, max, comm)
            total_column_size = MPI.Allreduce(global_column_range.stop, max, comm)
            if total_column_size != total_size
                error("Values of global_row_range ($global_row_range) and "
                      * "global_column_range($global_column_range) that were passed do "
                      * "not give the same total size ($total_size vs. "
                      * "$total_column_size).")
            end

            gather_A(A_local, total_size, global_row_range, global_column_range, comm)

            Alu = nothing
            rhs_buffer = zeros(data_type, 0)

            n = total_size
            m = total_column_size

            (n, m) = MPI.bcast(nothing, 0, comm)
            shared_comm != MPI.COMM_NULL && MPI.bcast((n,m), 0, shared_comm)
            global_vector_ranges = UnitRange{Int64}[]
            MPI.Send(global_row_range.start, comm; dest=0)
            MPI.Send(global_row_range.stop, comm; dest=0)
        else
            Alu = nothing
            rhs_buffer = zeros(data_type, 0)
            (n, m) = MPI.bcast(nothing, 0, shared_comm)
            global_vector_ranges = UnitRange{Int64}[]
        end

        return new{data_type,typeof(Alu)}(Alu, n, m, rhs_buffer, global_row_range,
                                          global_column_range, global_vector_ranges, comm,
                                          rank, nproc, shared_rank, use_sparse)
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
    if Alu.shared_rank != 0
        # Non-root shared-memory block ranks do not participate in fake MPI.
        return Alu
    end
    A = gather_A(A_local, Alu.n, Alu.global_row_range, Alu.global_column_range, Alu.comm)
    if Alu.rank == 0
        if Alu.use_sparse
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
    if Alu.shared_rank != 0
        # Non-root shared-memory block ranks do not participate in fake MPI.
        return x
    end
    comm = Alu.comm
    rank = Alu.rank
    nproc = Alu.nproc

    if rank == 0
        rhs_buffer = Alu.rhs_buffer

        rhs_buffer[Alu.global_row_range] .= b
        last_row_max = Alu.global_row_range.stop
        for iproc ∈ 1:nproc-1
            this_vector_range = Alu.global_vector_ranges[iproc+1]
            this_rhs_buffer = zeros(eltype(rhs_buffer), length(this_vector_range))
            MPI.Recv!(this_rhs_buffer, comm; source=iproc)

            # Any entries in overlapping parts of the vector are identical, so is OK to
            # overwirte them.
            rhs_buffer[this_vector_range] .= this_rhs_buffer
        end

        ldiv!(Alu.Alu, Alu.rhs_buffer)

        @views x .= rhs_buffer[Alu.global_row_range]
        for iproc ∈ 1:nproc-1
            @views MPI.Send(rhs_buffer[Alu.global_vector_ranges[iproc+1]], comm;
                            dest=iproc)
        end
    else
        MPI.Send(b, comm; dest=0)
        MPI.Recv!(x, comm; source=0)
    end

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
